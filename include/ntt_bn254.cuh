// =============================================================================
// ntt_bn254.cuh ??? BN254 Fr NTT (Number Theoretic Transform)
// Zyklop GPU prover ??? Forum ZK voting
//
// Algorithm : Stockham radix-2, self-sorting (no bit-reversal)
// Domain    : up to 2^23 elements  (log_n ??? 23)
// Field     : Fr scalar field of BN254, Montgomery representation
//
// API:
//   nttPrepare(log_n_max)          ??? precompute twiddle table (call once)
//   nttDestroy()                   ??? free twiddle table
//   ntt(d_a, log_n, inverse)       ??? in-place NTT/iNTT on d_a[2^log_n]
//   nttCosetMul(d_a, log_n, inv)   ??? multiply a[i] *= g^i  (g=7, coset shift)
//
// Twiddle table: 2^(log_n_max-1) entries, Montgomery Fr, ~128 MB for log_n_max=23
// omega_28 = 5^t mod r (snarkjs convention, verified)
//
// Requires: fp_bn254.cuh then fr_bn254.cuh included before this header
// =============================================================================
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

// Vector type for 256-bit load/store
struct __align__(32) zyklop_u256 {
    unsigned long long int x, y, z, w;
};

__device__ __forceinline__ zyklop_u256 make_zyklop_u256(unsigned long long int x, unsigned long long int y, unsigned long long int z, unsigned long long int w) {
    zyklop_u256 t;
    t.x = x; t.y = y; t.z = z; t.w = w;
    return t;
}

#define NTT_CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[NTT] CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// =============================================================================
// Global state
// =============================================================================
static uint64_t* g_ntt_twiddles  = nullptr;
static uint64_t* g_ntt_ninv = nullptr;
static uint64_t* g_ntt_buf       = nullptr;   // persistent ping-pong buffer (no cudaMalloc per call)
static int       g_ntt_log_n_max = 0;

// =============================================================================
// CPU Montgomery arithmetic (for twiddle + coset precomputation)
// =============================================================================

static const uint64_t R_SCALAR[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
static const uint64_t R2_FR_CPU[4] = {
    0x1bb8e645ae216da7ULL, 0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL, 0x0216d0b17f4e44a5ULL
};
static const uint64_t R_FR_CPU[4] = {   // 1 in Montgomery = 2^256 mod r
    0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
};
static const uint64_t FR_PRIME_CPU = 0xc2e1f593efffffffULL;

static void cpuFrMontMul(const uint64_t a[4], const uint64_t b[4], uint64_t res[4])
{
    typedef unsigned __int128 u128;
    uint64_t S[5] = {};
    for (int i = 0; i < 4; i++) {
        // t = (S[0] + a[i]*b[0]) * r'
        u128 t0 = (u128)S[0] + (u128)a[i] * b[0];
        uint64_t mm = (uint64_t)t0 * FR_PRIME_CPU;
        u128 carry = (t0 + (u128)mm * R_SCALAR[0]) >> 64;
        for (int j = 1; j < 4; j++) {
            u128 cur = (u128)S[j] + (u128)a[i] * b[j] + (u128)mm * R_SCALAR[j] + carry;
            S[j-1] = (uint64_t)cur;
            carry  = cur >> 64;
        }
        S[3] = (uint64_t)((u128)S[4] + carry);
        S[4] = (uint64_t)(((u128)S[4] + carry) >> 64);
    }
    // Conditional subtraction
    bool ge = false;
    if (S[4] > 0) {
        ge = true;
    } else {
        ge = true;
        for (int i = 3; i >= 0; i--) {
            if (S[i] > R_SCALAR[i]) { ge = true; break; }
            if (S[i] < R_SCALAR[i]) { ge = false; break; }
        }
    }
    if (ge) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            unsigned __int128 d = (u128)S[i] - R_SCALAR[i] - borrow;
            S[i] = (uint64_t)d;
            borrow = (d >> 127) & 1;
        }
    }
    res[0]=S[0]; res[1]=S[1]; res[2]=S[2]; res[3]=S[3];
}

static void cpuFrEncode(const uint64_t a[4], uint64_t out[4]) {
    cpuFrMontMul(a, R2_FR_CPU, out);
}

// =============================================================================
// CPU: compute (2^log_n)^{-1} mod r in Montgomery form
// 2^{-1} mod r = (r+1)/2  (r is odd)
// =============================================================================
static void cpuNInvMont(int log_n, uint64_t out[4]) {
    // canonical 2^{-1} mod r
    const uint64_t inv2_raw[4] = {
        0xa1f0fac9f8000001ULL, 0x9419f4243cdcb848ULL,
        0xdc2822db40c0ac2eULL, 0x183227397098d014ULL
    };
    uint64_t inv2m[4];
    cpuFrEncode(inv2_raw, inv2m);
    // Start from 1 in Montgomery
    out[0]=R_FR_CPU[0]; out[1]=R_FR_CPU[1]; out[2]=R_FR_CPU[2]; out[3]=R_FR_CPU[3];
    for (int i = 0; i < log_n; i++) {
        cpuFrMontMul(out, inv2m, out);
    }
}

// =============================================================================
// nttPrepare ??? build twiddle table on device
//
// twiddles[j] = omega_N^j  (Montgomery), j in [0, N/2)
// where N = 2^log_n_max, omega_N = omega_28^(2^(28-log_n_max))
//
// For a sub-domain of size 2^k, omega_k^j = twiddles[j * stride]
// where stride = 2^(log_n_max - k).
// =============================================================================
__global__ void k_encode(const uint64_t* __restrict__ in, uint64_t* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    frMontEncode(in + i*4, out + i*4);
}

__global__ void frSqr_kernel(uint64_t* a) {
    if (threadIdx.x == 0 && blockIdx.x == 0) frSqr(a, a);
}
__global__ void frMul_kernel(const uint64_t* a, const uint64_t* b, uint64_t* c) {
    if (threadIdx.x == 0 && blockIdx.x == 0) frMul(a, b, c);
}


// ?????? GPU twiddle table builder ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
// Computes twiddles[j] = omega^j in Montgomery using GPU frMul (no CPU __int128)
// 2-phase parallel twiddle build:
// Phase 1 (1 thread): fills ALL block-start entries tw[0], tw[B], tw[2B], ...
//   sequentially ??? only n_half/B multiplications needed.
// Phase 2 (parallel): block k, thread i fills tw[k*B+i] = tw[k*B] * tw[i % B]
//   where tw[i % B] was computed in phase 1 for block 0.
#define NTT_TW_BLOCK 1024
__global__ void ntt_build_twiddles_p1(uint64_t* tw, const uint64_t* omega_B, int n_half) {
    // omega_B = omega^B (stride between block starts)
    // tw[0] already set to 1. Compute tw[B], tw[2B], ... sequentially.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int n_blocks = (n_half + NTT_TW_BLOCK - 1) / NTT_TW_BLOCK;
    for (int k = 1; k < n_blocks; k++)
        frMul(tw + (k-1)*NTT_TW_BLOCK*4, omega_B, tw + k*NTT_TW_BLOCK*4);
}
// Phase 1b (1 thread): fill tw[1..B-1] for block 0
__global__ void ntt_build_twiddles_p1b(uint64_t* tw, const uint64_t* omega, int n_half) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int lim = min(n_half, NTT_TW_BLOCK);
    for (int j = 1; j < lim; j++)
        frMul(tw + (j-1)*4, omega, tw + j*4);
}
// Phase 2: block k>=1, thread i: tw[k*B+i] = tw[k*B] * tw[i]
__global__ void ntt_build_twiddles_p2(uint64_t* tw, int n_half) {
    int k = blockIdx.x + 1;
    int i = threadIdx.x;
    int idx = k * NTT_TW_BLOCK + i;
    if (idx >= n_half) return;
    frMul(tw + k*NTT_TW_BLOCK*4, tw + i*4, tw + idx*4);
}

inline void nttPrepare(int log_n_max = 23)
{
    if (g_ntt_twiddles) return;
    g_ntt_log_n_max = log_n_max;

    const uint64_t omega28_raw[4] = {
        0x9bd61b6e725b19f0ULL, 0x402d111e41112ed4ULL,
        0x00e0a7eb8ef62abcULL, 0x2a3c09f0a58a7e85ULL
    };

    uint64_t *d_w;
    NTT_CUDA_CHECK(cudaMalloc(&d_w, 32));

    // --- CORRECTION : Encodage Montgomery AVANT le squaring ---
    {
        uint64_t *d_tmp;
        NTT_CUDA_CHECK(cudaMalloc(&d_tmp, 32));
        NTT_CUDA_CHECK(cudaMemcpy(d_tmp, omega28_raw, 32, cudaMemcpyHostToDevice));
        
        // On transforme la valeur brute en Montgomery sur GPU
        k_encode<<<1,1>>>(d_tmp, d_w, 1);
        NTT_CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();
        cudaFree(d_tmp);
    }

    // Square (28-log_n_max) times sur la valeur DEJA en Montgomery
    for (int i = 0; i < (28 - log_n_max); i++) {
        frSqr_kernel<<<1,1>>>(d_w);
        NTT_CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();
    }
    // ---------------------------------------------------------

    size_t n_half = 1ULL << (log_n_max - 1);
    NTT_CUDA_CHECK(cudaMalloc(&g_ntt_twiddles, n_half * 32));
    NTT_CUDA_CHECK(cudaMalloc(&g_ntt_buf, (1ULL << log_n_max) * 32));
    NTT_CUDA_CHECK(cudaMemcpy(g_ntt_twiddles, R_FR_CPU, 32, cudaMemcpyHostToDevice));

    ntt_build_twiddles_p1b<<<1,1>>>(g_ntt_twiddles, d_w, (int)n_half);
    NTT_CUDA_CHECK(cudaGetLastError()); cudaDeviceSynchronize();

    if ((int)n_half > NTT_TW_BLOCK) {
        uint64_t *d_omB;
        NTT_CUDA_CHECK(cudaMalloc(&d_omB, 32));
        frMul_kernel<<<1,1>>>(g_ntt_twiddles + (NTT_TW_BLOCK-1)*4, d_w, d_omB);
        cudaDeviceSynchronize();
        ntt_build_twiddles_p1<<<1,1>>>(g_ntt_twiddles, d_omB, (int)n_half);
        cudaDeviceSynchronize();
        int nb = ((int)n_half + NTT_TW_BLOCK - 1) / NTT_TW_BLOCK - 1;
        ntt_build_twiddles_p2<<<nb, NTT_TW_BLOCK>>>(g_ntt_twiddles, (int)n_half);
        cudaDeviceSynchronize();
        cudaFree(d_omB);
    }
    cudaFree(d_w);
	
	// --- NOUVEAU : Cache des inverses (n^-1) ---
    if (!g_ntt_ninv) {
        NTT_CUDA_CHECK(cudaMalloc(&g_ntt_ninv, (size_t)(g_ntt_log_n_max + 1) * 4 * sizeof(uint64_t)));

        std::vector<uint64_t> h_ninv((size_t)(g_ntt_log_n_max + 1) * 4, 0);
        for (int ln = 0; ln <= g_ntt_log_n_max; ++ln) {
            uint64_t tmp[4];
            cpuNInvMont(ln, tmp);
            memcpy(&h_ninv[(size_t)ln * 4], tmp, 4 * sizeof(uint64_t));
        }

        NTT_CUDA_CHECK(cudaMemcpy(
            g_ntt_ninv,
            h_ninv.data(),
            (size_t)(g_ntt_log_n_max + 1) * 4 * sizeof(uint64_t),
            cudaMemcpyHostToDevice));
    }
}

inline void nttDestroy() {
    if (g_ntt_twiddles) { cudaFree(g_ntt_twiddles); g_ntt_twiddles=nullptr; }
    if (g_ntt_buf)      { cudaFree(g_ntt_buf); g_ntt_buf=nullptr; }
	if (g_ntt_ninv) { cudaFree(g_ntt_ninv); g_ntt_ninv = nullptr; }
}

// Returns the device pointer to the twiddle table and the log_n_max value.
// Use this instead of extern declarations which don't work across translation units
// because g_ntt_twiddles/g_ntt_log_n_max are static (internal linkage).
inline uint64_t* nttGetTwiddleTable(int* out_log_n_max) {
    if (out_log_n_max) *out_log_n_max = g_ntt_log_n_max;
    return g_ntt_twiddles;
}
// log_n must be <= g_ntt_log_n_max. Call after nttPrepare(>= log_n).
// tw[stride] = omega_{2^log_n_max}^stride = omega_{2^log_n} when stride = 2^(log_n_max-log_n)
inline void nttGetOmega2n(int log_n, uint64_t out[4]) {
    int stride = 1 << (g_ntt_log_n_max - log_n);
    NTT_CUDA_CHECK(cudaMemcpy(out, g_ntt_twiddles + stride*4, 32, cudaMemcpyDeviceToHost));
}

// =============================================================================
// Stockham (Swarztrauber) radix-2 butterfly kernels
//
// Correct Stockham formulation ??? pass p (1-indexed), n1=2^{p-1}, n2=N/(2*n1):
//   Thread idx handles element idx in [0, N/2).
//   m     = idx % n2        (inner position within group)
//   n_grp = idx / n2        (group index)
//   src_lo = n_grp * 2*n2 + m
//   src_hi = src_lo + n2
//   dst_lo = idx             (= n_grp*n2 + m)
//   dst_hi = idx + N/2
//   twiddle = omega_N^{n_grp * n2}
//           = twiddles[n_grp * (N_max >> p)]    (in our table indexed by omega_max^j)
//
// Forward: twiddle as above.
// Inverse: twiddle[N_half_max - tw_idx]  (conjugate = omega^{-k}) for tw_idx>0.
// =============================================================================
__global__ void ntt_stockham_fwd(
    const uint64_t* __restrict__ src,
    uint64_t*       __restrict__ dst,
    int pass,           // 1-indexed pass number
    int N,
    int log_n_max,      // for twiddle table stride
    const uint64_t* __restrict__ tw)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N/2) return;

    int n1 = 1 << (pass-1);    // = half_size
    int n2 = N / (2*n1);       // inner group size

    int m     = idx % n2;
    int n_grp = idx / n2;

    int src_lo = n_grp*2*n2 + m;
    int src_hi = src_lo + n2;
    int dst_lo = idx;
    int dst_hi = idx + N/2;

    // zyklop_u256 = 256-bit load/store (1 memory transaction per element)
    const zyklop_u256* src4 = reinterpret_cast<const zyklop_u256*>(src);
    zyklop_u256*       dst4 = reinterpret_cast<zyklop_u256*>(dst);
    const zyklop_u256* tw4  = reinterpret_cast<const zyklop_u256*>(tw);

    zyklop_u256 vu = src4[src_lo];
    zyklop_u256 vv = src4[src_hi];
    int tw_idx = n_grp * (1 << (log_n_max - pass));
    zyklop_u256 vw = tw4[tw_idx];

    uint64_t u[4]={vu.x,vu.y,vu.z,vu.w}, v[4]={vv.x,vv.y,vv.z,vv.w};
    uint64_t w[4]={vw.x,vw.y,vw.z,vw.w}, res[4], lo[4], hi[4];
    frMul(v, w, res);
    frAdd(u, res, lo);
    frSub(u, res, hi);

    dst4[dst_lo] = make_zyklop_u256(lo[0], lo[1], lo[2], lo[3]);
    dst4[dst_hi] = make_zyklop_u256(hi[0], hi[1], hi[2], hi[3]);
}

__global__ void ntt_stockham_inv(
    const uint64_t* __restrict__ src,
    uint64_t*       __restrict__ dst,
    int pass,
    int N,
    int log_n_max,
    const uint64_t* __restrict__ tw)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N/2) return;

    int n1 = 1 << (pass-1);
    int n2 = N / (2*n1);

    int m     = idx % n2;
    int n_grp = idx / n2;

    int src_lo = n_grp*2*n2 + m;
    int src_hi = src_lo + n2;
    int dst_lo = idx;
    int dst_hi = idx + N/2;

    const zyklop_u256* src4 = reinterpret_cast<const zyklop_u256*>(src);
    zyklop_u256*       dst4 = reinterpret_cast<zyklop_u256*>(dst);
    const zyklop_u256* tw4  = reinterpret_cast<const zyklop_u256*>(tw);

    zyklop_u256 vu = src4[src_lo];
    zyklop_u256 vv = src4[src_hi];
    int tw_fwd = n_grp * (1 << (log_n_max - pass));
    int N_half_max = 1 << (log_n_max - 1);
    int tw_idx = (tw_fwd == 0) ? 0 : (N_half_max - tw_fwd);
    zyklop_u256 vw = tw4[tw_idx];

    uint64_t u[4]={vu.x,vu.y,vu.z,vu.w}, v[4]={vv.x,vv.y,vv.z,vv.w};
    uint64_t w[4]={vw.x,vw.y,vw.z,vw.w}, res[4], lo[4], hi[4];
    // omega^{-k} = -omega^{N/2-k}  (since omega^{N/2}=-1)
    // tw_idx stores the "N/2-k" part; negate to get the true inverse twiddle
    if (tw_fwd != 0) frNeg(w, w);
    frMul(v, w, res);
    frAdd(u, res, lo);
    frSub(u, res, hi);

    dst4[dst_lo] = make_zyklop_u256(lo[0], lo[1], lo[2], lo[3]);
    dst4[dst_hi] = make_zyklop_u256(hi[0], hi[1], hi[2], hi[3]);
}

// Scale by N^{-1}, in-place (result already in d)
__global__ void ntt_scale(uint64_t* d, int N, const uint64_t* n_inv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    zyklop_u256* d4 = reinterpret_cast<zyklop_u256*>(d);
    zyklop_u256 v = d4[i];
    uint64_t a[4]={v.x,v.y,v.z,v.w}, tmp[4];
    frMul(a, n_inv, tmp);
    d4[i] = make_zyklop_u256(tmp[0], tmp[1], tmp[2], tmp[3]);
}

// Fused scale+copy: reads from src (ping-pong buf), writes scaled to dst (d_a)
// Saves one full D2D pass vs copy-then-scale (for iNTT when log_n is odd)
__global__ void ntt_scale_copy(const uint64_t* __restrict__ src,
                               uint64_t* __restrict__ dst,
                               int N, const uint64_t* n_inv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const zyklop_u256* s4 = reinterpret_cast<const zyklop_u256*>(src);
    zyklop_u256*       d4 = reinterpret_cast<zyklop_u256*>(dst);
    zyklop_u256 v = s4[i];
    uint64_t a[4]={v.x,v.y,v.z,v.w}, tmp[4];
    frMul(a, n_inv, tmp);
    d4[i] = make_zyklop_u256(tmp[0], tmp[1], tmp[2], tmp[3]);
}

// =============================================================================
// ntt() ??? in-place NTT (forward or inverse)
// d_a layout: d_a[i*4 .. i*4+3] = 4 limbs of Fr element i, all Montgomery
// =============================================================================
inline void ntt(uint64_t* d_a, int log_n, bool inverse)
{
    if (!g_ntt_twiddles) { fprintf(stderr,"[NTT] call nttPrepare() first\n"); exit(1); }
    if (log_n < 1 || log_n > g_ntt_log_n_max) {
        fprintf(stderr,"[NTT] log_n %d out of range\n", log_n); exit(1);
    }

    int N = 1 << log_n;
    int threads = 256;
    int blocks  = (N/2 + 255) / 256;

    // g_ntt_buf is pre-allocated for the max domain ??? no cudaMalloc per call
    uint64_t* src = d_a;
    uint64_t* dst = g_ntt_buf;

    for (int pass = 1; pass <= log_n; pass++) {
        if (!inverse)
            ntt_stockham_fwd<<<blocks,threads>>>(
                src, dst, pass, N, g_ntt_log_n_max, g_ntt_twiddles);
        else
            ntt_stockham_inv<<<blocks,threads>>>(
                src, dst, pass, N, g_ntt_log_n_max, g_ntt_twiddles);
        NTT_CUDA_CHECK(cudaGetLastError());
        uint64_t* tmp=src; src=dst; dst=tmp;
    }

    // Inverse: scale by N^{-1}
    if (inverse) {
        
		const uint64_t* d_ninv = g_ntt_ninv + (size_t)log_n * 4;
        if (src != d_a) {
            // Result in g_ntt_buf: fused scale+copy ??? d_a (saves one D2D pass)
            ntt_scale_copy<<<(N+255)/256, 256>>>(src, d_a, N, d_ninv);
        } else {
            // Result already in d_a: scale in-place
            ntt_scale<<<(N+255)/256, 256>>>(d_a, N, d_ninv);
        }
        NTT_CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Forward: if result ended up in g_ntt_buf, copy back to d_a
    if (src != d_a)
        NTT_CUDA_CHECK(cudaMemcpy(d_a, src,
                                  (size_t)N*4*sizeof(uint64_t), cudaMemcpyDeviceToDevice));

}

// =============================================================================
// nttCosetMul ??? a[i] *= g^i  (or g^{-i} for inverse)
// g = 7, primitive root of Fr*
// =============================================================================
static uint64_t* g_coset_pow     = nullptr;
static uint64_t* g_coset_pow_inv = nullptr;
static int       g_coset_N       = 0;

__global__ void ntt_coset_kernel(uint64_t* d, int N, const uint64_t* gpow) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    zyklop_u256* d4         = reinterpret_cast<zyklop_u256*>(d);
    const zyklop_u256* gpow4 = reinterpret_cast<const zyklop_u256*>(gpow);
    zyklop_u256 v = d4[i], g = gpow4[i];
    uint64_t a[4]={v.x,v.y,v.z,v.w}, b[4]={g.x,g.y,g.z,g.w}, tmp[4];
    frMul(a, b, tmp);
    d4[i] = make_zyklop_u256(tmp[0], tmp[1], tmp[2], tmp[3]);
}

inline void nttCosetMul(uint64_t* d_a, int log_n, bool inverse)
{
    int N = 1 << log_n;
    if (N != g_coset_N) {
        if (g_coset_pow)     cudaFree(g_coset_pow);
        if (g_coset_pow_inv) cudaFree(g_coset_pow_inv);

        // Encode g=7 and g^{-1} on GPU ??? avoids CPU __int128 -O3 bug
        const uint64_t g_can[4]    = {7,0,0,0};
        const uint64_t ginv_can[4] = {
            0x09b290cbfdb6db6eULL, 0x4ee2d80a5a8834a7ULL,
            0xac9dc0d0edede80dULL, 0x06e9c21069503b73ULL
        };
        uint64_t *d_gm, *d_ginvm, *d_tmp;
        NTT_CUDA_CHECK(cudaMalloc(&d_gm,   32));
        NTT_CUDA_CHECK(cudaMalloc(&d_ginvm, 32));
        NTT_CUDA_CHECK(cudaMalloc(&d_tmp,   32));
        NTT_CUDA_CHECK(cudaMemcpy(d_tmp, g_can,    32, cudaMemcpyHostToDevice));
        k_encode<<<1,1>>>(d_tmp, d_gm, 1);
        NTT_CUDA_CHECK(cudaMemcpy(d_tmp, ginv_can, 32, cudaMemcpyHostToDevice));
        k_encode<<<1,1>>>(d_tmp, d_ginvm, 1);
        cudaDeviceSynchronize(); cudaFree(d_tmp);

        NTT_CUDA_CHECK(cudaMalloc(&g_coset_pow,     (size_t)N*4*sizeof(uint64_t)));
        NTT_CUDA_CHECK(cudaMalloc(&g_coset_pow_inv, (size_t)N*4*sizeof(uint64_t)));
        NTT_CUDA_CHECK(cudaMemcpy(g_coset_pow,     R_FR_CPU, 32, cudaMemcpyHostToDevice));
        NTT_CUDA_CHECK(cudaMemcpy(g_coset_pow_inv, R_FR_CPU, 32, cudaMemcpyHostToDevice));

        // Build g^i / g^{-i} tables on GPU using 2-phase parallel builder
        ntt_build_twiddles_p1b<<<1,1>>>(g_coset_pow,     d_gm,    N);
        ntt_build_twiddles_p1b<<<1,1>>>(g_coset_pow_inv, d_ginvm, N);
        cudaDeviceSynchronize();
        if (N > NTT_TW_BLOCK) {
            uint64_t *d_gB, *d_ginvB;
            NTT_CUDA_CHECK(cudaMalloc(&d_gB,    32));
            NTT_CUDA_CHECK(cudaMalloc(&d_ginvB, 32));
            frMul_kernel<<<1,1>>>(g_coset_pow     + (NTT_TW_BLOCK-1)*4, d_gm,    d_gB);
            frMul_kernel<<<1,1>>>(g_coset_pow_inv + (NTT_TW_BLOCK-1)*4, d_ginvm, d_ginvB);
            cudaDeviceSynchronize();
            ntt_build_twiddles_p1<<<1,1>>>(g_coset_pow,     d_gB,    N);
            ntt_build_twiddles_p1<<<1,1>>>(g_coset_pow_inv, d_ginvB, N);
            cudaDeviceSynchronize();
            int nb = (N + NTT_TW_BLOCK - 1) / NTT_TW_BLOCK - 1;
            ntt_build_twiddles_p2<<<nb, NTT_TW_BLOCK>>>(g_coset_pow,     N);
            ntt_build_twiddles_p2<<<nb, NTT_TW_BLOCK>>>(g_coset_pow_inv, N);
            cudaDeviceSynchronize();
            cudaFree(d_gB); cudaFree(d_ginvB);
        }
        cudaFree(d_gm); cudaFree(d_ginvm);
        g_coset_N = N;
    }

    int threads=256, blocks=(N+255)/256;
    ntt_coset_kernel<<<blocks,threads>>>(d_a, N, inverse ? g_coset_pow_inv : g_coset_pow);
    NTT_CUDA_CHECK(cudaGetLastError());
}
