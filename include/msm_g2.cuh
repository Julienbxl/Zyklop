// =============================================================================
// msm_g2.cuh — BN254 G2 Multi-Scalar Multiplication (clean large-only backend)
// Zyklop GPU prover
//
// Algorithm : Pippenger with signed digits + histogram/prefix-sum/scatter
//             followed by a large-only bucket accumulation kernel.
//
// Rationale:
//   On real Zyklop workloads benchmarked so far, G2 buckets are effectively
//   always in the "large" regime. The small/medium classification machinery was
//   therefore removed to keep the code path simple and explicit.
//
// Tuning:
//   - ZK_G2_DEBUG=1      : print bucket stats + phase timings
//   - ZK_G2_LARGE_BLK=.. : optional override for large kernel threads/block
//                          Allowed values are effectively 32 / 64 / 128.
//                          When unset, an auto-heuristic is used:
//                            * <= 8 nonzero buckets  -> 128
//                            * otherwise             -> 32
//
// Requires: fp_bn254.cuh then fp2_bn254.cuh included before this header
// =============================================================================
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
using msm_u64x4 = ulonglong4_16a;
#else
using msm_u64x4 = ulonglong4;
#endif

#ifndef C_BITS
#define C_BITS 14
#endif

static constexpr int      G2_C             = C_BITS;
static constexpr int      G2_HALF_BUCKETS  = 1 << (G2_C - 1);         // 8192
static constexpr int      G2_STRIDE        = G2_HALF_BUCKETS + 1;     // 8193
static constexpr int      G2_N_WINDOWS     = (256 + G2_C - 1) / G2_C; // 19
static constexpr int      G2_TOTAL_BINS    = G2_N_WINDOWS * G2_STRIDE;
static constexpr int      G2_TOTAL_BUCKETS = G2_N_WINDOWS * G2_HALF_BUCKETS;
static constexpr uint32_t G2_NEG_FLAG      = 0x80000000u;
static constexpr uint32_t G2_IDX_MASK      = 0x7FFFFFFFu;

static constexpr int      G2_LARGE_BLK_DEFAULT = 0;   // 0 = auto heuristic

#define G2_CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[G2_MSM] CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// =============================================================================
// Host-side tuning / debug helpers
// =============================================================================
struct G2TuneConfig {
    int  debug;
    int  large_blk_override;
    bool large_blk_forced;
    int  small_thresh  = 0;    // 0 = large-only at C=14 (good default)
    int  medium_thresh = 0;    // set via ZK_G2_MEDIUM_THRESH or auto-detect
};

static inline int g2GetEnvInt(const char* name, int defv) {
    const char* s = std::getenv(name);
    if (!s || !*s) return defv;
    char* endp = nullptr;
    long v = std::strtol(s, &endp, 10);
    if (endp == s) return defv;
    return (int)v;
}

static inline int g2NormalizeLargeBlk(int blk) {
    if (blk >= 256) return 256;  // On autorise 256 !
    if (blk >= 128) return 128;
    if (blk >= 64)  return 64;
    return 32;
}

static inline int g2ChooseLargeBlk(const G2TuneConfig& cfg, uint32_t nonzero_count) {
    if (cfg.large_blk_forced) return cfg.large_blk_override;
    return 128; // Finie la limite à 32, on passe à 128 threads par défaut !
}

static inline G2TuneConfig g2GetTuneConfig() {
    G2TuneConfig cfg{};
    cfg.debug = g2GetEnvInt("ZK_G2_DEBUG", 0);

    const char* s = std::getenv("ZK_G2_LARGE_BLK");
    if (s && *s) {
        cfg.large_blk_forced = true;
        cfg.large_blk_override = g2NormalizeLargeBlk(std::atoi(s));
    } else {
        cfg.large_blk_forced = false;
        cfg.large_blk_override = 0;
    }
    cfg.small_thresh  = g2GetEnvInt("ZK_G2_SMALL_THRESH",  0);
    // Default thresholds tuned per C:
    //   C=14: mean ~330 pts/bucket → large-only optimal (small=0, medium=0)
    //   C=17: mean ~79 pts/bucket  → small kernel optimal (1 thread/bucket)
    const int default_small  = (G2_C >= 17) ? 600 : 0;
    const int default_medium = 0;  // medium never optimal in tested configs
    cfg.small_thresh  = g2GetEnvInt("ZK_G2_SMALL_THRESH",  default_small);
    cfg.medium_thresh = g2GetEnvInt("ZK_G2_MEDIUM_THRESH", default_medium);
    return cfg;
}

struct G2BucketStats {
    uint32_t nonzero = 0;
    uint64_t points = 0;
    double   mean_nonzero = 0.0;
    uint32_t max_count = 0;
    uint32_t p50 = 0, p90 = 0, p95 = 0, p99 = 0;
};

static inline uint32_t g2PickQuantile(std::vector<uint32_t>& v, double q) {
    if (v.empty()) return 0;
    size_t idx = (size_t)(q * (double)(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

static inline G2BucketStats g2CollectBucketStats(const uint32_t* d_hist) {
    std::vector<uint32_t> hist(G2_TOTAL_BINS);
    G2_CUDA_CHECK(cudaMemcpy(hist.data(), d_hist, (size_t)G2_TOTAL_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    G2BucketStats s{};
    std::vector<uint32_t> nz;
    nz.reserve(G2_TOTAL_BUCKETS);

    for (int w = 0; w < G2_N_WINDOWS; w++) {
        const size_t base = (size_t)w * (size_t)G2_STRIDE;
        for (int b = 0; b < G2_HALF_BUCKETS; b++) {
            const uint32_t c = hist[base + (size_t)b];
            if (c == 0u) continue;
            s.nonzero++;
            s.points += (uint64_t)c;
            if (c > s.max_count) s.max_count = c;
            nz.push_back(c);
        }
    }

    if (s.nonzero) s.mean_nonzero = (double)s.points / (double)s.nonzero;
    if (!nz.empty()) {
        auto v50 = nz, v90 = nz, v95 = nz, v99 = nz;
        s.p50 = g2PickQuantile(v50, 0.50);
        s.p90 = g2PickQuantile(v90, 0.90);
        s.p95 = g2PickQuantile(v95, 0.95);
        s.p99 = g2PickQuantile(v99, 0.99);
    }
    return s;
}

struct G2PhaseTimes {
    float extract = 0.f, hist = 0.f, scan = 0.f, scatter = 0.f, init = 0.f;
    float small = 0.f, medium = 0.f, large = 0.f, reduce = 0.f, combine = 0.f, jac2aff = 0.f, total = 0.f;
};

static inline float g2ElapsedMs(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    G2_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

static inline void g2PrintDebug(const G2TuneConfig& cfg,
                                const G2BucketStats& st,
                                uint32_t nonzero_count,
                                int      selected_large_blk,
                                const G2PhaseTimes& t)
{
    fprintf(stderr,
        "[g2] config: C=%d windows=%d half_buckets=%d large_blk=%d (%s)\n",
        G2_C, G2_N_WINDOWS, G2_HALF_BUCKETS, selected_large_blk,
        cfg.large_blk_forced ? "forced" : "auto");

    fprintf(stderr,
        "[g2] buckets: nonzero=%u/%d (%.1f%%) points=%llu mean_nonzero=%.2f max=%u\n",
        st.nonzero, G2_TOTAL_BUCKETS, 100.0 * (double)st.nonzero / (double)G2_TOTAL_BUCKETS,
        (unsigned long long)st.points, st.mean_nonzero, st.max_count);

    // On supprime la ligne qui mentait sur le "large-only path"

    fprintf(stderr,
        "[g2] kernel ms: extract=%7.1f hist=%7.1f scan=%7.1f scatter=%7.1f init=%7.1f\n",
        t.extract, t.hist, t.scan, t.scatter, t.init);

    fprintf(stderr,
        "[g2] kernel ms: small=%7.1f medium=%7.1f large=%7.1f reduce=%7.1f combine=%7.1f jac2aff=%7.1f total=%7.1f\n",
        t.small, t.medium, t.large, t.reduce, t.combine, t.jac2aff, t.total);
}

// =============================================================================
// Helpers
// =============================================================================
__device__ __forceinline__ void g2LoadAffSoA(
    const msm_u64x4* __restrict__ Pxv,
    const msm_u64x4* __restrict__ Pyv,
    uint32_t idx,
    Fp2PointAff& P)
{
    const size_t base = (size_t)idx * 2u;

    const msm_u64x4 x0 = Pxv[base + 0];
    const msm_u64x4 x1 = Pxv[base + 1];
    const msm_u64x4 y0 = Pyv[base + 0];
    const msm_u64x4 y1 = Pyv[base + 1];

    P.X[0] = x0.x; P.X[1] = x0.y; P.X[2] = x0.z; P.X[3] = x0.w;
    P.X[4] = x1.x; P.X[5] = x1.y; P.X[6] = x1.z; P.X[7] = x1.w;
    P.Y[0] = y0.x; P.Y[1] = y0.y; P.Y[2] = y0.z; P.Y[3] = y0.w;
    P.Y[4] = y1.x; P.Y[5] = y1.y; P.Y[6] = y1.z; P.Y[7] = y1.w;

    bool is_inf = true;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (P.X[j] != 0ull || P.Y[j] != 0ull) {
            is_inf = false;
            break;
        }
    }
    P.infinity = is_inf;
}

__device__ __forceinline__ void g2AccumulateValMixed(
    Fp2PointJac& acc,
    Fp2PointJac& tmp,
    const msm_u64x4* __restrict__ Pxv,
    const msm_u64x4* __restrict__ Pyv,
    uint32_t val)
{
    const bool is_neg = (val & G2_NEG_FLAG) != 0u;
    const uint32_t point_id = val & G2_IDX_MASK;

    Fp2PointAff P;
    g2LoadAffSoA(Pxv, Pyv, point_id, P);
    if (P.infinity) return;
    if (is_neg) fp2Neg(P.Y, P.Y);

    point2JacAddMixed(acc, P, tmp);
    acc = tmp;
}

// =============================================================================
// Phase 1 — Extract signed digits for ALL windows in ONE scalar pass
// =============================================================================
__global__ void msm_g2_extractAllSigned(
    const uint64_t* __restrict__ scalars,
    int             N,
    int32_t* __restrict__        d_sd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const msm_u64x4* sv = reinterpret_cast<const msm_u64x4*>(scalars);
    const msm_u64x4 sv4 = sv[i];
    const uint64_t s0 = sv4.x;
    const uint64_t s1 = sv4.y;
    const uint64_t s2 = sv4.z;
    const uint64_t s3 = sv4.w;

    uint32_t carry = 0u;

    #pragma unroll
    for (int w = 0; w < G2_N_WINDOWS; w++) {
        const int bit_start = w * G2_C;
        const int limb      = bit_start >> 6;
        const int bit_off   = bit_start & 63;

        uint32_t raw;
        if (bit_off + G2_C <= 64) {
            const uint64_t src =
                (limb == 0) ? s0 :
                (limb == 1) ? s1 :
                (limb == 2) ? s2 : s3;
            raw = (uint32_t)(src >> bit_off) & ((1u << G2_C) - 1u);
        } else {
            const uint64_t lo_src =
                (limb == 0) ? s0 :
                (limb == 1) ? s1 :
                (limb == 2) ? s2 : s3;

            const uint64_t hi_src =
                (limb + 1 == 1) ? s1 :
                (limb + 1 == 2) ? s2 :
                (limb + 1 == 3) ? s3 : 0ull;

            const uint32_t lo = (uint32_t)(lo_src >> bit_off);
            const uint32_t hi = (limb + 1 < 4) ? (uint32_t)(hi_src << (64 - bit_off)) : 0u;
            raw = (lo | hi) & ((1u << G2_C) - 1u);
        }

        raw += carry;
        carry = 0u;

        int digit = (int)raw;
        if (raw > (uint32_t)G2_HALF_BUCKETS) {
            digit = (int)((1u << G2_C) - raw);
            carry = 1u;
            digit = -digit;
        }

        d_sd[(size_t)w * (size_t)N + (size_t)i] = digit;
    }
}

__global__ void msm_g2_histogramWindow(
    const int32_t* __restrict__ d_sd_win,
    int            N,
    uint32_t* __restrict__      hist_win)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int d = (int)d_sd_win[i];
    if (d == 0) return;
    const uint32_t b = (d < 0) ? (uint32_t)(-d - 1) : (uint32_t)(d - 1);
    atomicAdd(&hist_win[b], 1u);
}

__global__ void msm_g2_scatterWindow(
    const int32_t* __restrict__ d_sd_win,
    int            N,
    uint32_t* __restrict__      write_win,
    uint32_t* __restrict__      d_perm)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int d = (int)d_sd_win[i];
    if (d == 0) return;
    const bool neg = (d < 0);
    const uint32_t b = neg ? (uint32_t)(-d - 1) : (uint32_t)(d - 1);
    const uint32_t pos = atomicAdd(&write_win[b], 1u);
    d_perm[pos] = (uint32_t)i | (neg ? G2_NEG_FLAG : 0u);
}

__global__ void msm_g2_classifyBuckets(
    const uint32_t* __restrict__ d_hist,
    uint32_t* __restrict__       d_small_list,
    uint32_t* __restrict__       d_medium_list,
    uint32_t* __restrict__       d_large_list,
    uint32_t* __restrict__       d_counts,   // [3]: small, medium, large
    int small_thresh,
    int medium_thresh)
{
    int gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (gb >= G2_TOTAL_BUCKETS) return;
    const int w = gb / G2_HALF_BUCKETS;
    const int b = gb - w * G2_HALF_BUCKETS;
    const uint32_t bin   = (uint32_t)w * G2_STRIDE + (uint32_t)b;
    const uint32_t count = d_hist[bin];
    if (count == 0u) return;
    if ((int)count <= small_thresh) {
        d_small_list[atomicAdd(&d_counts[0], 1u)] = (uint32_t)gb;
    } else if ((int)count <= medium_thresh) {
        d_medium_list[atomicAdd(&d_counts[1], 1u)] = (uint32_t)gb;
    } else {
        d_large_list[atomicAdd(&d_counts[2], 1u)] = (uint32_t)gb;
    }
}

__global__ void msm_g2_accumulate_small(
    const uint64_t* __restrict__ d_Px,
    const uint64_t* __restrict__ d_Py,
    const uint32_t* __restrict__ d_perm,
    const uint32_t* __restrict__ d_offsets,
    const uint32_t* __restrict__ d_hist,
    const uint32_t* __restrict__ d_small_list,
    uint32_t                     n_small,
    Fp2PointJac* __restrict__    d_buckets)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_small) return;
    const uint32_t gb    = d_small_list[tid];
    const int w          = (int)(gb / G2_HALF_BUCKETS);
    const int b          = (int)(gb - (uint32_t)w * G2_HALF_BUCKETS);
    const uint32_t bin   = (uint32_t)w * G2_STRIDE + (uint32_t)b;
    const uint32_t start = d_offsets[bin];
    const uint32_t count = d_hist[bin];
    const msm_u64x4* Pxv = reinterpret_cast<const msm_u64x4*>(d_Px);
    const msm_u64x4* Pyv = reinterpret_cast<const msm_u64x4*>(d_Py);
    Fp2PointJac acc, tmp;
    point2JacSetInfinity(acc);
    #pragma unroll 1
    for (uint32_t k = 0; k < count; k++)
        g2AccumulateValMixed(acc, tmp, Pxv, Pyv, d_perm[start + k]);
    d_buckets[gb] = acc;
}

__global__ void msm_g2_accumulate_medium(
    const uint64_t* __restrict__ d_Px,
    const uint64_t* __restrict__ d_Py,
    const uint32_t* __restrict__ d_perm,
    const uint32_t* __restrict__ d_offsets,
    const uint32_t* __restrict__ d_hist,
    const uint32_t* __restrict__ d_medium_list,
    uint32_t                     n_medium,
    Fp2PointJac* __restrict__    d_buckets)
{
    const int warp_in_block  = threadIdx.x >> 5;
    const int lane           = threadIdx.x & 31;
    const int warps_per_block= blockDim.x >> 5;
    const uint32_t bucket_idx= (uint32_t)blockIdx.x * (uint32_t)warps_per_block + (uint32_t)warp_in_block;
    if (bucket_idx >= n_medium) return;
    const uint32_t gb    = d_medium_list[bucket_idx];
    const int w          = (int)(gb / G2_HALF_BUCKETS);
    const int b          = (int)(gb - (uint32_t)w * G2_HALF_BUCKETS);
    const uint32_t bin   = (uint32_t)w * G2_STRIDE + (uint32_t)b;
    const uint32_t start = d_offsets[bin];
    const uint32_t count = d_hist[bin];
    const msm_u64x4* Pxv = reinterpret_cast<const msm_u64x4*>(d_Px);
    const msm_u64x4* Pyv = reinterpret_cast<const msm_u64x4*>(d_Py);
    Fp2PointJac acc, tmp;
    point2JacSetInfinity(acc);
    for (uint32_t k = (uint32_t)lane; k < count; k += 32u)
        g2AccumulateValMixed(acc, tmp, Pxv, Pyv, d_perm[start + k]);
    extern __shared__ unsigned char smem_raw[];
    Fp2PointJac* smem = reinterpret_cast<Fp2PointJac*>(smem_raw);
    smem[warp_in_block * 32 + lane] = acc;
    __syncwarp();
    if (lane == 0) {
        Fp2PointJac total, tmp2;
        point2JacSetInfinity(total);
        for (int t = 0; t < 32; t++) {
            if (!smem[warp_in_block * 32 + t].infinity) {
                point2JacAdd(total, smem[warp_in_block * 32 + t], tmp2);
                total = tmp2;
            }
        }
        d_buckets[gb] = total;
    }
}



__global__ void msm_g2_accumulate_large(
    const uint64_t* __restrict__ d_Px,
    const uint64_t* __restrict__ d_Py,
    const uint32_t* __restrict__ d_perm,
    const uint32_t* __restrict__ d_offsets,
    const uint32_t* __restrict__ d_hist,
    const uint32_t* __restrict__ d_large_list,
    uint32_t                     n_large,
    Fp2PointJac* __restrict__    d_buckets)
{
    const uint32_t bucket_idx = (uint32_t)blockIdx.x;
    if (bucket_idx >= n_large) return;

    const uint32_t gb = d_large_list[bucket_idx];
    const int w = (int)(gb / G2_HALF_BUCKETS);
    const int b = (int)(gb - (uint32_t)w * G2_HALF_BUCKETS);
    const uint32_t bin   = (uint32_t)w * G2_STRIDE + (uint32_t)b;
    const uint32_t start = d_offsets[bin];
    const uint32_t count = d_hist[bin];

    const msm_u64x4* Pxv = reinterpret_cast<const msm_u64x4*>(d_Px);
    const msm_u64x4* Pyv = reinterpret_cast<const msm_u64x4*>(d_Py);

    Fp2PointJac acc, tmp;
    point2JacSetInfinity(acc);

    for (uint32_t k = (uint32_t)threadIdx.x; k < count; k += (uint32_t)blockDim.x) {
        g2AccumulateValMixed(acc, tmp, Pxv, Pyv, d_perm[start + k]);
    }

    extern __shared__ unsigned char smem_raw[];
    Fp2PointJac* smem = reinterpret_cast<Fp2PointJac*>(smem_raw);
    smem[threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.x == 0) {
        Fp2PointJac total, tmp2;
        point2JacSetInfinity(total);
        for (int t = 0; t < blockDim.x; t++) {
            if (!smem[t].infinity) {
                point2JacAdd(total, smem[t], tmp2);
                total = tmp2;
            }
        }
        d_buckets[gb] = total;
    }
}

// =============================================================================
// =============================================================================
// G2 CHUNKED PIPPENGER REDUCE — portage direct du kernel G1 validé
// =============================================================================
//
// Formule identique à G1, avec Fp2PointJac au lieu de FpPointJac.
// CHUNK_THREADS = 64  → smem = 2 × 64 × 200B = 25KB < 48KB
// CHUNK_SIZE    = G2_HALF_BUCKETS / 64
//   C=14 : CHUNK_SIZE=128, scalar_mul 128*suffix → 8 bits
//   C=17 : CHUNK_SIZE=1024, scalar_mul 1024*suffix → 11 bits
// =============================================================================

static constexpr int G2_CHUNK_THREADS = 64;
static_assert(G2_HALF_BUCKETS % G2_CHUNK_THREADS == 0,
    "G2_HALF_BUCKETS must be divisible by G2_CHUNK_THREADS");
static constexpr int G2_CHUNK_SIZE = G2_HALF_BUCKETS / G2_CHUNK_THREADS;

__device__ __forceinline__ void g2_jacMulSmall(
    const Fp2PointJac& P, uint32_t scalar, Fp2PointJac& res)
{
    point2JacSetInfinity(res);
    if (scalar == 0u || P.infinity) return;
    Fp2PointJac base = P, tmp;
    while (scalar > 0u) {
        if (scalar & 1u) {
            if (res.infinity) { res = base; }
            else { point2JacAdd(res, base, tmp); res = tmp; }
        }
        scalar >>= 1u;
        if (scalar > 0u) { point2JacDouble(base, tmp); base = tmp; }
    }
}

// g2_reduce_p12 : phases 1 (triangle sum) + 2 (suffix scan, thread 0)
// Écrit d_smemR[w * G2_CHUNK_THREADS + k] en GMEM pour le kernel suivant.
// Mémoire intermédiaire : G2_N_WINDOWS × G2_CHUNK_THREADS × sizeof(Fp2PointJac) = ~228KB
__global__ void g2_reduce_p12(
    const Fp2PointJac* __restrict__ d_buckets,
    Fp2PointJac* __restrict__       d_smemR)
{
    const int w = (int)blockIdx.x;
    const int k = (int)threadIdx.x;

    extern __shared__ Fp2PointJac g2_smem_p12[];
    Fp2PointJac* smem_S = g2_smem_p12;
    Fp2PointJac* smem_R = g2_smem_p12 + G2_CHUNK_THREADS;

    const Fp2PointJac* win = d_buckets + (size_t)w * (size_t)G2_HALF_BUCKETS;
    const int base_idx = k * G2_CHUNK_SIZE;

    // --- PHASE 1 : triangle sum locale + somme simple ----------------------
    Fp2PointJac running, TL, tmp;
    point2JacSetInfinity(running);
    point2JacSetInfinity(TL);

    for (int j = G2_CHUNK_SIZE - 1; j >= 0; j--) {
        const int idx = base_idx + j;
        const Fp2PointJac& bkt = win[idx];
        if (!bkt.infinity) {
            if (running.infinity) { running = bkt; }
            else { point2JacAdd(running, bkt, tmp); running = tmp; }
        }
        if (!running.infinity) {
            if (TL.infinity) { TL = running; }
            else { point2JacAdd(TL, running, tmp); TL = tmp; }
        }
    }

    smem_S[k] = running;
    smem_R[k] = TL;
    __syncthreads();

    // --- PHASE 2 : suffix scan (thread 0) ----------------------------------
    if (k == 0) {
        Fp2PointJac suffix, acc, t2;
        point2JacSetInfinity(suffix);

        for (int i = G2_CHUNK_THREADS - 1; i >= 0; i--) {
            Fp2PointJac R_i      = smem_R[i];
            const Fp2PointJac Si = smem_S[i];

            if (!suffix.infinity) {
                g2_jacMulSmall(suffix, (uint32_t)G2_CHUNK_SIZE, acc);
                if (!acc.infinity) {
                    if (R_i.infinity) { R_i = acc; }
                    else { point2JacAdd(R_i, acc, t2); R_i = t2; }
                }
            }
            smem_R[i] = R_i;

            if (!Si.infinity) {
                if (suffix.infinity) { suffix = Si; }
                else { point2JacAdd(suffix, Si, t2); suffix = t2; }
            }
        }
    }
    __syncthreads();

    // Écrire smem_R en GMEM pour g2_reduce_p3
    d_smemR[(size_t)w * G2_CHUNK_THREADS + k] = smem_R[k];
}

// g2_reduce_p3 : phase 3 (tree-reduce) — ~52 regs, spills négligeables
__global__ void g2_reduce_p3(
    const Fp2PointJac* __restrict__ d_smemR,
    Fp2PointJac* __restrict__       d_window_sums)
{
    const int w = (int)blockIdx.x;
    const int k = (int)threadIdx.x;

    extern __shared__ Fp2PointJac g2_smem_p3[];
    g2_smem_p3[k] = d_smemR[(size_t)w * G2_CHUNK_THREADS + k];
    __syncthreads();

    Fp2PointJac tmp;
    for (int s = G2_CHUNK_THREADS / 2; s > 0; s >>= 1) {
        if (k < s) {
            Fp2PointJac& a       = g2_smem_p3[k];
            const Fp2PointJac& b = g2_smem_p3[k + s];
            if (!b.infinity) {
                if (a.infinity) { a = b; }
                else { point2JacAdd(a, b, tmp); a = tmp; }
            }
        }
        __syncthreads();
    }

    if (k == 0) d_window_sums[w] = g2_smem_p3[0];
}

inline void g2_reduce_chunked_launch(
    const Fp2PointJac* d_buckets,
    Fp2PointJac*       d_window_sums,
    Fp2PointJac*       d_smemR)
{
    const size_t smem_p12 = 2u * (size_t)G2_CHUNK_THREADS * sizeof(Fp2PointJac);
    const size_t smem_p3  =      (size_t)G2_CHUNK_THREADS * sizeof(Fp2PointJac);
    g2_reduce_p12<<<G2_N_WINDOWS, G2_CHUNK_THREADS, smem_p12>>>(d_buckets, d_smemR);
    g2_reduce_p3 <<<G2_N_WINDOWS, G2_CHUNK_THREADS, smem_p3 >>>(d_smemR, d_window_sums);
}

__global__ void msm_g2_combine(
    const Fp2PointJac* __restrict__ d_window_sums,
    Fp2PointJac* __restrict__       d_result)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    Fp2PointJac acc, tmp;
    point2JacSetInfinity(acc);

    for (int w = G2_N_WINDOWS - 1; w >= 0; w--) {
        #pragma unroll
        for (int d = 0; d < G2_C; d++) {
            point2JacDouble(acc, tmp);
            acc = tmp;
        }
        if (!d_window_sums[w].infinity) {
            point2JacAdd(acc, d_window_sums[w], tmp);
            acc = tmp;
        }
    }

    *d_result = acc;
}

__global__ void msm_g2_jacToAff(const Fp2PointJac* j, Fp2PointAff* a) {
    if (blockIdx.x == 0 && threadIdx.x == 0) point2JacToAff(*j, *a);
}

__global__ void g2_initBuckets(Fp2PointJac* buckets, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) point2JacSetInfinity(buckets[i]);
}

// =============================================================================
// Device-native API — preferred long term
// =============================================================================
inline void msm_g2_device(
    const uint64_t* d_Px,
    const uint64_t* d_Py,
    const uint64_t* d_scalars,
    int             N,
    Fp2PointAff&    h_result)
{
    const int BLK = 256;
    const size_t total_records = (size_t)N * (size_t)G2_N_WINDOWS;
    const G2TuneConfig cfg = g2GetTuneConfig();

    G2PhaseTimes times{};
    cudaEvent_t e0, e1;
    G2_CUDA_CHECK(cudaEventCreate(&e0));
    G2_CUDA_CHECK(cudaEventCreate(&e1));
    auto tic = [&]() { G2_CUDA_CHECK(cudaEventRecord(e0)); };
    auto toc = [&](float& dst) {
        G2_CUDA_CHECK(cudaEventRecord(e1));
        G2_CUDA_CHECK(cudaEventSynchronize(e1));
        dst += g2ElapsedMs(e0, e1);
    };

    int32_t* d_sd = nullptr;
    uint32_t *d_hist = nullptr, *d_offsets = nullptr, *d_write = nullptr;
    uint32_t* d_perm = nullptr;
    uint32_t* d_large_list = nullptr;
    uint32_t* d_nonzero_count = nullptr;
    Fp2PointJac *d_buckets = nullptr, *d_window_sums = nullptr, *d_result_jac = nullptr;
    Fp2PointAff *d_result_aff = nullptr;
    void* d_cub = nullptr;

    G2_CUDA_CHECK(cudaMalloc(&d_sd, total_records * sizeof(int32_t)));
    G2_CUDA_CHECK(cudaMalloc(&d_hist,    (size_t)G2_TOTAL_BINS * sizeof(uint32_t)));
    G2_CUDA_CHECK(cudaMalloc(&d_offsets, (size_t)G2_TOTAL_BINS * sizeof(uint32_t)));
    G2_CUDA_CHECK(cudaMalloc(&d_write,   (size_t)G2_TOTAL_BINS * sizeof(uint32_t)));
    G2_CUDA_CHECK(cudaMemset(d_hist, 0,  (size_t)G2_TOTAL_BINS * sizeof(uint32_t)));
    G2_CUDA_CHECK(cudaMalloc(&d_perm, total_records * sizeof(uint32_t)));
    G2_CUDA_CHECK(cudaMalloc(&d_large_list,  (size_t)G2_TOTAL_BUCKETS * sizeof(uint32_t)));
    G2_CUDA_CHECK(cudaMalloc(&d_nonzero_count, sizeof(uint32_t)));
    G2_CUDA_CHECK(cudaMemset(d_nonzero_count, 0, sizeof(uint32_t)));
    G2_CUDA_CHECK(cudaMalloc(&d_buckets,     (size_t)G2_TOTAL_BUCKETS * sizeof(Fp2PointJac)));
    G2_CUDA_CHECK(cudaMalloc(&d_window_sums, (size_t)G2_N_WINDOWS * sizeof(Fp2PointJac)));
    G2_CUDA_CHECK(cudaMalloc(&d_result_jac,  sizeof(Fp2PointJac)));
    G2_CUDA_CHECK(cudaMalloc(&d_result_aff,  sizeof(Fp2PointAff)));
    Fp2PointJac* d_smemR = nullptr;
    G2_CUDA_CHECK(cudaMalloc(&d_smemR, (size_t)G2_N_WINDOWS * G2_CHUNK_THREADS * sizeof(Fp2PointJac)));

    size_t scan_sz = 0;
    G2_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scan_sz, d_hist, d_offsets, G2_TOTAL_BINS));
    G2_CUDA_CHECK(cudaMalloc(&d_cub, scan_sz));

    tic();
    msm_g2_extractAllSigned<<<(N + BLK - 1) / BLK, BLK>>>(d_scalars, N, d_sd);
    G2_CUDA_CHECK(cudaGetLastError());
    toc(times.extract);

    tic();
    for (int w = 0; w < G2_N_WINDOWS; w++) {
        const int32_t* d_sd_win = d_sd + (size_t)w * (size_t)N;
        uint32_t* hist_win = d_hist + (size_t)w * (size_t)G2_STRIDE;
        msm_g2_histogramWindow<<<(N + BLK - 1) / BLK, BLK>>>(d_sd_win, N, hist_win);
        G2_CUDA_CHECK(cudaGetLastError());
    }
    G2_CUDA_CHECK(cudaDeviceSynchronize());
    toc(times.hist);

    tic();
    G2_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_cub, scan_sz, d_hist, d_offsets, G2_TOTAL_BINS));
    G2_CUDA_CHECK(cudaGetLastError());
    G2_CUDA_CHECK(cudaMemcpy(d_write, d_offsets, (size_t)G2_TOTAL_BINS * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    toc(times.scan);

    tic();
    for (int w = 0; w < G2_N_WINDOWS; w++) {
        const int32_t* d_sd_win = d_sd + (size_t)w * (size_t)N;
        uint32_t* write_win = d_write + (size_t)w * (size_t)G2_STRIDE;
        msm_g2_scatterWindow<<<(N + BLK - 1) / BLK, BLK>>>(d_sd_win, N, write_win, d_perm);
        G2_CUDA_CHECK(cudaGetLastError());
    }
    G2_CUDA_CHECK(cudaDeviceSynchronize());
    toc(times.scatter);

    cudaFree(d_sd);      d_sd = nullptr;
    cudaFree(d_write);   d_write = nullptr;
    cudaFree(d_cub);     d_cub = nullptr;

    G2BucketStats stats{};
    if (cfg.debug) stats = g2CollectBucketStats(d_hist);

    // Three-level classify
    uint32_t *d_small_list  = nullptr;
    uint32_t *d_medium_list = nullptr;
    uint32_t *d_counts      = nullptr;
    G2_CUDA_CHECK(cudaMalloc(&d_small_list,  (size_t)G2_TOTAL_BUCKETS * sizeof(uint32_t)));
    G2_CUDA_CHECK(cudaMalloc(&d_medium_list, (size_t)G2_TOTAL_BUCKETS * sizeof(uint32_t)));
    G2_CUDA_CHECK(cudaMalloc(&d_counts,      3 * sizeof(uint32_t)));
    G2_CUDA_CHECK(cudaMemset(d_counts, 0,    3 * sizeof(uint32_t)));

    tic();
    g2_initBuckets<<<(G2_TOTAL_BUCKETS + BLK - 1) / BLK, BLK>>>(d_buckets, G2_TOTAL_BUCKETS);
    G2_CUDA_CHECK(cudaGetLastError());
    msm_g2_classifyBuckets<<<(G2_TOTAL_BUCKETS + BLK - 1) / BLK, BLK>>>(
        d_hist, d_small_list, d_medium_list, d_large_list, d_counts,
        cfg.small_thresh, cfg.medium_thresh);
    G2_CUDA_CHECK(cudaGetLastError());
    G2_CUDA_CHECK(cudaDeviceSynchronize());
    toc(times.init);

    uint32_t h_counts[3] = {0, 0, 0};
    G2_CUDA_CHECK(cudaMemcpy(h_counts, d_counts, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    const uint32_t n_small  = h_counts[0];
    const uint32_t n_medium = h_counts[1];
    const uint32_t n_large  = h_counts[2];

    if (cfg.debug) {
        fprintf(stderr, "[g2] classified: small=%u medium=%u large=%u total=%u\n",
            n_small, n_medium, n_large, n_small + n_medium + n_large);
    }

    const int selected_large_blk = g2ChooseLargeBlk(cfg, n_large ? n_large : 1);

    if (n_small) {
        tic();
        msm_g2_accumulate_small<<<(n_small + 255) / 256, 256>>>(
            d_Px, d_Py, d_perm, d_offsets, d_hist, d_small_list, n_small, d_buckets);
        G2_CUDA_CHECK(cudaGetLastError());
        toc(times.small);
    }
    if (n_medium) {
        const int MBLK = 64;  // 2 warps/block — smaller smem for Fp2PointJac (192B)
        const int med_grids = (int)((n_medium + (MBLK/32) - 1) / (MBLK/32));
        const size_t smem_med = (size_t)MBLK * sizeof(Fp2PointJac);
        tic();
        msm_g2_accumulate_medium<<<med_grids, MBLK, smem_med>>>(
            d_Px, d_Py, d_perm, d_offsets, d_hist, d_medium_list, n_medium, d_buckets);
        G2_CUDA_CHECK(cudaGetLastError());
        toc(times.medium);
    }
    if (n_large) {
        tic();
        const size_t smem_large = (size_t)selected_large_blk * sizeof(Fp2PointJac);
        msm_g2_accumulate_large<<<n_large, selected_large_blk, smem_large>>>(
            d_Px, d_Py, d_perm, d_offsets, d_hist, d_large_list, n_large, d_buckets);
        G2_CUDA_CHECK(cudaGetLastError());
        toc(times.large);
    }

    cudaFree(d_perm);         d_perm = nullptr;
    cudaFree(d_offsets);      d_offsets = nullptr;
    cudaFree(d_hist);         d_hist = nullptr;
    cudaFree(d_small_list);   d_small_list = nullptr;
    cudaFree(d_medium_list);  d_medium_list = nullptr;
    cudaFree(d_large_list);   d_large_list = nullptr;
    cudaFree(d_counts);       d_counts = nullptr;
    cudaFree(d_nonzero_count); d_nonzero_count = nullptr;

    tic();
    g2_reduce_chunked_launch(d_buckets, d_window_sums, d_smemR);
    G2_CUDA_CHECK(cudaGetLastError());
    toc(times.reduce);
    cudaFree(d_smemR);   d_smemR   = nullptr;
    cudaFree(d_buckets); d_buckets = nullptr;

    tic();
    msm_g2_combine<<<1, 1>>>(d_window_sums, d_result_jac);
    G2_CUDA_CHECK(cudaGetLastError());
    toc(times.combine);
    cudaFree(d_window_sums); d_window_sums = nullptr;

    tic();
    msm_g2_jacToAff<<<1, 1>>>(d_result_jac, d_result_aff);
    G2_CUDA_CHECK(cudaGetLastError());
    G2_CUDA_CHECK(cudaDeviceSynchronize());
    toc(times.jac2aff);

    G2_CUDA_CHECK(cudaMemcpy(&h_result, d_result_aff, sizeof(Fp2PointAff), cudaMemcpyDeviceToHost));

    cudaFree(d_result_jac);
    cudaFree(d_result_aff);

    times.total = times.extract + times.hist + times.scan + times.scatter + times.init + times.small + times.medium + times.large + times.reduce + times.combine + times.jac2aff;
    if (cfg.debug) g2PrintDebug(cfg, stats, n_small + n_medium + n_large, selected_large_blk, times);

    G2_CUDA_CHECK(cudaEventDestroy(e0));
    G2_CUDA_CHECK(cudaEventDestroy(e1));
}

// =============================================================================
// Legacy drop-in API — preserves current call sites
// =============================================================================
inline void msmG2(
    const uint64_t* h_scalars,
    const uint64_t* h_Px,
    const uint64_t* h_Py,
    int             N,
    Fp2PointAff&    h_result)
{
    uint64_t *d_scalars = nullptr, *d_Px = nullptr, *d_Py = nullptr;
    G2_CUDA_CHECK(cudaMalloc(&d_scalars, (size_t)N * 4 * sizeof(uint64_t)));
    G2_CUDA_CHECK(cudaMalloc(&d_Px,      (size_t)N * 8 * sizeof(uint64_t)));
    G2_CUDA_CHECK(cudaMalloc(&d_Py,      (size_t)N * 8 * sizeof(uint64_t)));

    G2_CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars, (size_t)N * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    G2_CUDA_CHECK(cudaMemcpy(d_Px, h_Px, (size_t)N * 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    G2_CUDA_CHECK(cudaMemcpy(d_Py, h_Py, (size_t)N * 8 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    msm_g2_device(d_Px, d_Py, d_scalars, N, h_result);

    cudaFree(d_scalars);
    cudaFree(d_Px);
    cudaFree(d_Py);
}
