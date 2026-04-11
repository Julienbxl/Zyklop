// =============================================================================
// msm_g1.cuh — BN254 G1 Multi-Scalar Multiplication
// Zyklop GPU prover
//
// Production G1 backend used for:
//   - pi_A
//   - pi_C (C-part)
//   - pi_C (H-part)
//
// Design:
//   - signed digits, C = MSM_G1_C_BITS (default 14)
//   - histogram / prefix-sum / scatter (no global radix sort)
//   - THREE-LEVEL bucket backend: small / medium / large
//     small  (count <= ZK_G1_SMALL_THRESH, default 16)  : 1 thread / bucket
//     medium (count <= ZK_G1_MEDIUM_THRESH, default 512): 1 warp  / bucket
//     large  (count >  ZK_G1_MEDIUM_THRESH)             : 1 CTA   / bucket
//
// Debug / tuning:
//   - ZK_G1_DEBUG=1              : print bucket stats + phase timings
//   - ZK_G1_DEBUG=2              : add per-window summaries
//   - ZK_G1_LARGE_BLK=<int>     : CTA size for large buckets (default: 64)
//   - ZK_G1_SMALL_THRESH=<int>  : small/medium boundary (default: 16)
//   - ZK_G1_MEDIUM_THRESH=<int> : medium/large boundary (default: 512)
//
// Requires: fp_bn254.cuh included before this header
// =============================================================================
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <numeric>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
using msm_u64x4 = ulonglong4_16a;
#else
using msm_u64x4 = ulonglong4;
#endif

#ifndef MSM_G1_C_BITS
#define MSM_G1_C_BITS 14
#endif

namespace msm_g1_ns {

static constexpr int      G1_C             = MSM_G1_C_BITS;
static constexpr int      G1_HALF_BUCKETS  = 1 << (G1_C - 1);           // 8192
static constexpr int      G1_STRIDE        = G1_HALF_BUCKETS + 1;       // 8193
static constexpr int      G1_N_WINDOWS     = (256 + G1_C - 1) / G1_C;   // 19
static constexpr int      G1_TOTAL_BINS    = G1_N_WINDOWS * G1_STRIDE;
static constexpr int      G1_TOTAL_BUCKETS = G1_N_WINDOWS * G1_HALF_BUCKETS;
static constexpr uint32_t G1_NEG_FLAG      = 0x80000000u;
static constexpr uint32_t G1_IDX_MASK      = 0x7FFFFFFFu;

static constexpr int      G1_LARGE_BLK_DEFAULT = 64;

#define G1_CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[G1_MSM] CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

struct G1Config {
    int large_blk    = G1_LARGE_BLK_DEFAULT;
    int small_thresh = 16;
    int medium_thresh = 512;
    int debug        = 0;
};

struct G1PhaseTimes {
    float extract_ms   = 0.0f;
    float hist_ms      = 0.0f;
    float scan_ms      = 0.0f;
    float scatter_ms   = 0.0f;
    float classify_ms  = 0.0f;
    float init_ms      = 0.0f;
    float small_ms     = 0.0f;
    float medium_ms    = 0.0f;
    float large_ms     = 0.0f;
    float reduce_ms    = 0.0f;
    float combine_ms   = 0.0f;
    float jac2aff_ms   = 0.0f;
    float total_ms     = 0.0f;
};


inline int g1GetEnvInt(const char* name, int fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    return std::atoi(s);
}

inline int g1ClampPositive(int v, int fallback) {
    return (v > 0) ? v : fallback;
}

inline G1Config g1LoadConfigFromEnv() {
    G1Config cfg;
    cfg.debug        = g1GetEnvInt("ZK_G1_DEBUG", g1GetEnvInt("ZK_G1H_DEBUG", 0));
    cfg.large_blk    = g1ClampPositive(g1GetEnvInt("ZK_G1_LARGE_BLK", g1GetEnvInt("ZK_G1H_LARGE_BLK", G1_LARGE_BLK_DEFAULT)), G1_LARGE_BLK_DEFAULT);
    // Auto-tune thresholds per C:
    //   C=14: mean ~330 pts/bucket → medium optimal (small=16, medium=512)
    //   C=17: mean ~79 pts/bucket  → small optimal  (small=600, medium=0)
    const int default_small  = (G1_C >= 17) ? 600 : 16;
    const int default_medium = (G1_C >= 17) ?   0 : 512;
    cfg.small_thresh = g1GetEnvInt("ZK_G1_SMALL_THRESH",  default_small);
    cfg.medium_thresh= g1GetEnvInt("ZK_G1_MEDIUM_THRESH", default_medium);
    return cfg;
}

inline float g1ElapsedMs(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    G1_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

inline void g1PrintPhaseTimes(const G1PhaseTimes& t) {
    std::fprintf(stderr,
        "[g1] kernel ms: extract=%7.1f hist=%7.1f scan=%7.1f scatter=%7.1f classify=%7.1f init=%7.1f\n",
        t.extract_ms, t.hist_ms, t.scan_ms, t.scatter_ms, t.classify_ms, t.init_ms);
    std::fprintf(stderr,
        "[g1] kernel ms: small=%7.1f medium=%7.1f large=%7.1f reduce=%7.1f combine=%7.1f jac2aff=%7.1f total=%7.1f\n",
        t.small_ms, t.medium_ms, t.large_ms, t.reduce_ms, t.combine_ms, t.jac2aff_ms, t.total_ms);
}

inline void g1AnalyzeAndPrintHistogram(
    const std::vector<uint32_t>& hist,
    int debug_level,
    int large_blk,
    int small_thresh  = 16,
    int medium_thresh = 512)
{
    std::vector<uint32_t> nonzero_sizes;
    nonzero_sizes.reserve(G1_TOTAL_BUCKETS);

    uint64_t total_points = 0;
    uint32_t nonzero_buckets = 0;
    uint32_t max_bucket = 0;
    uint32_t max_window_nonzero = 0;
    uint64_t max_window_points = 0;
    uint32_t min_window_nonzero = UINT32_MAX;
    uint64_t min_window_points = UINT64_MAX;
    std::vector<uint32_t> win_nonzero(G1_N_WINDOWS, 0);
    std::vector<uint64_t> win_points(G1_N_WINDOWS, 0);

    for (int w = 0; w < G1_N_WINDOWS; ++w) {
        uint32_t nz = 0;
        uint64_t pts = 0;
        const size_t base = (size_t)w * (size_t)G1_STRIDE;
        for (int b = 0; b < G1_HALF_BUCKETS; ++b) {
            const uint32_t c = hist[base + (size_t)b];
            if (!c) continue;
            ++nz;
            pts += c;
            ++nonzero_buckets;
            total_points += c;
            max_bucket = std::max(max_bucket, c);
            nonzero_sizes.push_back(c);
        }
        win_nonzero[w] = nz;
        win_points[w] = pts;
        max_window_nonzero = std::max(max_window_nonzero, nz);
        max_window_points = std::max(max_window_points, pts);
        min_window_nonzero = std::min(min_window_nonzero, nz);
        min_window_points = std::min(min_window_points, pts);
    }

    if (nonzero_buckets == 0) {
        std::fprintf(stderr, "[g1] bucket stats: all buckets empty\n");
        return;
    }

    std::sort(nonzero_sizes.begin(), nonzero_sizes.end());
    auto pct = [&](double p) -> uint32_t {
        size_t idx = (size_t)(p * (double)(nonzero_sizes.size() - 1));
        return nonzero_sizes[idx];
    };

    const double mean_nonzero = (double)total_points / (double)nonzero_buckets;
    const double fill_pct = 100.0 * (double)nonzero_buckets / (double)G1_TOTAL_BUCKETS;

    // Count small/medium/large with current thresholds
    uint32_t n_small_stat = 0, n_medium_stat = 0, n_large_stat = 0;
    for (uint32_t sz : nonzero_sizes) {
        if      (sz <= (uint32_t)small_thresh)  ++n_small_stat;
        else if (sz <= (uint32_t)medium_thresh) ++n_medium_stat;
        else                                     ++n_large_stat;
    }

    std::fprintf(stderr,
        "[g1] config: C=%d windows=%d half_buckets=%d backend=small/medium/large"
        " thresholds=(%d/%d) large_blk=%d\n",
        G1_C, G1_N_WINDOWS, G1_HALF_BUCKETS, small_thresh, medium_thresh, large_blk);

    std::fprintf(stderr,
        "[g1] buckets: nonzero=%u/%d (%.1f%%) points=%llu mean_nonzero=%.2f max=%u p50=%u p90=%u p95=%u p99=%u\n",
        nonzero_buckets, G1_TOTAL_BUCKETS, fill_pct,
        (unsigned long long)total_points, mean_nonzero, max_bucket,
        pct(0.50), pct(0.90), pct(0.95), pct(0.99));

    std::fprintf(stderr,
        "[g1] classify: small=%u (%.1f%%) medium=%u (%.1f%%) large=%u (%.1f%%)\n",
        n_small_stat,  100.0f * n_small_stat  / (float)nonzero_buckets,
        n_medium_stat, 100.0f * n_medium_stat / (float)nonzero_buckets,
        n_large_stat,  100.0f * n_large_stat  / (float)nonzero_buckets);

    if (debug_level >= 2) {
        const double mean_win_nonzero = std::accumulate(win_nonzero.begin(), win_nonzero.end(), 0.0) / (double)G1_N_WINDOWS;
        const double mean_win_points  = std::accumulate(win_points.begin(),  win_points.end(),  0.0) / (double)G1_N_WINDOWS;
        std::fprintf(stderr,
            "[g1] windows: nonzero min=%u mean=%.1f max=%u | points min=%llu mean=%.1f max=%llu\n",
            min_window_nonzero, mean_win_nonzero, max_window_nonzero,
            (unsigned long long)min_window_points, mean_win_points, (unsigned long long)max_window_points);
        for (int w = 0; w < G1_N_WINDOWS; ++w) {
            std::fprintf(stderr, "[g1] window[%02d]: nonzero=%u points=%llu\n",
                w, win_nonzero[w], (unsigned long long)win_points[w]);
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================
__device__ __forceinline__ void g1LoadAffSoA(
    const msm_u64x4* __restrict__ Pxv,
    const msm_u64x4* __restrict__ Pyv,
    uint32_t idx,
    FpPointAff& P)
{
    const msm_u64x4 px4 = Pxv[idx];
    const msm_u64x4 py4 = Pyv[idx];

    P.X[0] = px4.x; P.X[1] = px4.y; P.X[2] = px4.z; P.X[3] = px4.w;
    P.Y[0] = py4.x; P.Y[1] = py4.y; P.Y[2] = py4.z; P.Y[3] = py4.w;
    P.infinity = !(P.X[0] | P.X[1] | P.X[2] | P.X[3] |
                   P.Y[0] | P.Y[1] | P.Y[2] | P.Y[3]);
}

__device__ __forceinline__ void g1AccumulateValMixed(
    FpPointJac& acc,
    FpPointJac& tmp,
    const msm_u64x4* __restrict__ Pxv,
    const msm_u64x4* __restrict__ Pyv,
    uint32_t val)
{
    const bool is_neg = (val & G1_NEG_FLAG) != 0u;
    const uint32_t point_id = val & G1_IDX_MASK;

    FpPointAff P;
    g1LoadAffSoA(Pxv, Pyv, point_id, P);
    if (P.infinity) return;
    if (is_neg) fpNeg(P.Y, P.Y);

    pointMixedAdd(acc, P, tmp);
    acc = tmp;
}



// =============================================================================
// Phase 1 — Extract signed digits for all windows in one pass
// =============================================================================
__global__ void g1_extractAllSigned(
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
    for (int w = 0; w < G1_N_WINDOWS; w++) {
        const int bit_start = w * G1_C;
        const int limb      = bit_start >> 6;
        const int bit_off   = bit_start & 63;

        uint32_t raw;
        if (bit_off + G1_C <= 64) {
            const uint64_t src =
                (limb == 0) ? s0 :
                (limb == 1) ? s1 :
                (limb == 2) ? s2 : s3;
            raw = (uint32_t)(src >> bit_off) & ((1u << G1_C) - 1u);
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
            raw = (lo | hi) & ((1u << G1_C) - 1u);
        }

        raw += carry;
        carry = 0u;

        int digit = (int)raw;
        if (raw > (uint32_t)G1_HALF_BUCKETS) {
            digit = (int)((1u << G1_C) - raw);
            carry = 1u;
            digit = -digit;
        }

        d_sd[(size_t)w * (size_t)N + (size_t)i] = digit;
    }
}

__global__ void g1_histogramWindow(
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

__global__ void g1_scatterWindow(
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
    d_perm[pos] = (uint32_t)i | (neg ? G1_NEG_FLAG : 0u);
}

// Classify buckets into small / medium / large lists
// d_counts[0]=n_small, d_counts[1]=n_medium, d_counts[2]=n_large
__global__ void g1_classifyBuckets(
    const uint32_t* __restrict__ d_hist,
    uint32_t* __restrict__       d_small_list,
    uint32_t* __restrict__       d_medium_list,
    uint32_t* __restrict__       d_large_list,
    uint32_t* __restrict__       d_counts,   // [3]: small, medium, large
    int small_thresh,
    int medium_thresh)
{
    int gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (gb >= G1_TOTAL_BUCKETS) return;

    const int w = gb / G1_HALF_BUCKETS;
    const int b = gb - w * G1_HALF_BUCKETS;
    const uint32_t bin   = (uint32_t)w * G1_STRIDE + (uint32_t)b;
    const uint32_t count = d_hist[bin];
    if (count == 0u) return;

    if ((int)count <= small_thresh) {
        const uint32_t pos = atomicAdd(&d_counts[0], 1u);
        d_small_list[pos] = (uint32_t)gb;
    } else if ((int)count <= medium_thresh) {
        const uint32_t pos = atomicAdd(&d_counts[1], 1u);
        d_medium_list[pos] = (uint32_t)gb;
    } else {
        const uint32_t pos = atomicAdd(&d_counts[2], 1u);
        d_large_list[pos] = (uint32_t)gb;
    }
}


// Small buckets: 1 thread per bucket, sequential loop
__global__ void g1_accumulate_small(
    const uint64_t* __restrict__ d_Px,
    const uint64_t* __restrict__ d_Py,
    const uint32_t* __restrict__ d_perm,
    const uint32_t* __restrict__ d_offsets,
    const uint32_t* __restrict__ d_hist,
    const uint32_t* __restrict__ d_small_list,
    uint32_t                     n_small,
    FpPointJac* __restrict__     d_buckets)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_small) return;

    const uint32_t gb    = d_small_list[tid];
    const int w          = (int)(gb / G1_HALF_BUCKETS);
    const int b          = (int)(gb - (uint32_t)w * G1_HALF_BUCKETS);
    const uint32_t bin   = (uint32_t)w * G1_STRIDE + (uint32_t)b;
    const uint32_t start = d_offsets[bin];
    const uint32_t count = d_hist[bin];

    const msm_u64x4* Pxv = reinterpret_cast<const msm_u64x4*>(d_Px);
    const msm_u64x4* Pyv = reinterpret_cast<const msm_u64x4*>(d_Py);

    FpPointJac acc, tmp;
    pointJacSetInfinity(acc);
    #pragma unroll 1
    for (uint32_t k = 0; k < count; k++) {
        g1AccumulateValMixed(acc, tmp, Pxv, Pyv, d_perm[start + k]);
    }
    d_buckets[gb] = acc;
}

// Medium buckets: 1 warp (32 threads) per bucket, warp-level reduction
__global__ void g1_accumulate_medium(
    const uint64_t* __restrict__ d_Px,
    const uint64_t* __restrict__ d_Py,
    const uint32_t* __restrict__ d_perm,
    const uint32_t* __restrict__ d_offsets,
    const uint32_t* __restrict__ d_hist,
    const uint32_t* __restrict__ d_medium_list,
    uint32_t                     n_medium,
    FpPointJac* __restrict__     d_buckets)
{
    const int warp_in_block  = threadIdx.x >> 5;
    const int lane           = threadIdx.x & 31;
    const int warps_per_block= blockDim.x >> 5;
    const uint32_t bucket_idx= (uint32_t)blockIdx.x * (uint32_t)warps_per_block + (uint32_t)warp_in_block;
    if (bucket_idx >= n_medium) return;

    const uint32_t gb    = d_medium_list[bucket_idx];
    const int w          = (int)(gb / G1_HALF_BUCKETS);
    const int b          = (int)(gb - (uint32_t)w * G1_HALF_BUCKETS);
    const uint32_t bin   = (uint32_t)w * G1_STRIDE + (uint32_t)b;
    const uint32_t start = d_offsets[bin];
    const uint32_t count = d_hist[bin];

    const msm_u64x4* Pxv = reinterpret_cast<const msm_u64x4*>(d_Px);
    const msm_u64x4* Pyv = reinterpret_cast<const msm_u64x4*>(d_Py);

    FpPointJac acc, tmp;
    pointJacSetInfinity(acc);
    for (uint32_t k = (uint32_t)lane; k < count; k += 32u) {
        g1AccumulateValMixed(acc, tmp, Pxv, Pyv, d_perm[start + k]);
    }

    // Store partial sums in shared memory then reduce in lane 0
    extern __shared__ unsigned char smem_raw[];
    FpPointJac* smem = reinterpret_cast<FpPointJac*>(smem_raw);
    smem[warp_in_block * 32 + lane] = acc;
    __syncwarp();

    if (lane == 0) {
        FpPointJac total, tmp2;
        pointJacSetInfinity(total);
        for (int t = 0; t < 32; t++) {
            if (!smem[warp_in_block * 32 + t].infinity) {
                pointJacAdd(total, smem[warp_in_block * 32 + t], tmp2);
                total = tmp2;
            }
        }
        d_buckets[gb] = total;
    }
}

__global__ void g1_accumulate_large(
    const uint64_t* __restrict__ d_Px,
    const uint64_t* __restrict__ d_Py,
    const uint32_t* __restrict__ d_perm,
    const uint32_t* __restrict__ d_offsets,
    const uint32_t* __restrict__ d_hist,
    const uint32_t* __restrict__ d_bucket_list,
    uint32_t                     n_large,
    FpPointJac* __restrict__     d_buckets)
{
    const uint32_t bucket_idx = (uint32_t)blockIdx.x;
    if (bucket_idx >= n_large) return;

    const uint32_t gb = d_bucket_list[bucket_idx];
    const int w = (int)(gb / G1_HALF_BUCKETS);
    const int b = (int)(gb - (uint32_t)w * G1_HALF_BUCKETS);
    const uint32_t bin   = (uint32_t)w * G1_STRIDE + (uint32_t)b;
    const uint32_t start = d_offsets[bin];
    const uint32_t count = d_hist[bin];

    const msm_u64x4* Pxv = reinterpret_cast<const msm_u64x4*>(d_Px);
    const msm_u64x4* Pyv = reinterpret_cast<const msm_u64x4*>(d_Py);

    FpPointJac acc, tmp;
    pointJacSetInfinity(acc);

    for (uint32_t k = (uint32_t)threadIdx.x; k < count; k += (uint32_t)blockDim.x) {
        g1AccumulateValMixed(acc, tmp, Pxv, Pyv, d_perm[start + k]);
    }

    extern __shared__ unsigned char smem_raw[];
    FpPointJac* smem = reinterpret_cast<FpPointJac*>(smem_raw);
    smem[threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.x == 0) {
        FpPointJac total, tmp2;
        pointJacSetInfinity(total);
        for (int t = 0; t < blockDim.x; t++) {
            if (!smem[t].infinity) {
                pointJacAdd(total, smem[t], tmp2);
                total = tmp2;
            }
        }
        d_buckets[gb] = total;
    }
}

// g1_reduce (sequential 1-thread/window) — supprimé, remplacé par g1_reduce_chunked.

// =============================================================================
// CHUNKED PIPPENGER REDUCE — single kernel, smem, ~15ms
// =============================================================================
//
// Formule validée (Python + référence GPU 2-pass) :
//
// PHASE 1 (128 threads parallèles, chaque thread k indépendant) :
//   Chunk k = buckets [k*M .. k*M+M-1], parcours décroissant j=M-1..0.
//   TOUS les buckets sont traités, bucket 0 inclus (pas de skip sur idx).
//     if B[idx] non-nul : running += B[idx]
//     if running non-nul : TL += running   ← à CHAQUE position, même vide
//   → S[k]  = running final (somme simple des buckets du chunk)
//   → TL[k] = Σ running_j pour j valides (triangle sum locale)
//   Écriture : smem_S[k] = S[k], smem_R[k] = TL[k]
//   __syncthreads()
//
// PHASE 2 (thread 0 uniquement, suffix scan) :
//   Itère i = C-1 downto 0 :
//     suffix = S[i+1] + ... + S[C-1]  (mis à jour à chaque itération)
//     R[i] = TL[i] + M * suffix        (nb = M pour TOUS les chunks)
//     smem_R[i] = R[i]
//   __syncthreads()
//
// PHASE 3 (128 threads, tree-reduce sur smem_R) :
//   log2(128) = 7 étapes, résultat dans smem_R[0].
//
// SMEM : smem_S[128] | smem_R[128]  = 2 × 128 × 104B = 26KB < 48KB
// LAUNCH : <<<G1_N_WINDOWS, G1_CHUNK_THREADS, smem>>>
// =============================================================================

static constexpr int G1_CHUNK_THREADS = 128;
static_assert(G1_HALF_BUCKETS % G1_CHUNK_THREADS == 0,
    "G1_HALF_BUCKETS must be divisible by G1_CHUNK_THREADS");
static constexpr int G1_CHUNK_SIZE = G1_HALF_BUCKETS / G1_CHUNK_THREADS;

// Scalar mul point EC × petit entier (max 10 bits pour M=512).
__device__ __forceinline__ void g1_jacMulSmall(
    const FpPointJac& P, uint32_t scalar, FpPointJac& res)
{
    pointJacSetInfinity(res);
    if (scalar == 0u || P.infinity) return;
    FpPointJac base = P, tmp;
    while (scalar > 0u) {
        if (scalar & 1u) {
            if (res.infinity) { res = base; }
            else { pointJacAdd(res, base, tmp); res = tmp; }
        }
        scalar >>= 1u;
        if (scalar > 0u) { pointJacDouble(base, tmp); base = tmp; }
    }
}

__global__ void g1_reduce_chunked(
    const FpPointJac* __restrict__ d_buckets,
    FpPointJac* __restrict__       d_window_sums)
{
    const int w = (int)blockIdx.x;
    const int k = (int)threadIdx.x;

    // SMEM layout : smem_S[0..C-1] | smem_R[0..C-1]
    extern __shared__ FpPointJac g1_smem[];
    FpPointJac* smem_S = g1_smem;
    FpPointJac* smem_R = g1_smem + G1_CHUNK_THREADS;

    const FpPointJac* win = d_buckets + (size_t)w * (size_t)G1_HALF_BUCKETS;
    const int base_idx = k * G1_CHUNK_SIZE;

    // --- PHASE 1 : triangle sum locale + somme simple ----------------------
    FpPointJac running, TL, tmp;
    pointJacSetInfinity(running);
    pointJacSetInfinity(TL);

    for (int j = G1_CHUNK_SIZE - 1; j >= 0; j--) {
        const int idx = base_idx + j;   // idx ∈ [0, HALF-1], bucket 0 VALIDE

        const FpPointJac& bkt = win[idx];
        if (!bkt.infinity) {
            if (running.infinity) { running = bkt; }
            else { pointJacAdd(running, bkt, tmp); running = tmp; }
        }
        // Accumuler TL à chaque position (même si bkt est infinity) :
        // running_global[idx] = running_local + suffix, et on accumule
        // running_local à chaque step pour que la formule R=TL+M*suffix soit exacte.
        if (!running.infinity) {
            if (TL.infinity) { TL = running; }
            else { pointJacAdd(TL, running, tmp); TL = tmp; }
        }
    }

    smem_S[k] = running;
    smem_R[k] = TL;
    __syncthreads();

    // --- PHASE 2 : suffix scan (thread 0) ----------------------------------
    if (k == 0) {
        FpPointJac suffix, acc, t2;
        pointJacSetInfinity(suffix);

        for (int i = G1_CHUNK_THREADS - 1; i >= 0; i--) {
            // Lire TL[i] et S[i] avant toute écriture sur smem_R[i]
            FpPointJac R_i = smem_R[i];   // copie locale de TL[i]
            const FpPointJac Si = smem_S[i]; // copie locale de S[i]

            // R[i] = TL[i] + M * suffix[i+1]
            if (!suffix.infinity) {
                g1_jacMulSmall(suffix, (uint32_t)G1_CHUNK_SIZE, acc);
                if (!acc.infinity) {
                    if (R_i.infinity) { R_i = acc; }
                    else { pointJacAdd(R_i, acc, t2); R_i = t2; }
                }
            }
            smem_R[i] = R_i;

            // suffix[i] = S[i] + suffix[i+1]
            if (!Si.infinity) {
                if (suffix.infinity) { suffix = Si; }
                else { pointJacAdd(suffix, Si, t2); suffix = t2; }
            }
        }
    }
    // Barrière : thread 0 a fini d'écrire smem_R avant la tree-reduce
    __syncthreads();

    // --- PHASE 3 : tree-reduce sur smem_R ----------------------------------
    for (int s = G1_CHUNK_THREADS / 2; s > 0; s >>= 1) {
        if (k < s) {
            FpPointJac& a       = smem_R[k];
            const FpPointJac& b = smem_R[k + s];
            if (!b.infinity) {
                if (a.infinity) { a = b; }
                else { pointJacAdd(a, b, tmp); a = tmp; }
            }
        }
        __syncthreads();
    }

    if (k == 0) d_window_sums[w] = smem_R[0];
}

inline void g1_reduce_chunked_launch(
    const FpPointJac* d_buckets,
    FpPointJac*       d_window_sums)
{
    const size_t smem = 2u * (size_t)G1_CHUNK_THREADS * sizeof(FpPointJac);
    g1_reduce_chunked<<<G1_N_WINDOWS, G1_CHUNK_THREADS, smem>>>(d_buckets, d_window_sums);
}

__global__ void g1_combine(
    const FpPointJac* __restrict__ d_window_sums,
    FpPointJac* __restrict__       d_result)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    FpPointJac acc, tmp;
    pointJacSetInfinity(acc);

    for (int w = G1_N_WINDOWS - 1; w >= 0; w--) {
        #pragma unroll
        for (int d = 0; d < G1_C; d++) {
            pointJacDouble(acc, tmp);
            acc = tmp;
        }
        if (!d_window_sums[w].infinity) {
            pointJacAdd(acc, d_window_sums[w], tmp);
            acc = tmp;
        }
    }

    *d_result = acc;
}

__global__ void g1_jacToAff(const FpPointJac* j, FpPointAff* a)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) pointJacToAff(*j, *a);
}

__global__ void g1_initBuckets(FpPointJac* buckets, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) pointJacSetInfinity(buckets[i]);
}

inline void msm_g1(
    const uint64_t* d_Px,
    const uint64_t* d_Py,
    const uint64_t* d_scalars,
    int             N,
    FpPointAff&     h_result)
{
    const G1Config cfg = g1LoadConfigFromEnv();
    const int BLK = 256;
    const size_t total_records = (size_t)N * (size_t)G1_N_WINDOWS;

    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    G1PhaseTimes phase;
    if (cfg.debug) {
        G1_CUDA_CHECK(cudaEventCreate(&ev0));
        G1_CUDA_CHECK(cudaEventCreate(&ev1));
    }

    auto measure = [&](float& slot, auto launch_block) {
        if (!cfg.debug) {
            launch_block();
            return;
        }
        G1_CUDA_CHECK(cudaEventRecord(ev0));
        launch_block();
        G1_CUDA_CHECK(cudaGetLastError());
        G1_CUDA_CHECK(cudaEventRecord(ev1));
        G1_CUDA_CHECK(cudaEventSynchronize(ev1));
        slot += g1ElapsedMs(ev0, ev1);
    };

    int32_t* d_sd = nullptr;
    G1_CUDA_CHECK(cudaMalloc(&d_sd, total_records * sizeof(int32_t)));

    uint32_t *d_hist = nullptr, *d_offsets = nullptr, *d_write = nullptr;
    G1_CUDA_CHECK(cudaMalloc(&d_hist,    (size_t)G1_TOTAL_BINS * sizeof(uint32_t)));
    G1_CUDA_CHECK(cudaMalloc(&d_offsets, (size_t)G1_TOTAL_BINS * sizeof(uint32_t)));
    G1_CUDA_CHECK(cudaMalloc(&d_write,   (size_t)G1_TOTAL_BINS * sizeof(uint32_t)));
    G1_CUDA_CHECK(cudaMemset(d_hist, 0,  (size_t)G1_TOTAL_BINS * sizeof(uint32_t)));

    uint32_t* d_perm = nullptr;
    G1_CUDA_CHECK(cudaMalloc(&d_perm, total_records * sizeof(uint32_t)));

    // Three-level classification lists
    uint32_t *d_small_list  = nullptr, *d_medium_list = nullptr, *d_large_list = nullptr;
    uint32_t *d_counts      = nullptr;  // [3]: n_small, n_medium, n_large
    G1_CUDA_CHECK(cudaMalloc(&d_small_list,  (size_t)G1_TOTAL_BUCKETS * sizeof(uint32_t)));
    G1_CUDA_CHECK(cudaMalloc(&d_medium_list, (size_t)G1_TOTAL_BUCKETS * sizeof(uint32_t)));
    G1_CUDA_CHECK(cudaMalloc(&d_large_list,  (size_t)G1_TOTAL_BUCKETS * sizeof(uint32_t)));
    G1_CUDA_CHECK(cudaMalloc(&d_counts,      3 * sizeof(uint32_t)));
    G1_CUDA_CHECK(cudaMemset(d_counts, 0,    3 * sizeof(uint32_t)));

    FpPointJac *d_buckets = nullptr, *d_window_sums = nullptr, *d_result_jac = nullptr;
    FpPointAff *d_result_aff = nullptr;
    G1_CUDA_CHECK(cudaMalloc(&d_buckets,     (size_t)G1_TOTAL_BUCKETS * sizeof(FpPointJac)));
    G1_CUDA_CHECK(cudaMalloc(&d_window_sums, (size_t)G1_N_WINDOWS * sizeof(FpPointJac)));
    G1_CUDA_CHECK(cudaMalloc(&d_result_jac,  sizeof(FpPointJac)));
    G1_CUDA_CHECK(cudaMalloc(&d_result_aff,  sizeof(FpPointAff)));

    size_t scan_sz = 0;
    G1_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_sz, d_hist, d_offsets, G1_TOTAL_BINS));
    void* d_cub = nullptr;
    G1_CUDA_CHECK(cudaMalloc(&d_cub, scan_sz));

    if (cfg.debug) {
        std::fprintf(stderr, "[g1] config env: C=%d large_blk=%d small_thresh=%d medium_thresh=%d\n",
            G1_C, cfg.large_blk, cfg.small_thresh, cfg.medium_thresh);
    }

    measure(phase.extract_ms, [&] {
        g1_extractAllSigned<<<(N + BLK - 1) / BLK, BLK>>>(d_scalars, N, d_sd);
    });

    measure(phase.hist_ms, [&] {
        for (int w = 0; w < G1_N_WINDOWS; w++) {
            const int32_t* d_sd_win = d_sd + (size_t)w * (size_t)N;
            uint32_t* hist_win = d_hist + (size_t)w * (size_t)G1_STRIDE;
            g1_histogramWindow<<<(N + BLK - 1) / BLK, BLK>>>(d_sd_win, N, hist_win);
        }
    });

    if (cfg.debug) {
        std::vector<uint32_t> h_hist(G1_TOTAL_BINS);
        G1_CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist,
            (size_t)G1_TOTAL_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        g1AnalyzeAndPrintHistogram(h_hist, cfg.debug, cfg.large_blk,
            cfg.small_thresh, cfg.medium_thresh);
    }

    measure(phase.scan_ms, [&] {
        G1_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            d_cub, scan_sz, d_hist, d_offsets, G1_TOTAL_BINS));
        G1_CUDA_CHECK(cudaMemcpy(
            d_write, d_offsets,
            (size_t)G1_TOTAL_BINS * sizeof(uint32_t),
            cudaMemcpyDeviceToDevice));
    });

    measure(phase.scatter_ms, [&] {
        for (int w = 0; w < G1_N_WINDOWS; w++) {
            const int32_t* d_sd_win = d_sd + (size_t)w * (size_t)N;
            uint32_t* write_win = d_write + (size_t)w * (size_t)G1_STRIDE;
            g1_scatterWindow<<<(N + BLK - 1) / BLK, BLK>>>(d_sd_win, N, write_win, d_perm);
        }
    });

    cudaFree(d_sd);
    cudaFree(d_write);
    cudaFree(d_cub);

    measure(phase.classify_ms, [&] {
        g1_classifyBuckets<<<(G1_TOTAL_BUCKETS + BLK - 1) / BLK, BLK>>>(
            d_hist,
            d_small_list, d_medium_list, d_large_list, d_counts,
            cfg.small_thresh, cfg.medium_thresh);
    });

    uint32_t h_counts[3] = {0, 0, 0};
    G1_CUDA_CHECK(cudaMemcpy(h_counts, d_counts, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    const uint32_t n_small  = h_counts[0];
    const uint32_t n_medium = h_counts[1];
    const uint32_t n_large  = h_counts[2];
    if (cfg.debug) {
        std::fprintf(stderr,
            "[g1] classified: small=%u medium=%u large=%u total_nonzero=%u\n",
            n_small, n_medium, n_large, n_small + n_medium + n_large);
    }

    measure(phase.init_ms, [&] {
        g1_initBuckets<<<(G1_TOTAL_BUCKETS + BLK - 1) / BLK, BLK>>>(d_buckets, G1_TOTAL_BUCKETS);
    });

    // Small: 1 thread/bucket, packed into warps
    if (n_small) {
        const int SBLK = 256;
        measure(phase.small_ms, [&] {
            g1_accumulate_small<<<(n_small + SBLK - 1) / SBLK, SBLK>>>(
                d_Px, d_Py, d_perm, d_offsets, d_hist, d_small_list, n_small, d_buckets);
        });
    }

    // Medium: 1 warp (32 threads) per bucket
    if (n_medium) {
        const int MBLK = 128;  // 4 warps per block
        const int med_grids = (int)((n_medium + (MBLK/32) - 1) / (MBLK/32));
        const size_t smem_med = (size_t)MBLK * sizeof(FpPointJac);
        measure(phase.medium_ms, [&] {
            g1_accumulate_medium<<<med_grids, MBLK, smem_med>>>(
                d_Px, d_Py, d_perm, d_offsets, d_hist, d_medium_list, n_medium, d_buckets);
        });
    }

    // Large: 1 CTA/bucket
    if (n_large) {
        const size_t smem_large = (size_t)cfg.large_blk * sizeof(FpPointJac);
        measure(phase.large_ms, [&] {
            g1_accumulate_large<<<n_large, cfg.large_blk, smem_large>>>(
                d_Px, d_Py, d_perm, d_offsets, d_hist, d_large_list, n_large, d_buckets);
        });
    }

    cudaFree(d_perm);
    cudaFree(d_offsets);
    cudaFree(d_hist);
    cudaFree(d_small_list);
    cudaFree(d_medium_list);
    cudaFree(d_large_list);
    cudaFree(d_counts);

    measure(phase.reduce_ms, [&] {
        g1_reduce_chunked_launch(d_buckets, d_window_sums);
    });
    cudaFree(d_buckets);

    measure(phase.combine_ms, [&] {
        g1_combine<<<1, 1>>>(d_window_sums, d_result_jac);
    });
    cudaFree(d_window_sums);

    measure(phase.jac2aff_ms, [&] {
        g1_jacToAff<<<1, 1>>>(d_result_jac, d_result_aff);
    });

    G1_CUDA_CHECK(cudaDeviceSynchronize());
    G1_CUDA_CHECK(cudaMemcpy(&h_result, d_result_aff, sizeof(FpPointAff), cudaMemcpyDeviceToHost));

    if (cfg.debug) {
        phase.total_ms = phase.extract_ms + phase.hist_ms + phase.scan_ms + phase.scatter_ms +
                         phase.classify_ms + phase.init_ms +
                         phase.small_ms + phase.medium_ms + phase.large_ms +
                         phase.reduce_ms + phase.combine_ms + phase.jac2aff_ms;
        g1PrintPhaseTimes(phase);
        G1_CUDA_CHECK(cudaEventDestroy(ev0));
        G1_CUDA_CHECK(cudaEventDestroy(ev1));
    }

    cudaFree(d_result_jac);
    cudaFree(d_result_aff);
}

} // namespace msm_g1_ns


inline void msm_g1(
    const uint64_t* d_Px,
    const uint64_t* d_Py,
    const uint64_t* d_scalars,
    int             N,
    FpPointAff&     h_result)
{
    msm_g1_ns::msm_g1(d_Px, d_Py, d_scalars, N, h_result);
}
