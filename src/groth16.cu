/*
 * ===========================================================================
 * Forum / Zyklop ??? groth16.cu
 * BN254 Groth16 prover ??? impl??mentation GPU
 * ===========================================================================
 *
 * +-------------------------------------------------------------------------+
 * |  LESSONS LEARNED ??? d??bogage pi_C (session 2025-03)                     |
 * |  Toutes les subtilit??s d??couvertes en comparant avec snarkjs prove      |
 * |                                                                         |
 * |  1. ENCODAGE DES COEFFICIENTS ??? double-Montgomery (section 4)           |
 * |     Les coefficients QAP en section 4 du zkey sont stock??s sous         |
 * |     la forme  coef * R^2  (DEUX facteurs Montgomery), pas  coef * R.     |
 * |     cpuFrDecodeCanon doit multiplier par R^-^2 pour obtenir la forme      |
 * |     canonique.  R^-^2 mod r est une constante hardcod??e.                  |
 * |     Sans ce d??codage, les Aw/Bw sont faux et tout pi_C est faux.        |
 * |                                                                         |
 * |  2. POLYN??ME H ??? base "coset impair" de snarkjs                         |
 * |     Section 9 stocke : H[i] = tau^(2i+1) * delta^-^1 * G_1                     |
 * |     (puissances impaires de tau, divis??es par delta lors de la c??r??monie).    |
 * |     Les scalaires MSM sont les n ??valuations de (A*B ??? C) aux points    |
 * |     impairs du coset omega_{2n} :                                           |
 * |       h_odd[j] = A(omega_{2n}^{2j+1}) * B(omega_{2n}^{2j+1})                  |
 * |                ??? C(omega_{2n}^{2j+1})                                       |
 * |     SANS division par Z (snarkjs ne divise pas par Z = ???2 malgr??        |
 * |     Z(omega_{2n}^{2j+1}) = ???2 constant pour tout j).                        |
 * |     qap_joinABC calcule juste A*B ??? C en Montgomery, puis               |
 * |     frm_batchFromMontgomery d??code vers canonique.                       |
 * |                                                                         |
 * |  3. G??N??RATEUR DU COSET ??? omega_{2n}, PAS g = 7                            |
 * |     snarkjs fait : IFFT_n(Aw) -> coef[i] *= omega_{2n}^i -> FFT_n           |
 * |     -> ??valuation en A(omega_{2n}^{2j+1}).                                   |
 * |     Le g??n??rateur est omega_{2n} = 5^{(r???1)/(2n)} mod r,                   |
 * |     PAS le g??n??rateur de coset standard g = 7 utilis?? dans notre        |
 * |     impl??mentation pr??c??dente.                                           |
 * |     Dans la table de twiddle nttPrepare(ld+1) :                         |
 * |       omega_{2n} = tw[stride]  o?? stride = 2^(log_n_max ??? (log_n+1))       |
 * |       -> accessible via nttGetOmega2n(log_n+1, out).                     |
 * |                                                                         |
 * |  4. NOMBRE DE POINTS H ??? n, pas n???1                                     |
 * |     snarkjs g??n??re n points H[0..n???1] (= domainSize points).            |
 * |     On lit n points en section 9 et on passe n scalaires au MSM.        |
 * |     L'ancienne impl??mentation utilisait n???1 (trop peu).                 |
 * |                                                                         |
 * |  5. pi_A ET pi_B ??? alpha/beta doivent ??tre inclus explicitement         |
 * |     Les sections 5/7 ne contiennent PAS alpha_1/beta_2 (ce sont les points       |
 * |     A_i = alpha*A_i(tau) + ??? / delta, pas le terme alpha_1 seul).                     |
 * |     Il faut ajouter :                                                    |
 * |       pi_A = alpha_1  +  MSM(section 5, w[0..m])                            |
 * |       pi_B = beta_2  +  MSM(section 7, w[0..m])                            |
 * |     Impl??ment?? en pr??pendant alpha_1/beta_2 comme un point virtuel scalar=1.    |
 * |                                                                         |
 * |  6. ORDRE DES COORDONN??ES pi_B EN JSON                                  |
 * |     snarkjs sort les coordonn??es G_2 dans l'ordre [x.a0, x.a1]           |
 * |     (coefficient de degr?? 0 en premier), pas [x.a1, x.a0].              |
 * |                                                                         |
 * |  7. ENCODAGE DU WITNESS                                                  |
 * |     Le fichier .wtns stocke les valeurs en Fr CANONIQUE.                 |
 * |     On encode vers Montgomery via k_fr_encode_batch avant la NTT.       |
 * +-------------------------------------------------------------------------+
 *
 *  Dependances :
 *    include/ : fp_bn254.cuh, fr_bn254.cuh, fp2_bn254.cuh,
 *               msm_g1.cuh, msm_g2.cuh, ntt_bn254.cuh, groth16.cuh
 *    src/     : binfile_utils.{cpp,hpp}, zkey_utils.{cpp,hpp},
 *               wtns_utils.{cpp,hpp}   (fork iden3/rapidsnark, LGPL-3)
 *
 *  Compile :
 *    nvcc -O3 -std=c++17 -arch=sm_120             \
 *         -I../include                             \
 *         groth16.cu binfile_utils.cpp             \
 *         zkey_utils.cpp wtns_utils.cpp            \
 *         -o prover
 * ===========================================================================
 */

#include "../include/fp_bn254.cuh"
#include "../include/fp2_bn254.cuh"
#include "../include/fr_bn254.cuh"
#include "../include/msm_g1.cuh"
// Experimental medium affine-batch v0 lives behind ZK_G1_MEDIUM_AFFINE_V0 in msm_g1.cuh.
#include "../include/msm_g2.cuh"
#include "../include/ntt_bn254.cuh"
#include "../include/groth16.cuh"
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "wtns_utils.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <future>
#include <thread>
#include <omp.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

// ============================================================================
// Macros d'erreur CUDA
// ============================================================================
#define G16_CUDA_CHECK(x) do {                                              \
    cudaError_t _e = (x);                                                   \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr,"[groth16] CUDA error %s:%d : %s\n",                 \
                __FILE__,__LINE__,cudaGetErrorString(_e));                  \
        exit(1);                                                             \
    }                                                                        \
} while(0)

// Forward declarations
static void cpuFrAddCanon(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
// Decode Montgomery form to canonical: out = a * R^{-1} mod Fr_r
static void cpuFrMulCanon(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]);
static void cpuFrDecodeCanon(const uint64_t a[4], uint64_t out[4]);
void g16_g1_add_host(const FpPointAff* A, const FpPointAff* B, FpPointAff* R);


struct G16CRSSectionStats {
    uint32_t points = 0;
    uint64_t raw_bytes = 0;
    double start_ms = 0.0;
    double read_raw_ms = 0.0;
    double parse_pack_ms = 0.0;
    double end_ms = 0.0;
    double total_ms = 0.0;
};

static inline double g16NowMs();

// ============================================================================
// 1. KERNELS GPU
// ============================================================================

/*
 * k_sparse_mvm_direct — MVM direct sans décodage préalable
 * coef_vals_raw : Valeurs brutes (c * R^2)
 * witness_canon : Témoins normaux (w)
 * Sortie : (c * w * R) -> Automatiquement en Montgomery !
 */
__global__ void k_sparse_mvm_direct(
    const uint64_t* __restrict__ coef_vals_raw,
    const uint32_t* __restrict__ coef_sigs,
    const uint32_t* __restrict__ row_start,
    const uint64_t* __restrict__ witness_canon,
    uint64_t* out,
    uint32_t                     domain_size)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= domain_size) return;

    uint64_t acc[4] = {0,0,0,0};
    const uint32_t start = row_start[row];
    const uint32_t end   = row_start[row + 1];

    for (uint32_t k = start; k < end; k++) {
        uint64_t prod[4];
        // frMul(c*R^2, w) = c * w * R
        frMul(coef_vals_raw + k*4, witness_canon + coef_sigs[k]*4, prod);
        frAdd(acc, prod, acc);
    }
    out[row*4+0]=acc[0]; out[row*4+1]=acc[1];
    out[row*4+2]=acc[2]; out[row*4+3]=acc[3];
}

/*
 * k_fr_pointwise_mul ??? multiplication pointwise de deux tableaux Fr
 *  out[i] = a[i] * b[i]  (Montgomery)
 *  N : nombre d'??l??ments Fr
 */
__global__ void k_fr_pointwise_mul(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t*                    out,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    frMul(a + i*4, b + i*4, out + i*4);
}

/*
 * k_fr_pointwise_sub ??? soustraction pointwise
 *  out[i] = a[i] - b[i]  (Montgomery)
 */
__global__ void k_fr_pointwise_sub(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t*                    out,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    frSub(a + i*4, b + i*4, out + i*4);
}

/*
 * k_fr_encode_batch ??? encode un tableau Fr canonique -> Montgomery
 *  out[i] = in[i] * R mod r
 *  in et out peuvent ??tre le m??me buffer (in-place)
 */
__global__ void k_fr_encode_batch(
    const uint64_t* __restrict__ in,
    uint64_t*                    out,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    frMontEncode(in + i*4, out + i*4);
}

/*
 * k_fr_decode_batch ??? d??code Montgomery -> canonique
 */
__global__ void k_fr_decode_batch(
    const uint64_t* __restrict__ in,
    uint64_t*                    out,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    frMontDecode(in + i*4, out + i*4);
}

/*
 * k_fp_decode_batch ??? d??code Montgomery Fp -> canonique
 *  Utilis?? pour extraire x,y des G1 points avant s??rialisation
 */
__global__ void k_fp_decode_batch(
    const uint64_t* __restrict__ in,
    uint64_t*                    out,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    fpMontDecode(in + i*4, out + i*4);
}

/*
 * k_unpack_g1_aos_to_soa — Transforme [X, Y, X, Y...] (AoS) en [X,X...][Y,Y...] (SoA)
 */
__global__ void k_unpack_g1_aos_to_soa(
    const uint64_t* __restrict__ aos,
    uint64_t* __restrict__ Px,
    uint64_t* __restrict__ Py,
    uint32_t offset,
    uint32_t N_chunk)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_chunk) return;

    // aos a 8 uint64_t par point (64 bytes)
    const uint64_t* b = aos + i * 8;
    uint32_t out_idx = offset + i;

    Px[out_idx*4 + 0] = b[0];
    Px[out_idx*4 + 1] = b[1];
    Px[out_idx*4 + 2] = b[2];
    Px[out_idx*4 + 3] = b[3];

    Py[out_idx*4 + 0] = b[4];
    Py[out_idx*4 + 1] = b[5];
    Py[out_idx*4 + 2] = b[6];
    Py[out_idx*4 + 3] = b[7];
}

/*
 * k_unpack_g2_aos_to_soa — Transforme [x.a0, x.a1, y.a0, y.a1...] en SoA
 */
__global__ void k_unpack_g2_aos_to_soa(
    const uint64_t* __restrict__ aos,
    uint64_t* __restrict__ Px,
    uint64_t* __restrict__ Py,
    uint32_t offset,
    uint32_t N_chunk)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_chunk) return;

    // G2 a 16 uint64_t par point (128 bytes)
    const uint64_t* b = aos + i * 16;
    uint32_t out_idx = offset + i;

    Px[out_idx*8 + 0] = b[0]; Px[out_idx*8 + 1] = b[1];
    Px[out_idx*8 + 2] = b[2]; Px[out_idx*8 + 3] = b[3];
    Px[out_idx*8 + 4] = b[4]; Px[out_idx*8 + 5] = b[5];
    Px[out_idx*8 + 6] = b[6]; Px[out_idx*8 + 7] = b[7];

    Py[out_idx*8 + 0] = b[8]; Py[out_idx*8 + 1] = b[9];
    Py[out_idx*8 + 2] = b[10]; Py[out_idx*8 + 3] = b[11];
    Py[out_idx*8 + 4] = b[12]; Py[out_idx*8 + 5] = b[13];
    Py[out_idx*8 + 6] = b[14]; Py[out_idx*8 + 7] = b[15];
}

/*
/*
 * k_calc_inv_z ??? calcule inv_z[i] = 1 / Z(g*omega_2n^i)  pour i=0..N2-1
 *
 * Z(g*omega_2n^i) = (g*omega_2n^i)^n - 1 = g^n*(-1)^i - 1
 *   i pair  : Z = g^n - 1
 *   i impair: Z = -g^n - 1 = r - g^n - 1
 *
 * N  = domaine standard (n)
 * N2 = domaine ??tendu (2n)
 * Lanc?? avec N2 threads.
 */
__global__ void k_calc_inv_z(int N, int N2, uint64_t* d_inv_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N2) return;

    // g = 7 (constante coset bn254) en Montgomery
    uint64_t g[4] = {7, 0, 0, 0};
    uint64_t g_mont[4];
    frMontEncode(g, g_mont);

    // gn = g^N en Montgomery
    uint64_t gn[4];
    frPow(g_mont, (uint64_t)N, gn);

    // one_mont = Montgomery(1)
    uint64_t one[4] = {1, 0, 0, 0};
    uint64_t one_mont[4];
    frMontEncode(one, one_mont);

    // Z_i = g^n * (-1)^i - 1
    uint64_t zn[4];
    if (i % 2 == 0) {
        // Z = g^n - 1
        frSub(gn, one_mont, zn);
    } else {
        // Z = -g^n - 1 = -(g^n + 1)
        uint64_t gn_plus_1[4];
        frAdd(gn, one_mont, gn_plus_1);
        frNeg(gn_plus_1, zn);
    }

    // inv_z[i] = 1 / Z_i
    frInv(zn, d_inv_z + i * 4);
}

/*
 * k_h_poly_sub_and_div ??? h[i] = (AwBw[i] - Cw[i]) * inv_z[i]
 * inv_z est maintenant un tableau de N2 ??l??ments Montgomery.
 */
__global__ void k_h_poly_sub_and_div(
    const uint64_t* __restrict__ AwBw,
    const uint64_t* __restrict__ Cw,
    const uint64_t* __restrict__ inv_z,  // N2 ??l??ments
    uint64_t* out_h,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint64_t num[4];
    frSub(AwBw + i*4, Cw + i*4, num);
    frMul(num, inv_z + i*4, out_h + i*4);  // inv_z[i] sp??cifique
}

// ============================================================================
// 2. UTILITAIRES CPU ??? lectures .zkey / .wtns via rapidsnark parsers
// ============================================================================

void groth16ReadZkeyHeader(const char* zkey_path, Groth16ZkeyHeader& hdr)
{
    BinFileUtils::BinFile zkey(zkey_path, "zkey", 1);
    auto zkh = ZKeyUtils::loadHeader(&zkey);

    hdr.n_constraints = zkh->domainSize;
    hdr.n_vars        = zkh->nVars;         // m+1
    hdr.n_public      = zkh->nPublic;       // l

    // domain_size = next power-of-2 ??? n_constraints
    uint32_t d = 1; int log_d = 0;
    while (d < hdr.n_constraints) { d <<= 1; log_d++; }
    hdr.domain_size = d;
    hdr.log_domain  = log_d;

    // V??rifie le champ scalaire BN254
    const uint64_t BN254_R[4] = {
        0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
        0xb85045b68181585dULL, 0x30644e72e131a029ULL
    };
    memcpy(hdr.prime, BN254_R, 32);
}

// ============================================================================
// 3. Arithm??tique Fr sur CPU
// ============================================================================

static const uint64_t FR_MOD[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};

// Addition Fr canonique sur CPU (borrow-safe)
// Addition Fr canonique sur CPU (100% borrow-safe)
static void cpuFrAddCanon(const uint64_t a[4], const uint64_t b[4], uint64_t r[4])
{
    uint64_t tmp[4];
    uint64_t carry = 0;
    
    // 1. Addition classique sur 256 bits
    for (int i = 0; i < 4; i++) {
        unsigned __int128 sum = (unsigned __int128)a[i] + b[i] + carry;
        tmp[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }
    
    // 2. V??rification si tmp >= FR_MOD
    bool geq = carry > 0;
    if (!geq) {
        for (int i = 3; i >= 0; i--) {
            if (tmp[i] > FR_MOD[i]) { geq = true;  break; }
            if (tmp[i] < FR_MOD[i]) { geq = false; break; }
        }
    }
    
    // 3. Soustraction de FR_MOD si d??passement
    if (geq) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            // ICI on soustrait directement le borrow, sans bitshift farfelu !
            unsigned __int128 d = (unsigned __int128)tmp[i] - FR_MOD[i] - borrow;
            r[i] = (uint64_t)d;
            borrow = (uint64_t)(d >> 64) ? 1 : 0;
        }
    } else {
        r[0] = tmp[0]; r[1] = tmp[1]; r[2] = tmp[2]; r[3] = tmp[3];
    }
}


// Multiplication Fr canonique sur CPU ??? schoolbook 4??4 limbs, r??duction Barrett
// ATTENTION : __int128 est utilis?? ici mais en -O2 (pas -O3 sur cette fonction).
// Solution : pragma optimize ou utiliser __uint128_t sans __int128.
// On utilise __uint128_t uniquement (unsigned) qui est correct sous -O3.
// Multiplication Fr canonique sur CPU (Division longue 100% robuste)
static void cpuFrMulCanon(const uint64_t a[4], const uint64_t b[4], uint64_t r[4])
{
    uint64_t t[8] = {0};
    for(int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for(int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)a[i] * b[j] + t[i+j] + carry;
            t[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[i+4] = carry;
    }

    // R??duction bit-??-bit stricte
    for (int shift = 255; shift >= 0; shift--) {
        uint64_t shifted_r[8] = {0};
        int word_shift = shift / 64;
        int bit_shift = shift % 64;
        if (bit_shift == 0) {
            for(int i = 0; i < 4; i++) shifted_r[i+word_shift] = FR_MOD[i];
        } else {
            uint64_t carry = 0;
            for(int i = 0; i < 4; i++) {
                shifted_r[i+word_shift] = (FR_MOD[i] << bit_shift) | carry;
                carry = FR_MOD[i] >> (64 - bit_shift);
            }
            shifted_r[4+word_shift] = carry;
        }

        bool geq = false;
        for (int i = 7; i >= 0; i--) {
            if (t[i] > shifted_r[i]) { geq = true; break; }
            if (t[i] < shifted_r[i]) { geq = false; break; }
        }
        if (geq) {
            uint64_t borrow = 0;
            for (int i = 0; i < 8; i++) {
                unsigned __int128 diff = (unsigned __int128)t[i] - shifted_r[i] - borrow;
                t[i] = (uint64_t)diff;
                borrow = (diff >> 64) ? 1 : 0;
            }
        }
    }
    r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
}

// -- cpuFrDecodeCanon ----------------------------------------------------------
// D??code un ??l??ment Fr depuis la forme DOUBLE-MONTGOMERY vers canonique.
//
// POURQUOI double-Montgomery ?
//   snarkjs (via ffjavascript) utilise la multiplication de Montgomery :
//     frm_mul(a, b) = a * b * R^-^1  mod r
//   Quand il g??n??re le zkey (section 4), les coefficients sont calcul??s
//   comme  coef_raw * R^2  dans le code de setup. R??sultat : la valeur
//   stock??e sur le disque est  coef_canonical * R^2.
//
//   Notre cpuFrMulCanon est une multiplication ordinaire (a*b mod r),
//   donc pour r??cup??rer coef_canonical depuis coef_raw, il faut :
//     coef_canonical = coef_raw * R^-^2  mod r
//
//   R^-^2 = (2^2?????? mod r)^-^2 mod r  (constante pour BN254 Fr)
//     = 0x12d5f775e436631e_e065f3e379a1edeb_52f28270b38e2428_ae12ba81d3c71148
//
// V??rifi?? exp??rimentalement : avec ce d??codage, les scalaires h_odd[j]
// calcul??s par notre GPU correspondent exactement aux buffPodd_T de snarkjs.
static void cpuFrDecodeCanon(const uint64_t a[4], uint64_t out[4]) {
    // value stored in zkey section 4 = coef * R^2 (double-Montgomery)
    // cpuFrMontMul(x, 1) = x * R^-1  (one Montgomery step)
    // Apply twice: coef*R^2 -> coef*R -> coef
    // This is ~15x faster than cpuFrMulCanon (schoolbook Barrett)
    static const uint64_t one[4] = {1ULL, 0ULL, 0ULL, 0ULL};
    uint64_t tmp[4];
    cpuFrMontMul(a,   one, tmp);  // coef*R^2 * 1 * R^-1 = coef*R
    cpuFrMontMul(tmp, one, out);  // coef*R   * 1 * R^-1 = coef
}

// ============================================================================
// 3.2 GPU Sparse MVM
// ============================================================================
//
// Coefs après readCoefficients = canonique Fr (cpuFrDecodeCanon a été appliqué)
// Witness = canonique Fr (.wtns)
//
// Stratégie : encoder les deux en Montgomery sur GPU, puis frMul.
//   frMul(a*R, b*R) = a*b*R  (même résultat que cpuFrMulCanon + k_fr_encode_batch)
//
// Format CSR : coefs déjà triés par constraint dans le zkey.
//   1 thread par constraint, itère sur ses coefs.
//   Pas d'atomics.

__global__ void k_sparse_mvm(
    const uint64_t* __restrict__ coef_vals_mont,
    const uint32_t* __restrict__ coef_sigs,
    const uint32_t* __restrict__ row_start,
    const uint64_t* __restrict__ witness_mont,
    uint64_t*                    out,
    uint32_t                     domain_size)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= domain_size) return;

    uint64_t acc[4] = {0,0,0,0};
    const uint32_t start = row_start[row];
    const uint32_t end   = row_start[row + 1];

    for (uint32_t k = start; k < end; k++) {
        uint64_t prod[4];
        frMul(coef_vals_mont + k*4, witness_mont + coef_sigs[k]*4, prod);
        frAdd(acc, prod, acc);
    }
    out[row*4+0]=acc[0]; out[row*4+1]=acc[1];
    out[row*4+2]=acc[2]; out[row*4+3]=acc[3];
}

// ============================================================================
// 4. Lecture sections .zkey (points G1/G2) et witness
// ============================================================================

/*
 * readAndUnpackG1Chunked — Pipeline agressif CPU->GPU
 * Double buffering avec mémoire bloquée (Pinned) + Unpack asynchrone GPU.
 */
static void readAndUnpackG1Chunked(
    BinFileUtils::BinFile& zkey,
    int section_id,
    uint32_t N,
    uint64_t* d_Px,
    uint64_t* d_Py,
    cudaStream_t stream)
{
    // Pointeur direct vers la section mappée du fichier (zéro-copy depuis le SSD)
    const uint64_t* raw_mmap = (const uint64_t*)zkey.getSectionData(section_id);

    // Chunk size : 512k points = 33.5 MB par chunk
    uint32_t chunk_pts = 512 * 1024;
    uint32_t chunk_bytes = chunk_pts * 64;

    uint64_t* h_pinned[2];
    uint64_t* d_raw[2];
    cudaEvent_t events[2];

    // Initialisation des 2 pipelines
    for (int i = 0; i < 2; i++) {
        G16_CUDA_CHECK(cudaMallocHost(&h_pinned[i], chunk_bytes)); // Mémoire RAM hyper-rapide
        G16_CUDA_CHECK(cudaMalloc(&d_raw[i], chunk_bytes));        // VRAM temporaire brute
        G16_CUDA_CHECK(cudaEventCreate(&events[i]));
        G16_CUDA_CHECK(cudaEventRecord(events[i], stream));        // Marque comme "prêt"
    }

    for (uint32_t offset = 0; offset < N; offset += chunk_pts) {
        uint32_t pts = std::min(chunk_pts, N - offset);
        uint32_t bytes = pts * 64;
        int b = (offset / chunk_pts) % 2; // Alterne entre buffer 0 et 1

        // 1. Le CPU attend que le GPU ait fini avec le buffer 'b'
        G16_CUDA_CHECK(cudaEventSynchronize(events[b]));

        // 2. Le CPU copie les données brutes dans le buffer Pinned (Très rapide)
		#pragma omp parallel for num_threads(8)
        for (uint32_t i = 0; i < bytes / 64; i++) {
            memcpy(h_pinned[b] + i * 8, raw_mmap + offset * 8 + i * 8, 64);
        }

        // 3. Envoi DMA asynchrone vers le GPU (Le CPU passe immédiatement à la suite)
        G16_CUDA_CHECK(cudaMemcpyAsync(d_raw[b], h_pinned[b], bytes, cudaMemcpyHostToDevice, stream));

        // 4. Tri GPU asynchrone
        int threads = 256;
        int blocks = (pts + threads - 1) / threads;
        k_unpack_g1_aos_to_soa<<<blocks, threads, 0, stream>>>(d_raw[b], d_Px, d_Py, offset, pts);

        // 5. On place un marqueur pour dire "Le buffer b peut être écrasé au prochain tour"
        G16_CUDA_CHECK(cudaEventRecord(events[b], stream));
    }

    // Attendre que le dernier chunk soit fini avant de désallouer
    G16_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < 2; i++) {
        cudaFreeHost(h_pinned[i]);
        cudaFree(d_raw[i]);
        cudaEventDestroy(events[i]);
    }
}

/*
 * readAndUnpackG2Chunked — Pipeline agressif CPU->GPU pour G2
 */
static void readAndUnpackG2Chunked(
    BinFileUtils::BinFile& zkey,
    int section_id,
    uint32_t N,
    uint64_t* d_Px,
    uint64_t* d_Py,
    cudaStream_t stream)
{
    const uint64_t* raw_mmap = (const uint64_t*)zkey.getSectionData(section_id);

    uint32_t chunk_pts = 256 * 1024;
    uint32_t chunk_bytes = chunk_pts * 128; // 33.5 MB par chunk Pinned

    uint64_t* h_pinned[2];
    uint64_t* d_raw[2];
    cudaEvent_t events[2];

    for (int i = 0; i < 2; i++) {
        G16_CUDA_CHECK(cudaMallocHost(&h_pinned[i], chunk_bytes));
        G16_CUDA_CHECK(cudaMalloc(&d_raw[i], chunk_bytes));
        G16_CUDA_CHECK(cudaEventCreate(&events[i]));
        G16_CUDA_CHECK(cudaEventRecord(events[i], stream));
    }

    for (uint32_t offset = 0; offset < N; offset += chunk_pts) {
        uint32_t pts = std::min(chunk_pts, N - offset);
        uint32_t bytes = pts * 128;
        int b = (offset / chunk_pts) % 2;

        G16_CUDA_CHECK(cudaEventSynchronize(events[b]));

        #pragma omp parallel for num_threads(8)
        for (uint32_t i = 0; i < bytes / 128; i++) {
            memcpy(h_pinned[b] + i * 16, raw_mmap + offset * 16 + i * 16, 128);
        }

        G16_CUDA_CHECK(cudaMemcpyAsync(d_raw[b], h_pinned[b], bytes, cudaMemcpyHostToDevice, stream));

        int threads = 256;
        int blocks = (pts + threads - 1) / threads;
        k_unpack_g2_aos_to_soa<<<blocks, threads, 0, stream>>>(d_raw[b], d_Px, d_Py, offset, pts);

        G16_CUDA_CHECK(cudaEventRecord(events[b], stream));
    }

    G16_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < 2; i++) {
        cudaFreeHost(h_pinned[i]);
        cudaFree(d_raw[i]);
        cudaEventDestroy(events[i]);
    }
}

// ============================================================================
// 4.bis AUDIT / PARSING PARALLELE DE LA SECTION 4
// ============================================================================

struct G16Section4Stats {
    uint64_t nCoefs           = 0;
    uint64_t raw_bytes        = 0;
    uint64_t chunk_entries    = 0;
    uint64_t chunks           = 0;
    uint32_t threads          = 0;

    uint64_t count_A          = 0;
    uint64_t count_B          = 0;
    uint64_t count_other      = 0;

    uint64_t cap_growth_A     = 0;
    uint64_t cap_growth_B     = 0;

    double start_ms           = 0.0;
    double header_ms          = 0.0;
    double reserve_ms         = 0.0;
    double read_raw_ms        = 0.0;
    double count_ms           = 0.0;
    double parse_fill_ms      = 0.0;
    double end_ms             = 0.0;
    double total_ms           = 0.0;
};

static inline double g16NowMs() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
        clk::now().time_since_epoch()).count();
}

static int g16GetEnvInt(const char* name, int defval, int lo, int hi) {
    const char* s = std::getenv(name);
    if (!s || !*s) return defval;
    long v = std::strtol(s, nullptr, 10);
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return (int)v;
}

template <typename Fn>
static void g16ParallelForChunks(size_t n_chunks, int n_threads, Fn fn) {
    if (n_chunks == 0) return;
    #pragma omp parallel for schedule(dynamic,1) num_threads(n_threads)
    for (int c = 0; c < (int)n_chunks; c++) fn((size_t)c);
}

/*
 * readAndComputeMVM — Remplace readCoefficients + gpuSparseMvm
 * Zéro-allocation CPU superflue, Zéro-décodage Montgomery.
 */
static void readAndComputeMVM(
    BinFileUtils::BinFile& zkey,
    const std::vector<uint64_t>& h_witness,
    uint32_t n_vars,
    uint32_t domain_size,
    uint64_t** d_Aw_out,
    uint64_t** d_Bw_out)
{
    double t0 = g16NowMs();

    zkey.startReadSection(4);
    uint32_t nCoefs = zkey.readU32LE();
    // Zero-copy read de tout le buffer
    const uint8_t* raw = (const uint8_t*)zkey.read((size_t)nCoefs * 44ull);

    std::vector<uint32_t> row_start_A(domain_size + 1, 0);
    std::vector<uint32_t> row_start_B(domain_size + 1, 0);

    // 1. Comptage rapide
    uint32_t total_A = 0, total_B = 0;
    const uint8_t* p = raw;
    for (size_t i = 0; i < nCoefs; i++, p += 44) {
        uint32_t mat = *(const uint32_t*)p;
        uint32_t row = *(const uint32_t*)(p + 4);
        if (mat == 0)      { row_start_A[row + 1]++; total_A++; }
        else if (mat == 1) { row_start_B[row + 1]++; total_B++; }
    }

    // 2. Prefix sum (Création de l'index CSR)
    for (uint32_t i = 0; i < domain_size; i++) {
        row_start_A[i+1] += row_start_A[i];
        row_start_B[i+1] += row_start_B[i];
    }

    // 3. Allocations Pinned Memory
    uint64_t *h_vals_A, *h_vals_B;
    uint32_t *h_sigs_A, *h_sigs_B;
    G16_CUDA_CHECK(cudaMallocHost(&h_vals_A, total_A * 32));
    G16_CUDA_CHECK(cudaMallocHost(&h_sigs_A, total_A * 4));
    G16_CUDA_CHECK(cudaMallocHost(&h_vals_B, total_B * 32));
    G16_CUDA_CHECK(cudaMallocHost(&h_sigs_B, total_B * 4));

    std::vector<uint32_t> write_pos_A = row_start_A;
    std::vector<uint32_t> write_pos_B = row_start_B;

    // 4. Remplissage multi-threadé avec variables atomiques
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < nCoefs; i++) {
        const uint8_t* p_coef = raw + i * 44;
        uint32_t mat = *(const uint32_t*)p_coef;
        uint32_t row = *(const uint32_t*)(p_coef + 4);
        uint32_t sig = *(const uint32_t*)(p_coef + 8);

        if (mat == 0) {
            uint32_t idx;
            #pragma omp atomic capture
            idx = write_pos_A[row]++;
            
            // Copie brute : c * R^2 (Pas de décodage !)
            memcpy(h_vals_A + idx * 4, p_coef + 12, 32);
            h_sigs_A[idx] = sig;
        } else if (mat == 1) {
            uint32_t idx;
            #pragma omp atomic capture
            idx = write_pos_B[row]++;
            memcpy(h_vals_B + idx * 4, p_coef + 12, 32);
            h_sigs_B[idx] = sig;
        }
    }

    zkey.endReadSection(false);
    double t1 = g16NowMs();
    fprintf(stderr, "[sec4] OpenMP Fast Parse total: %.1f ms\n", t1 - t0);

    // 5. Uploads et MVM sur GPU
    uint64_t *d_vals_A, *d_vals_B, *d_witness;
    uint32_t *d_sigs_A, *d_sigs_B, *d_rowstart_A, *d_rowstart_B;

    G16_CUDA_CHECK(cudaMalloc(&d_vals_A, total_A * 32));
    G16_CUDA_CHECK(cudaMalloc(&d_sigs_A, total_A * 4));
    G16_CUDA_CHECK(cudaMalloc(&d_rowstart_A, (domain_size + 1) * 4));

    G16_CUDA_CHECK(cudaMalloc(&d_vals_B, total_B * 32));
    G16_CUDA_CHECK(cudaMalloc(&d_sigs_B, total_B * 4));
    G16_CUDA_CHECK(cudaMalloc(&d_rowstart_B, (domain_size + 1) * 4));

    G16_CUDA_CHECK(cudaMalloc(&d_witness, n_vars * 32));

    G16_CUDA_CHECK(cudaMalloc(d_Aw_out, domain_size * 32));
    G16_CUDA_CHECK(cudaMalloc(d_Bw_out, domain_size * 32));
    G16_CUDA_CHECK(cudaMemset(*d_Aw_out, 0, domain_size * 32));
    G16_CUDA_CHECK(cudaMemset(*d_Bw_out, 0, domain_size * 32));

    cudaStream_t sA, sB;
    cudaStreamCreate(&sA); cudaStreamCreate(&sB);

    G16_CUDA_CHECK(cudaMemcpyAsync(d_vals_A, h_vals_A, total_A * 32, cudaMemcpyHostToDevice, sA));
    G16_CUDA_CHECK(cudaMemcpyAsync(d_sigs_A, h_sigs_A, total_A * 4, cudaMemcpyHostToDevice, sA));
    G16_CUDA_CHECK(cudaMemcpyAsync(d_rowstart_A, row_start_A.data(), (domain_size + 1) * 4, cudaMemcpyHostToDevice, sA));

    G16_CUDA_CHECK(cudaMemcpyAsync(d_vals_B, h_vals_B, total_B * 32, cudaMemcpyHostToDevice, sB));
    G16_CUDA_CHECK(cudaMemcpyAsync(d_sigs_B, h_sigs_B, total_B * 4, cudaMemcpyHostToDevice, sB));
    G16_CUDA_CHECK(cudaMemcpyAsync(d_rowstart_B, row_start_B.data(), (domain_size + 1) * 4, cudaMemcpyHostToDevice, sB));

    // Témoin envoyé sur le stream A
    G16_CUDA_CHECK(cudaMemcpyAsync(d_witness, h_witness.data(), n_vars * 32, cudaMemcpyHostToDevice, sA));

    int blk = 256;
    int grid = (domain_size + blk - 1) / blk;

    // Lancement asynchrone des MVM directes !
    k_sparse_mvm_direct<<<grid, blk, 0, sA>>>(d_vals_A, d_sigs_A, d_rowstart_A, d_witness, *d_Aw_out, domain_size);
    k_sparse_mvm_direct<<<grid, blk, 0, sB>>>(d_vals_B, d_sigs_B, d_rowstart_B, d_witness, *d_Bw_out, domain_size);

    G16_CUDA_CHECK(cudaStreamSynchronize(sA));
    G16_CUDA_CHECK(cudaStreamSynchronize(sB));

    // Nettoyage temporaire
    cudaStreamDestroy(sA); cudaStreamDestroy(sB);
    cudaFreeHost(h_vals_A); cudaFreeHost(h_sigs_A);
    cudaFreeHost(h_vals_B); cudaFreeHost(h_sigs_B);
    cudaFree(d_vals_A); cudaFree(d_sigs_A); cudaFree(d_rowstart_A);
    cudaFree(d_vals_B); cudaFree(d_sigs_B); cudaFree(d_rowstart_B);
    cudaFree(d_witness);
}

/*
 * readWitness ??? lit le fichier witness.wtns.
 *  Retourne un vecteur de n_vars ?? 4 uint64_t en Fr canonique.
 */
static std::vector<uint64_t> readWitness(
    const char* wtns_path,
    uint32_t    n_vars)
{
    BinFileUtils::BinFile wtns(wtns_path, "wtns", 2);
    auto wtnsH = WtnsUtils::loadHeader(&wtns);

    uint32_t n_witness = wtnsH->nVars;
    if (n_witness < n_vars) {
        fprintf(stderr,"[groth16] witness has %u entries, need %u\n", n_witness, n_vars);
        exit(1);
    }

    std::vector<uint64_t> w(n_vars * 4, 0);

    wtns.startReadSection(2);
    for (uint32_t i = 0; i < n_vars; i++) {
        const uint8_t* p = (const uint8_t*)wtns.read(32);
        for (int j = 0; j < 4; j++) {
            uint64_t v = 0;
            for (int k = 0; k < 8; k++) v |= ((uint64_t)p[j*8+k]) << (k*8);
            w[i*4+j] = v;
        }
    }
    wtns.endReadSection(false);
    return w;
}

// ============================================================================
// 5. CALCUL DU POLYN??ME H
// ============================================================================


/*
 * nttOmega2nCosetMul ??? a[i] *= omega_{2n}^i   (coset shift by omega_{2n})
 * omega_{2n} is the primitive 2n-th root = g^{(r-1)/(2n)} in Montgomery form.
 * We read it directly from the twiddle table: tw[1] = omega_{2^log_n_max}^1.
 * For the domain we use, log_n_max >= log_n+1, so tw[1] = omega_{2^log_n_max}.
 * We need omega_{2n} = omega_{2^(log_n+1)}, which requires adjusting the exponent.
 * Simpler: compute omega_{2n} via frPow on GPU once, then multiply.
 */
// k_coset_twiddle -- parallel coset shift using the NTT twiddle table.
// Computes d[i] *= omega_{2n}^i for all i in [0, N).
// Uses g_ntt_twiddles directly: twiddles[i * stride] = omega_{2n}^i
// where stride = 2^(log_n_max - (log_n+1)).
// This replaces the single-threaded k_build_omega2n_table which would
// timeout CUDA watchdog for large N (e.g. N=65536 on Windows/WSL).
__global__ void k_coset_twiddle(uint64_t* d, int N,
                                 const uint64_t* tw, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint64_t tmp[4];
    frMul(d + i*4, tw + (size_t)i * stride * 4, tmp);
    memcpy(d + i*4, tmp, 32);
}

// k_h_from_coset ??? calcule h_odd[j] = A_odd[j]*B_odd[j] ??? C_odd[j]
//
// !! IMPORTANT : on ne divise PAS par Z(omega_{2n}^{2j+1}) = ???2.
//
// snarkjs (qap_joinABC) fait exactement :
//   output[j] = frm_mul(A_odd[j], B_odd[j]) ??? C_odd[j]
// puis frm_batchFromMontgomery pour d??coder vers canonique.
// L'absence de division par Z est intentionnelle : les H points dans le
// zkey (section 9) compensent via leur structure tau^{2i+1}/delta.
//
// Entr??es et sortie en MONTGOMERY Fr.
// La sortie sera d??cod??e vers canonique par k_fr_decode_batch juste apr??s.
__global__ void k_h_from_coset(
    const uint64_t* Aodd,  // A(omega_{2n}^{2j+1}) en Montgomery
    const uint64_t* Bodd,  // B(omega_{2n}^{2j+1}) en Montgomery
    const uint64_t* Codd,  // C(omega_{2n}^{2j+1}) en Montgomery  (C = A*B sur domaine standard)
    uint64_t*       h_out, // sortie : (A*B ??? C)(omega_{2n}^{2j+1}) en Montgomery
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint64_t ab[4], num[4];
    frMul(Aodd + i*4, Bodd + i*4, ab);   // ab = Montgomery(A*B)
    frSub(ab, Codd + i*4, num);           // num = Montgomery(A*B ??? C)
    memcpy(h_out + i*4, num, 32);
}


struct G16HPolyStats {
    double pointwise_mul_ms = 0.0;
    double ifft_a_ms = 0.0;
    double ifft_b_ms = 0.0;
    double ifft_c_ms = 0.0;
    double ifft_total_ms = 0.0;
    double coset_a_ms = 0.0;
    double coset_b_ms = 0.0;
    double coset_c_ms = 0.0;
    double coset_total_ms = 0.0;
    double fft_a_ms = 0.0;
    double fft_b_ms = 0.0;
    double fft_c_ms = 0.0;
    double fft_total_ms = 0.0;
    double h_from_coset_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
};

struct G16CRSReadStats {
    G16CRSSectionStats sec5;
    G16CRSSectionStats sec7;
    G16CRSSectionStats sec8;
    G16CRSSectionStats sec9;
    double raw_total_ms = 0.0;
    double parse_total_ms = 0.0;
    double end_total_ms = 0.0;
    double total_ms = 0.0;
};

static bool g16EnvEnabled(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] && std::strcmp(v, "0") != 0;
}

static void printHPolyStats(const G16HPolyStats& s) {
    fprintf(stderr,
            "[hpoly] pointwise=%7.1f | ifft(a/b/c)=%7.1f / %7.1f / %7.1f | ifft_total=%7.1f ms\n",
            s.pointwise_mul_ms, s.ifft_a_ms, s.ifft_b_ms, s.ifft_c_ms, s.ifft_total_ms);
    fprintf(stderr,
            "[hpoly] coset(a/b/c)=%7.1f / %7.1f / %7.1f | coset_total=%7.1f ms\n",
            s.coset_a_ms, s.coset_b_ms, s.coset_c_ms, s.coset_total_ms);
    fprintf(stderr,
            "[hpoly] fft(a/b/c)=%7.1f / %7.1f / %7.1f | fft_total=%7.1f ms\n",
            s.fft_a_ms, s.fft_b_ms, s.fft_c_ms, s.fft_total_ms);
    fprintf(stderr,
            "[hpoly] h_from_coset=%7.1f | decode=%7.1f | total=%7.1f ms\n",
            s.h_from_coset_ms, s.decode_ms, s.total_ms);
}

static void printCRSSectionStats(const char* label, const G16CRSSectionStats& s) {
    fprintf(stderr,
            "[%s] points=%u raw=%.1f MiB | start=%7.1f read_raw=%7.1f parse_pack=%7.1f end=%7.1f total=%7.1f ms\n",
            label,
            s.points,
            (double)s.raw_bytes / (1024.0 * 1024.0),
            s.start_ms,
            s.read_raw_ms,
            s.parse_pack_ms,
            s.end_ms,
            s.total_ms);
}

static void printCRSReadStats(const G16CRSReadStats& s) {
    printCRSSectionStats("crs5", s.sec5);
    printCRSSectionStats("crs7", s.sec7);
    printCRSSectionStats("crs8", s.sec8);
    printCRSSectionStats("crs9", s.sec9);
    fprintf(stderr,
            "[crs] raw_total=%7.1f | parse_total=%7.1f | end_total=%7.1f | total=%7.1f ms\n",
            s.raw_total_ms, s.parse_total_ms, s.end_total_ms, s.total_ms);
}

/*
 * computeHPoly — calcule les évaluations de h(x) aux n points impairs omega_{2n}^{2j+1}
 *
 * Algorithme snarkjs-compatible :
 * Input : d_Aw, d_Bw = évaluations A(omega_n^i), B(omega_n^i), i=0..n-1 (Montgomery)
 * 1. Cw[i] = Aw[i]*Bw[i]
 * 2. IFFT_n -> polynomial coefficients
 * 3. CosetShift by omega_{2n}: coef[i] *= omega_{2n}^i
 * 4. FFT_n -> evaluations at odd coset {omega_{2n}^{2j+1}}
 * 5. h_odd[j] = (A_odd[j]*B_odd[j] - C_odd[j]) * (-2)^{-1}
 *
 * d_h : output, n canonical Fr values (h_odd[0..n-1]), used directly as MSM scalars
 */
static void computeHPoly(
    uint64_t* d_Aw,    // n values, Montgomery Fr
    uint64_t* d_Bw,    // n values, Montgomery Fr
    uint64_t* d_Cw,    // n values, scratch
    int        log_n,
    uint64_t* d_h,    // output: n canonical Fr values
    G16HPolyStats* stats = nullptr)
{
    const int N       = 1 << log_n;
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    // 1. Création des marqueurs temporels GPU (Events)
    cudaEvent_t ev_start, ev_pwise, ev_ifft, ev_coset, ev_fft, ev_h_from, ev_decode;
    cudaEventCreate(&ev_start);  cudaEventCreate(&ev_pwise);
    cudaEventCreate(&ev_ifft);   cudaEventCreate(&ev_coset);
    cudaEventCreate(&ev_fft);    cudaEventCreate(&ev_h_from);
    cudaEventCreate(&ev_decode);

    // TOP CHRONO (On jette le premier marqueur dans la file du GPU)
    cudaEventRecord(ev_start);

    // --- 1. Pointwise Cw = Aw * Bw ---
    k_fr_pointwise_mul<<<blocks, threads>>>(d_Aw, d_Bw, d_Cw, N);
    cudaEventRecord(ev_pwise);

    // --- 2. IFFT ---
    ntt(d_Aw, log_n, true);
    ntt(d_Bw, log_n, true);
    ntt(d_Cw, log_n, true);
    cudaEventRecord(ev_ifft);

    // --- 3. Coset Shift by omega_{2n} ---
    int log_n_max;
    uint64_t* tw = nttGetTwiddleTable(&log_n_max);
    int stride = 1 << (log_n_max - (log_n + 1));
    k_coset_twiddle<<<blocks, threads>>>(d_Aw, N, tw, stride);
    k_coset_twiddle<<<blocks, threads>>>(d_Bw, N, tw, stride);
    k_coset_twiddle<<<blocks, threads>>>(d_Cw, N, tw, stride);
    cudaEventRecord(ev_coset);

    // --- 4. FFT ---
    ntt(d_Aw, log_n, false);
    ntt(d_Bw, log_n, false);
    ntt(d_Cw, log_n, false);
    cudaEventRecord(ev_fft);

    // --- 5. h_odd = (A*B - C) ---
    k_h_from_coset<<<blocks, threads>>>(d_Aw, d_Bw, d_Cw, d_h, N);
    cudaEventRecord(ev_h_from);

    // --- 6. Decode ---
    k_fr_decode_batch<<<blocks, threads>>>(d_h, d_h, N);
    cudaEventRecord(ev_decode);

    // ==========================================================
    // RÉCUPÉRATION DES STATISTIQUES (Si demandé via ZK_HPIPE_DEBUG=1)
    // ==========================================================
    if (stats) {
        // Le CPU attend UNIQUEMENT ici que le dernier kernel soit terminé !
        cudaEventSynchronize(ev_decode);

        float t_pwise=0, t_ifft=0, t_coset=0, t_fft=0, t_h=0, t_decode=0, t_total=0;
        
        cudaEventElapsedTime(&t_pwise,  ev_start,  ev_pwise);
        cudaEventElapsedTime(&t_ifft,   ev_pwise,  ev_ifft);
        cudaEventElapsedTime(&t_coset,  ev_ifft,   ev_coset);
        cudaEventElapsedTime(&t_fft,    ev_coset,  ev_fft);
        cudaEventElapsedTime(&t_h,      ev_fft,    ev_h_from);
        cudaEventElapsedTime(&t_decode, ev_h_from, ev_decode);
        cudaEventElapsedTime(&t_total,  ev_start,  ev_decode);

        stats->pointwise_mul_ms = t_pwise;
        stats->ifft_total_ms    = t_ifft;
        stats->coset_total_ms   = t_coset;
        stats->fft_total_ms     = t_fft;
        stats->h_from_coset_ms  = t_h;
        stats->decode_ms        = t_decode;
        stats->total_ms         = t_total;
        
        // On met à 0 les temps détaillés individuels car on a packé les kernels
        stats->ifft_a_ms = 0; stats->ifft_b_ms = 0; stats->ifft_c_ms = 0;
        stats->coset_a_ms = 0; stats->coset_b_ms = 0; stats->coset_c_ms = 0;
        stats->fft_a_ms = 0; stats->fft_b_ms = 0; stats->fft_c_ms = 0;
    }

    // Nettoyage des événements
    cudaEventDestroy(ev_start);  cudaEventDestroy(ev_pwise);
    cudaEventDestroy(ev_ifft);   cudaEventDestroy(ev_coset);
    cudaEventDestroy(ev_fft);    cudaEventDestroy(ev_h_from);
    cudaEventDestroy(ev_decode);
}


// ============================================================================
// 6. GROTH16 PROVE ??? fonction principale
// ============================================================================


// ── Timing helpers ────────────────────────────────────────────────────────────
using g16_clock = std::chrono::steady_clock;
static g16_clock::time_point g16_t0;
#define G16_TIMER_START() do { g16_t0 = g16_clock::now(); } while(0)
#define G16_TIMER_STEP(label) do {     cudaDeviceSynchronize();     auto _t1 = g16_clock::now();     double _ms = std::chrono::duration<double,std::milli>(_t1 - g16_t0).count();     fprintf(stderr, "[timer] %-38s %7.1f ms\n", label, _ms);     g16_t0 = g16_clock::now(); } while(0)
// ─────────────────────────────────────────────────────────────────────────────

void groth16Prove(
    const char*   zkey_path,
    const char*   wtns_path,
    Groth16Proof& proof)
{
    // ---- 6.1 Lecture header .zkey ----
    printf("[groth16] Opening %s\n", zkey_path);
    G16_TIMER_START();
    BinFileUtils::BinFile zkey(zkey_path, "zkey", 1);
    auto zkh = ZKeyUtils::loadHeader(&zkey);

    Groth16ZkeyHeader hdr;
    hdr.n_constraints = zkh->domainSize;
    hdr.n_vars        = zkh->nVars;
    hdr.n_public      = zkh->nPublic;
    uint32_t d = 1; int log_d = 0;
    while (d < hdr.n_constraints) { d <<= 1; log_d++; }
    hdr.domain_size = d;
    hdr.log_domain  = log_d;

    const uint32_t n  = hdr.domain_size;   // taille domaine NTT
    const uint32_t m1 = hdr.n_vars;        // m+1 wires total
    const uint32_t l  = hdr.n_public;      // wires publics
    const int      ld = hdr.log_domain;

    printf("[groth16] n_constraints=%u domain=%u (log=%d) n_vars=%u n_public=%u\n",
           hdr.n_constraints, n, ld, m1, l);

    // ---- 6.2 nttPrepare ----
    nttPrepare(ld + 1);
    G16_TIMER_STEP("open zkey + nttPrepare");

	// ---- 6.3 readWitness ----
    printf("[groth16] Reading witness %s\n", wtns_path);
    std::vector<uint64_t> h_witness = readWitness(wtns_path, m1);
    G16_TIMER_STEP("read witness");

    // ---- 6.4 + 6.5 + 6.6 : Fast Parse Section 4 & GPU MVM ----
    printf("[groth16] Fast Parse Section 4 & GPU MVM\n");
    uint64_t *d_Aw, *d_Bw;
    readAndComputeMVM(zkey, h_witness, m1, n, &d_Aw, &d_Bw);
    G16_TIMER_STEP("sparse MVM A*w + B*w (GPU)");

    uint64_t *d_Cw, *d_h;
    G16_CUDA_CHECK(cudaMalloc(&d_Cw, (size_t)n * 4 * sizeof(uint64_t)));
    G16_CUDA_CHECK(cudaMalloc(&d_h,  (size_t)n * 4 * sizeof(uint64_t)));
    G16_CUDA_CHECK(cudaMemset(d_Cw, 0, (size_t)n * 4 * sizeof(uint64_t)));
    G16_CUDA_CHECK(cudaMemset(d_h, 0,  (size_t)n * 4 * sizeof(uint64_t)));

    // ---- 6.7 Calcul du polynôme H + lecture CRS + PRE-UPLOAD GPU ----
    G16_TIMER_STEP("upload + Montgomery encode");

    const bool hpipe_debug = g16EnvEnabled("ZK_HPIPE_DEBUG");
    printf("[groth16] Reading CRS points (sections 5,7,8,9)\n");

    // Dimensions
    const uint64_t* alpha1_raw = (const uint64_t*)zkh->vk_alpha1;
    const uint64_t* beta2_raw  = (const uint64_t*)zkh->vk_beta2;
    const uint32_t m1e = m1 + 1;   // extended: prepend alpha1/beta2 with scalar=1
    const uint32_t N_C = m1 - l - 1;
    const uint32_t N_H = n;

    // ── PRE-ALLOCATE all GPU buffers BEFORE the async thread ──────────────────
    // This avoids cudaMalloc inside the MSM call sites (malloc is synchronous
    // and serialises with the GPU, adding latency).

    // Extended A points (G1): alpha1 + sec5[0..m1-1]
    uint64_t *d_A_Px_ext = nullptr, *d_A_Py_ext = nullptr;
    G16_CUDA_CHECK(cudaMalloc(&d_A_Px_ext, (size_t)m1e * 4 * 8));
    G16_CUDA_CHECK(cudaMalloc(&d_A_Py_ext, (size_t)m1e * 4 * 8));

    // Extended B2 points (G2): beta2 + sec7[0..m1-1]
    uint64_t *d_B2_Px_ext = nullptr, *d_B2_Py_ext = nullptr;
    G16_CUDA_CHECK(cudaMalloc(&d_B2_Px_ext, (size_t)m1e * 8 * 8));
    G16_CUDA_CHECK(cudaMalloc(&d_B2_Py_ext, (size_t)m1e * 8 * 8));

    // C points (G1): sec8[0..N_C-1]
    uint64_t *d_C_Px = nullptr, *d_C_Py = nullptr;
    G16_CUDA_CHECK(cudaMalloc(&d_C_Px, (size_t)N_C * 4 * 8));
    G16_CUDA_CHECK(cudaMalloc(&d_C_Py, (size_t)N_C * 4 * 8));

    // H points (G1): sec9[0..N_H-1]
    uint64_t *d_H_Px = nullptr, *d_H_Py = nullptr;
    G16_CUDA_CHECK(cudaMalloc(&d_H_Px, (size_t)N_H * 4 * 8));
    G16_CUDA_CHECK(cudaMalloc(&d_H_Py, (size_t)N_H * 4 * 8));

    // Upload stream for CRS transfers (runs concurrently with computeHPoly on default stream)
    cudaStream_t crs_upload_stream;
    G16_CUDA_CHECK(cudaStreamCreate(&crs_upload_stream));

    // ── Upload alpha1/beta2 leader points to GPU extended buffers ─────────────
    // These are tiny (32-64 bytes each) — synchronous is fine, they're done instantly.
    // alpha1 = G1 point: [x[4], y[4]] already in Montgomery Fp
    G16_CUDA_CHECK(cudaMemcpy(d_A_Px_ext, alpha1_raw,     4 * 8, cudaMemcpyHostToDevice));
    G16_CUDA_CHECK(cudaMemcpy(d_A_Py_ext, alpha1_raw + 4, 4 * 8, cudaMemcpyHostToDevice));
    // beta2 = G2 point: [x.a0[4], x.a1[4], y.a0[4], y.a1[4]]
    G16_CUDA_CHECK(cudaMemcpy(d_B2_Px_ext, beta2_raw,     8 * 8, cudaMemcpyHostToDevice));
    G16_CUDA_CHECK(cudaMemcpy(d_B2_Py_ext, beta2_raw + 8, 8 * 8, cudaMemcpyHostToDevice));

	// ── Async thread: read CRS from zkey + unpack on GPU ─────────────────────
    G16CRSReadStats crs_stats{};
    G16HPolyStats   hpoly_stats{};
    using hclk = std::chrono::steady_clock;
    auto overlap_t0 = hclk::now();

    auto crs_future = std::async(std::launch::async, [&]() {
        auto t0 = hclk::now();

        // --- Section 5: A points (G1) ---
        auto s5_t0 = hclk::now();
        readAndUnpackG1Chunked(zkey, 5, m1, d_A_Px_ext + 4, d_A_Py_ext + 4, crs_upload_stream);
        auto s5_t1 = hclk::now();
        crs_stats.sec5.points = m1;
        crs_stats.sec5.total_ms = std::chrono::duration<double,std::milli>(s5_t1 - s5_t0).count();

        // --- Section 7: B2 points (G2) ---
        auto s7_t0 = hclk::now();
        readAndUnpackG2Chunked(zkey, 7, m1, d_B2_Px_ext + 8, d_B2_Py_ext + 8, crs_upload_stream);
        auto s7_t1 = hclk::now();
        crs_stats.sec7.points = m1;
        crs_stats.sec7.total_ms = std::chrono::duration<double,std::milli>(s7_t1 - s7_t0).count();

        // --- Section 8: C points (G1) ---
        auto s8_t0 = hclk::now();
        readAndUnpackG1Chunked(zkey, 8, N_C, d_C_Px, d_C_Py, crs_upload_stream);
        auto s8_t1 = hclk::now();
        crs_stats.sec8.points = N_C;
        crs_stats.sec8.total_ms = std::chrono::duration<double,std::milli>(s8_t1 - s8_t0).count();

        // --- Section 9: H points (G1) ---
        auto s9_t0 = hclk::now();
        readAndUnpackG1Chunked(zkey, 9, N_H, d_H_Px, d_H_Py, crs_upload_stream);
        auto s9_t1 = hclk::now();
        crs_stats.sec9.points = N_H;
        crs_stats.sec9.total_ms = std::chrono::duration<double,std::milli>(s9_t1 - s9_t0).count();

        auto t1 = hclk::now();
        crs_stats.total_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    });

    // ── computeHPoly runs on the DEFAULT stream while CRS uploads on crs_upload_stream ──
    printf("[groth16] Computing H polynomial (NTT)\n");
    auto hpoly_t0 = hclk::now();
    computeHPoly(d_Aw, d_Bw, d_Cw, ld, d_h, hpipe_debug ? &hpoly_stats : nullptr);
    auto hpoly_t1 = hclk::now();
    double hpoly_wall_ms = std::chrono::duration<double,std::milli>(hpoly_t1 - hpoly_t0).count();

    // Wait for CRS read + upload to complete
    crs_future.get();
    G16_CUDA_CHECK(cudaStreamSynchronize(crs_upload_stream));
    G16_CUDA_CHECK(cudaStreamDestroy(crs_upload_stream));

    auto overlap_t1 = hclk::now();
    double overlap_total_ms = std::chrono::duration<double,std::milli>(overlap_t1 - overlap_t0).count();
    double wait_after_h_ms  = std::chrono::duration<double,std::milli>(overlap_t1 - hpoly_t1).count();
    if (wait_after_h_ms < 0.0) wait_after_h_ms = 0.0;
    double hidden_ms = hpoly_wall_ms + crs_stats.total_ms - overlap_total_ms;
    if (hidden_ms < 0.0) hidden_ms = 0.0;

    if (hpipe_debug) {
        printHPolyStats(hpoly_stats);
        printCRSReadStats(crs_stats);
        fprintf(stderr,
                "[hpipe] hpoly_wall=%7.1f | crs_total=%7.1f | overlap_total=%7.1f | wait_after_h=%7.1f | hidden=%7.1f ms\n",
                hpoly_wall_ms, crs_stats.total_ms, overlap_total_ms, wait_after_h_ms, hidden_ms);
    }

    G16_TIMER_STEP("H poly (NTT) + CRS read+upload (async overlap)");

	// ---- 6.9 Witness upload sur GPU (canonical Fr) ----
    printf("[groth16] Uploading witness\n");

    // We keep a SINGLE device witness buffer: d_we = [1 || witness[0..m1-1]]
    uint64_t* d_we;
    G16_CUDA_CHECK(cudaMalloc(&d_we, (size_t)m1e * 4 * 8));

    // Write leading scalar 1
    const uint64_t h_one_scalar[4] = {1ULL, 0ULL, 0ULL, 0ULL};
    G16_CUDA_CHECK(cudaMemcpy(d_we, h_one_scalar, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Append canonical witness right after the leading 1
    G16_CUDA_CHECK(cudaMemcpy(d_we + 4, h_witness.data(), (size_t)m1 * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    G16_CUDA_CHECK(cudaDeviceSynchronize());
    G16_TIMER_STEP("upload witness to GPU");

    // ── pi_A : points ALREADY on GPU, just run MSM ──────────────────────────
    printf("[groth16] MSM G1 pi_A (N=%u)\n", m1e);
    {
        FpPointAff result;
        msm_g1(d_A_Px_ext, d_A_Py_ext, d_we, (int)m1e, result);

        memcpy(proof.pi_A,     result.X, 32);
        memcpy(proof.pi_A + 4, result.Y, 32);
        G16_TIMER_STEP("MSM G1 pi_A");
    }

    // ── pi_B : points ALREADY on GPU, just run MSM ──────────────────────────
    printf("[groth16] MSM G2 pi_B (N=%u)\n", m1e);
    {
        Fp2PointAff result;
        msm_g2_device(d_B2_Px_ext, d_B2_Py_ext, d_we, (int)m1e, result);

        memcpy(proof.pi_B,      result.X,     32);
        memcpy(proof.pi_B + 4,  result.X + 4, 32);
        memcpy(proof.pi_B + 8,  result.Y,     32);
        memcpy(proof.pi_B + 12, result.Y + 4, 32);
        G16_TIMER_STEP("MSM G2 pi_B");
    }

    cudaFree(d_A_Px_ext); cudaFree(d_A_Py_ext);
    cudaFree(d_B2_Px_ext); cudaFree(d_B2_Py_ext);

    // ── pi_C = MSM_G1(C) + MSM_G1(H) : points ALREADY on GPU ───────────────
    printf("[groth16] MSM G1 pi_C (C: N=%u, H: N=%u)\n", N_C, N_H);
    {
        auto piC_t0 = g16_clock::now();
        FpPointAff result_C, result_H;

        G16_TIMER_STEP("MSM G1 pi_C upload C points");  // ~0ms — already on GPU
		msm_g1(d_C_Px, d_C_Py, d_we + (size_t)(l + 2) * 4, N_C, result_C);
        G16_TIMER_STEP("MSM G1 pi_C (C part)");
        cudaFree(d_C_Px); cudaFree(d_C_Py);

        G16_TIMER_STEP("MSM G1 pi_C upload H points");  // ~0ms — already on GPU
        msm_g1(d_H_Px, d_H_Py, d_h, N_H, result_H);
        G16_TIMER_STEP("MSM G1 pi_C (H backend)");
        cudaFree(d_H_Px); cudaFree(d_H_Py);

        // pi_C = result_C + result_H
        FpPointAff piC;
        if (result_C.infinity) {
            piC = result_H;
        } else if (result_H.infinity) {
            piC = result_C;
        } else {
            g16_g1_add_host(&result_C, &result_H, &piC);
        }
        G16_TIMER_STEP("MSM G1 pi_C final add");

        memcpy(proof.pi_C,     piC.X, 32);
        memcpy(proof.pi_C + 4, piC.Y, 32);

        auto piC_t1 = g16_clock::now();
        double piC_ms = std::chrono::duration<double,std::milli>(piC_t1 - piC_t0).count();
        fprintf(stderr, "[timer] %-38s %7.1f ms\n", "MSM G1 pi_C total", piC_ms);
    }

    // ---- 6.13 Cleanup ----
    printf("[groth16] Done. Cleaning up.\n");
    cudaFree(d_Aw); cudaFree(d_Bw); cudaFree(d_Cw); cudaFree(d_h);
	cudaFree(d_we);

    nttDestroy();
}


// Kernel + helper pour l'addition G1 finale (??_C = C_part + H_part)
__global__ void k_g1_add(
    const FpPointAff* __restrict__ A,
    const FpPointAff* __restrict__ B,
    FpPointAff*                    R)
{
    if (blockIdx.x || threadIdx.x) return;
    FpPointJac Aj, tmp;
    pointAffToJac(*A, Aj);
    pointMixedAdd(Aj, *B, tmp);
    pointJacToAff(tmp, *R);
}

void g16_g1_add_host(const FpPointAff* A, const FpPointAff* B, FpPointAff* R)
{
    FpPointAff *d_A, *d_B, *d_R;
    G16_CUDA_CHECK(cudaMalloc(&d_A, sizeof(FpPointAff)));
    G16_CUDA_CHECK(cudaMalloc(&d_B, sizeof(FpPointAff)));
    G16_CUDA_CHECK(cudaMalloc(&d_R, sizeof(FpPointAff)));
    G16_CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(FpPointAff), cudaMemcpyHostToDevice));
    G16_CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(FpPointAff), cudaMemcpyHostToDevice));
    k_g1_add<<<1,1>>>(d_A, d_B, d_R);
    G16_CUDA_CHECK(cudaDeviceSynchronize());
    G16_CUDA_CHECK(cudaMemcpy(R, d_R, sizeof(FpPointAff), cudaMemcpyDeviceToHost));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_R);
}

// ============================================================================
// 7. S??RIALISATION
// ============================================================================

// D??code Montgomery Fp -> canonique, formate en d??cimal (comme snarkjs)
static void fpDecodeToDecStr(const uint64_t mont[4], char* buf, size_t buf_size)
{
    // Decode via un kernel GPU
    uint64_t *d_in, *d_out;
    uint64_t h_out[4];
    G16_CUDA_CHECK(cudaMalloc(&d_in,  32));
    G16_CUDA_CHECK(cudaMalloc(&d_out, 32));
    G16_CUDA_CHECK(cudaMemcpy(d_in, mont, 32, cudaMemcpyHostToDevice));
    k_fp_decode_batch<<<1,256>>>(d_in, d_out, 1);
    G16_CUDA_CHECK(cudaDeviceSynchronize());
    G16_CUDA_CHECK(cudaMemcpy(h_out, d_out, 32, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out);

    // Convertit 256-bit LE en d??cimal (big number -> string)
    // Algorithme : division r??p??t??e par 10
    // h_out[0] = least significant, h_out[3] = most significant
    uint64_t n[4]; memcpy(n, h_out, 32);
    char tmp[80]; int len = 0;
    do {
        // Division par 10, reste = digit
        uint64_t rem = 0;
        for (int i = 3; i >= 0; i--) {
            __uint128_t d = ((__uint128_t)rem << 64) | n[i];
            n[i] = (uint64_t)(d / 10);
            rem  = (uint64_t)(d % 10);
        }
        tmp[len++] = '0' + (char)rem;
    } while (n[0] || n[1] || n[2] || n[3]);

    // Inverse
    if ((size_t)len + 1 > buf_size) len = (int)buf_size - 1;
    for (int i = 0; i < len; i++) buf[i] = tmp[len-1-i];
    buf[len] = '\0';
}

void groth16ProofToSnarkjsJson(
    const Groth16Proof& proof,
    char*               out,
    size_t              out_size)
{
    // D??code les 6 coordonn??es Fp + 4 coordonn??es Fp2
    char ax[80], ay[80];
    char bx0[80], bx1[80], by0[80], by1[80];
    char cx[80], cy[80];

    fpDecodeToDecStr(proof.pi_A,     ax, sizeof(ax));
    fpDecodeToDecStr(proof.pi_A + 4, ay, sizeof(ay));

    // pi_B : [x.a0[4], x.a1[4], y.a0[4], y.a1[4]]
    // snarkjs JSON attend [a1, a0] (imaginaire en premier)
    fpDecodeToDecStr(proof.pi_B,      bx0, sizeof(bx0));  // x.a0
    fpDecodeToDecStr(proof.pi_B + 4,  bx1, sizeof(bx1));  // x.a1
    fpDecodeToDecStr(proof.pi_B + 8,  by0, sizeof(by0));  // y.a0
    fpDecodeToDecStr(proof.pi_B + 12, by1, sizeof(by1));  // y.a1

    fpDecodeToDecStr(proof.pi_C,     cx, sizeof(cx));
    fpDecodeToDecStr(proof.pi_C + 4, cy, sizeof(cy));

    snprintf(out, out_size,
        "{\n"
        " \"pi_a\": [\"%s\",\"%s\",\"1\"],\n"
        " \"pi_b\": [[\"%s\",\"%s\"],[\"%s\",\"%s\"],[\"1\",\"0\"]],\n"
        " \"pi_c\": [\"%s\",\"%s\",\"1\"],\n"
        " \"protocol\": \"groth16\",\n"
        " \"curve\": \"bn128\"\n"
        "}",
        ax, ay,
        bx0, bx1,  // [a0, a1] ??? snarkjs c0+c1*i convention
        by0, by1,
        cx, cy);
}

// Conversion 256-bit LE -> big-endian bytes (32 bytes)
static void leToBeBytes(const uint64_t le[4], uint8_t be[32])
{
    for (int i = 0; i < 4; i++) {
        uint64_t v = le[3-i]; // big-endian word order
        for (int j = 0; j < 8; j++)
            be[i*8 + j] = (uint8_t)(v >> ((7-j)*8));
    }
}

void groth16ProofToCalldata(
    const Groth16Proof& proof,
    uint8_t*            out,
    size_t              out_size)
{
    if (out_size < 256) { fprintf(stderr,"groth16ProofToCalldata: buffer too small\n"); return; }

    // D??code Montgomery -> canonique sur GPU
    uint64_t *d_in, *d_out;
    uint64_t h_A[8], h_B[16], h_C[8];
    G16_CUDA_CHECK(cudaMalloc(&d_in,  16*4*8)); // max 16 limbs
    G16_CUDA_CHECK(cudaMalloc(&d_out, 16*4*8));

    // pi_A (2 Fp)
    G16_CUDA_CHECK(cudaMemcpy(d_in, proof.pi_A, 64, cudaMemcpyHostToDevice));
    k_fp_decode_batch<<<1,256>>>(d_in, d_out, 2);
    G16_CUDA_CHECK(cudaDeviceSynchronize());
    G16_CUDA_CHECK(cudaMemcpy(h_A, d_out, 64, cudaMemcpyDeviceToHost));

    // pi_B (4 Fp ??? Fp2 coords)
    G16_CUDA_CHECK(cudaMemcpy(d_in, proof.pi_B, 128, cudaMemcpyHostToDevice));
    k_fp_decode_batch<<<1,256>>>(d_in, d_out, 4);
    G16_CUDA_CHECK(cudaDeviceSynchronize());
    G16_CUDA_CHECK(cudaMemcpy(h_B, d_out, 128, cudaMemcpyDeviceToHost));

    // pi_C (2 Fp)
    G16_CUDA_CHECK(cudaMemcpy(d_in, proof.pi_C, 64, cudaMemcpyHostToDevice));
    k_fp_decode_batch<<<1,256>>>(d_in, d_out, 2);
    G16_CUDA_CHECK(cudaDeviceSynchronize());
    G16_CUDA_CHECK(cudaMemcpy(h_C, d_out, 64, cudaMemcpyDeviceToHost));

    cudaFree(d_in); cudaFree(d_out);

    // ABI encoding : pi_A.x, pi_A.y, pi_B.x[1], pi_B.x[0], pi_B.y[1], pi_B.y[0], pi_C.x, pi_C.y
    // Chaque coordonn??e = 32 bytes big-endian
    size_t off = 0;
    leToBeBytes(h_A,     out+off); off+=32; // pi_A.x
    leToBeBytes(h_A+4,   out+off); off+=32; // pi_A.y
    leToBeBytes(h_B+4,   out+off); off+=32; // pi_B.x.a1 (big-endian Fp2: a1 first)
    leToBeBytes(h_B,     out+off); off+=32; // pi_B.x.a0
    leToBeBytes(h_B+12,  out+off); off+=32; // pi_B.y.a1
    leToBeBytes(h_B+8,   out+off); off+=32; // pi_B.y.a0
    leToBeBytes(h_C,     out+off); off+=32; // pi_C.x
    leToBeBytes(h_C+4,   out+off);          // pi_C.y
}