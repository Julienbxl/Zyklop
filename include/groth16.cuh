/*
 * ===========================================================================
 * Forum / Zyklop — groth16.cuh
 * BN254 Groth16 prover — public API
 * ===========================================================================
 *
 *  Proof equation (snarkjs convention, affine Montgomery):
 *
 *    π_A = MSM_G1( τ·A_i[0..m],   w[0..m] )
 *    π_B = MSM_G2( τ·B2_i[0..m],  w[0..m] )
 *    π_C = MSM_G1( τ·C_i[l+1..m] + τ·H_i[0..n-1],
 *                  w[l+1..m]       + h[0..n-1] )
 *
 *  H polynomial (evaluated on coset g·Ω):
 *    a(x)  = IFFT( A·w )   (size n, Fr)
 *    b(x)  = IFFT( B·w )
 *    c(x)  = IFFT( C·w )
 *    h'(x) = a(x)·b(x) - c(x)          (pointwise in coset domain)
 *    h(x)  = h'(x) / Z(x)  where Z(x) = x^n - 1
 *    h_i   = FFT( h(x) ) [0..n-2]
 *
 *  .zkey section layout (snarkjs binary format):
 *    §1  Header Groth16   → n, m, l (nConstraints, nVars, nPublic)
 *    §3  IC points        → G1 (l+1 points, vérificateur)
 *    §4  Coefficients A,B,C → sparse (constraint, wire, value)
 *    §5  τ·A_i            → G1, m+1 points
 *    §6  τ·B1_i           → G1, m+1 points  (unused in prover)
 *    §7  τ·B2_i           → G2, m+1 points  ← seul G2 MSM
 *    §8  τ·C_i            → G1, m-l points
 *    §9  τ·H_i            → G1, n points
 *
 *  Usage:
 *    nttPrepare(log2(n) + 1);      // une seule fois
 *    Groth16Proof proof;
 *    groth16Prove("circuit.zkey", "witness.wtns", proof);
 *    nttDestroy();
 * ===========================================================================
 */

#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

// groth16.cuh n'inclut PAS les .cuh d'implémentation (fp_bn254, msm, ntt).
// Ces headers contiennent des définitions inline, des static globals et des
// __constant__ — les inclure ici provoquerait des doubles définitions et des
// états dupliqués si plusieurs .cu incluent groth16.cuh.
//
// groth16.cu les inclut directement. main.cu n'a besoin que de ce header.

// ============================================================================
// 1. STRUCTS
// ============================================================================

/*
 * Groth16Proof — éléments en représentation affine Montgomery.
 *
 *  pi_A, pi_C : G1 affine → 2 × 4 × uint64_t (x, y)
 *  pi_B       : G2 affine → 2 × 8 × uint64_t (x Fp2, y Fp2)
 *
 *  Convention stockage (snarkjs / rapidsnark) :
 *    G1 : [x[0..3], y[0..3]]     — 32+32 bytes
 *    G2 : [x.a0[0..3], x.a1[0..3], y.a0[0..3], y.a1[0..3]]  — 64+64 bytes
 *
 *  Tout est en Montgomery — appeler groth16ProofToSnarkjsJson pour
 *  la conversion canonique en sortie.
 */
struct Groth16Proof {
    uint64_t pi_A[8];   // G1 affine : [x[4], y[4]], Montgomery Fp
    uint64_t pi_B[16];  // G2 affine : [x.a0[4], x.a1[4], y.a0[4], y.a1[4]], Montgomery Fp
    uint64_t pi_C[8];   // G1 affine : [x[4], y[4]], Montgomery Fp
};

/*
 * Groth16Zkey — header parsé du .zkey (sections §1, §5-§9).
 * Utilisé en interne par groth16Prove et exposé pour debug/tests.
 */
struct Groth16ZkeyHeader {
    uint32_t n_constraints;  // n  — taille domaine NTT
    uint32_t n_vars;         // m+1 — nombre total de wires (public + privé + 1)
    uint32_t n_public;       // l  — wires publics (hors wire 0)
    uint32_t domain_size;    // n = prochain power-of-2 ≥ n_constraints
    int      log_domain;     // log2(domain_size)
    uint64_t prime[4];       // champ scalaire (doit être BN254 r)
};

/*
 * Groth16CoefEntry — une entrée de la section §4 (coefficients A/B/C).
 * Layout binaire snarkjs : matrix (1 byte), constraint (4 bytes LE),
 *                          signal (4 bytes LE), value (32 bytes LE Fr).
 */
struct Groth16CoefEntry {
    uint8_t  matrix;       // 0=A, 1=B, 2=C
    uint32_t constraint;   // ligne (0..n_constraints-1)
    uint32_t signal;       // colonne / wire (0..n_vars-1)
    uint64_t value[4];     // Fr, Little-Endian canonique (NON Montgomery)
};

// ============================================================================
// 2. API PRINCIPALE
// ============================================================================

/*
 * groth16Prove — calcule un proof Groth16 BN254 sur GPU.
 *
 *  zkey_path : chemin vers le fichier .zkey (format snarkjs / rapidsnark)
 *  wtns_path : chemin vers le fichier witness.wtns
 *  proof     : [out] résultat
 *
 *  Pré-condition : nttPrepare(log_domain + 1) doit avoir été appelé
 *                  avec un log_n_max suffisant (typiquement log_domain + 1).
 *
 *  Appels internes :
 *    - msmPippenger (G1) × 3
 *    - msmG2        (G2) × 1
 *    - ntt/intt     (Fr) × 3 + coset
 */
void groth16Prove(
    const char*   zkey_path,
    const char*   wtns_path,
    Groth16Proof& proof
);

// ============================================================================
// 3. SÉRIALISATION
// ============================================================================

/*
 * groth16ProofToSnarkjsJson — sérialise en JSON compatible snarkjs.
 *
 *  Format attendu par snarkjs verify / rapidsnark :
 *  {
 *    "pi_a": ["<x>", "<y>", "1"],
 *    "pi_b": [["<x.a1>","<x.a0>"], ["<y.a1>","<y.a0>"], ["1","0"]],
 *    "pi_c": ["<x>", "<y>", "1"],
 *    "protocol": "groth16",
 *    "curve": "bn128"
 *  }
 *
 *  Note BN254/BN128 : snarkjs appelle la courbe "bn128" dans les JSON,
 *  c'est la même courbe que BN254 (même paramètres).
 *
 *  Note G2 : snarkjs stocke G2 en ordre [a1, a0] (imaginaire en premier) —
 *  le contraire de notre convention interne [a0, a1].
 *
 *  out_size : taille du buffer out (recommandé : 2048 bytes)
 */
void groth16ProofToSnarkjsJson(
    const Groth16Proof& proof,
    char*               out,
    size_t              out_size
);

/*
 * groth16ProofToCalldata — sérialise en ABI calldata EVM (Solidity verifier).
 *
 *  Format : bytes représentant les 3 points G1/G2 en big-endian non-Montgomery,
 *  prêts à être passés à un verifier Solidity généré par snarkjs.
 *
 *  out_size : recommandé 768 bytes (3 × 256 bytes)
 */
void groth16ProofToCalldata(
    const Groth16Proof& proof,
    uint8_t*            out,
    size_t              out_size
);

// ============================================================================
// 4. UTILITAIRES INTERNES (exposés pour tests unitaires)
// ============================================================================

/*
 * groth16ReadZkeyHeader — lit seulement la section §1 du .zkey.
 *  Utile pour déterminer le log_domain avant d'appeler nttPrepare().
 */
void groth16ReadZkeyHeader(
    const char*         zkey_path,
    Groth16ZkeyHeader&  hdr
);

/*
 * groth16SparseMvm — multiplie matrice creuse × vecteur dense sur CPU.
 *
 *  Implémente A·w (ou B·w, C·w) depuis les Groth16CoefEntry.
 *  Tous les éléments sont en Fr canonique (NON Montgomery) en entrée,
 *  résultat en Fr canonique.
 *
 *  coefs      : tableau de Groth16CoefEntry trié par matrix puis constraint
 *  n_coefs    : nombre d'entrées pour le matrix sélectionné
 *  matrix_sel : 0=A, 1=B, 2=C
 *  witness    : w[0..n_vars-1] en Fr canonique
 *  n_vars     : nombre total de wires
 *  out        : [out] résultat, taille domain_size × 4 uint64_t (zéro-paddé)
 *  domain_size: taille du domaine NTT (power of 2)
 */
void groth16SparseMvm(
    const Groth16CoefEntry* coefs,
    size_t                  n_coefs,
    uint8_t                 matrix_sel,
    const uint64_t*         witness,    // n_vars × 4 limbs
    uint32_t                n_vars,
    uint64_t*               out,        // domain_size × 4 limbs
    uint32_t                domain_size
);
