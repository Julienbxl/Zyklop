/*
 * ===========================================================================
 * ptx_bn254.h — Primitives PTX partagées pour BN254 (Fp et Fr)
 * ===========================================================================
 *
 * Ce fichier centralise les macros PTX et helpers CUDA utilisés par :
 *   - fp_bn254.cuh  (arithmetic mod p, Fp)
 *   - fr_bn254.cuh  (arithmetic mod r, Fr)
 *   - fp2_bn254.cuh (arithmetic mod p^2, Fp2 — bénéficie via fp_bn254)
 *
 * Philosophie (héritée de Cyclope/Barracuda) :
 *   Les macros PTX utilisent les instructions CUDA natives :
 *     add.cc.u64  : addition 64 bits avec carry-out vers CC
 *     addc.cc.u64 : addition 64 bits avec carry-in/out depuis/vers CC
 *     sub.cc.u64  : soustraction 64 bits avec borrow-out vers CC
 *     subc.cc.u64 : soustraction 64 bits avec borrow-in/out
 *     mad.lo.cc   : multiply-add (partie basse) avec carry
 *     madc.hi.cc  : multiply-add (partie haute) avec carry
 *
 * Réduction Montgomery BN254 :
 *   Algorithme CIOS (Coarsely Integrated Operand Scanning).
 *   Mul et reduce sont inséparables : CIOS entrelace les deux colonne par colonne.
 *   Contrairement à secp256k1 (p = 2^256 - K), BN254 n'a pas de forme spéciale
 *   permettant un "fold" — CIOS est l'unique voie optimale.
 *
 * Sources :
 *   - Barracuda ECC.h : structure _mul + ptx_mad_accumulate
 *   - Cyclope ECC.h   : blocs PTX monolithiques, lazy carry
 *   - IOSAD CIOS      : https://eprint.iacr.org/2011/255.pdf
 * ===========================================================================
 */

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// 1. MACROS PTX FONDAMENTALES
// ============================================================================
// Nommage : U = unsigned 64-bit, O = "opening" (set CC), C = "continuing" (use+set CC)
// Compatibles avec Cyclope et Barracuda.

// Addition avec carry
#define PTX_UADDO(r,a,b)  asm volatile("add.cc.u64  %0,%1,%2;" :"=l"(r):"l"(a),"l"(b):"memory")
#define PTX_UADDC(r,a,b)  asm volatile("addc.cc.u64 %0,%1,%2;" :"=l"(r):"l"(a),"l"(b):"memory")
#define PTX_UADD(r,a,b)   asm volatile("addc.u64    %0,%1,%2;" :"=l"(r):"l"(a),"l"(b))

// Addition in-place avec carry
#define PTX_UADDO1(c,a)   asm volatile("add.cc.u64  %0,%0,%1;" :"+l"(c):"l"(a):"memory")
#define PTX_UADDC1(c,a)   asm volatile("addc.cc.u64 %0,%0,%1;" :"+l"(c):"l"(a):"memory")
#define PTX_UADD1(c,a)    asm volatile("addc.u64    %0,%0,%1;" :"+l"(c):"l"(a))

// Soustraction avec borrow
#define PTX_USUBO(r,a,b)  asm volatile("sub.cc.u64  %0,%1,%2;" :"=l"(r):"l"(a),"l"(b):"memory")
#define PTX_USUBC(r,a,b)  asm volatile("subc.cc.u64 %0,%1,%2;" :"=l"(r):"l"(a),"l"(b):"memory")
#define PTX_USUB(r,a,b)   asm volatile("subc.u64    %0,%1,%2;" :"=l"(r):"l"(a),"l"(b))

// Soustraction in-place avec borrow
#define PTX_USUBO1(c,a)   asm volatile("sub.cc.u64  %0,%0,%1;" :"+l"(c):"l"(a):"memory")
#define PTX_USUBC1(c,a)   asm volatile("subc.cc.u64 %0,%0,%1;" :"+l"(c):"l"(a):"memory")
#define PTX_USUB1(c,a)    asm volatile("subc.u64    %0,%0,%1;" :"+l"(c):"l"(a))

// Multiplication 64x64 -> 128 bits
#define PTX_UMULLO(lo,a,b) asm volatile("mul.lo.u64  %0,%1,%2;" :"=l"(lo):"l"(a),"l"(b))
#define PTX_UMULHI(hi,a,b) asm volatile("mul.hi.u64  %0,%1,%2;" :"=l"(hi):"l"(a),"l"(b))

// Multiply-Add (partie haute) avec carry -- instruction clé pour CIOS
#define PTX_MADDO(r,a,b,c)  asm volatile("mad.hi.cc.u64  %0,%1,%2,%3;" :"=l"(r):"l"(a),"l"(b),"l"(c):"memory")
#define PTX_MADDC(r,a,b,c)  asm volatile("madc.hi.cc.u64 %0,%1,%2,%3;" :"=l"(r):"l"(a),"l"(b),"l"(c):"memory")
#define PTX_MADD(r,a,b,c)   asm volatile("madc.hi.u64    %0,%1,%2,%3;" :"=l"(r):"l"(a),"l"(b),"l"(c))

// ============================================================================
// 2. ADDITION / SOUSTRACTION 256 BITS (blocs monolithiques)
// ============================================================================

// out = a + b (256 bits), retourne carry 32 bits (0 ou 1)
__device__ __forceinline__ uint32_t
ptx_add256(uint64_t out[4], const uint64_t a[4], const uint64_t b[4])
{
    uint32_t carry;
    asm volatile(
        "add.cc.u64  %0,%4,%8;\n\t"
        "addc.cc.u64 %1,%5,%9;\n\t"
        "addc.cc.u64 %2,%6,%10;\n\t"
        "addc.cc.u64 %3,%7,%11;\n\t"
        "addc.u32    %12,0,0;\n\t"
        : "=l"(out[0]),"=l"(out[1]),"=l"(out[2]),"=l"(out[3]),"=r"(carry)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
          "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3])
    );
    return carry;
}

// out = a - b (256 bits), retourne borrow 32 bits (0 ou 1)
__device__ __forceinline__ uint32_t
ptx_sub256(uint64_t out[4], const uint64_t a[4], const uint64_t b[4])
{
    uint32_t borrow;
    asm volatile(
        "sub.cc.u64  %0,%4,%8;\n\t"
        "subc.cc.u64 %1,%5,%9;\n\t"
        "subc.cc.u64 %2,%6,%10;\n\t"
        "subc.cc.u64 %3,%7,%11;\n\t"
        "subc.u32    %12,0,0;\n\t"
        : "=l"(out[0]),"=l"(out[1]),"=l"(out[2]),"=l"(out[3]),"=r"(borrow)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
          "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3])
    );
    return borrow;
}

// Comparaison 256 bits : retourne -1, 0, 1
__device__ __forceinline__ int
ptx_cmp256(const uint64_t a[4], const uint64_t b[4])
{
    if (a[3]!=b[3]) return (a[3]<b[3])?-1:1;
    if (a[2]!=b[2]) return (a[2]<b[2])?-1:1;
    if (a[1]!=b[1]) return (a[1]<b[1])?-1:1;
    if (a[0]!=b[0]) return (a[0]<b[0])?-1:1;
    return 0;
}

// ============================================================================
// 3. HELPER CIOS : accumulateur colonne (s0,s1,s2) += a * b
// ============================================================================
//
// Utilise mad.lo.cc + madc.hi.cc sur le MÊME produit a*b :
//   s0 += lo(a*b)        [set CC]
//   s1 += hi(a*b) + CC   [set CC]
//   s2 += CC
//
// La chaîne CC est propre car les deux instructions concernent le même produit.
// C'est le cœur de l'approche Barracuda -- réutilisé ici pour BN254 Fp et Fr.

__device__ __forceinline__ void
ptx_mad_acc(uint64_t &s0, uint64_t &s1, uint64_t &s2,
            const uint64_t a, const uint64_t b)
{
    asm volatile(
        "mad.lo.cc.u64  %0, %3, %4, %0;\n\t"   // s0 += lo(a*b), set CC
        "madc.hi.cc.u64 %1, %3, %4, %1;\n\t"   // s1 += hi(a*b) + CC, set CC
        "addc.u64       %2, %2, 0;\n\t"          // s2 += CC
        : "+l"(s0), "+l"(s1), "+l"(s2)
        : "l"(a), "l"(b)
        : "memory");
}

// ============================================================================
// 4. RÉDUCTION FINALE CIOS : T in [0,2p) -> [0,p)
// ============================================================================
// Partagée entre fpMul (mod p) et frMul (mod r) — seule la constante du modulo change.
// Usage : ptx_final_reduce(t0,t1,t2,t3, MOD[4], r[4])

__device__ __forceinline__ void
ptx_final_reduce(uint64_t t0, uint64_t t1, uint64_t t2, uint64_t t3,
                 const uint64_t mod[4], uint64_t r[4])
{
    uint64_t s0,s1,s2,s3,borrow;
    asm volatile(
        "sub.cc.u64  %0,%5,%9;\n\t"
        "subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t"
        "subc.cc.u64 %3,%8,%12;\n\t"
        "subc.u64    %4,0,0;\n\t"
        : "=l"(s0),"=l"(s1),"=l"(s2),"=l"(s3),"=l"(borrow)
        : "l"(t0),"l"(t1),"l"(t2),"l"(t3),
          "l"(mod[0]),"l"(mod[1]),"l"(mod[2]),"l"(mod[3]));
    uint64_t keep = borrow & 1ULL, mask = 0ULL - keep;
    r[0]=(t0&mask)|(s0&~mask); r[1]=(t1&mask)|(s1&~mask);
    r[2]=(t2&mask)|(s2&~mask); r[3]=(t3&mask)|(s3&~mask);
}

// ============================================================================
// 5. NOTE ARCHITECTURALE : pourquoi pas de "lazy reduction" pour BN254
// ============================================================================
//
// Cyclope (secp256k1) utilise une réduction "lazy" (fold) car :
//   p_secp = 2^256 - K  avec K = 2^32 + 977  (forme spéciale, K petit)
//   Réduire x * 2^256 = x * K, quasi-gratuit.
//
// BN254 : p = 0x30644e72e131a029... sans forme spéciale.
//   -> Réduction "lazy" impossible.
//   -> CIOS obligatoire : mul et reduce entrelacés colonne par colonne.
//   -> fp_cios_iter() dans fp_bn254.cuh est la forme optimale.
//
// Conséquence : on ne peut PAS séparer _mul et _reduce comme dans Barracuda.
// CIOS est déjà le déroulage optimal de mul+reduce pour BN254.
//
// ============================================================================
