/*
 * ===========================================================================
 * Forum — fp_bn254.cuh
 * BN254 prime field arithmetic — CUDA sm_120 (Blackwell RTX 5060)
 * ===========================================================================
 *
 *  p  = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
 *  p' = -p^{-1} mod 2^64 = 0x87d20782e4866389
 *  R  = 2^256 mod p
 *  R2 = R*R mod p
 *
 * Representation Montgomery : tout element stocke comme a*R mod p.
 * ===========================================================================
 */

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// 1. MACROS PTX
// ============================================================================

#define UADDO(r,a,b)      asm volatile("add.cc.u64  %0,%1,%2;" :"=l"(r):"l"(a),"l"(b):"memory");
#define UADDC(r,a,b)      asm volatile("addc.cc.u64 %0,%1,%2;" :"=l"(r):"l"(a),"l"(b):"memory");
#define UADD(r,a,b)       asm volatile("addc.u64    %0,%1,%2;" :"=l"(r):"l"(a),"l"(b));
#define UADDO1(c,a)       asm volatile("add.cc.u64  %0,%0,%1;" :"+l"(c):"l"(a):"memory");
#define UADDC1(c,a)       asm volatile("addc.cc.u64 %0,%0,%1;" :"+l"(c):"l"(a):"memory");
#define UADD1(c,a)        asm volatile("addc.u64    %0,%0,%1;" :"+l"(c):"l"(a));
#define USUBO(r,a,b)      asm volatile("sub.cc.u64  %0,%1,%2;" :"=l"(r):"l"(a),"l"(b):"memory");
#define USUBC(r,a,b)      asm volatile("subc.cc.u64 %0,%1,%2;" :"=l"(r):"l"(a),"l"(b):"memory");
#define USUB(r,a,b)       asm volatile("subc.u64    %0,%1,%2;" :"=l"(r):"l"(a),"l"(b));
#define USUBO1(c,a)       asm volatile("sub.cc.u64  %0,%0,%1;" :"+l"(c):"l"(a):"memory");
#define USUBC1(c,a)       asm volatile("subc.cc.u64 %0,%0,%1;" :"+l"(c):"l"(a):"memory");
#define USUB1(c,a)        asm volatile("subc.u64    %0,%0,%1;" :"+l"(c):"l"(a));
#define CM_UMULLO(lo,a,b) asm volatile("mul.lo.u64  %0,%1,%2;" :"=l"(lo):"l"(a),"l"(b));
#define CM_UMULHI(hi,a,b) asm volatile("mul.hi.u64  %0,%1,%2;" :"=l"(hi):"l"(a),"l"(b));
#define CM_MADDO(r,a,b,c) asm volatile("mad.hi.cc.u64  %0,%1,%2,%3;" :"=l"(r):"l"(a),"l"(b),"l"(c):"memory");
#define CM_MADDC(r,a,b,c) asm volatile("madc.hi.cc.u64 %0,%1,%2,%3;" :"=l"(r):"l"(a),"l"(b),"l"(c):"memory");
#define CM_MADD(r,a,b,c)  asm volatile("madc.hi.u64    %0,%1,%2,%3;" :"=l"(r):"l"(a),"l"(b),"l"(c));
#define __sleft128(a,b,n)  (((b)<<(n))|((a)>>(64-(n))))
#define __sright128(a,b,n) (((a)>>(n))|((b)<<(64-(n))))

// ============================================================================
// 2. CONSTANTES BN254
// ============================================================================

__constant__ uint64_t BN254_P[4] = {
    0x3c208c16d87cfd47ULL, 0x97816a916871ca8dULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};
__constant__ uint64_t BN254_R_MOD[4] = {
    0xd35d438dc58f0d9dULL, 0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL, 0x0e0a77c19a07df2fULL
};
__constant__ uint64_t BN254_R2[4] = {
    0xf32cfc5b538afa89ULL, 0xb5e71911d44501fbULL,
    0x47ab1eff0a417ff6ULL, 0x06d89f71cab8351fULL
};
static constexpr uint64_t BN254_P_PRIME = 0x87d20782e4866389ULL;

__constant__ uint64_t BN254_G1X[4] = {
    0xd35d438dc58f0d9dULL, 0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL, 0x0e0a77c19a07df2fULL
};
__constant__ uint64_t BN254_G1Y[4] = {
    0xa6ba871b8b1e1b3aULL, 0x14f1d651eb8e167bULL,
    0xccdd46def0f28c58ULL, 0x1c14ef83340fbe5eULL
};

// ============================================================================
// 3. OPERATIONS DE CHAMP
// ============================================================================

__device__ __forceinline__ void fpCopy(const uint64_t a[4], uint64_t r[4]) {
    r[0]=a[0]; r[1]=a[1]; r[2]=a[2]; r[3]=a[3];
}

__device__ __forceinline__ int fpCmp(const uint64_t a[4], const uint64_t b[4]) {
    if(a[3]!=b[3]) return (a[3]<b[3])?-1:1;
    if(a[2]!=b[2]) return (a[2]<b[2])?-1:1;
    if(a[1]!=b[1]) return (a[1]<b[1])?-1:1;
    if(a[0]!=b[0]) return (a[0]<b[0])?-1:1;
    return 0;
}

__device__ __forceinline__ void fpAdd(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t r0,r1,r2,r3, s0,s1,s2,s3, borrow;
    asm volatile(
        "add.cc.u64  %0,%4,%8;\n\t"
        "addc.cc.u64 %1,%5,%9;\n\t"
        "addc.cc.u64 %2,%6,%10;\n\t"
        "addc.u64    %3,%7,%11;\n\t"
        :"=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
         "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    asm volatile(
        "sub.cc.u64  %0,%5,%9;\n\t"
        "subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t"
        "subc.cc.u64 %3,%8,%12;\n\t"
        "subc.u64    %4,0,0;\n\t"
        :"=l"(s0),"=l"(s1),"=l"(s2),"=l"(s3),"=l"(borrow)
        :"l"(r0),"l"(r1),"l"(r2),"l"(r3),
         "l"(BN254_P[0]),"l"(BN254_P[1]),"l"(BN254_P[2]),"l"(BN254_P[3]));
    uint64_t keep=borrow&1ULL, mask=0ULL-keep;
    r[0]=(r0&mask)|(s0&~mask); r[1]=(r1&mask)|(s1&~mask);
    r[2]=(r2&mask)|(s2&~mask); r[3]=(r3&mask)|(s3&~mask);
}

__device__ __forceinline__ void fpSub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0,t1,t2,t3,bor;
    asm volatile(
        "sub.cc.u64  %0,%5,%9;\n\t"
        "subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t"
        "subc.cc.u64 %3,%8,%12;\n\t"
        "subc.u64    %4,0,0;\n\t"
        :"=l"(t0),"=l"(t1),"=l"(t2),"=l"(t3),"=l"(bor)
        :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
         "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    uint64_t m=0ULL-(bor&1ULL);
    uint64_t p0=BN254_P[0]&m, p1=BN254_P[1]&m, p2=BN254_P[2]&m, p3=BN254_P[3]&m;
    asm volatile(
        "add.cc.u64  %0,%4,%8;\n\t"
        "addc.cc.u64 %1,%5,%9;\n\t"
        "addc.cc.u64 %2,%6,%10;\n\t"
        "addc.u64    %3,%7,%11;\n\t"
        :"=l"(r[0]),"=l"(r[1]),"=l"(r[2]),"=l"(r[3])
        :"l"(t0),"l"(t1),"l"(t2),"l"(t3),
         "l"(p0),"l"(p1),"l"(p2),"l"(p3));
}

__device__ __forceinline__ void fpNeg(const uint64_t a[4], uint64_t r[4]) {
    uint64_t t0,t1,t2,t3;
    USUBO(t0,BN254_P[0],a[0]); USUBC(t1,BN254_P[1],a[1]);
    USUBC(t2,BN254_P[2],a[2]); USUB (t3,BN254_P[3],a[3]);
    uint64_t nz=a[0]|a[1]|a[2]|a[3], mask=0ULL-(uint64_t)(nz!=0);
    r[0]=t0&mask; r[1]=t1&mask; r[2]=t2&mask; r[3]=t3&mask;
}

__device__ __forceinline__ void fpNormalize(uint64_t x[4]) {
    uint64_t t0,t1,t2,t3,bor;
    asm volatile(
        "sub.cc.u64  %0,%5,%9;\n\t"
        "subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t"
        "subc.cc.u64 %3,%8,%12;\n\t"
        "subc.u64    %4,0,0;\n\t"
        :"=l"(t0),"=l"(t1),"=l"(t2),"=l"(t3),"=l"(bor)
        :"l"(x[0]),"l"(x[1]),"l"(x[2]),"l"(x[3]),
         "l"(BN254_P[0]),"l"(BN254_P[1]),"l"(BN254_P[2]),"l"(BN254_P[3]));
    uint64_t keep=0ULL-(1ULL-(bor&1ULL));
    x[0]=(x[0]&~keep)|(t0&keep); x[1]=(x[1]&~keep)|(t1&keep);
    x[2]=(x[2]&~keep)|(t2&keep); x[3]=(x[3]&~keep)|(t3&keep);
}

// ============================================================================
// fpMul : multiplication de Montgomery CIOS
//
// Chaque iteration i contient exactement 2 blocs asm volatile :
//   Bloc A : T += a[i] * b   (8 instructions madc enchainees)
//   Bloc B : T += m   * p    (8 instructions madc enchainees)
//
// Instructions utilisees :
//   mad.lo.cc.u64  rd, a, b, c  =>  rd = lo(a*b)+c,      set CC
//   madc.hi.cc.u64 rd, a, b, c  =>  rd = hi(a*b)+c+CC,   set CC
//   madc.lo.cc.u64 rd, a, b, c  =>  rd = lo(a*b)+c+CC,   set CC
//   madc.hi.u64    rd, a, b, c  =>  rd = hi(a*b)+c+CC,   pas CC (fin)
//
// Ces instructions sont ATOMIQUES (mul+addc fusionnes) :
// CC n'est jamais ecrase entre deux addc de la chaine.
// ============================================================================


// ============================================================================
// fpMul : multiplication de Montgomery CIOS — version PTX v2
//
// Point cle : le carry entre limbs est un MOT 64-bit complet, pas un simple
// carry flag reinjecte dans l'addition suivante.
// Pour chaque limb j>=1 on calcule exactement :
//   s1 = acc + carry_prev
//   s2 = low(s1) + lo(prod)
//   carry_next = hi(prod) + carry_out(s1) + carry_out(s2)
// ============================================================================

__device__ __forceinline__ void fpMulCiosLimb0(
    uint64_t &acc, uint64_t &carry,
    const uint64_t x, const uint64_t y)
{
    uint64_t lo, hi;
    CM_UMULLO(lo, x, y);
    CM_UMULHI(hi, x, y);
    asm volatile(
        "add.cc.u64  %0, %0, %2;\n\t"
        "addc.u64    %1, %3, 0;\n\t"
        : "+l"(acc), "=l"(carry)
        : "l"(lo), "l"(hi)
        : "memory");
}

__device__ __forceinline__ void fpMulCiosLimbN(
    uint64_t &acc, uint64_t &carry,
    const uint64_t x, const uint64_t y)
{
    uint64_t lo, hi, ctmp;
    CM_UMULLO(lo, x, y);
    CM_UMULHI(hi, x, y);
	asm volatile(
		"add.cc.u64  %0, %0, %2;\n\t"
		"addc.u64    %1, 0, 0;\n\t"
		"add.cc.u64  %0, %0, %3;\n\t"
		"addc.u64    %2, %4, %1;\n\t"
		: "+l"(acc), "=l"(ctmp), "+l"(carry)
		: "l"(lo), "l"(hi)
		: "memory");
	(void)ctmp;	
}

__device__ __forceinline__ void fpMulCiosIterPTX(
    uint64_t &t0, uint64_t &t1, uint64_t &t2, uint64_t &t3, uint64_t &t4,
    const uint64_t ai,
    const uint64_t b0, const uint64_t b1, const uint64_t b2, const uint64_t b3)
{
    uint64_t carry, mm;

    // Phase A : T += ai * b
    fpMulCiosLimb0(t0, carry, ai, b0);
    fpMulCiosLimbN(t1, carry, ai, b1);
    fpMulCiosLimbN(t2, carry, ai, b2);
    fpMulCiosLimbN(t3, carry, ai, b3);
    asm volatile("add.u64 %0, %0, %1;" : "+l"(t4) : "l"(carry) : "memory");

    // mm = t0 * p' mod 2^64
    asm volatile("mul.lo.u64 %0,%1,%2;" : "=l"(mm) : "l"(t0), "l"(BN254_P_PRIME));

    // Phase B : T += mm * p
    fpMulCiosLimb0(t0, carry, mm, BN254_P[0]);
    fpMulCiosLimbN(t1, carry, mm, BN254_P[1]);
    fpMulCiosLimbN(t2, carry, mm, BN254_P[2]);
    fpMulCiosLimbN(t3, carry, mm, BN254_P[3]);
    asm volatile("add.u64 %0, %0, %1;" : "+l"(t4) : "l"(carry) : "memory");

    // Shift Montgomery
    t0 = t1;
    t1 = t2;
    t2 = t3;
    t3 = t4;
    t4 = 0;
}

__device__ __forceinline__ void fpMul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4])
{
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;

    fpMulCiosIterPTX(t0, t1, t2, t3, t4, a[0], b[0], b[1], b[2], b[3]);
    fpMulCiosIterPTX(t0, t1, t2, t3, t4, a[1], b[0], b[1], b[2], b[3]);
    fpMulCiosIterPTX(t0, t1, t2, t3, t4, a[2], b[0], b[1], b[2], b[3]);
    fpMulCiosIterPTX(t0, t1, t2, t3, t4, a[3], b[0], b[1], b[2], b[3]);

    // reduction finale : T in [0,2p) -> [0,p)
    uint64_t s0,s1,s2,s3,borrow;
    asm volatile(
        "sub.cc.u64  %0,%5,%9;\n\t"
        "subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t"
        "subc.cc.u64 %3,%8,%12;\n\t"
        "subc.u64    %4,0,0;\n\t"
        : "=l"(s0), "=l"(s1), "=l"(s2), "=l"(s3), "=l"(borrow)
        : "l"(t0), "l"(t1), "l"(t2), "l"(t3),
          "l"(BN254_P[0]), "l"(BN254_P[1]), "l"(BN254_P[2]), "l"(BN254_P[3]));
    uint64_t keep = borrow & 1ULL, mask = 0ULL - keep;
    r[0] = (t0 & mask) | (s0 & ~mask);
    r[1] = (t1 & mask) | (s1 & ~mask);
    r[2] = (t2 & mask) | (s2 & ~mask);
    r[3] = (t3 & mask) | (s3 & ~mask);
}


__device__ __forceinline__ void fpAdd128At(uint64_t t[9], int idx, uint64_t lo, uint64_t hi)
{
    unsigned __int128 z = (unsigned __int128)t[idx] + lo;
    t[idx] = (uint64_t)z;
    uint64_t carry = (uint64_t)(z >> 64);

    z = (unsigned __int128)t[idx + 1] + hi + carry;
    t[idx + 1] = (uint64_t)z;
    carry = (uint64_t)(z >> 64);

    int k = idx + 2;
    while (carry) {
        z = (unsigned __int128)t[k] + carry;
        t[k] = (uint64_t)z;
        carry = (uint64_t)(z >> 64);
        ++k;
    }
}

__device__ __forceinline__ void fpAdd129At(uint64_t t[9], int idx, uint64_t lo, uint64_t hi, uint64_t extra)
{
    unsigned __int128 z = (unsigned __int128)t[idx] + lo;
    t[idx] = (uint64_t)z;
    uint64_t carry = (uint64_t)(z >> 64);

    z = (unsigned __int128)t[idx + 1] + hi + carry;
    t[idx + 1] = (uint64_t)z;
    carry = (uint64_t)(z >> 64);

    z = (unsigned __int128)t[idx + 2] + extra + carry;
    t[idx + 2] = (uint64_t)z;
    carry = (uint64_t)(z >> 64);

    int k = idx + 3;
    while (carry) {
        z = (unsigned __int128)t[k] + carry;
        t[k] = (uint64_t)z;
        carry = (uint64_t)(z >> 64);
        ++k;
    }
}

__device__ __forceinline__ void fpMontReduce512(uint64_t t[9], uint64_t r[4])
{
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint64_t m;
        CM_UMULLO(m, t[i], BN254_P_PRIME);

        unsigned __int128 z = (unsigned __int128)m * BN254_P[0] + t[i + 0];
        t[i + 0] = (uint64_t)z;
        uint64_t carry = (uint64_t)(z >> 64);

        z = (unsigned __int128)m * BN254_P[1] + t[i + 1] + carry;
        t[i + 1] = (uint64_t)z;
        carry = (uint64_t)(z >> 64);

        z = (unsigned __int128)m * BN254_P[2] + t[i + 2] + carry;
        t[i + 2] = (uint64_t)z;
        carry = (uint64_t)(z >> 64);

        z = (unsigned __int128)m * BN254_P[3] + t[i + 3] + carry;
        t[i + 3] = (uint64_t)z;
        carry = (uint64_t)(z >> 64);

        int k = i + 4;
        while (carry) {
            z = (unsigned __int128)t[k] + carry;
            t[k] = (uint64_t)z;
            carry = (uint64_t)(z >> 64);
            ++k;
        }
    }

    uint64_t x0 = t[4], x1 = t[5], x2 = t[6], x3 = t[7];
    uint64_t s0, s1, s2, s3, borrow;
    asm volatile(
        "sub.cc.u64  %0,%5,%9;\n\t"
        "subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t"
        "subc.cc.u64 %3,%8,%12;\n\t"
        "subc.u64    %4,0,0;\n\t"
        : "=l"(s0), "=l"(s1), "=l"(s2), "=l"(s3), "=l"(borrow)
        : "l"(x0), "l"(x1), "l"(x2), "l"(x3),
          "l"(BN254_P[0]), "l"(BN254_P[1]), "l"(BN254_P[2]), "l"(BN254_P[3]));
    uint64_t ge = ((t[8] != 0ULL) | (1ULL - (borrow & 1ULL)));
    uint64_t mask = 0ULL - ge;
    r[0] = (x0 & ~mask) | (s0 & mask);
    r[1] = (x1 & ~mask) | (s1 & mask);
    r[2] = (x2 & ~mask) | (s2 & mask);
    r[3] = (x3 & ~mask) | (s3 & mask);
}
/*
__device__ __forceinline__ void fpSqr(const uint64_t a[4], uint64_t r[4])
{
    uint64_t t[9] = {0,0,0,0,0,0,0,0,0};
    uint64_t lo, hi;

    // Diagonales: a0^2, a1^2, a2^2, a3^2
    CM_UMULLO(lo, a[0], a[0]); CM_UMULHI(hi, a[0], a[0]); fpAdd128At(t, 0, lo, hi);
    CM_UMULLO(lo, a[1], a[1]); CM_UMULHI(hi, a[1], a[1]); fpAdd128At(t, 2, lo, hi);
    CM_UMULLO(lo, a[2], a[2]); CM_UMULHI(hi, a[2], a[2]); fpAdd128At(t, 4, lo, hi);
    CM_UMULLO(lo, a[3], a[3]); CM_UMULHI(hi, a[3], a[3]); fpAdd128At(t, 6, lo, hi);

    // Produits croisés doublés: 2*a_i*a_j
    uint64_t extra;
    CM_UMULLO(lo, a[0], a[1]); CM_UMULHI(hi, a[0], a[1]);
    extra = hi >> 63; hi = (hi << 1) | (lo >> 63); lo <<= 1; fpAdd129At(t, 1, lo, hi, extra);

    CM_UMULLO(lo, a[0], a[2]); CM_UMULHI(hi, a[0], a[2]);
    extra = hi >> 63; hi = (hi << 1) | (lo >> 63); lo <<= 1; fpAdd129At(t, 2, lo, hi, extra);

    CM_UMULLO(lo, a[0], a[3]); CM_UMULHI(hi, a[0], a[3]);
    extra = hi >> 63; hi = (hi << 1) | (lo >> 63); lo <<= 1; fpAdd129At(t, 3, lo, hi, extra);

    CM_UMULLO(lo, a[1], a[2]); CM_UMULHI(hi, a[1], a[2]);
    extra = hi >> 63; hi = (hi << 1) | (lo >> 63); lo <<= 1; fpAdd129At(t, 3, lo, hi, extra);

    CM_UMULLO(lo, a[1], a[3]); CM_UMULHI(hi, a[1], a[3]);
    extra = hi >> 63; hi = (hi << 1) | (lo >> 63); lo <<= 1; fpAdd129At(t, 4, lo, hi, extra);

    CM_UMULLO(lo, a[2], a[3]); CM_UMULHI(hi, a[2], a[3]);
    extra = hi >> 63; hi = (hi << 1) | (lo >> 63); lo <<= 1; fpAdd129At(t, 5, lo, hi, extra);

    fpMontReduce512(t, r);
}
*/
// ============================================================================
// fpSqr_cios_symm_proto
//
// Prototype de square spécialisé BN254, toujours en Montgomery CIOS,
// mais en exploitant la symétrie du carré :
//   4 diagonales  : a0^2, a1^2, a2^2, a3^2
//   6 croisés×2  : 2*a0*a1, 2*a0*a2, 2*a0*a3, 2*a1*a2, 2*a1*a3, 2*a2*a3
//
// Différence clé vs proto non-CIOS précédent :
//   - PAS de t[8]/t[9] complet
//   - PAS de REDC séparée
//   - réduction Montgomery toujours entrelacée à chaque itération
//
// État interne : T[6] et non T[5], car 2*(x*y) peut faire 129 bits.
// ============================================================================

struct FpSqrCIOSState {
    uint64_t t[6];
};

__device__ __forceinline__ void fpSqrStateInit(FpSqrCIOSState &S) {
    #pragma unroll
    for (int i = 0; i < 6; i++) S.t[i] = 0ULL;
}

// Ajoute un mot x à partir du limb OFF, avec propagation branchless jusqu'à T5.
// Pas de while(carry), boucle fixe et unrollée.
template<int OFF>
__device__ __forceinline__ void fpSqrAddWord(FpSqrCIOSState &S, uint64_t x) {
    uint64_t carry = x;
    #pragma unroll
    for (int k = OFF; k < 6; ++k) {
        uint64_t out, c;
        asm volatile(
            "add.cc.u64  %0, %2, %3;\n\t"
            "addc.u64    %1, 0, 0;\n\t"
            : "=l"(out), "=l"(c)
            : "l"(S.t[k]), "l"(carry)
            : "memory");
        S.t[k] = out;
        carry  = c;
    }
    // Le proto suppose que T[6] n'est pas nécessaire si le scheduling est correct.
}

// Ajoute un produit 128-bit simple x*y à partir du limb OFF.
template<int OFF>
__device__ __forceinline__ void fpSqrAddProd1x(FpSqrCIOSState &S, uint64_t x, uint64_t y) {
    uint64_t lo, hi;
    CM_UMULLO(lo, x, y);
    CM_UMULHI(hi, x, y);
    fpSqrAddWord<OFF    >(S, lo);
    fpSqrAddWord<OFF + 1>(S, hi);
}

// Ajoute 2*(x*y) à partir du limb OFF.
// Important : 2*(128-bit) peut faire 129 bits.
// Donc on ajoute trois morceaux : lo2, hi2, topbit.
template<int OFF>
__device__ __forceinline__ void fpSqrAddProd2x(FpSqrCIOSState &S, uint64_t x, uint64_t y) {
    uint64_t lo, hi;
    CM_UMULLO(lo, x, y);
    CM_UMULHI(hi, x, y);

    const uint64_t lo2    = lo << 1;
    const uint64_t hi2    = (hi << 1) | (lo >> 63);
    const uint64_t hi_top = (hi >> 63);   // 0 ou 1

    fpSqrAddWord<OFF    >(S, lo2);
    fpSqrAddWord<OFF + 1>(S, hi2);
    fpSqrAddWord<OFF + 2>(S, hi_top);
}

// Une étape REDC Montgomery branchless :
//   m = T0 * p' mod 2^64
//   T += m * p
//   shift d'un limb
__device__ __forceinline__ void fpSqrRedcStep(FpSqrCIOSState &S) {
    uint64_t mm;
    CM_UMULLO(mm, S.t[0], BN254_P_PRIME);

    fpSqrAddProd1x<0>(S, mm, BN254_P[0]);
    fpSqrAddProd1x<1>(S, mm, BN254_P[1]);
    fpSqrAddProd1x<2>(S, mm, BN254_P[2]);
    fpSqrAddProd1x<3>(S, mm, BN254_P[3]);

    // Le choix de mm garantit que le limb bas est annulé modulo 2^64.
    // On "drop" donc T0 via le shift Montgomery.
    S.t[0] = S.t[1];
    S.t[1] = S.t[2];
    S.t[2] = S.t[3];
    S.t[3] = S.t[4];
    S.t[4] = S.t[5];
    S.t[5] = 0ULL;
}

// -----------------------------------------------------------------------------
// Itérations spécialisées du carré
// -----------------------------------------------------------------------------

__device__ __forceinline__ void fpSqrIter0(FpSqrCIOSState &S, const uint64_t a[4]) {
    fpSqrAddProd1x<0>(S, a[0], a[0]);   // a0^2
    fpSqrAddProd2x<1>(S, a[0], a[1]);   // 2*a0*a1
    fpSqrAddProd2x<2>(S, a[0], a[2]);   // 2*a0*a2
    fpSqrAddProd2x<3>(S, a[0], a[3]);   // 2*a0*a3
    fpSqrRedcStep(S);
}

__device__ __forceinline__ void fpSqrIter1(FpSqrCIOSState &S, const uint64_t a[4]) {
    fpSqrAddProd1x<1>(S, a[1], a[1]);   // a1^2
    fpSqrAddProd2x<2>(S, a[1], a[2]);   // 2*a1*a2
    fpSqrAddProd2x<3>(S, a[1], a[3]);   // 2*a1*a3
    fpSqrRedcStep(S);
}

__device__ __forceinline__ void fpSqrIter2(FpSqrCIOSState &S, const uint64_t a[4]) {
    fpSqrAddProd1x<2>(S, a[2], a[2]);   // a2^2
    fpSqrAddProd2x<3>(S, a[2], a[3]);   // 2*a2*a3
    fpSqrRedcStep(S);
}

__device__ __forceinline__ void fpSqrIter3(FpSqrCIOSState &S, const uint64_t a[4]) {
    fpSqrAddProd1x<3>(S, a[3], a[3]);   // a3^2
    fpSqrRedcStep(S);
}

// -----------------------------------------------------------------------------
// Proto principal
// -----------------------------------------------------------------------------

__device__ __forceinline__ void fpSqr(const uint64_t a[4], uint64_t r[4]) {
    FpSqrCIOSState S;
    fpSqrStateInit(S);

    fpSqrIter0(S, a);
    fpSqrIter1(S, a);
    fpSqrIter2(S, a);
    fpSqrIter3(S, a);

    // S.t[0..3] contient le résultat Montgomery avant sub conditionnelle finale
    uint64_t s0, s1, s2, s3, borrow;
    asm volatile(
        "sub.cc.u64  %0,%5,%9;\n\t"
        "subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t"
        "subc.cc.u64 %3,%8,%12;\n\t"
        "subc.u64    %4,0,0;\n\t"
        : "=l"(s0), "=l"(s1), "=l"(s2), "=l"(s3), "=l"(borrow)
        : "l"(S.t[0]), "l"(S.t[1]), "l"(S.t[2]), "l"(S.t[3]),
          "l"(BN254_P[0]), "l"(BN254_P[1]), "l"(BN254_P[2]), "l"(BN254_P[3]));

    const uint64_t keep = borrow & 1ULL;
    const uint64_t mask = 0ULL - keep;

    r[0] = (S.t[0] & mask) | (s0 & ~mask);
    r[1] = (S.t[1] & mask) | (s1 & ~mask);
    r[2] = (S.t[2] & mask) | (s2 & ~mask);
    r[3] = (S.t[3] & mask) | (s3 & ~mask);
}

__device__ __forceinline__ void fpMontEncode(const uint64_t a[4], uint64_t r[4]) { fpMul(a,BN254_R2,r); }
__device__ __forceinline__ void fpMontDecode(const uint64_t a[4], uint64_t r[4]) {
    const uint64_t one[4]={1ULL,0ULL,0ULL,0ULL};
    fpMul(a,one,r);
}

__device__ __forceinline__ void fieldAdd(const uint64_t a[4],const uint64_t b[4],uint64_t r[4]){fpAdd(a,b,r);}
__device__ __forceinline__ void fieldSub(const uint64_t a[4],const uint64_t b[4],uint64_t r[4]){fpSub(a,b,r);}
__device__ __forceinline__ void fieldMul(const uint64_t a[4],const uint64_t b[4],uint64_t r[4]){fpMul(a,b,r);}
__device__ __forceinline__ void fieldSqr(const uint64_t a[4],uint64_t r[4]){fpSqr(a,r);}
__device__ __forceinline__ void fieldNeg(const uint64_t a[4],uint64_t r[4]){fpNeg(a,r);}
__device__ __forceinline__ void fieldNormalize(uint64_t x[4]){fpNormalize(x);}
__device__ __forceinline__ void fieldCopy(const uint64_t a[4],uint64_t r[4]){fpCopy(a,r);}

// ============================================================================
// 4. INVERSION MODULAIRE — Bernstein-Yang
// ============================================================================

#define BY_NBBLOCK 5
#define BY_IsPositive(x) (((int64_t)((x)[4]))>=0LL)
#define BY_IsNegative(x) (((int64_t)((x)[4]))<0LL)
#define BY_IsZero(a)     (((a)[4]|(a)[3]|(a)[2]|(a)[1]|(a)[0])==0ULL)
#define BY_IsOne(a)      (((a)[4]==0ULL)&&((a)[3]==0ULL)&&((a)[2]==0ULL)&&((a)[1]==0ULL)&&((a)[0]==1ULL))

static constexpr uint64_t BY_MM64_BN254 = 0x87d20782e4866389ULL & 0x3FFFFFFFFFFFFFFFULL;
static constexpr uint64_t BY_MSK62      = 0x3FFFFFFFFFFFFFFFULL;

template<typename T>
__device__ __forceinline__ void by_swap(T &a,T &b){T t=a;a=b;b=t;}

__device__ __forceinline__ void BY_AddP(uint64_t r[5]){
    UADDO1(r[0],0x3c208c16d87cfd47ULL); UADDC1(r[1],0x97816a916871ca8dULL);
    UADDC1(r[2],0xb85045b68181585dULL); UADDC1(r[3],0x30644e72e131a029ULL);
    UADD1 (r[4],0ULL);
}
__device__ __forceinline__ void BY_SubP(uint64_t r[5]){
    USUBO1(r[0],0x3c208c16d87cfd47ULL); USUBC1(r[1],0x97816a916871ca8dULL);
    USUBC1(r[2],0xb85045b68181585dULL); USUBC1(r[3],0x30644e72e131a029ULL);
    USUB1 (r[4],0ULL);
}
__device__ __forceinline__ void BY_Neg(uint64_t r[5]){
    USUBO(r[0],0ULL,r[0]); USUBC(r[1],0ULL,r[1]);
    USUBC(r[2],0ULL,r[2]); USUBC(r[3],0ULL,r[3]); USUB(r[4],0ULL,r[4]);
}
__device__ __forceinline__ void BY_Load(uint64_t r[5],const uint64_t a[5]){
    #pragma unroll
    for(int i=0;i<5;i++) r[i]=a[i];
}
__device__ __forceinline__ uint32_t BY_ctz(uint64_t x){
    uint32_t n;
    asm("{ .reg .u64 tmp; brev.b64 tmp,%1; clz.b64 %0,tmp; }":"=r"(n):"l"(x));
    return n;
}
__device__ __forceinline__ void BY_ShiftR62(uint64_t r[5]){
    r[0]=(r[1]<<2)|(r[0]>>62); r[1]=(r[2]<<2)|(r[1]>>62);
    r[2]=(r[3]<<2)|(r[2]>>62); r[3]=(r[4]<<2)|(r[3]>>62);
    r[4]=(int64_t)(r[4])>>62;
}
__device__ __forceinline__ void BY_ShiftR62_Carry(uint64_t dest[5],const uint64_t r[5],uint64_t carry){
    dest[0]=(r[1]<<2)|(r[0]>>62); dest[1]=(r[2]<<2)|(r[1]>>62);
    dest[2]=(r[3]<<2)|(r[2]>>62); dest[3]=(r[4]<<2)|(r[3]>>62);
    dest[4]=(carry<<2)|(uint64_t)((int64_t)r[4]>>62);
}
__device__ __forceinline__ uint64_t BY_IMultC(uint64_t r[5],uint64_t a[5],int64_t b){
    uint64_t t[BY_NBBLOCK],carry;
    if(b<0){b=-b; USUBO(t[0],0ULL,a[0]); USUBC(t[1],0ULL,a[1]);
                   USUBC(t[2],0ULL,a[2]); USUBC(t[3],0ULL,a[3]); USUB(t[4],0ULL,a[4]);}
    else BY_Load(t,a);
    CM_UMULLO(r[0],t[0],(uint64_t)b);
    CM_UMULLO(r[1],t[1],(uint64_t)b); CM_MADDO(r[1],t[0],(uint64_t)b,r[1]);
    CM_UMULLO(r[2],t[2],(uint64_t)b); CM_MADDC(r[2],t[1],(uint64_t)b,r[2]);
    CM_UMULLO(r[3],t[3],(uint64_t)b); CM_MADDC(r[3],t[2],(uint64_t)b,r[3]);
    CM_UMULLO(r[4],t[4],(uint64_t)b); CM_MADDC(r[4],t[3],(uint64_t)b,r[4]);
    asm volatile("madc.hi.s64 %0,%1,%2,%3;":"=l"(carry):"l"(t[4]),"l"((uint64_t)b),"l"(0ULL));
    return carry;
}
__device__ __forceinline__ void BY_IMult(uint64_t r[5],uint64_t a[5],int64_t b){
    uint64_t t[BY_NBBLOCK];
    if(b<0){b=-b; USUBO(t[0],0ULL,a[0]); USUBC(t[1],0ULL,a[1]);
                   USUBC(t[2],0ULL,a[2]); USUBC(t[3],0ULL,a[3]); USUB(t[4],0ULL,a[4]);}
    else BY_Load(t,a);
    CM_UMULLO(r[0],t[0],(uint64_t)b);
    CM_UMULLO(r[1],t[1],(uint64_t)b); CM_MADDO(r[1],t[0],(uint64_t)b,r[1]);
    CM_UMULLO(r[2],t[2],(uint64_t)b); CM_MADDC(r[2],t[1],(uint64_t)b,r[2]);
    CM_UMULLO(r[3],t[3],(uint64_t)b); CM_MADDC(r[3],t[2],(uint64_t)b,r[3]);
    CM_UMULLO(r[4],t[4],(uint64_t)b); CM_MADD (r[4],t[3],(uint64_t)b,r[4]);
}
__device__ __forceinline__ void BY_MatrixVecMul(uint64_t u[5],uint64_t v[5],
                                                 int64_t _11,int64_t _12,int64_t _21,int64_t _22){
    uint64_t t1[BY_NBBLOCK],t2[BY_NBBLOCK],t3[BY_NBBLOCK],t4[BY_NBBLOCK];
    BY_IMult(t1,u,_11); BY_IMult(t2,v,_12);
    BY_IMult(t3,u,_21); BY_IMult(t4,v,_22);
    UADDO(u[0],t1[0],t2[0]); UADDC(u[1],t1[1],t2[1]); UADDC(u[2],t1[2],t2[2]);
    UADDC(u[3],t1[3],t2[3]); UADD (u[4],t1[4],t2[4]);
    UADDO(v[0],t3[0],t4[0]); UADDC(v[1],t3[1],t4[1]); UADDC(v[2],t3[2],t4[2]);
    UADDC(v[3],t3[3],t4[3]); UADD (v[4],t3[4],t4[4]);
}
__device__ __forceinline__ void BY_MatrixVecMulHalf(uint64_t dest[5],uint64_t u[5],uint64_t v[5],
                                                     int64_t _11,int64_t _12,uint64_t *carry){
    uint64_t t1[BY_NBBLOCK],t2[BY_NBBLOCK],c1,c2,cout;
    c1=BY_IMultC(t1,u,_11); c2=BY_IMultC(t2,v,_12);
    asm volatile(
        "add.cc.u64  %0,%6,%11;\n\t" "addc.cc.u64 %1,%7,%12;\n\t"
        "addc.cc.u64 %2,%8,%13;\n\t" "addc.cc.u64 %3,%9,%14;\n\t"
        "addc.cc.u64 %4,%10,%15;\n\t" "addc.u64 %5,0,0;\n\t"
        :"=l"(dest[0]),"=l"(dest[1]),"=l"(dest[2]),"=l"(dest[3]),"=l"(dest[4]),"=l"(cout)
        :"l"(t1[0]),"l"(t1[1]),"l"(t1[2]),"l"(t1[3]),"l"(t1[4]),
         "l"(t2[0]),"l"(t2[1]),"l"(t2[2]),"l"(t2[3]),"l"(t2[4]));
    *carry=c1+c2+cout;
}
__device__ __forceinline__ void BY_MulP(uint64_t r[5],uint64_t m){
    typedef unsigned __int128 u128;
    u128 carry;
    carry  = (u128)m * (u128)BN254_P[0];
    r[0]   = (uint64_t)carry;
    carry  = (u128)m * (u128)BN254_P[1] + (carry >> 64);
    r[1]   = (uint64_t)carry;
    carry  = (u128)m * (u128)BN254_P[2] + (carry >> 64);
    r[2]   = (uint64_t)carry;
    carry  = (u128)m * (u128)BN254_P[3] + (carry >> 64);
    r[3]   = (uint64_t)carry;
    r[4]   = (uint64_t)(carry >> 64);
}
__device__ __forceinline__ uint64_t BY_AddCh(uint64_t r[5],const uint64_t a[5],uint64_t carry){
    uint64_t cout;
    asm volatile(
        "add.cc.u64  %0,%0,%6;\n\t" "addc.cc.u64 %1,%1,%7;\n\t"
        "addc.cc.u64 %2,%2,%8;\n\t" "addc.cc.u64 %3,%3,%9;\n\t"
        "addc.cc.u64 %4,%4,%10;\n\t" "addc.u64 %5,%11,0;\n\t"
        :"+l"(r[0]),"+l"(r[1]),"+l"(r[2]),"+l"(r[3]),"+l"(r[4]),"=l"(cout)
        :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(a[4]),"l"(carry));
    return cout;
}
__device__ __forceinline__ void BY_DivStep62(uint64_t u[5],uint64_t v[5],int32_t *pos,
                                              int64_t *uu,int64_t *uv,int64_t *vu,int64_t *vv){
    *uu=1;*uv=0;*vu=0;*vv=1;
    uint32_t bitCount=62,zeros;
    uint64_t u0=u[0],v0=v[0],uh,vh;
    while(*pos>0&&(u[*pos]|v[*pos])==0)(*pos)--;
    if(*pos==0){uh=u[0];vh=v[0];}
    else{
        uint32_t s=__clzll(u[*pos]|v[*pos]);
        if(s==0){uh=u[*pos];vh=v[*pos];}
        else{uh=__sleft128(u[*pos-1],u[*pos],s);vh=__sleft128(v[*pos-1],v[*pos],s);}
    }
    while(true){
        zeros=BY_ctz(v0|(1ULL<<bitCount));
        v0>>=zeros;vh>>=zeros;*uu<<=zeros;*uv<<=zeros;bitCount-=zeros;
        if(bitCount==0)break;
        if(vh<uh){by_swap(uh,vh);by_swap(u0,v0);by_swap(*uu,*vu);by_swap(*uv,*vv);}
        vh-=uh;v0-=u0;*vv-=*uv;*vu-=*uu;
    }
}
__device__ __noinline__ void BY_ModInv5(uint64_t R[5]){
    int64_t uu,uv,vu,vv;
    uint64_t mr0,ms0,carryR,carryS;
    int32_t pos=BY_NBBLOCK-1;
    uint64_t u[BY_NBBLOCK],v[BY_NBBLOCK],r[BY_NBBLOCK],s[BY_NBBLOCK];
    uint64_t tr[BY_NBBLOCK],tmp[BY_NBBLOCK],s0[BY_NBBLOCK];
    u[0]=0x3c208c16d87cfd47ULL; u[1]=0x97816a916871ca8dULL;
    u[2]=0xb85045b68181585dULL; u[3]=0x30644e72e131a029ULL; u[4]=0;
    BY_Load(v,R);
    r[0]=0;r[1]=r[2]=r[3]=r[4]=0;
    s[0]=1;s[1]=s[2]=s[3]=s[4]=0;
    while(true){
        BY_DivStep62(u,v,&pos,&uu,&uv,&vu,&vv);
        BY_MatrixVecMul(u,v,uu,uv,vu,vv);
        if(BY_IsNegative(u)){BY_Neg(u);uu=-uu;uv=-uv;}
        if(BY_IsNegative(v)){BY_Neg(v);vu=-vu;vv=-vv;}
        BY_ShiftR62(u); BY_ShiftR62(v);
        BY_MatrixVecMulHalf(tr,r,s,uu,uv,&carryR);
        mr0=(tr[0]*BY_MM64_BN254)&BY_MSK62;
        BY_MulP(tmp,mr0); carryR=BY_AddCh(tr,tmp,carryR);
        if(BY_IsZero(v)){BY_ShiftR62_Carry(r,tr,carryR);break;}
        else{
            BY_MatrixVecMulHalf(tmp,r,s,vu,vv,&carryS);
            ms0=(tmp[0]*BY_MM64_BN254)&BY_MSK62;
            BY_MulP(s0,ms0); carryS=BY_AddCh(tmp,s0,carryS);
        }
        BY_ShiftR62_Carry(r,tr,carryR);
        BY_ShiftR62_Carry(s,tmp,carryS);
    }
    if(!BY_IsOne(u)){R[0]=R[1]=R[2]=R[3]=R[4]=0;return;}
    while(BY_IsNegative(r))BY_AddP(r);
    while(!BY_IsNegative(r))BY_SubP(r);
    BY_AddP(r);
    BY_Load(R,r);
}
// fpInvNormal : a (representation normale) -> a^{-1} mod p (normale)
// Utilise directement BY_ModInv5 — pas de conversion Montgomery.
__device__ __forceinline__ void fpInvNormal(const uint64_t a[4],uint64_t r[4]){
    if((a[0]|a[1]|a[2]|a[3])==0ULL){r[0]=r[1]=r[2]=r[3]=0;return;}
    uint64_t t[5]={a[0],a[1],a[2],a[3],0};
    BY_ModInv5(t);
    r[0]=t[0];r[1]=t[1];r[2]=t[2];r[3]=t[3];
}

// fpInv : a*R (Montgomery) -> a^{-1}*R (Montgomery)
// Decode -> invert -> encode pour rester dans le domaine Montgomery.
// Utilise par pointAddAffine / pointDoubleAffine.
__device__ __forceinline__ void fpInv(const uint64_t a[4],uint64_t r[4]){
    if((a[0]|a[1]|a[2]|a[3])==0ULL){r[0]=r[1]=r[2]=r[3]=0;return;}
    uint64_t tmp[4];
    fpMontDecode(a, tmp);       // tmp = a (normale)
    fpInvNormal(tmp, tmp);      // tmp = a^{-1} (normale)
    fpMontEncode(tmp, r);       // r   = a^{-1}*R (Montgomery)
}
__device__ __forceinline__ void fieldInv(const uint64_t a[4],uint64_t r[4]){fpInv(a,r);}
__device__ __forceinline__ void _ModInv(uint64_t R[4]){fpInv(R,R);}

// ============================================================================
// 5. POINT EC (G1, y2 = x3 + 3, a=0)
// ============================================================================

struct FpPoint { uint64_t X[4],Y[4]; bool infinity; };

__device__ __forceinline__ void pointSetInfinity(FpPoint &P){
    P.infinity=true;
    P.X[0]=P.X[1]=P.X[2]=P.X[3]=0;
    P.Y[0]=P.Y[1]=P.Y[2]=P.Y[3]=0;
}
__device__ __forceinline__ void pointSetG1(FpPoint &P){
    P.infinity=false; fpCopy(BN254_G1X,P.X); fpCopy(BN254_G1Y,P.Y);
}
__device__ void pointDoubleAffine(const FpPoint &P, FpPoint &R){
    if(P.infinity){pointSetInfinity(R);return;}
    uint64_t t0[4],t1[4],t2[4],t3[4];
    fpSqr(P.X,t0); fpAdd(t0,t0,t1); fpAdd(t1,t0,t0);
    fpAdd(P.Y,P.Y,t1); fpInv(t1,t2); fpMul(t0,t2,t1);
    fpSqr(t1,t2); fpAdd(P.X,P.X,t3); fpSub(t2,t3,R.X);
    fpSub(P.X,R.X,t2); fpMul(t1,t2,t3); fpSub(t3,P.Y,R.Y);
    R.infinity=false;
}
__device__ void pointAddAffine(const FpPoint &P, const FpPoint &Q, FpPoint &R){
    if(P.infinity){R=Q;return;}
    if(Q.infinity){R=P;return;}
    if(fpCmp(P.X,Q.X)==0){
        if(fpCmp(P.Y,Q.Y)==0) pointDoubleAffine(P,R);
        else pointSetInfinity(R);
        return;
    }
    uint64_t t0[4],t1[4],t2[4],t3[4];
    fpSub(Q.X,P.X,t1); fpSub(Q.Y,P.Y,t2); fpInv(t1,t3); fpMul(t2,t3,t0);
    fpSqr(t0,t1); fpSub(t1,P.X,t2); fpSub(t2,Q.X,R.X);
    fpSub(P.X,R.X,t1); fpMul(t0,t1,t2); fpSub(t2,P.Y,R.Y);
    R.infinity=false;
}

// ============================================================================
// 6. JACOBI POINT ARITHMETIC  (for MSM — no field inversion in inner loop)
// ============================================================================

// Jacobi point: (X:Y:Z), affine = (X/Z^2, Y/Z^3)
struct FpPointJac {
    uint64_t X[4], Y[4], Z[4];
    bool infinity;
};

// FpPointAff : already covered by FpPoint above (X,Y only, no Z)
// Alias for clarity in MSM context:
typedef FpPoint FpPointAff;

// ── pointJacSetInfinity ──────────────────────────────────────────────────────
__device__ __forceinline__ void pointJacSetInfinity(FpPointJac &P) {
    P.infinity = true;
    P.X[0]=P.X[1]=P.X[2]=P.X[3]=0;
    P.Y[0]=P.Y[1]=P.Y[2]=P.Y[3]=0;
    P.Z[0]=P.Z[1]=P.Z[2]=P.Z[3]=0;
}

// ── pointAffToJac : lift affine to Jacobi (Z=1 in Montgomery = R mod p) ──────
__device__ __forceinline__ void pointAffToJac(const FpPointAff &P, FpPointJac &Q) {
    if (P.infinity) { pointJacSetInfinity(Q); return; }
    fpCopy(P.X, Q.X);
    fpCopy(P.Y, Q.Y);
    // Z = 1 in Montgomery representation = BN254_R_MOD
    fpCopy(BN254_R_MOD, Q.Z);
    Q.infinity = false;
}

// ── pointJacToAff : convert Jacobi → affine (requires fpInv) ─────────────────
// Use only at the END of MSM, not in the inner loop.
__device__ void pointJacToAff(const FpPointJac &P, FpPointAff &Q) {
    if (P.infinity) { pointSetInfinity(Q); return; }
    uint64_t Zinv[4], Zinv2[4], Zinv3[4];
    fpInv(P.Z, Zinv);          // Zinv  = Z^{-1}
    fpSqr(Zinv, Zinv2);        // Zinv2 = Z^{-2}
    fpMul(Zinv, Zinv2, Zinv3); // Zinv3 = Z^{-3}
    fpMul(P.X, Zinv2, Q.X);    // x = X * Z^{-2}
    fpMul(P.Y, Zinv3, Q.Y);    // y = Y * Z^{-3}
    Q.infinity = false;
}

// ── pointJacDouble : 2P in Jacobi coordinates (a=0, BN254) ──────────────────
// Algorithm: dbl-2009-l (Lange 2009, a=0 specialisation)
// https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
// Cost: 1M + 5S + 10add  (no inversion, no branches after infinity check)
//
//   A = X1^2,  B = Y1^2,  C = B^2
//   D = 2*((X1+B)^2 - A - C)  [= 4*X1*B]
//   E = 3*A,  F = E^2
//   X3 = F - 2*D
//   Y3 = E*(D - X3) - 8*C
//   Z3 = 2*Y1*Z1
__device__ __forceinline__ void pointJacDouble(
    const FpPointJac &P, FpPointJac &R)
{
    if (P.infinity) { R = P; return; }

    uint64_t A[4], B[4], C[4], D[4], E[4], F[4], t0[4];

    fpSqr(P.X, A);
    fpSqr(P.Y, B);
    fpSqr(B,   C);

    fpAdd(P.X, B, t0);
    fpSqr(t0,  t0);
    fpSub(t0,  A,  t0);
    fpSub(t0,  C,  t0);         // t0 = 2*X1*B
    fpAdd(t0,  t0, D);          // D  = 4*X1*B

    fpAdd(A,   A,  E);
    fpAdd(E,   A,  E);          // E  = 3*A

    fpSqr(E,   F);

    fpSub(F,   D,  R.X);
    fpSub(R.X, D,  R.X);        // X3 = F - 2*D

    fpSub(D,   R.X, t0);
    fpMul(E,   t0,  R.Y);       // Y3 = E*(D-X3)
    fpAdd(C,   C,   t0);
    fpAdd(t0,  t0,  t0);
    fpAdd(t0,  t0,  t0);        // t0 = 8*C
    fpSub(R.Y, t0,  R.Y);       // Y3 = E*(D-X3) - 8*C

    fpMul(P.Y, P.Z, t0);
    fpAdd(t0,  t0,  R.Z);       // Z3 = 2*Y1*Z1

    R.infinity = false;
}

// ── pointMixedAdd : Jacobi + Affine → Jacobi  (MSM inner loop) ───────────────
// P1 = (X1:Y1:Z1) Jacobi, P2 = (X2:Y2:1) Affine
// Cost: 8M + 3S + 6add  (no inversion)
//
// Formulas (complete, handles P1=∞ and P2=∞):
//   Z1Z1 = Z1^2
//   U2   = X2 * Z1Z1
//   S2   = Y2 * Z1 * Z1Z1
//   H    = U2 - X1
//   HH   = H^2
//   I    = 4 * HH
//   J    = H * I
//   r    = 2 * (S2 - Y1)
//   V    = X1 * I
//   X3   = r^2 - J - 2*V
//   Y3   = r*(V - X3) - 2*Y1*J
//   Z3   = (Z1+H)^2 - Z1Z1 - HH  [= 2*Z1*H, but this avoids extra mul]
//
// NOTE: All values in Montgomery domain (a*R mod p).
// Handles: P1=∞ → return P2 as Jacobi; P2=∞ → return P1.
// Does NOT handle P1 == P2 (use pointJacDouble in that case).
// In MSM context, collisions (same bucket, same point) are extremely rare.
__device__ __forceinline__ void pointMixedAdd(
    const FpPointJac &P1, const FpPointAff &P2, FpPointJac &P3)
{
    if (P1.infinity) { pointAffToJac(P2, P3); return; }
    if (P2.infinity) { P3 = P1; return; }

    uint64_t Z1Z1[4], U2[4], S2[4], H[4], HH[4], I[4], J[4];
    uint64_t rr[4], V[4], t0[4], t1[4];

    fpSqr(P1.Z, Z1Z1);              // Z1Z1 = Z1^2
    fpMul(P2.X, Z1Z1, U2);          // U2   = X2 * Z1Z1
    fpMul(P1.Z, Z1Z1, t0);          // t0   = Z1^3
    fpMul(P2.Y, t0, S2);            // S2   = Y2 * Z1^3
    fpSub(U2, P1.X, H);             // H    = U2 - X1
    fpSub(S2, P1.Y, t0);
    fpAdd(t0, t0, rr);              // rr   = 2*(S2 - Y1)  (computed early for guard)

    // ── Complete addition: handle P1==P2 and P1==-P2 ─────────────────────────
    if ((H[0]|H[1]|H[2]|H[3]) == 0) {
        if ((rr[0]|rr[1]|rr[2]|rr[3]) == 0)
            pointJacDouble(P1, P3);   // P1 == P2 (same point)
        else
            pointJacSetInfinity(P3);  // P1 == -P2
        return;
    }

    fpSqr(H, HH);                   // HH   = H^2
    fpAdd(HH, HH, t0);
    fpAdd(t0, t0, I);               // I    = 4*HH
    fpMul(H, I, J);                 // J    = H*I
    // rr already = 2*(S2-Y1), matches original formula
    fpMul(P1.X, I, V);              // V    = X1*I
    fpSqr(rr, t0);
    fpSub(t0, J, t1);
    fpAdd(V, V, t0);
    fpSub(t1, t0, P3.X);            // X3   = r^2 - J - 2V
    fpSub(V, P3.X, t0);
    fpMul(rr, t0, t1);
    fpAdd(P1.Y, P1.Y, t0);
    fpMul(t0, J, rr);
    fpSub(t1, rr, P3.Y);            // Y3   = r*(V-X3) - 2*Y1*J
    fpAdd(P1.Z, H, t0);
    fpSqr(t0, t1);
    fpSub(t1, Z1Z1, t0);
    fpSub(t0, HH, P3.Z);            // Z3   = (Z1+H)^2 - Z1Z1 - HH
    P3.infinity = false;
}


// ── pointJacAdd : Jacobi + Jacobi  (for final bucket reduction) ──────────────
// Cost: 12M + 4S + 6add
// Used at end of MSM window, NOT in the main accumulation loop.
__device__ __forceinline__ void pointJacAdd(
    const FpPointJac &P1, const FpPointJac &P2, FpPointJac &P3)
{
    if (P1.infinity) { P3 = P2; return; }
    if (P2.infinity) { P3 = P1; return; }

    uint64_t Z1Z1[4], Z2Z2[4], U1[4], U2[4], S1[4], S2[4];
    uint64_t H[4], I[4], J[4], rr[4], V[4], t0[4], t1[4];

    fpSqr(P1.Z, Z1Z1);
    fpSqr(P2.Z, Z2Z2);
    fpMul(P1.X, Z2Z2, U1);
    fpMul(P2.X, Z1Z1, U2);
    fpMul(P1.Y, P2.Z, t0); fpMul(t0, Z2Z2, S1);
    fpMul(P2.Y, P1.Z, t0); fpMul(t0, Z1Z1, S2);
    fpSub(U2, U1, H);
    fpSub(S2, S1, t0); fpAdd(t0, t0, rr);  // rr = 2*(S2-S1)

    // ── Complete addition: handle P1==P2 and P1==-P2 ─────────────────────────
    // H==0  ⟺  U1==U2  ⟺  same projective X
    // rr==0 ⟺  S1==S2  ⟺  same projective Y  → P1==P2, use doubling
    // rr!=0              → P1==-P2, result is point at infinity
    if ((H[0]|H[1]|H[2]|H[3]) == 0) {
        if ((rr[0]|rr[1]|rr[2]|rr[3]) == 0)
            pointJacDouble(P1, P3);   // P1 == P2
        else
            pointJacSetInfinity(P3);  // P1 == -P2
        return;
    }

    fpAdd(H, H, t0); fpSqr(t0, I);
    fpMul(H, I, J);
    fpMul(U1, I, V);
    fpSqr(rr, t0); fpSub(t0, J, t1); fpAdd(V, V, t0); fpSub(t1, t0, P3.X);
    fpSub(V, P3.X, t0); fpMul(rr, t0, t1);
    fpAdd(S1, S1, t0); fpMul(t0, J, rr); fpSub(t1, rr, P3.Y);
    fpAdd(P1.Z, P2.Z, t0); fpSqr(t0, t1);
    fpSub(t1, Z1Z1, t0); fpSub(t0, Z2Z2, t1); fpMul(t1, H, P3.Z);
    P3.infinity = false;
}
