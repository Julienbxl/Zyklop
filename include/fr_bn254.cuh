/*
 * ===========================================================================
 * Forum — fr_bn254.cuh
 * BN254 scalar field Fr arithmetic — CUDA sm_120 (Blackwell RTX 5060)
 * ===========================================================================
 *
 *  r   = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
 *  r'  = -r^{-1} mod 2^64 = 0xc2e1f593efffffff   (CIOS Montgomery)
 *  R   = 2^256 mod r       = 0x0e0a77c19a07df2f666ea36f7879462e36fc76959f60cd29ac96341c4ffffffb
 *  R2  = R^2 mod r         = 0x0216d0b17f4e44a58c49833d53bb808553fe3ab1e35c59e31bb8e645ae216da7
 *  BY_MM64_FR = -r^{-1} mod 2^62 = 0x02e1f593efffffff
 *
 *  2-adicity: r-1 = 2^28 * t  →  NTT max size 2^28 elements
 *  omega_17^{2^17} mod r = 1  ✓  (verified by Python)
 *
 *  USAGE: NTT polynomial coefficients, MSM scalars, Groth16 witness.
 *         NOT for curve coordinates (use fp_bn254.cuh for those).
 *
 *  INCLUDE ORDER: #include "fp_bn254.cuh" THEN #include "fr_bn254.cuh"
 *  fp_bn254.cuh defines all PTX macros and field-generic BY_ helpers.
 *  fr_bn254.cuh adds Fr-specific constants + frModInv5.
 * ===========================================================================
 */

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// 1. Fr CONSTANTS (scalar field of BN254)
// All verified by Python — see CONTEXT.md Section 7.
// ============================================================================

__constant__ uint64_t BN254_R_SCALAR[4] = {
    0x43e1f593f0000001ULL, 0x2833e84879b97091ULL,
    0xb85045b68181585dULL, 0x30644e72e131a029ULL
};

// R_Fr = 2^256 mod r  ("1" in Montgomery representation for Fr)
__constant__ uint64_t BN254_R_FR[4] = {
    0xac96341c4ffffffbULL, 0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL, 0x0e0a77c19a07df2fULL
};

// R2_Fr = R_Fr^2 mod r  (for frMontEncode: frMul(a, R2_Fr))
__constant__ uint64_t BN254_R2_FR[4] = {
    0x1bb8e645ae216da7ULL, 0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL, 0x0216d0b17f4e44a5ULL
};

// r' = -r^{-1} mod 2^64  (CIOS inner reduction constant)
static constexpr uint64_t BN254_FR_PRIME = 0xc2e1f593efffffffULL;

// BY_MM64_FR = -r^{-1} mod 2^62  (Bernstein-Yang reduction constant for Fr)
static constexpr uint64_t BY_MM64_FR = 0x02e1f593efffffffULL;

// omega_17 = primitive 2^17-th root of unity mod r (normal representation)
// = 0x1bf82deba7d74902c3708cc6e70e61f30512eca95655210e276e5858ce8f58e5
// Verified: omega_17^{2^17} mod r = 1
__constant__ uint64_t BN254_OMEGA_17_RAW[4] = {
    0x276e5858ce8f58e5ULL, 0x0512eca95655210eULL,
    0xc3708cc6e70e61f3ULL, 0x1bf82deba7d74902ULL
};

// omega_28 = primitive 2^28-th root of unity (normal) — max NTT size
__constant__ uint64_t BN254_OMEGA_28_RAW[4] = {
    0x9bd61b6e725b19f0ULL, 0x402d111e41112ed4ULL,
    0x00e0a7eb8ef62abcULL, 0x2a3c09f0a58a7e85ULL
};

// ============================================================================
// 2. Fr-specific BY helpers (field-generic helpers live in fp_bn254.cuh)
// ============================================================================

// BY_FR_MulR : r[5] = m * BN254_R_SCALAR  (BY internal reduction, uses r not p)
__device__ __forceinline__ void BY_FR_MulR(uint64_t res[5], uint64_t m) {
    typedef unsigned __int128 u128;
    u128 carry;
    carry  = (u128)m * (u128)BN254_R_SCALAR[0];
    res[0] = (uint64_t)carry;
    carry  = (u128)m * (u128)BN254_R_SCALAR[1] + (carry >> 64);
    res[1] = (uint64_t)carry;
    carry  = (u128)m * (u128)BN254_R_SCALAR[2] + (carry >> 64);
    res[2] = (uint64_t)carry;
    carry  = (u128)m * (u128)BN254_R_SCALAR[3] + (carry >> 64);
    res[3] = (uint64_t)carry;
    res[4] = (uint64_t)(carry >> 64);
}

// BY_FR_AddR / BY_FR_SubR  (BY convergence helpers for scalar field)
__device__ __forceinline__ void BY_FR_AddR(uint64_t r[5]) {
    UADDO(r[0],r[0],BN254_R_SCALAR[0]); UADDC(r[1],r[1],BN254_R_SCALAR[1]);
    UADDC(r[2],r[2],BN254_R_SCALAR[2]); UADDC(r[3],r[3],BN254_R_SCALAR[3]);
    asm volatile("addc.u64 %0,%0,0;":"+l"(r[4]));
}
__device__ __forceinline__ void BY_FR_SubR(uint64_t r[5]) {
    USUBO(r[0],r[0],BN254_R_SCALAR[0]); USUBC(r[1],r[1],BN254_R_SCALAR[1]);
    USUBC(r[2],r[2],BN254_R_SCALAR[2]); USUBC(r[3],r[3],BN254_R_SCALAR[3]);
    asm volatile("subc.u64 %0,%0,0;":"+l"(r[4]));
}

// BY_FR_AddCh : r[5] += a[5] + carry  (same as BY_AddCh but separate to avoid collision)
__device__ __forceinline__ uint64_t BY_FR_AddCh(uint64_t r[5],const uint64_t a[5],uint64_t carry){
    uint64_t cout;
    asm volatile(
        "add.cc.u64  %0,%0,%6;\n\t" "addc.cc.u64 %1,%1,%7;\n\t"
        "addc.cc.u64 %2,%2,%8;\n\t" "addc.cc.u64 %3,%3,%9;\n\t"
        "addc.cc.u64 %4,%4,%10;\n\t" "addc.u64 %5,%11,0;\n\t"
        :"+l"(r[0]),"+l"(r[1]),"+l"(r[2]),"+l"(r[3]),"+l"(r[4]),"=l"(cout)
        :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(a[4]),"l"(carry));
    return cout;
}

// ============================================================================
// 3. frModInv5 : Bernstein-Yang inversion on Fr
//    Exact copy of BY_ModInv5 (fp_bn254.cuh) with:
//      BN254_P   → BN254_R_SCALAR
//      BY_MM64_BN254 → BY_MM64_FR
//      BY_MulP   → BY_FR_MulR
//      BY_AddP   → BY_FR_AddR
//      BY_SubP   → BY_FR_SubR
//      BY_AddCh  → BY_FR_AddCh
//    All other helpers (BY_DivStep62, BY_MatrixVecMul*, BY_ShiftR62*,
//    BY_IMultC, BY_Neg, BY_Load etc.) are field-independent.
// ============================================================================
__device__ __noinline__ void frModInv5(uint64_t R[5]) {
    int64_t uu,uv,vu,vv;
    uint64_t mr0,ms0,carryR,carryS;
    int32_t pos = BY_NBBLOCK - 1;
    uint64_t u[BY_NBBLOCK],v[BY_NBBLOCK],r[BY_NBBLOCK],s[BY_NBBLOCK];
    uint64_t tr[BY_NBBLOCK],tmp[BY_NBBLOCK],s0[BY_NBBLOCK];

    // Initialize: u = r (scalar modulus), v = input, r = 0, s = 1
    u[0]=0x43e1f593f0000001ULL; u[1]=0x2833e84879b97091ULL;
    u[2]=0xb85045b68181585dULL; u[3]=0x30644e72e131a029ULL; u[4]=0;
    BY_Load(v, R);
    r[0]=0; r[1]=r[2]=r[3]=r[4]=0;
    s[0]=1; s[1]=s[2]=s[3]=s[4]=0;

    while (true) {
        BY_DivStep62(u, v, &pos, &uu, &uv, &vu, &vv);
        BY_MatrixVecMul(u, v, uu, uv, vu, vv);
        if (BY_IsNegative(u)) { BY_Neg(u); uu=-uu; uv=-uv; }
        if (BY_IsNegative(v)) { BY_Neg(v); vu=-vu; vv=-vv; }
        BY_ShiftR62(u); BY_ShiftR62(v);

        BY_MatrixVecMulHalf(tr, r, s, uu, uv, &carryR);
        mr0 = (tr[0] * BY_MM64_FR) & BY_MSK62;
        BY_FR_MulR(tmp, mr0); carryR = BY_FR_AddCh(tr, tmp, carryR);

        if (BY_IsZero(v)) { BY_ShiftR62_Carry(r, tr, carryR); break; }
        else {
            BY_MatrixVecMulHalf(tmp, r, s, vu, vv, &carryS);
            ms0 = (tmp[0] * BY_MM64_FR) & BY_MSK62;
            BY_FR_MulR(s0, ms0); carryS = BY_FR_AddCh(tmp, s0, carryS);
        }
        BY_ShiftR62_Carry(r, tr, carryR);
        BY_ShiftR62_Carry(s, tmp, carryS);
    }

    if (!BY_IsOne(u)) { R[0]=R[1]=R[2]=R[3]=R[4]=0; return; }
    while (BY_IsNegative(r)) BY_FR_AddR(r);
    while (!BY_IsNegative(r)) BY_FR_SubR(r);
    BY_FR_AddR(r);
    BY_Load(R, r);
}

// ============================================================================
// 4. Fr arithmetic (Montgomery domain: all values stored as a*R mod r)
// ============================================================================

// CIOS_Fr macro — identical structure to CIOS_C in fp_bn254.cuh, uses r constants
#define CIOS_Fr(T0,T1,T2,T3,T4,ai) {                                               \
    typedef unsigned __int128 u128;                                                 \
    u128 carry;                                                                     \
    carry  = (u128)(ai)*(u128)(b[0]) + (T0);    T0 = (uint64_t)carry;              \
    carry  = (u128)(ai)*(u128)(b[1]) + (T1) + (carry>>64); T1 = (uint64_t)carry;  \
    carry  = (u128)(ai)*(u128)(b[2]) + (T2) + (carry>>64); T2 = (uint64_t)carry;  \
    carry  = (u128)(ai)*(u128)(b[3]) + (T3) + (carry>>64); T3 = (uint64_t)carry;  \
    T4 += (uint64_t)(carry>>64);                                                    \
    uint64_t mm;                                                                    \
    asm volatile("mul.lo.u64 %0,%1,%2;":"=l"(mm):"l"(T0),"l"(BN254_FR_PRIME));     \
    carry  = (u128)(mm)*(u128)(BN254_R_SCALAR[0]) + (T0); T0 = (uint64_t)carry;   \
    carry  = (u128)(mm)*(u128)(BN254_R_SCALAR[1]) + (T1) + (carry>>64); T1 = (uint64_t)carry; \
    carry  = (u128)(mm)*(u128)(BN254_R_SCALAR[2]) + (T2) + (carry>>64); T2 = (uint64_t)carry; \
    carry  = (u128)(mm)*(u128)(BN254_R_SCALAR[3]) + (T3) + (carry>>64); T3 = (uint64_t)carry; \
    T4 += (uint64_t)(carry>>64);                                                    \
    T0=T1; T1=T2; T2=T3; T3=T4; T4=0;                                              \
}

__device__ __forceinline__ void frMul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t t0=0,t1=0,t2=0,t3=0,t4=0;
    CIOS_Fr(t0,t1,t2,t3,t4, a[0])
    CIOS_Fr(t0,t1,t2,t3,t4, a[1])
    CIOS_Fr(t0,t1,t2,t3,t4, a[2])
    CIOS_Fr(t0,t1,t2,t3,t4, a[3])
    uint64_t s0,s1,s2,s3,bor;
    asm volatile(
        "sub.cc.u64  %0,%5,%9;\n\t" "subc.cc.u64 %1,%6,%10;\n\t"
        "subc.cc.u64 %2,%7,%11;\n\t" "subc.cc.u64 %3,%8,%12;\n\t"
        "subc.u64    %4,0,0;\n\t"
        :"=l"(s0),"=l"(s1),"=l"(s2),"=l"(s3),"=l"(bor)
        :"l"(t0),"l"(t1),"l"(t2),"l"(t3),
         "l"(BN254_R_SCALAR[0]),"l"(BN254_R_SCALAR[1]),
         "l"(BN254_R_SCALAR[2]),"l"(BN254_R_SCALAR[3]));
    uint64_t keep = (bor & 1ULL) & (t4 == 0);
    uint64_t mask = 0ULL - keep;
    r[0]=(t0&mask)|(s0&~mask); r[1]=(t1&mask)|(s1&~mask);
    r[2]=(t2&mask)|(s2&~mask); r[3]=(t3&mask)|(s3&~mask);
}

__device__ __forceinline__ void frSqr(const uint64_t a[4], uint64_t r[4]) { frMul(a,a,r); }

// frAdd / frSub / frNeg
__device__ __forceinline__ void frAdd(const uint64_t a[4], const uint64_t b[4], uint64_t res[4]) {
    uint64_t s0,s1,s2,s3;
    uint32_t bor;
    // 5 outputs (%0--%4), inputs: a=%5..%8, b=%9..%12, r=%13..%16
    asm volatile(
        "add.cc.u64   %0,%5,%9;\n\t"
        "addc.cc.u64  %1,%6,%10;\n\t"
        "addc.cc.u64  %2,%7,%11;\n\t"
        "addc.cc.u64  %3,%8,%12;\n\t"
        "sub.cc.u64   %0,%0,%13;\n\t"
        "subc.cc.u64  %1,%1,%14;\n\t"
        "subc.cc.u64  %2,%2,%15;\n\t"
        "subc.cc.u64  %3,%3,%16;\n\t"
        "subc.u32     %4,0,0;\n\t"
        :"=l"(s0),"=l"(s1),"=l"(s2),"=l"(s3),"=r"(bor)
        :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
         "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]),
         "l"(BN254_R_SCALAR[0]),"l"(BN254_R_SCALAR[1]),
         "l"(BN254_R_SCALAR[2]),"l"(BN254_R_SCALAR[3]));
    // 4 outputs (%0--%3), inputs: s=%4..%7, r&mask=%8..%11
    uint64_t mask = 0ULL - (uint64_t)(bor & 1u);
    asm volatile(
        "add.cc.u64   %0,%4,%8;\n\t"
        "addc.cc.u64  %1,%5,%9;\n\t"
        "addc.cc.u64  %2,%6,%10;\n\t"
        "addc.cc.u64  %3,%7,%11;\n\t"
        :"=l"(res[0]),"=l"(res[1]),"=l"(res[2]),"=l"(res[3])
        :"l"(s0),"l"(s1),"l"(s2),"l"(s3),
         "l"(BN254_R_SCALAR[0]&mask),"l"(BN254_R_SCALAR[1]&mask),
         "l"(BN254_R_SCALAR[2]&mask),"l"(BN254_R_SCALAR[3]&mask));
}

__device__ __forceinline__ void frSub(const uint64_t a[4], const uint64_t b[4], uint64_t res[4]) {
    uint64_t r0,r1,r2,r3;
    uint32_t bor;
    // 5 outputs (%0--%4), inputs: a=%5..%8, b=%9..%12
    asm volatile(
        "sub.cc.u64   %0,%5,%9;\n\t"
        "subc.cc.u64  %1,%6,%10;\n\t"
        "subc.cc.u64  %2,%7,%11;\n\t"
        "subc.cc.u64  %3,%8,%12;\n\t"
        "subc.u32     %4,0,0;\n\t"
        :"=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3),"=r"(bor)
        :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
         "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    // 4 outputs (%0--%3), inputs: r=%4..%7, r&mask=%8..%11
    uint64_t mask = 0ULL - (uint64_t)(bor & 1u);
    asm volatile(
        "add.cc.u64   %0,%4,%8;\n\t"
        "addc.cc.u64  %1,%5,%9;\n\t"
        "addc.cc.u64  %2,%6,%10;\n\t"
        "addc.cc.u64  %3,%7,%11;\n\t"
        :"=l"(res[0]),"=l"(res[1]),"=l"(res[2]),"=l"(res[3])
        :"l"(r0),"l"(r1),"l"(r2),"l"(r3),
         "l"(BN254_R_SCALAR[0]&mask),"l"(BN254_R_SCALAR[1]&mask),
         "l"(BN254_R_SCALAR[2]&mask),"l"(BN254_R_SCALAR[3]&mask));
}

__device__ __forceinline__ void frNeg(const uint64_t a[4], uint64_t res[4]) {
    if ((a[0]|a[1]|a[2]|a[3])==0ULL){res[0]=res[1]=res[2]=res[3]=0;return;}
    USUBO(res[0],BN254_R_SCALAR[0],a[0]); USUBC(res[1],BN254_R_SCALAR[1],a[1]);
    USUBC(res[2],BN254_R_SCALAR[2],a[2]); USUB (res[3],BN254_R_SCALAR[3],a[3]);
}

// Montgomery encode/decode
__device__ __forceinline__ void frMontEncode(const uint64_t a[4], uint64_t r[4]) { frMul(a,BN254_R2_FR,r); }
__device__ __forceinline__ void frMontDecode(const uint64_t a[4], uint64_t r[4]) {
    uint64_t one[4]={1ULL,0ULL,0ULL,0ULL}; frMul(a,one,r);
}

// frInvNormal / frInv (same split as fpInvNormal / fpInv)
__device__ __forceinline__ void frInvNormal(const uint64_t a[4], uint64_t res[4]) {
    if ((a[0]|a[1]|a[2]|a[3])==0ULL){res[0]=res[1]=res[2]=res[3]=0;return;}
    uint64_t t[5]={a[0],a[1],a[2],a[3],0};
    frModInv5(t);
    res[0]=t[0]; res[1]=t[1]; res[2]=t[2]; res[3]=t[3];
}
__device__ __forceinline__ void frInv(const uint64_t a[4], uint64_t res[4]) {
    if ((a[0]|a[1]|a[2]|a[3])==0ULL){res[0]=res[1]=res[2]=res[3]=0;return;}
    uint64_t tmp[4]; frMontDecode(a,tmp); frInvNormal(tmp,tmp); frMontEncode(tmp,res);
}

// ============================================================================
// 5. NTT helpers
// ============================================================================

// CT butterfly : a,b in-place
//   a' = a + b*omega
//   b' = a - b*omega
__device__ __forceinline__ void frNTT_CT(uint64_t a[4], uint64_t b[4], const uint64_t omega[4]) {
    uint64_t t[4];
    frMul(b, omega, t);
    frSub(a, t, b);
    frAdd(a, t, a);
}

// frPow : base^exp mod r (inputs/outputs in Montgomery domain)
__device__ __forceinline__ void frPow(const uint64_t base[4], uint64_t exp, uint64_t res[4]) {
    uint64_t acc[4]={BN254_R_FR[0],BN254_R_FR[1],BN254_R_FR[2],BN254_R_FR[3]};
    uint64_t b[4]  ={base[0],base[1],base[2],base[3]};
    while(exp>0){
        if(exp&1) frMul(acc,b,acc);
        frMul(b,b,b);
        exp>>=1;
    }
    res[0]=acc[0]; res[1]=acc[1]; res[2]=acc[2]; res[3]=acc[3];
}

// Utility
__device__ __forceinline__ bool frIsZero(const uint64_t a[4]){ return (a[0]|a[1]|a[2]|a[3])==0ULL; }
__device__ __forceinline__ void frCopy(const uint64_t src[4],uint64_t dst[4]){
    dst[0]=src[0];dst[1]=src[1];dst[2]=src[2];dst[3]=src[3];
}
