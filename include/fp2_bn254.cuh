/*
 * ===========================================================================
 * Forum — fp2_bn254.cuh
 * BN254 Fp2 extension field arithmetic — CUDA sm_120 (Blackwell RTX 5060)
 * ===========================================================================
 *
 *  Fp2 = Fp[u] / (u^2 + 1)   (the standard BN254 tower, u^2 = -1)
 *
 *  Element a ∈ Fp2 : a = a0 + a1*u
 *    a0, a1 ∈ Fp   (Montgomery representation, 4 × uint64_t each)
 *    stored as: uint64_t a[8] = {a0[0..3], a1[0..3]}
 *
 *  G2 generator (Montgomery, verified against snarkjs/ffjavascript):
 *    Gx = (BN254_G2X0, BN254_G2X1)   (a0 = BN254_G2X0, a1 = BN254_G2X1)
 *    Gy = (BN254_G2Y0, BN254_G2Y1)
 *
 *  Include order:
 *    #include "fp_bn254.cuh"    <- Fp arithmetic + PTX macros
 *    #include "fp2_bn254.cuh"   <- this file
 *
 *  Naming conventions (match fp_bn254.cuh):
 *    fp2Add, fp2Sub, fp2Mul, fp2Sqr, fp2Neg, fp2Inv   — field ops
 *    Fp2Point, Fp2PointJac, Fp2PointAff               — G2 structs
 *    point2JacSetInfinity, point2JacAdd, point2JacDouble,
 *    point2JacToAff, point2AffToJac                   — G2 ops
 *
 * ===========================================================================
 */

#pragma once
#include <cstdint>
#include <cuda_runtime.h>
// fp_bn254.cuh MUST be included before this file (provides fpAdd/fpSub/fpMul etc.)

// ============================================================================
// 1. G2 GENERATOR CONSTANTS (Montgomery Fp, verified by Python / snarkjs)
// ============================================================================

// G2.x = x0 + x1*u  (x0 = real part, x1 = imaginary part)
__constant__ uint64_t BN254_G2X0[4] = {
    0x8e83b5d102bc2026ULL, 0xdceb1935497b0172ULL,
    0xfbb8264797811adaULL, 0x19573841af96503bULL
};
__constant__ uint64_t BN254_G2X1[4] = {
    0xafb4737da84c6140ULL, 0x6043dd5a5802d8c4ULL,
    0x09e950fc52a02f86ULL, 0x14fef0833aea7b6bULL
};

// G2.y = y0 + y1*u
__constant__ uint64_t BN254_G2Y0[4] = {
    0x619dfa9d886be9f6ULL, 0xfe7fd297f59e9b78ULL,
    0xff9e1a62231b7dfeULL, 0x28fd7eebae9e4206ULL
};
__constant__ uint64_t BN254_G2Y1[4] = {
    0x64095b56c71856eeULL, 0xdc57f922327d3cbbULL,
    0x55f935be33351076ULL, 0x0da4a0e693fd6482ULL
};

// ============================================================================
// 2. Fp2 ELEMENT ACCESSORS
// ============================================================================

// An Fp2 element is 8 × uint64_t: a[0..3] = a0 (real), a[4..7] = a1 (imag)
// These helpers make kernel code readable.

__device__ __forceinline__       uint64_t* fp2Re(uint64_t a[8])       { return a;     }
__device__ __forceinline__ const uint64_t* fp2Re(const uint64_t a[8]) { return a;     }
__device__ __forceinline__       uint64_t* fp2Im(uint64_t a[8])       { return a + 4; }
__device__ __forceinline__ const uint64_t* fp2Im(const uint64_t a[8]) { return a + 4; }

__device__ __forceinline__ void fp2Copy(const uint64_t a[8], uint64_t r[8]) {
    fpCopy(a,   r);
    fpCopy(a+4, r+4);
}

__device__ __forceinline__ void fp2Zero(uint64_t r[8]) {
    r[0]=r[1]=r[2]=r[3]=r[4]=r[5]=r[6]=r[7]=0ULL;
}

__device__ __forceinline__ bool fp2IsZero(const uint64_t a[8]) {
    return (a[0]|a[1]|a[2]|a[3]|a[4]|a[5]|a[6]|a[7]) == 0ULL;
}

__device__ __forceinline__ bool fp2Eq(const uint64_t a[8], const uint64_t b[8]) {
    return (fpCmp(a, b) == 0) && (fpCmp(a+4, b+4) == 0);
}

// ============================================================================
// 3. Fp2 FIELD ARITHMETIC
//
// Fp2 = Fp[u]/(u^2+1)  =>  mul uses Karatsuba (3 Fp-muls instead of 4)
// All operands and results in Montgomery representation.
// ============================================================================

// fp2Add : (a0+a1*u) + (b0+b1*u) = (a0+b0) + (a1+b1)*u
__device__ __forceinline__ void fp2Add(const uint64_t a[8], const uint64_t b[8], uint64_t r[8]) {
    fpAdd(a,   b,   r);
    fpAdd(a+4, b+4, r+4);
}

// fp2Sub : (a0+a1*u) - (b0+b1*u) = (a0-b0) + (a1-b1)*u
__device__ __forceinline__ void fp2Sub(const uint64_t a[8], const uint64_t b[8], uint64_t r[8]) {
    fpSub(a,   b,   r);
    fpSub(a+4, b+4, r+4);
}

// fp2Neg : -(a0+a1*u) = (-a0) + (-a1)*u
__device__ __forceinline__ void fp2Neg(const uint64_t a[8], uint64_t r[8]) {
    fpNeg(a,   r);
    fpNeg(a+4, r+4);
}

// fp2MulFp : scale Fp2 element by an Fp scalar: (a0+a1*u)*k = (a0*k) + (a1*k)*u
__device__ __forceinline__ void fp2MulFp(const uint64_t a[8], const uint64_t k[4], uint64_t r[8]) {
    fpMul(a,   k, r);
    fpMul(a+4, k, r+4);
}

// fp2Mul : Karatsuba-3 multiplication in Fp[u]/(u^2+1)
//
// (a0 + a1*u)(b0 + b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
//
// Karatsuba:
//   t0 = a0*b0
//   t1 = a1*b1
//   t2 = (a0+a1)*(b0+b1)
//   re = t0 - t1
//   im = t2 - t0 - t1
// → 3 Fp-muls + 5 Fp-adds (vs 4 muls naively)
__device__ __forceinline__ void fp2Mul(const uint64_t a[8], const uint64_t b[8], uint64_t r[8]) {
    uint64_t t0[4], t1[4], t2[4], sa[4], sb[4];
    fpMul(a,   b,   t0);    // t0 = a0*b0
    fpMul(a+4, b+4, t1);    // t1 = a1*b1
    fpAdd(a,   a+4, sa);    // sa = a0+a1
    fpAdd(b,   b+4, sb);    // sb = b0+b1
    fpMul(sa,  sb,  t2);    // t2 = (a0+a1)*(b0+b1)
    // re = t0 - t1
    fpSub(t0, t1, r);
    // im = t2 - t0 - t1
    fpSub(t2, t0, t2);
    fpSub(t2, t1, r+4);
}

// fp2Sqr : (a0+a1*u)^2 = (a0^2 - a1^2) + (2*a0*a1)*u
//
// Optimized:
//   t0 = a0+a1
//   t1 = a0-a1
//   re = t0*t1     (= a0^2 - a1^2)
//   im = 2*a0*a1   (via (a0+a1)^2 - a0^2 - a1^2 trick... but simpler below)
//
// Actually cheapest in Fp2/(u^2+1):
//   re = (a0-a1)*(a0+a1)
//   im = 2*a0*a1
// 2 Fp-muls + 3 Fp-adds
__device__ __forceinline__ void fp2Sqr(const uint64_t a[8], uint64_t r[8]) {
    uint64_t t0[4], t1[4], t2[4];
    fpAdd(a,   a+4, t0);    // t0 = a0+a1
    fpSub(a,   a+4, t1);    // t1 = a0-a1
    fpMul(t0,  t1,  r);     // re = (a0+a1)(a0-a1) = a0^2 - a1^2
    fpMul(a,   a+4, t2);    // t2 = a0*a1
    fpAdd(t2,  t2,  r+4);   // im = 2*a0*a1
}

// fp2Inv : (a0+a1*u)^{-1} in Fp2 = (a0-a1*u)/(a0^2+a1^2)
//   norm  = a0^2 + a1^2  ∈ Fp
//   inv_n = norm^{-1}    ∈ Fp
//   re    = a0 * inv_n
//   im    = (-a1) * inv_n
__device__ __forceinline__ void fp2Inv(const uint64_t a[8], uint64_t r[8]) {
    if (fp2IsZero(a)) { fp2Zero(r); return; }
    uint64_t n0[4], n1[4], norm[4], inv_n[4];
    fpSqr(a,   n0);         // n0 = a0^2
    fpSqr(a+4, n1);         // n1 = a1^2
    fpAdd(n0, n1, norm);    // norm = a0^2 + a1^2
    fpInv(norm, inv_n);     // inv_n = 1/(a0^2+a1^2)
    fpMul(a,   inv_n, r);   // re = a0 * inv_n
    fpNeg(a+4, n0);         // n0 = -a1
    fpMul(n0,  inv_n, r+4); // im = -a1 * inv_n
}

// ============================================================================
// 4. G2 POINT STRUCTS
// ============================================================================

// Fp2Point (affine) : X, Y ∈ Fp2  (8 limbs each)
struct Fp2Point {
    uint64_t X[8];   // X = X0 + X1*u
    uint64_t Y[8];   // Y = Y0 + Y1*u
    bool infinity;
};
typedef Fp2Point Fp2PointAff;

// Fp2PointJac (Jacobi) : (X:Y:Z) with Z ∈ Fp2
// Affine = (X/Z^2, Y/Z^3)
struct Fp2PointJac {
    uint64_t X[8];
    uint64_t Y[8];
    uint64_t Z[8];
    bool infinity;
};

// ============================================================================
// 5. G2 POINT OPERATIONS
// ============================================================================

__device__ __forceinline__ void point2JacSetInfinity(Fp2PointJac &P) {
    P.infinity = true;
    fp2Zero(P.X); fp2Zero(P.Y); fp2Zero(P.Z);
}

// point2AffToJac : lift affine G2 to Jacobi (Z = 1 in Montgomery = R mod p)
__device__ __forceinline__ void point2AffToJac(const Fp2PointAff &P, Fp2PointJac &Q) {
    if (P.infinity) { point2JacSetInfinity(Q); return; }
    fp2Copy(P.X, Q.X);
    fp2Copy(P.Y, Q.Y);
    // Z = (1, 0) in Fp2, with 1 = R_MOD in Montgomery
    fpCopy(BN254_R_MOD, Q.Z);
    Q.Z[4]=Q.Z[5]=Q.Z[6]=Q.Z[7]=0ULL;
    Q.infinity = false;
}

// point2JacToAff : convert Jacobi → affine  (requires fp2Inv, costs 1 Fp-inv)
__device__ void point2JacToAff(const Fp2PointJac &P, Fp2PointAff &Q) {
    if (P.infinity) { Q.infinity = true; fp2Zero(Q.X); fp2Zero(Q.Y); return; }
    uint64_t Zinv[8], Zinv2[8], Zinv3[8];
    fp2Inv(P.Z,    Zinv);        // Zinv  = Z^{-1}
    fp2Mul(Zinv,   Zinv, Zinv2); // Zinv2 = Z^{-2}
    fp2Mul(Zinv2,  Zinv, Zinv3); // Zinv3 = Z^{-3}
    fp2Mul(P.X, Zinv2, Q.X);    // x = X * Z^{-2}
    fp2Mul(P.Y, Zinv3, Q.Y);    // y = Y * Z^{-3}
    Q.infinity = false;
}

// ── point2JacDouble : Jacobi doubling for G2 ─────────────────────────────────
// Full Jacobi doubling: a=0 (BN254 G2 curve: y^2 = x^3 + b/xi, a=0)
// Formula (cost: 3S + 6M + 5add in Fp2, but entirely in fp2):
//
//   W  = 3*X1^2         (since a=0)
//   S  = Y1*Z1
//   B  = X1*Y1*S
//   H  = W^2 - 8*B
//   X3 = 2*H*S
//   Y3 = W*(4*B - H) - 8*Y1^2*S^2
//   Z3 = 8*S^3
//
// (using Jacobi coordinates where P = (X:Y:Z) means affine (X/Z^2, Y/Z^3))
__device__ void point2JacDouble(const Fp2PointJac &P, Fp2PointJac &R) {
    if (P.infinity) { point2JacSetInfinity(R); return; }

    uint64_t A[8], B[8], C[8], D[8], E[8], F[8], tmp[8];

    // A = X1^2
    fp2Sqr(P.X, A);
    // B = Y1^2
    fp2Sqr(P.Y, B);
    // C = B^2  (= Y1^4)
    fp2Sqr(B, C);
    // D = 2*((X1+B)^2 - A - C)  = 4*X1*Y1^2
    fp2Add(P.X, B, tmp);
    fp2Sqr(tmp, D);
    fp2Sub(D, A, D);
    fp2Sub(D, C, D);
    fp2Add(D, D, D);
    // E = 3*A  (since curve a=0)
    fp2Add(A, A, E);
    fp2Add(E, A, E);
    // F = E^2
    fp2Sqr(E, F);
    // X3 = F - 2*D
    fp2Add(D, D, tmp);
    fp2Sub(F, tmp, R.X);
    // Y3 = E*(D - X3) - 8*C
    fp2Sub(D, R.X, tmp);
    fp2Mul(E, tmp, R.Y);
    // 8*C
    fp2Add(C, C, tmp); fp2Add(tmp, tmp, tmp); fp2Add(tmp, tmp, tmp);
    fp2Sub(R.Y, tmp, R.Y);
    // Z3 = 2*Y1*Z1
    fp2Mul(P.Y, P.Z, R.Z);
    fp2Add(R.Z, R.Z, R.Z);
    R.infinity = false;
}

// ── point2JacAdd : Jacobi addition for G2 ────────────────────────────────────
// Standard Jacobi mixed/full add (handles infinity correctly).
// Cost: 11M + 5S + 9add in Fp2.
//
// Algorithm (Cohen-Miyaji-Ono 1998):
//   U1=X1*Z2^2, U2=X2*Z1^2
//   S1=Y1*Z2^3, S2=Y2*Z1^3
//   H=U2-U1, R=S2-S1
//   X3 = R^2 - H^3 - 2*U1*H^2
//   Y3 = R*(U1*H^2 - X3) - S1*H^3
//   Z3 = H*Z1*Z2
__device__ void point2JacAdd(const Fp2PointJac &P, const Fp2PointJac &Q, Fp2PointJac &R) {
    if (P.infinity) { R = Q; return; }
    if (Q.infinity) { R = P; return; }

    uint64_t Z1sq[8], Z2sq[8], U1[8], U2[8];
    uint64_t Z1cu[8], Z2cu[8], S1[8], S2[8];
    uint64_t H[8], Hsq[8], Hcu[8], Rr[8];
    uint64_t tmp[8], tmp2[8];

    fp2Sqr(P.Z, Z1sq);          // Z1^2
    fp2Sqr(Q.Z, Z2sq);          // Z2^2
    fp2Mul(P.X, Z2sq, U1);      // U1 = X1*Z2^2
    fp2Mul(Q.X, Z1sq, U2);      // U2 = X2*Z1^2
    fp2Mul(P.Z, Z1sq, Z1cu);    // Z1^3
    fp2Mul(Q.Z, Z2sq, Z2cu);    // Z2^3
    fp2Mul(P.Y, Z2cu, S1);      // S1 = Y1*Z2^3
    fp2Mul(Q.Y, Z1cu, S2);      // S2 = Y2*Z1^3

    fp2Sub(U2, U1, H);           // H = U2-U1
    fp2Sub(S2, S1, Rr);          // Rr = S2-S1  (called R in the formula, renamed to avoid collision)

    // Check for special cases: H=0 means X coords equal
    if (fp2IsZero(H)) {
        if (fp2IsZero(Rr)) {
            // P == Q → double
            point2JacDouble(P, R);
        } else {
            // P == -Q → infinity
            point2JacSetInfinity(R);
        }
        return;
    }

    fp2Sqr(H, Hsq);              // H^2
    fp2Mul(H, Hsq, Hcu);         // H^3

    // X3 = Rr^2 - Hcu - 2*U1*Hsq
    fp2Sqr(Rr, R.X);
    fp2Sub(R.X, Hcu, R.X);
    fp2Mul(U1, Hsq, tmp);
    fp2Add(tmp, tmp, tmp2);
    fp2Sub(R.X, tmp2, R.X);

    // Y3 = Rr*(U1*Hsq - X3) - S1*Hcu
    fp2Mul(U1, Hsq, tmp);
    fp2Sub(tmp, R.X, tmp);
    fp2Mul(Rr, tmp, R.Y);
    fp2Mul(S1, Hcu, tmp);
    fp2Sub(R.Y, tmp, R.Y);

    // Z3 = H*Z1*Z2
    fp2Mul(H, P.Z, R.Z);
    fp2Mul(R.Z, Q.Z, R.Z);

    R.infinity = false;
}

// ── point2JacAddMixed : Jacobi + Affine (Z2 = 1) ────────────────────────────
// Same as point2JacAdd but Q.Z = (1,0) → saves ~3 fp2Mul
// Used in MSM scatter where base points are affine.
__device__ void point2JacAddMixed(const Fp2PointJac &P, const Fp2PointAff &Q, Fp2PointJac &R) {
    if (P.infinity) { point2AffToJac(Q, R); return; }
    if (Q.infinity) { R = P; return; }

    uint64_t Z1sq[8], U2[8], S2[8];
    uint64_t H[8], Hsq[8], Hcu[8], Rr[8];
    uint64_t tmp[8], tmp2[8];

    fp2Sqr(P.Z, Z1sq);          // Z1^2
    // U1 = X1 (Z2=1 so Z2^2=1)
    // U2 = X2*Z1^2
    fp2Mul(Q.X, Z1sq, U2);
    // S1 = Y1 (Z2=1 so Z2^3=1)
    // S2 = Y2*Z1^3
    fp2Mul(Q.Y, Z1sq, S2);
    fp2Mul(S2, P.Z, S2);        // S2 = Y2*Z1^2*Z1 = Y2*Z1^3

    fp2Sub(U2,   P.X,  H);      // H = U2-U1 = X2*Z1^2 - X1
    fp2Sub(S2,   P.Y,  Rr);     // Rr = S2-S1 = Y2*Z1^3 - Y1

    if (fp2IsZero(H)) {
        if (fp2IsZero(Rr)) { point2JacDouble(P, R); }
        else               { point2JacSetInfinity(R); }
        return;
    }

    fp2Sqr(H, Hsq);
    fp2Mul(H, Hsq, Hcu);

    // X3 = Rr^2 - Hcu - 2*X1*Hsq
    fp2Sqr(Rr, R.X);
    fp2Sub(R.X, Hcu, R.X);
    fp2Mul(P.X, Hsq, tmp);
    fp2Add(tmp, tmp, tmp2);
    fp2Sub(R.X, tmp2, R.X);

    // Y3 = Rr*(X1*Hsq - X3) - Y1*Hcu
    fp2Mul(P.X, Hsq, tmp);
    fp2Sub(tmp, R.X, tmp);
    fp2Mul(Rr, tmp, R.Y);
    fp2Mul(P.Y, Hcu, tmp);
    fp2Sub(R.Y, tmp, R.Y);

    // Z3 = H*Z1  (Z2=1)
    fp2Mul(H, P.Z, R.Z);

    R.infinity = false;
}

// ============================================================================
// 6. G2 GENERATOR + UTILITY
// ============================================================================

// Load G2 generator into an affine G2 point (from __constant__ memory)
__device__ __forceinline__ void point2SetG2(Fp2PointAff &P) {
    P.infinity = false;
    fpCopy(BN254_G2X0, P.X);
    fpCopy(BN254_G2X1, P.X + 4);
    fpCopy(BN254_G2Y0, P.Y);
    fpCopy(BN254_G2Y1, P.Y + 4);
}

// Negate a G2 affine point (negate Y)
__device__ __forceinline__ void point2AffNeg(const Fp2PointAff &P, Fp2PointAff &R) {
    fp2Copy(P.X, R.X);
    fp2Neg(P.Y, R.Y);
    R.infinity = P.infinity;
}

// Negate Fp2 Y for Jacobi point (for signed-digit MSM)
__device__ __forceinline__ void point2JacNegY(Fp2PointJac &P) {
    fp2Neg(P.Y, P.Y);
}
