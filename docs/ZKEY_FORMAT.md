# snarkjs zkey File Format

This document describes the binary format of `.zkey` files produced by snarkjs
for BN254 Groth16 circuits. It was reverse-engineered by reading snarkjs source
code and validating each field against live circuit data.

The format is a specialization of the snarkjs BinFile container format, which
stores typed sections identified by integer IDs.

---

## Container Format

Every snarkjs binary file (`.zkey`, `.wtns`, `.r1cs`) uses the same outer container:

```
[magic: 4 bytes]           "zkey" in ASCII = 0x7a 0x6b 0x65 0x79
[version: uint32 LE]       = 2
[n_sections: uint32 LE]    number of sections that follow

For each section:
  [type: uint32 LE]        section identifier
  [size: uint64 LE]        byte length of section data
  [data: size bytes]       section payload
```

Sections can appear in any order. Implementations must seek by section ID.

---

## Section Overview

| ID | Name | Contents |
|----|------|----------|
| 1  | Header | protocol type identifier |
| 2  | Groth16 header | curve, domain, dimensions, VK points |
| 3  | IC points | verification key IC array (G1) |
| 4  | QAP coefficients | sparse A and B matrices |
| 5  | A points | CRS tau*A_i(tau) / delta (G1) |
| 6  | B1 points | CRS tau*B_i(tau) / delta (G1) |
| 7  | B2 points | CRS tau*B2_i(tau) / delta (G2) |
| 8  | C points | CRS tau*C_i(tau) / delta (G1) |
| 9  | H points | CRS tau^{2i+1} / delta (G1) |
| 10 | Contributions | ceremony contribution hashes |

---

## Section 1 — Protocol Header

```
[protocol_type: uint32 LE]    = 1 for Groth16
```

---

## Section 2 — Groth16 Header

```
[n8q: uint32 LE]              byte width of Fp elements = 32 (for BN254)
[q: n8q bytes LE]             Fp modulus p
[n8r: uint32 LE]              byte width of Fr elements = 32 (for BN254)
[r: n8r bytes LE]             Fr modulus r
[nVars: uint32 LE]            total wire count (public + private + constant)
[nPublic: uint32 LE]          public wire count (excluding wire 0 = constant 1)
[domainSize: uint32 LE]       n = smallest power of 2 >= n_constraints
[alpha1: 64 bytes]            vk_alpha1, G1 point, Montgomery Fp
[beta1: 64 bytes]             vk_beta1,  G1 point, Montgomery Fp
[delta1: 64 bytes]            vk_delta1, G1 point, Montgomery Fp
[beta2: 128 bytes]            vk_beta2,  G2 point, Montgomery Fp
[gamma2: 128 bytes]           vk_gamma2, G2 point, Montgomery Fp
[delta2: 128 bytes]           vk_delta2, G2 point, Montgomery Fp
```

**Point encoding:** G1 affine points are stored as `[x: 32 bytes LE][y: 32 bytes LE]`
where x and y are Fp values in **Montgomery form** (`x * R mod p`).

G2 affine points are stored as `[x.a0: 32 bytes][x.a1: 32 bytes][y.a0: 32 bytes][y.a1: 32 bytes]`
in the same Montgomery Fp encoding. The Fp2 element is `x.a0 + x.a1 * u`.

**JSON output:** snarkjs serializes G2 coordinates in the order `[x.a0, x.a1]`
(degree-0 coefficient first). Note: some early documentation incorrectly states
`[x.a1, x.a0]`.

---

## Section 3 — IC Points (Verification Key)

```
[(nPublic + 1) G1 points, each 64 bytes]
```

IC[i] = `tau * L_i(tau) / gamma * G1` for i = 0..nPublic.
Used by the verifier to compute `vk_x = sum_i public_i * IC[i]`.

---

## Section 4 — QAP Coefficients

This section encodes the sparse A and B QAP matrices.

```
[nCoeffs: uint32 LE]          total number of non-zero entries (A + B combined)

For each entry:
  [matrix: uint8]             0 = matrix A, 1 = matrix B
  [constraint: uint32 LE]     row index (0..n_constraints-1)
  [signal: uint32 LE]         column index / wire index (0..nVars-1)
  [value: 32 bytes LE]        Fr element (see encoding note below)
```

### CRITICAL: Double-Montgomery Encoding

The coefficient values in section 4 are stored as `coef * R^2 mod r`,
**not** as canonical Fr values and **not** as single-Montgomery (`coef * R`).

This is undocumented in snarkjs source. It arises because snarkjs processes
coefficients through its internal `frm_mul` (Montgomery multiply) during setup,
which accumulates two Montgomery factors.

To recover the canonical value:

```
coef_canonical = raw_value * R^{-2} mod r
```

where `R = 2^256` and the constant for BN254 Fr is:

```
R^{-2} mod r = 0x12d5f775e436631e_e065f3e379a1edeb_52f28270b38e2428_ae12ba81d3c71148
               (256-bit value, little-endian limbs)
```

Any implementation that treats section 4 values as canonical or single-Montgomery
will compute incorrect `A*w` and `B*w` evaluations, producing a wrong h polynomial
and therefore a wrong pi_C.

---

## Section 5 — A Points

```
[nVars G1 points, each 64 bytes]
```

`A[i] = tau * A_i(tau) / delta * G1`

Used in the MSM for pi_A: `pi_A = alpha1 + sum_i w[i] * A[i]`

Note: alpha1 is NOT included in section 5. It must be added explicitly from
the vk_alpha1 field in section 2.

---

## Section 6 — B1 Points (G1 version of B)

```
[nVars G1 points, each 64 bytes]
```

`B1[i] = tau * B_i(tau) / delta * G1`

Used for pi_C in some formulations. In the standard snarkjs implementation,
pi_C uses section 8 (C points) and section 9 (H points) instead.

---

## Section 7 — B2 Points

```
[nVars G2 points, each 128 bytes]
```

`B2[i] = tau * B_i(tau) / delta * G2`

Used in the MSM for pi_B: `pi_B = beta2 + sum_i w[i] * B2[i]`

Note: beta2 is NOT included in section 7. It must be added explicitly from
the vk_beta2 field in section 2.

---

## Section 8 — C Points

```
[(nVars - nPublic - 1) G1 points, each 64 bytes]
```

`C[i] = tau * L_{nPublic+1+i}(tau) / delta * G1`

Used in the MSM for pi_C (private wire contribution):
`MSM_C = sum_i w[nPublic+1+i] * C[i]`

The first nPublic wires (public inputs) and wire 0 (the constant) are excluded,
hence the count is `nVars - nPublic - 1`.

---

## Section 9 — H Points

```
[domainSize G1 points, each 64 bytes]
```

`H[i] = tau^{2i+1} / delta * G1`   for i = 0..domainSize-1

**This is the odd-powers-of-tau representation**, not the standard `tau^i * Z(tau) / delta`.

The MSM scalars for H are the evaluations of `(A*B - C)(omega_{2n}^{2j+1})` at the n
odd points of the 2n-th roots of unity, without dividing by Z:

```
h_odd[j] = A(omega_{2n}^{2j+1}) * B(omega_{2n}^{2j+1}) - C(omega_{2n}^{2j+1})
```

where `omega_{2n} = 5^{(r-1)/(2n)} mod r` is the primitive 2n-th root of unity.

Note that `Z(omega_{2n}^{2j+1}) = -2` for all j on BN254, but snarkjs does NOT
divide by Z. The algebraic relationship between the odd-power H points and these
scalars ensures the pairing equation holds without an explicit Z division.

The MSM for the H contribution to pi_C is:

```
MSM_H = sum_{j=0}^{n-1} h_odd[j] * H[j]
```

The final pi_C is `MSM_C + MSM_H`.

### Computing h_odd

The odd-coset evaluation pipeline (compatible with snarkjs):

```
1. Aw[i] = sum_j coefA[constraint=i][signal=j] * w[j]   (sparse MVM, canonical Fr)
2. Bw[i] = sum_j coefB[constraint=i][signal=j] * w[j]   (sparse MVM, canonical Fr)
3. Encode Aw, Bw to Montgomery Fr
4. Cw[i] = Aw[i] * Bw[i]                                (pointwise, Montgomery)
5. IFFT_n(Aw), IFFT_n(Bw), IFFT_n(Cw)                   -> polynomial coefficients
6. coef[i] *= omega_{2n}^i                               (coset shift, Montgomery)
7. FFT_n(Aw), FFT_n(Bw), FFT_n(Cw)                      -> odd-coset evaluations
8. h_odd[j] = Aw[j] * Bw[j] - Cw[j]                    (no Z division)
9. Decode h_odd from Montgomery to canonical Fr
```

---

## Section 10 — Contributions

Records the ceremony contribution hashes. Not required for proving.

---

## Witness File Format (.wtns)

For completeness, the witness format used by `.wtns` files:

```
magic: "wtns" (4 bytes)
version: uint32 LE = 2
n_sections: uint32 LE

Section 1 (header, size=40):
  n8: uint32 LE = 32           (byte width of Fr elements)
  prime[32]: LE                (Fr modulus r)
  n_witness: uint32 LE         (total number of witness values)

Section 2 (values, size = n_witness * 32):
  w[0]: 32 bytes LE            = 1 (constant wire, canonical Fr)
  w[1]: 32 bytes LE            = public input 1
  ...
  w[nPublic]: 32 bytes LE      = public input nPublic
  w[nPublic+1]: 32 bytes LE    = first private wire
  ...
  w[n_witness-1]: 32 bytes LE  = last private wire
```

All values are **canonical Fr** (not Montgomery-encoded).

Section 2 has no count prefix. The count is read from section 1.

---

## BN254 Constants

```
Fp modulus p:
  0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47

Fr modulus r:
  0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001

Montgomery constant R = 2^256 mod p (Fp):
  0x0e0a77c19a07df2f666ea36f7879462e36fc76959f60cd29ac96341c4ffffffb

Montgomery constant R = 2^256 mod r (Fr):
  0x0e0a77c19a07df2f666ea36f7879462e36fc76959f60cd29ac96341c4ffffffb

R^{-2} mod r (used to decode section 4 coefficients):
  0x12d5f775e436631ee065f3e379a1edeb52f28270b38e2428ae12ba81d3c71148
  (256-bit LE: limb0=0xae12ba81d3c71148, limb1=0x52f28270b38e2428,
               limb2=0xe065f3e379a1edeb, limb3=0x12d5f775e436631e)

Generator for coset shift (omega_{2n}):
  omega_{2n} = 5^{(r-1)/(2n)} mod r
  (NOT g=7, which is the multiplicative generator for the field)
```

---

## Reference Implementation

The file parsers in this repository (`src/binfile_utils.cpp`, `src/zkey_utils.cpp`,
`src/wtns_utils.cpp`) are adapted from iden3/rapidsnark and implement this format.
The GPU prover (`src/groth16.cu`) documents the double-Montgomery decoding and
the odd-coset h polynomial computation in detail.

See also [BUGS.md](BUGS.md) for a full list of implementation pitfalls discovered
during development, especially Bug 14 (double-Montgomery) and Bugs 11-12
(h polynomial coset convention).
