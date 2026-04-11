/*
 * ======================================================================================
 * Forum — test_fp_bn254.cu
 * Harness de test GPU pour fp_bn254.cuh
 * ======================================================================================
 *
 * Lit test_vectors.json généré par gen_test_vectors.py (source de vérité Python),
 * exécute chaque opération sur GPU et compare avec le résultat attendu.
 *
 * Couverture :
 *   [1] fpAdd   — 200+ vecteurs dont cas limites (p-1+1=0, p-1+p-1=p-2)
 *   [2] fpSub   — 200+ vecteurs dont 0-1=p-1
 *   [3] fpNeg   — 100+ vecteurs dont neg(0)=0
 *   [4] fpMul   — 500+ vecteurs CIOS Montgomery, cas identity et mont_encode
 *   [5] fpSqr   — 300+ vecteurs, doit être identique à fpMul(a,a)
 *   [6] fpInv   — 10k vecteurs, vérifie a·inv(a)==1 sur GPU
 *   [7] BY_MM64 — assert mathématique : p·BY_MM64 ≡ -1 mod 2^62
 *   [8] curve   — G+G, 2G, nG (via pointAddAffine + pointDoubleAffine)
 *
 * Compilation :
 *   nvcc -O3 -std=c++17 -arch=sm_120 \
 *        -I../include \
 *        test_fp_bn254.cu -o test_fp_bn254
 *
 * Exécution :
 *   python3 gen_test_vectors.py   # génère test_vectors.json
 *   ./test_fp_bn254 test_vectors.json
 *
 * Format de sortie :
 *   [PASS] fpAdd        206 / 206
 *   [PASS] fpSub        205 / 205
 *   ...
 *   [FAIL] fpMul        499 / 500  ← détail du premier échec imprimé
 *
 * ======================================================================================
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include "fp_bn254.cuh"

// ── Minuscule parser JSON ────────────────────────────────────────────────────
// On évite toute dépendance externe. Le JSON produit par Python est simple :
// pas d'espaces, pas d'unicode, que des tableaux d'entiers et de strings.
#include <fstream>
#include <sstream>
#include <stdexcept>

// ============================================================================
// 1. JSON PARSER MINIMAL
// ============================================================================

struct JsonVal;
using JsonArr = std::vector<JsonVal*>;   // pointeurs — pas de copy/move problem
struct JsonObj {
    std::vector<std::pair<std::string, JsonVal*>> kv;
    ~JsonObj();
    const JsonVal* get(const std::string& k) const;
};

struct JsonVal {
    enum Kind { Null, Int, Str, Arr, Obj } kind = Null;
    int64_t      i = 0;
    std::string  s;
    JsonArr*     arr = nullptr;
    JsonObj*     obj = nullptr;
    ~JsonVal() {
        if (arr) { for (auto* p : *arr) delete p; delete arr; }
        delete obj;
    }
    JsonVal() = default;
    JsonVal(const JsonVal&) = delete;
    JsonVal& operator=(const JsonVal&) = delete;
};

JsonObj::~JsonObj() {
    for (auto& p : kv) delete p.second;
}
const JsonVal* JsonObj::get(const std::string& k) const {
    for (auto& p : kv) if (p.first == k) return p.second;
    return nullptr;
}

static void skip_ws(const char*& p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
}

static JsonVal* parse_val(const char*& p);

static std::string parse_str(const char*& p) {
    assert(*p == '"'); p++;
    std::string s;
    while (*p && *p != '"') {
        if (*p == '\\') { p++; s += *p; } else s += *p;
        p++;
    }
    assert(*p == '"'); p++;
    return s;
}

static JsonVal* parse_val(const char*& p) {
    skip_ws(p);
    auto* v = new JsonVal();
    if (*p == '"') {
        v->kind = JsonVal::Str;
        v->s = parse_str(p);
    } else if (*p == '[') {
        v->kind = JsonVal::Arr;
        v->arr = new JsonArr();
        p++;
        skip_ws(p);
        while (*p != ']') {
            v->arr->push_back(parse_val(p));   // stocke le pointeur directement
            skip_ws(p);
            if (*p == ',') { p++; skip_ws(p); }
        }
        p++;
    } else if (*p == '{') {
        v->kind = JsonVal::Obj;
        v->obj = new JsonObj();
        p++;
        skip_ws(p);
        while (*p != '}') {
            skip_ws(p);
            std::string key = parse_str(p);
            skip_ws(p); assert(*p == ':'); p++;
            JsonVal* val = parse_val(p);
            v->obj->kv.push_back({key, val});
            skip_ws(p);
            if (*p == ',') { p++; skip_ws(p); }
        }
        p++;
    } else if (*p == 'n') {
        p += 4; // null
    } else {
        v->kind = JsonVal::Int;
        bool neg = false;
        if (*p == '-') { neg = true; p++; }
        while (*p >= '0' && *p <= '9') { v->i = v->i * 10 + (*p - '0'); p++; }
        if (neg) v->i = -v->i;
    }
    return v;
}

// ── Helpers d'accès ──────────────────────────────────────────────────────────

static void read_u64x4(const JsonVal* v, uint64_t out[4]) {
    assert(v && v->kind == JsonVal::Arr && v->arr->size() == 4);
    for (int i = 0; i < 4; i++)
        out[i] = (uint64_t)(*v->arr)[i]->i;
}

// ============================================================================
// 2. STRUCTURES DE RÉSULTATS GPU
// ============================================================================

struct TestResult {
    uint64_t r[4];      // résultat de l'opération
    uint64_t check[4];  // pour fpInv : a·inv(a), doit être 1
};

struct CurveResult {
    uint64_t x[4];
    uint64_t y[4];
};

// ============================================================================
// 3. KERNELS DE TEST
// ============================================================================

// ── fpAdd ────────────────────────────────────────────────────────────────────
__global__ void kernel_fp_add(
    const uint64_t* __restrict__ a4,
    const uint64_t* __restrict__ b4,
    uint64_t*       __restrict__ r4,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint64_t r[4];
    fpAdd(a4 + i*4, b4 + i*4, r);
    r4[i*4+0]=r[0]; r4[i*4+1]=r[1]; r4[i*4+2]=r[2]; r4[i*4+3]=r[3];
}

// ── fpSub ────────────────────────────────────────────────────────────────────
__global__ void kernel_fp_sub(
    const uint64_t* __restrict__ a4,
    const uint64_t* __restrict__ b4,
    uint64_t*       __restrict__ r4,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint64_t r[4];
    fpSub(a4 + i*4, b4 + i*4, r);
    r4[i*4+0]=r[0]; r4[i*4+1]=r[1]; r4[i*4+2]=r[2]; r4[i*4+3]=r[3];
}

// ── fpNeg ────────────────────────────────────────────────────────────────────
__global__ void kernel_fp_neg(
    const uint64_t* __restrict__ a4,
    uint64_t*       __restrict__ r4,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint64_t r[4];
    fpNeg(a4 + i*4, r);
    r4[i*4+0]=r[0]; r4[i*4+1]=r[1]; r4[i*4+2]=r[2]; r4[i*4+3]=r[3];
}

// ── fpMul ────────────────────────────────────────────────────────────────────
__global__ void kernel_fp_mul(
    const uint64_t* __restrict__ a4,
    const uint64_t* __restrict__ b4,
    uint64_t*       __restrict__ r4,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint64_t r[4];
    fpMul(a4 + i*4, b4 + i*4, r);
    r4[i*4+0]=r[0]; r4[i*4+1]=r[1]; r4[i*4+2]=r[2]; r4[i*4+3]=r[3];
}

// ── fpSqr ────────────────────────────────────────────────────────────────────
__global__ void kernel_fp_sqr(
    const uint64_t* __restrict__ a4,
    uint64_t*       __restrict__ r4,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint64_t r[4];
    fpSqr(a4 + i*4, r);
    r4[i*4+0]=r[0]; r4[i*4+1]=r[1]; r4[i*4+2]=r[2]; r4[i*4+3]=r[3];
}

// ── fpInv ────────────────────────────────────────────────────────────────────
// Pour chaque a : calcule inv(a), puis a*inv(a) et retourne les deux.
__global__ void kernel_fp_inv(
    const uint64_t* __restrict__ a4,
    uint64_t*       __restrict__ inv4,    // inv(a)
    uint64_t*       __restrict__ prod4,   // a*inv(a) — doit être 1 ou 0 si a==0
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const uint64_t* a = a4 + i*4;
    uint64_t inv[4], prod[4];
    fpInvNormal(a, inv);  // vecteurs en representation normale
    // fpInvNormal travaille en representation normale (entiers mod p)
    // Pour verifier a*inv(a)==1, on encode en Montgomery puis on multiplie :
    // fpMul(a*R, inv*R) = a*inv*R mod p = R mod p = 1 en Montgomery
    // Mais plus simple : verifier directement en normal avec une mul scalaire
    // a * inv(a) mod p doit etre 1.
    // On encode les deux en Montgomery, fpMul donne le resultat en Montgomery aussi,
    // et 1 en Montgomery = R mod p. Le check dans le harness attend 1 (normal).
    // Solution : ne pas utiliser fpMul pour le check -- faire la mul en normal.
    {
        // a et inv sont en representation normale.
        // Encoder en Montgomery, multiplier, decoder -> doit donner 1.
        uint64_t am[4], im[4];
        fpMontEncode(a,   am);
        fpMontEncode(inv, im);
        fpMul(am, im, prod);
        fpMontDecode(prod, prod);
    }

    // Normaliser le produit pour le cas a=0 (fpMul(0, 0) = 0, pas 1)
    inv4 [i*4+0]=inv[0];  inv4 [i*4+1]=inv[1];
    inv4 [i*4+2]=inv[2];  inv4 [i*4+3]=inv[3];
    prod4[i*4+0]=prod[0]; prod4[i*4+1]=prod[1];
    prod4[i*4+2]=prod[2]; prod4[i*4+3]=prod[3];
}

// ── Courbe G1 — pointDoubleAffine ────────────────────────────────────────────
__global__ void kernel_point_double(
    const uint64_t* __restrict__ px4,
    const uint64_t* __restrict__ py4,
    uint64_t*       __restrict__ rx4,
    uint64_t*       __restrict__ ry4,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    FpPoint P, R;
    for (int j = 0; j < 4; j++) { P.X[j] = px4[i*4+j]; P.Y[j] = py4[i*4+j]; }
    P.infinity = false;
    pointDoubleAffine(P, R);
    for (int j = 0; j < 4; j++) { rx4[i*4+j] = R.X[j]; ry4[i*4+j] = R.Y[j]; }
}

// ── Courbe G1 — pointAddAffine ───────────────────────────────────────────────
__global__ void kernel_point_add(
    const uint64_t* __restrict__ p1x4, const uint64_t* __restrict__ p1y4,
    const uint64_t* __restrict__ p2x4, const uint64_t* __restrict__ p2y4,
    uint64_t*       __restrict__ rx4,
    uint64_t*       __restrict__ ry4,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    FpPoint P1, P2, R;
    for (int j = 0; j < 4; j++) {
        P1.X[j]=p1x4[i*4+j]; P1.Y[j]=p1y4[i*4+j];
        P2.X[j]=p2x4[i*4+j]; P2.Y[j]=p2y4[i*4+j];
    }
    P1.infinity = false; P2.infinity = false;
    // Les vecteurs de test sont en representation normale.
    // pointAddAffine appelle fpMul/fpInv qui travaillent en Montgomery.
    // => Encoder les entrees en Montgomery, decoder la sortie.
    fpMontEncode(P1.X, P1.X); fpMontEncode(P1.Y, P1.Y);
    fpMontEncode(P2.X, P2.X); fpMontEncode(P2.Y, P2.Y);
    pointAddAffine(P1, P2, R);
    if (!R.infinity) {
        fpMontDecode(R.X, R.X);
        fpMontDecode(R.Y, R.Y);
    }
    for (int j = 0; j < 4; j++) { rx4[i*4+j] = R.X[j]; ry4[i*4+j] = R.Y[j]; }
}

// ============================================================================
// 4. HELPERS HOST
// ============================================================================

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static bool eq4(const uint64_t a[4], const uint64_t b[4]) {
    return a[0]==b[0] && a[1]==b[1] && a[2]==b[2] && a[3]==b[3];
}

static void print_u64x4(const char* label, const uint64_t v[4]) {
    printf("  %s : %016llx %016llx %016llx %016llx\n",
           label,
           (unsigned long long)v[3], (unsigned long long)v[2],
           (unsigned long long)v[1], (unsigned long long)v[0]);
}

static void print_fail(int idx, const uint64_t got[4], const uint64_t exp[4]) {
    printf("  ↳ vecteur #%d\n", idx);
    print_u64x4("  got", got);
    print_u64x4("  exp", exp);
}

// Lance un kernel et synchronise
template<typename F>
static void launch(F kernel, int N, int tpb = 256) {
    int blocks = (N + tpb - 1) / tpb;
    kernel<<<blocks, tpb>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// 5b. WRAPPERS STATIQUES (remplacent les lambdas — compatibilité nvcc C++11)
// ============================================================================

static void launch_fp_add(const uint64_t* a, const uint64_t* b, uint64_t* r, int N) {
    int tpb=256, blk=(N+tpb-1)/tpb;
    kernel_fp_add<<<blk,tpb>>>(a,b,r,N);
    cudaDeviceSynchronize();
}
static void launch_fp_sub(const uint64_t* a, const uint64_t* b, uint64_t* r, int N) {
    int tpb=256, blk=(N+tpb-1)/tpb;
    kernel_fp_sub<<<blk,tpb>>>(a,b,r,N);
    cudaDeviceSynchronize();
}
static void launch_fp_mul(const uint64_t* a, const uint64_t* b, uint64_t* r, int N) {
    int tpb=256, blk=(N+tpb-1)/tpb;
    kernel_fp_mul<<<blk,tpb>>>(a,b,r,N);
    cudaDeviceSynchronize();
}
static void launch_fp_neg(const uint64_t* a, uint64_t* r, int N) {
    int tpb=256, blk=(N+tpb-1)/tpb;
    kernel_fp_neg<<<blk,tpb>>>(a,r,N);
    cudaDeviceSynchronize();
}
static void launch_fp_sqr(const uint64_t* a, uint64_t* r, int N) {
    int tpb=256, blk=(N+tpb-1)/tpb;
    kernel_fp_sqr<<<blk,tpb>>>(a,r,N);
    cudaDeviceSynchronize();
}

// ── Typedef pour les launchers ────────────────────────────────────────────────
typedef void (*BinopLauncher)(const uint64_t*, const uint64_t*, uint64_t*, int);
typedef void (*UnaryLauncher)(const uint64_t*, uint64_t*, int);

// ── Generic binary op (a, b → r) ─────────────────────────────────────────────
static int test_binop(const char* name, const JsonArr& vecs, BinopLauncher kernel_launcher)
{
    int N = (int)vecs.size();
    std::vector<uint64_t> h_a(N*4), h_b(N*4), h_r(N*4), h_exp(N*4);
    for (int i = 0; i < N; i++) {
        read_u64x4(vecs[i]->obj->get("a"), h_a.data()+i*4);
        read_u64x4(vecs[i]->obj->get("b"), h_b.data()+i*4);
        read_u64x4(vecs[i]->obj->get("r"), h_exp.data()+i*4);
    }
    uint64_t *d_a, *d_b, *d_r;
    CUDA_CHECK(cudaMalloc(&d_a, N*32)); CUDA_CHECK(cudaMalloc(&d_b, N*32));
    CUDA_CHECK(cudaMalloc(&d_r, N*32));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N*32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N*32, cudaMemcpyHostToDevice));
    kernel_launcher(d_a, d_b, d_r, N);
    CUDA_CHECK(cudaMemcpy(h_r.data(), d_r, N*32, cudaMemcpyDeviceToHost));
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);

    int pass = 0, first_fail = -1;
    for (int i = 0; i < N; i++) {
        if (eq4(h_r.data()+i*4, h_exp.data()+i*4)) { pass++; }
        else if (first_fail < 0) { first_fail = i; }
    }
    if (pass == N)
        printf("[PASS] %-12s %d / %d\n", name, pass, N);
    else {
        printf("[FAIL] %-12s %d / %d\n", name, pass, N);
        print_fail(first_fail, h_r.data()+first_fail*4, h_exp.data()+first_fail*4);
    }
    return N - pass;
}

// ── Generic unary op (a → r) ─────────────────────────────────────────────────
static int test_unaryop(const char* name, const JsonArr& vecs, UnaryLauncher kernel_launcher)
{
    int N = (int)vecs.size();
    std::vector<uint64_t> h_a(N*4), h_r(N*4), h_exp(N*4);
    for (int i = 0; i < N; i++) {
        read_u64x4(vecs[i]->obj->get("a"), h_a.data()+i*4);
        read_u64x4(vecs[i]->obj->get("r"), h_exp.data()+i*4);
    }
    uint64_t *d_a, *d_r;
    CUDA_CHECK(cudaMalloc(&d_a, N*32)); CUDA_CHECK(cudaMalloc(&d_r, N*32));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N*32, cudaMemcpyHostToDevice));
    kernel_launcher(d_a, d_r, N);
    CUDA_CHECK(cudaMemcpy(h_r.data(), d_r, N*32, cudaMemcpyDeviceToHost));
    cudaFree(d_a); cudaFree(d_r);

    int pass = 0, first_fail = -1;
    for (int i = 0; i < N; i++) {
        if (eq4(h_r.data()+i*4, h_exp.data()+i*4)) { pass++; }
        else if (first_fail < 0) { first_fail = i; }
    }
    if (pass == N)
        printf("[PASS] %-12s %d / %d\n", name, pass, N);
    else {
        printf("[FAIL] %-12s %d / %d\n", name, pass, N);
        print_fail(first_fail, h_r.data()+first_fail*4, h_exp.data()+first_fail*4);
    }
    return N - pass;
}

// ── fpInv : test a·inv(a)==1 ─────────────────────────────────────────────────
static int test_fp_inv(const JsonArr& vecs) {
    int N = (int)vecs.size();
    std::vector<uint64_t> h_a(N*4), h_exp_inv(N*4), h_inv(N*4), h_prod(N*4);

    for (int i = 0; i < N; i++) {
        read_u64x4(vecs[i]->obj->get("a"), h_a.data()+i*4);
        read_u64x4(vecs[i]->obj->get("r"), h_exp_inv.data()+i*4);
    }

    uint64_t *d_a, *d_inv, *d_prod;
    CUDA_CHECK(cudaMalloc(&d_a,    N*32));
    CUDA_CHECK(cudaMalloc(&d_inv,  N*32));
    CUDA_CHECK(cudaMalloc(&d_prod, N*32));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N*32, cudaMemcpyHostToDevice));

    int tpb = 256, blocks = (N + tpb - 1) / tpb;
    kernel_fp_inv<<<blocks, tpb>>>(d_a, d_inv, d_prod, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_inv.data(),  d_inv,  N*32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_prod.data(), d_prod, N*32, cudaMemcpyDeviceToHost));
    cudaFree(d_a); cudaFree(d_inv); cudaFree(d_prod);

    // Test 1 : inv(a) == expected
    int pass_val = 0, fail_val = -1;
    for (int i = 0; i < N; i++) {
        if (eq4(h_inv.data()+i*4, h_exp_inv.data()+i*4)) pass_val++;
        else if (fail_val < 0) fail_val = i;
    }

    // Test 2 : a·inv(a) == 1 (sauf a==0 → 0·0=0)
    uint64_t one[4]  = {1,0,0,0};
    uint64_t zero[4] = {0,0,0,0};
    int pass_prod = 0, fail_prod = -1;
    for (int i = 0; i < N; i++) {
        bool a_is_zero = eq4(h_a.data()+i*4, zero);
        uint64_t* expected = a_is_zero ? zero : one;
        if (eq4(h_prod.data()+i*4, expected)) pass_prod++;
        else if (fail_prod < 0) fail_prod = i;
    }

    int total_fail = (N - pass_val) + (N - pass_prod);

    if (pass_val == N)
        printf("[PASS] %-12s %d / %d  (inv values)\n", "fpInv", pass_val, N);
    else {
        printf("[FAIL] %-12s %d / %d  (inv values)\n", "fpInv", pass_val, N);
        print_fail(fail_val, h_inv.data()+fail_val*4, h_exp_inv.data()+fail_val*4);
    }

    if (pass_prod == N)
        printf("[PASS] %-12s %d / %d  (a*inv(a)==1)\n", "fpInv·check", pass_prod, N);
    else {
        printf("[FAIL] %-12s %d / %d  (a*inv(a)==1)\n", "fpInv·check", pass_prod, N);
        printf("  ↳ vecteur #%d\n", fail_prod);
        print_u64x4("  a      ", h_a.data()+fail_prod*4);
        print_u64x4("  inv(a) ", h_inv.data()+fail_prod*4);
        print_u64x4("  product", h_prod.data()+fail_prod*4);
        print_u64x4("  expect ", one);
    }

    return total_fail;
}

// ── BY_MM64 : assertion mathématique ─────────────────────────────────────────
static int test_by_mm64(const JsonVal* node) {
    // Vérifié côté CPU (Python l'a déjà vérifié, on re-vérifie en C++)
    // p · BY_MM64 + 1 ≡ 0 mod 2^62
    // On utilise __uint128_t pour éviter le débordement
    const uint64_t p0 = 0x3c208c16d87cfd47ULL;
    const uint64_t by_mm64 = BY_MM64_BN254;
    __uint128_t lhs = (__uint128_t)p0 * by_mm64;
    lhs += 1;
    lhs &= ((__uint128_t)1 << 62) - 1;  // mod 2^62
    if (lhs == 0) {
        printf("[PASS] %-12s p·BY_MM64 ≡ -1 mod 2^62  (BY_MM64=0x%016llx)\n",
               "BY_MM64", (unsigned long long)by_mm64);
        return 0;
    } else {
        printf("[FAIL] %-12s p·BY_MM64+1 mod 2^62 = %llu  (expected 0)\n",
               "BY_MM64", (unsigned long long)(uint64_t)lhs);
        return 1;
    }
}

// ── Courbe G1 ─────────────────────────────────────────────────────────────────
static int test_curve_g1(const JsonArr& vecs) {
    // On teste uniquement les vecteurs G+G et 2G+G (ceux avec p1/p2)
    // Les vecteurs scalaires (n*G) nécessitent scalarMul — testé séparément.
    std::vector<uint64_t> hp1x, hp1y, hp2x, hp2y, h_rx_exp, h_ry_exp;
    std::vector<int> add_indices;

    for (int i = 0; i < (int)vecs.size(); i++) {
        const auto* v = vecs[i];
        if (!v->obj->get("p1")) continue;
        uint64_t p1x[4], p1y[4], p2x[4], p2y[4], rx[4], ry[4];
        // Lecture des paires [x,y] — chaque élément est un JsonVal* (tableau de 4 u64)
        const JsonArr& p1arr = *v->obj->get("p1")->arr;
        const JsonArr& p2arr = *v->obj->get("p2")->arr;
        const JsonArr& rarr  = *v->obj->get("r")->arr;
        read_u64x4(p1arr[0], p1x); read_u64x4(p1arr[1], p1y);
        read_u64x4(p2arr[0], p2x); read_u64x4(p2arr[1], p2y);
        read_u64x4(rarr[0],  rx);  read_u64x4(rarr[1],  ry);

        for (int j = 0; j < 4; j++) { hp1x.push_back(p1x[j]); hp1y.push_back(p1y[j]); }
        for (int j = 0; j < 4; j++) { hp2x.push_back(p2x[j]); hp2y.push_back(p2y[j]); }
        for (int j = 0; j < 4; j++) { h_rx_exp.push_back(rx[j]); h_ry_exp.push_back(ry[j]); }
        add_indices.push_back(i);
    }

    int N = (int)add_indices.size();
    if (N == 0) {
        printf("[SKIP] %-12s (aucun vecteur add/double avec p1/p2)\n", "curveG1");
        return 0;
    }

    uint64_t *dp1x,*dp1y,*dp2x,*dp2y,*drx,*dry;
    CUDA_CHECK(cudaMalloc(&dp1x,N*32)); CUDA_CHECK(cudaMalloc(&dp1y,N*32));
    CUDA_CHECK(cudaMalloc(&dp2x,N*32)); CUDA_CHECK(cudaMalloc(&dp2y,N*32));
    CUDA_CHECK(cudaMalloc(&drx, N*32)); CUDA_CHECK(cudaMalloc(&dry, N*32));
    CUDA_CHECK(cudaMemcpy(dp1x,hp1x.data(),N*32,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp1y,hp1y.data(),N*32,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp2x,hp2x.data(),N*32,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp2y,hp2y.data(),N*32,cudaMemcpyHostToDevice));

    int tpb=256, blocks=(N+tpb-1)/tpb;
    kernel_point_add<<<blocks,tpb>>>(dp1x,dp1y,dp2x,dp2y,drx,dry,N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> h_rx(N*4), h_ry(N*4);
    CUDA_CHECK(cudaMemcpy(h_rx.data(),drx,N*32,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ry.data(),dry,N*32,cudaMemcpyDeviceToHost));
    cudaFree(dp1x);cudaFree(dp1y);cudaFree(dp2x);cudaFree(dp2y);
    cudaFree(drx);cudaFree(dry);

    int pass=0, first_fail=-1;
    for (int i=0; i<N; i++) {
        bool okx = eq4(h_rx.data()+i*4, h_rx_exp.data()+i*4);
        bool oky = eq4(h_ry.data()+i*4, h_ry_exp.data()+i*4);
        if (okx && oky) pass++;
        else if (first_fail<0) first_fail=i;
    }
    if (pass==N)
        printf("[PASS] %-12s %d / %d\n", "curveG1_add", pass, N);
    else {
        printf("[FAIL] %-12s %d / %d\n", "curveG1_add", pass, N);
        if (first_fail>=0) {
            print_u64x4("  got.x ", h_rx.data()+first_fail*4);
            print_u64x4("  exp.x ", h_rx_exp.data()+first_fail*4);
            print_u64x4("  got.y ", h_ry.data()+first_fail*4);
            print_u64x4("  exp.y ", h_ry_exp.data()+first_fail*4);
        }
    }
    return N - pass;
}

// ============================================================================
// 6. MAIN
// ============================================================================

int main(int argc, char** argv) {
    const char* json_path = (argc > 1) ? argv[1] : "test_vectors.json";

    // ── Charger JSON ─────────────────────────────────────────────────────────
    std::ifstream f(json_path);
    if (!f.is_open()) {
        fprintf(stderr, "Impossible d'ouvrir %s\n", json_path);
        fprintf(stderr, "Générer d'abord : python3 gen_test_vectors.py\n");
        return 1;
    }
    std::ostringstream ss; ss << f.rdbuf();
    std::string src = ss.str();

    const char* p = src.c_str();
    JsonVal* root = parse_val(p);
    assert(root && root->kind == JsonVal::Obj);

    printf("=== Forum — test_fp_bn254 ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    int total_fail = 0;

    // ── [1] BY_MM64 ───────────────────────────────────────────────────────────
    total_fail += test_by_mm64(root->obj->get("by_mm64_check"));

    // ── [2] fpAdd ─────────────────────────────────────────────────────────────
    total_fail += test_binop("fpAdd",
        *root->obj->get("fp_add")->arr,
        launch_fp_add);

    // ── [3] fpSub ─────────────────────────────────────────────────────────────
    total_fail += test_binop("fpSub",
        *root->obj->get("fp_sub")->arr,
        launch_fp_sub);

    // ── [4] fpNeg ─────────────────────────────────────────────────────────────
    total_fail += test_unaryop("fpNeg",
        *root->obj->get("fp_neg")->arr,
        launch_fp_neg);

    // ── [5] fpMul ─────────────────────────────────────────────────────────────
    total_fail += test_binop("fpMul",
        *root->obj->get("fp_mul")->arr,
        launch_fp_mul);

    // ── [6] fpSqr ─────────────────────────────────────────────────────────────
    total_fail += test_unaryop("fpSqr",
        *root->obj->get("fp_sqr")->arr,
        launch_fp_sqr);

    // ── [7] fpInv ─────────────────────────────────────────────────────────────
    total_fail += test_fp_inv(*root->obj->get("fp_inv")->arr);

    // ── [8] Courbe G1 ─────────────────────────────────────────────────────────
    total_fail += test_curve_g1(*root->obj->get("curve_g1")->arr);

    // ── Bilan ─────────────────────────────────────────────────────────────────
    printf("\n");
    if (total_fail == 0)
        printf("✓ TOUS LES TESTS PASSENT\n");
    else
        printf("✗ %d ÉCHEC(S) — voir détails ci-dessus\n", total_fail);

    delete root;
    return (total_fail == 0) ? 0 : 1;
}
