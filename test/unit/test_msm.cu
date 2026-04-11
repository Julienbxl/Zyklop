// test_msm.cu — unit tests for msm_pippenger.cuh
// Compile: nvcc -O3 -std=c++17 -arch=sm_120 -I../include test_msm.cu -o test_msm
// Run:     ./test_msm test_vectors_msm.json

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#include "fp_bn254.cuh"
#include "msm_g1.cuh"

// =============================================================================
// Minimal JSON parser — same style as test_fp_bn254 / test_fr_bn254
// =============================================================================
static uint64_t parseU64(const std::string& s, size_t pos, size_t len) {
    return (uint64_t)std::stoull(s.substr(pos, len));
}

// Advance pos past whitespace
static void skipWS(const std::string& s, size_t& pos) {
    while (pos < s.size() && (s[pos]==' '||s[pos]=='\n'||s[pos]=='\r'||s[pos]=='\t')) pos++;
}

// Parse a JSON array of 4 uint64 into limbs[4]; advances pos past ']'
static void parseLimbs4(const std::string& s, size_t& pos, uint64_t limbs[4]) {
    while (pos < s.size() && s[pos] != '[') pos++;
    pos++; // '['
    for (int i = 0; i < 4; i++) {
        skipWS(s, pos);
        size_t start = pos;
        while (pos < s.size() && (isdigit(s[pos]) || s[pos]=='-')) pos++;
        limbs[i] = parseU64(s, start, pos-start);
        skipWS(s, pos);
        if (pos < s.size() && s[pos]==',') pos++;
    }
    while (pos < s.size() && s[pos] != ']') pos++;
    if (pos < s.size()) pos++; // ']'
}

// Parse array of N limbs-4 arrays; returns vector of size N*4 (flat)
static std::vector<uint64_t> parseLimbsArray(const std::string& s, size_t& pos, int N) {
    // Expects: [[a,b,c,d], [a,b,c,d], ...]
    while (pos < s.size() && s[pos] != '[') pos++;
    pos++; // outer '['
    std::vector<uint64_t> out(N*4);
    for (int i = 0; i < N; i++) {
        parseLimbs4(s, pos, &out[i*4]);
        skipWS(s, pos);
        if (pos < s.size() && s[pos]==',') pos++;
    }
    while (pos < s.size() && s[pos] != ']') pos++;
    if (pos < s.size()) pos++; // outer ']'
    return out;
}

struct TestVector {
    std::string          label;
    int                  N;
    std::vector<uint64_t> Px;      // N*4 limbs, Montgomery Fp
    std::vector<uint64_t> Py;      // N*4 limbs, Montgomery Fp
    std::vector<uint64_t> scalars; // N*4 limbs, canonical Fr
    uint64_t             out_x[4];
    uint64_t             out_y[4];
};

static std::vector<TestVector> loadVectors(const char* path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    std::stringstream ss; ss << f.rdbuf();
    std::string s = ss.str();

    std::vector<TestVector> vecs;
    size_t pos = 0;

    while (true) {
        size_t lp = s.find("\"label\"", pos);
        if (lp == std::string::npos) break;

        TestVector v;

        // label
        lp = s.find(':', lp) + 1;
        while (lp < s.size() && (s[lp]=='"'||s[lp]==' ')) lp++;
        size_t le = s.find('"', lp);
        v.label = s.substr(lp, le - lp);
        pos = le + 1;

        // N
        size_t np = s.find("\"N\"", pos);
        np = s.find(':', np) + 1;
        skipWS(s, np);
        size_t ns = np;
        while (np < s.size() && isdigit(s[np])) np++;
        v.N = std::stoi(s.substr(ns, np-ns));
        pos = np;

        // Px, Py, scalars
        v.Px      = parseLimbsArray(s, pos, v.N);
        v.Py      = parseLimbsArray(s, pos, v.N);
        v.scalars = parseLimbsArray(s, pos, v.N);

        // out_x, out_y (single limbs4 each)
        parseLimbs4(s, pos, v.out_x);
        parseLimbs4(s, pos, v.out_y);

        vecs.push_back(std::move(v));
    }
    return vecs;
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "test_vectors_msm.json";

    printf("=== Forum — msm_g1 tests ===\n");
    printf("Loading %s...\n", path);

    auto vecs = loadVectors(path);
    printf("%d vectors loaded\n\n", (int)vecs.size());

    int total_pass = 0, total_fail = 0;

    for (auto& v : vecs) {
        // Upload points and scalars to device
        uint64_t *d_Px, *d_Py, *d_scalars;
        size_t sz_pts     = (size_t)v.N * 4 * sizeof(uint64_t);
        size_t sz_scalars = sz_pts;

        cudaMalloc(&d_Px,      sz_pts);
        cudaMalloc(&d_Py,      sz_pts);
        cudaMalloc(&d_scalars, sz_scalars);

        cudaMemcpy(d_Px,      v.Px.data(),      sz_pts,     cudaMemcpyHostToDevice);
        cudaMemcpy(d_Py,      v.Py.data(),      sz_pts,     cudaMemcpyHostToDevice);
        cudaMemcpy(d_scalars, v.scalars.data(), sz_scalars, cudaMemcpyHostToDevice);

        // Run MSM
        FpPointAff result;
        msm_g1(d_Px, d_Py, d_scalars, v.N, result);

        cudaFree(d_Px); cudaFree(d_Py); cudaFree(d_scalars);

        // Compare
        bool ok = !result.infinity;
        for (int j = 0; j < 4 && ok; j++) {
            if (result.X[j] != v.out_x[j]) ok = false;
            if (result.Y[j] != v.out_y[j]) ok = false;
        }

        if (ok) {
            printf("[PASS] %s\n", v.label.c_str());
            total_pass++;
        } else {
            printf("[FAIL] %s\n", v.label.c_str());
            printf("  exp X: %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
                v.out_x[3], v.out_x[2], v.out_x[1], v.out_x[0]);
            printf("  got X: %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
                result.X[3], result.X[2], result.X[1], result.X[0]);
            printf("  exp Y: %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
                v.out_y[3], v.out_y[2], v.out_y[1], v.out_y[0]);
            printf("  got Y: %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
                result.Y[3], result.Y[2], result.Y[1], result.Y[0]);
            total_fail++;
        }
    }

    printf("\n[%s] msm_g1        %d / %d\n",
        total_fail==0 ? "PASS":"FAIL", total_pass, (int)vecs.size());

    if (total_fail == 0) printf("\n✓ ALL TESTS PASS\n");
    else                 printf("\n✗ FAILED — %d vector(s) wrong\n", total_fail);

    return total_fail > 0 ? 1 : 0;
}
