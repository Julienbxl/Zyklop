// test_ntt.cu — Zyklop NTT correctness tests
// Encoding/decoding done on GPU to avoid CPU __int128 optimisation bugs.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "fp_bn254.cuh"
#include "fr_bn254.cuh"
#include "ntt_bn254.cuh"

// ── GPU encode/decode kernels ────────────────────────────────────────────────
__global__ void k_decode(const uint64_t* __restrict__ in, uint64_t* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    frMontDecode(in + i*4, out + i*4);
}

static void gpu_encode(const uint64_t* h_in, uint64_t* h_out, int N) {
    uint64_t *d_in, *d_out;
    cudaMalloc(&d_in,  N*4*8); cudaMalloc(&d_out, N*4*8);
    cudaMemcpy(d_in, h_in, N*4*8, cudaMemcpyHostToDevice);
    k_encode<<<(N+255)/256,256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4*8, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
}
static void gpu_decode(const uint64_t* h_in, uint64_t* h_out, int N) {
    uint64_t *d_in, *d_out;
    cudaMalloc(&d_in,  N*4*8); cudaMalloc(&d_out, N*4*8);
    cudaMemcpy(d_in, h_in, N*4*8, cudaMemcpyHostToDevice);
    k_decode<<<(N+255)/256,256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4*8, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
}

// ── JSON parser (unchanged) ──────────────────────────────────────────────────
struct Fr4 { uint64_t v[4]; };
static std::vector<Fr4> parseFrArray(const std::string& s, size_t& pos) {
    std::vector<Fr4> out;
    while (pos < s.size() && s[pos] != '[') pos++;
    pos++;
    while (pos < s.size()) {
        while (pos < s.size() && (s[pos]==' '||s[pos]=='\n'||s[pos]=='\r'||s[pos]==',')) pos++;
        if (s[pos] == ']') { pos++; break; }
        if (s[pos] == '[') {
            pos++;
            Fr4 f; int li=0;
            while (pos < s.size() && li < 4) {
                while (pos<s.size()&&(s[pos]==' '||s[pos]==','||s[pos]=='\n')) pos++;
                if (s[pos]==']') break;
                uint64_t val=0;
                while (pos<s.size()&&s[pos]>='0'&&s[pos]<='9') {
                    val = val*10 + (s[pos]-'0'); pos++;
                }
                f.v[li++] = val;
            }
            while (pos<s.size()&&s[pos]!=']') pos++;
            pos++;
            out.push_back(f);
        }
    }
    return out;
}

int main(int argc, char** argv) {
    const char* vecfile = (argc>1) ? argv[1] : "test_vectors_ntt.json";
    printf("=== Zyklop — ntt_bn254 tests ===\n");
    printf("Loading %s...\n", vecfile);

    std::ifstream ifs(vecfile);
    if (!ifs) { fprintf(stderr,"Cannot open %s\n",vecfile); return 1; }
    std::string json((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());

    int pass_count=0, fail_count=0;
    size_t pos=0;

    nttPrepare(13);

    while (pos < json.size()) {
        size_t obj_start = json.find("\"log_n\"", pos);
        if (obj_start == std::string::npos) break;

        size_t colon = json.find(":", obj_start);
        size_t num_start = colon+1;
        while(json[num_start]==' ') num_start++;
        int log_n=0;
        while(json[num_start]>='0'&&json[num_start]<='9') {
            log_n=log_n*10+(json[num_start]-'0'); num_start++;
        }

        size_t desc_pos = json.find("\"desc\"", num_start);
        size_t dq1 = json.find("\"", desc_pos+7);
        size_t dq2 = json.find("\"", dq1+1);
        std::string desc = json.substr(dq1+1, dq2-dq1-1);

        size_t inp_pos = json.find("\"input\"", dq2);
        size_t inp_arr = json.find("[", inp_pos+7);
        std::vector<Fr4> input_raw = parseFrArray(json, inp_arr);

        size_t exp_pos = json.find("\"expected\"", inp_arr);
        size_t exp_arr = json.find("[", exp_pos+9);
        std::vector<Fr4> expected_raw = parseFrArray(json, exp_arr);

        pos = exp_arr;

        int N = 1 << log_n;
        if ((int)input_raw.size() != N || (int)expected_raw.size() != N) {
            printf("[SKIP] %s (parse error: got %zu/%zu, expected %d)\n",
                   desc.c_str(), input_raw.size(), expected_raw.size(), N);
            continue;
        }

        // Encode inputs and expected on GPU (no CPU CIOS)
        std::vector<uint64_t> h_in_can(N*4), h_exp_can(N*4);
        for(int i=0;i<N;i++) {
            h_in_can[i*4+0]=input_raw[i].v[0]; h_in_can[i*4+1]=input_raw[i].v[1];
            h_in_can[i*4+2]=input_raw[i].v[2]; h_in_can[i*4+3]=input_raw[i].v[3];
            h_exp_can[i*4+0]=expected_raw[i].v[0]; h_exp_can[i*4+1]=expected_raw[i].v[1];
            h_exp_can[i*4+2]=expected_raw[i].v[2]; h_exp_can[i*4+3]=expected_raw[i].v[3];
        }
        std::vector<uint64_t> h_in(N*4), h_exp(N*4);
        gpu_encode(h_in_can.data(), h_in.data(), N);
        gpu_encode(h_exp_can.data(), h_exp.data(), N);

        // For testing inverse: encode inputs, then run ntt(true), and compare with expected.
        uint64_t *d_a;
        cudaMalloc(&d_a, (size_t)N*4*8);
        // INVERSE TEST: we copy h_exp (which is the output of FWD) and expect h_in (the input of FWD)
        cudaMemcpy(d_a, h_exp.data(), N*4*8, cudaMemcpyHostToDevice);
        
        ntt(d_a, log_n, true);
        cudaDeviceSynchronize();
        
        std::vector<uint64_t> h_out(N*4);
        cudaMemcpy(h_out.data(), d_a, N*4*8, cudaMemcpyDeviceToHost);
        cudaFree(d_a);

        // Compare in Montgomery domain
        bool ok = true;
        for(int i=0;i<N&&ok;i++)
            for(int l=0;l<4&&ok;l++)
                if(h_out[i*4+l] != h_in[i*4+l]) ok=false; // We expect h_in now!

        // Print if it's the neg12_inv
        if (desc == "neg12_inv") {
            printf("neg12_inv IFFT raw output:\n");
            for(int i=0; i<N; i++) {
                printf("  [%d] 0x%016llx%016llx%016llx%016llx\n", i,
                    (unsigned long long)h_out[i*4+3],(unsigned long long)h_out[i*4+2],
                    (unsigned long long)h_out[i*4+1],(unsigned long long)h_out[i*4+0]);
            }
        }


        if (ok) {
            printf("[PASS] %s\n", desc.c_str());
            pass_count++;
        } else {
            printf("[FAIL] %s\n", desc.c_str());
            fail_count++;
            // Show first mismatch decoded
            std::vector<uint64_t> h_out_dec(N*4), h_exp_dec(N*4);
            gpu_decode(h_out.data(), h_out_dec.data(), N);
            gpu_decode(h_exp.data(), h_exp_dec.data(), N);
            for(int i=0;i<N;i++) {
                bool m=false;
                for(int l=0;l<4;l++) if(h_out[i*4+l]!=h_exp[i*4+l]) m=true;
                if(m) {
                    printf("  [%d] got  0x%016llx%016llx%016llx%016llx\n", i,
                        (unsigned long long)h_out_dec[i*4+3],(unsigned long long)h_out_dec[i*4+2],
                        (unsigned long long)h_out_dec[i*4+1],(unsigned long long)h_out_dec[i*4+0]);
                    printf("  [%d] want 0x%016llx%016llx%016llx%016llx\n", i,
                        (unsigned long long)h_exp_dec[i*4+3],(unsigned long long)h_exp_dec[i*4+2],
                        (unsigned long long)h_exp_dec[i*4+1],(unsigned long long)h_exp_dec[i*4+0]);
                    break;
                }
            }
        }
    }

    printf("ntt_bn254  %d / %d\n", pass_count, pass_count+fail_count);
    nttDestroy();
    return fail_count > 0 ? 1 : 0;
}
