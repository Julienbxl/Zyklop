/*
 * main.cu -- Zyklop GPU prover entry point
 *
 * Usage:
 *   ./prover <circuit.zkey> <witness.wtns> <proof.json> <public.json>
 *
 * Outputs:
 *   proof.json  -- Groth16 proof in snarkjs JSON format (pi_a, pi_b, pi_c)
 *   public.json -- public signals array, required by snarkjs groth16 verify
 *
 * Verify with:
 *   snarkjs groth16 verify verification_key.json public.json proof.json
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "../include/groth16.cuh"

// ---------------------------------------------------------------------------
// Write public.json from witness file
//
// The public signals are w[1..n_public] from the .wtns file (section 2).
// w[0] = 1 is the constant wire and is skipped.
// Values are stored as 256-bit little-endian Fr elements (canonical form).
// ---------------------------------------------------------------------------
static bool writePublicJson(const char* wtns_path,
                            uint32_t    n_public,
                            FILE*       out)
{
    std::ifstream f(wtns_path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[main] Cannot open %s\n", wtns_path);
        return false;
    }

    // wtns format:
    //   Section 1 (header, size=40): n8(4) | prime[32] | n_witness(4)
    //   Section 2 (values):          w[0] | w[1] | ...   (32 bytes each, NO count prefix)
    f.seekg(12);  // skip magic(4) + version(4) + nsections(4)
    uint64_t sec1_offset = 0, sec2_offset = 0;
    for (int i = 0; i < 5; i++) {
        uint32_t type; uint64_t size;
        f.read((char*)&type, 4);
        f.read((char*)&size, 8);
        uint64_t data_start = (uint64_t)f.tellg();
        if (type == 1) sec1_offset = data_start;
        if (type == 2) sec2_offset = data_start;
        f.seekg(size, std::ios::cur);
        if (sec1_offset && sec2_offset) break;
    }
    if (!sec1_offset || !sec2_offset) {
        fprintf(stderr, "[main] wtns: missing section 1 or 2\n");
        return false;
    }

    // n_witness is in section 1 at offset: n8(4) + prime(32) = skip 36 bytes
    f.seekg(sec1_offset + 4 + 32);
    uint32_t n_witness;
    f.read((char*)&n_witness, 4);
    if (n_witness < n_public + 1) {
        fprintf(stderr, "[main] witness has %u entries, need >= %u\n",
                n_witness, n_public + 1);
        return false;
    }

    // Section 2 has no count prefix -- skip w[0]=1 (constant wire)
    f.seekg(sec2_offset + 32);

    // Read and format w[1..n_public] as decimal strings
    fprintf(out, "[\n");
    for (uint32_t i = 0; i < n_public; i++) {
        uint8_t bytes[32];
        f.read((char*)bytes, 32);

        // Parse 256-bit LE integer into four 64-bit limbs
        uint64_t limbs[4] = {0};
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 8; k++)
                limbs[j] |= ((uint64_t)bytes[j*8+k]) << (k*8);

        // Convert to decimal by repeated division by 10
        char buf[80];
        int  len = 0;
        bool zero = !(limbs[0] | limbs[1] | limbs[2] | limbs[3]);
        if (zero) {
            buf[len++] = '0';
        } else {
            while (limbs[0] | limbs[1] | limbs[2] | limbs[3]) {
                uint64_t rem = 0;
                for (int j = 3; j >= 0; j--) {
                    __uint128_t cur = ((__uint128_t)rem << 64) | limbs[j];
                    limbs[j] = (uint64_t)(cur / 10);
                    rem      = (uint64_t)(cur % 10);
                }
                buf[len++] = '0' + (char)rem;
            }
            // Digits were accumulated least-significant first -- reverse
            for (int j = 0; j < len / 2; j++) {
                char tmp = buf[j]; buf[j] = buf[len-1-j]; buf[len-1-j] = tmp;
            }
        }
        buf[len] = '\0';

        fprintf(out, " \"%s\"%s\n", buf, (i + 1 < n_public) ? "," : "");
    }
    fprintf(out, "]\n");
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage:   %s <circuit.zkey> <witness.wtns> <proof.json> <public.json>\n",
            argv[0]);
        fprintf(stderr,
            "Example: %s circuit_final.zkey witness.wtns proof.json public.json\n",
            argv[0]);
        return 1;
    }

    const char* zkey_path   = argv[1];
    const char* wtns_path   = argv[2];
    const char* proof_path  = argv[3];
    const char* public_path = argv[4];

    // Generate proof (NTT setup/teardown handled inside groth16Prove)
    Groth16Proof proof;
    groth16Prove(zkey_path, wtns_path, proof);

    // Serialize proof to snarkjs-compatible JSON
    char json[4096];
    groth16ProofToSnarkjsJson(proof, json, sizeof(json));
    printf("%s\n", json);

    // Write proof.json
    {
        FILE* f = fopen(proof_path, "w");
        if (!f) {
            fprintf(stderr, "[main] Cannot write to %s\n", proof_path);
            return 1;
        }
        fprintf(f, "%s\n", json);
        fclose(f);
        printf("[main] Proof written to %s\n", proof_path);
    }

    // Write public.json
    // n_public is read from the zkey header; values from the witness file.
    {
        Groth16ZkeyHeader hdr;
        groth16ReadZkeyHeader(zkey_path, hdr);

        FILE* f = fopen(public_path, "w");
        if (!f) {
            fprintf(stderr, "[main] Cannot write to %s\n", public_path);
            return 1;
        }
        bool ok = writePublicJson(wtns_path, hdr.n_public, f);
        fclose(f);
        if (!ok) return 1;
        printf("[main] Public signals written to %s\n", public_path);
    }

    return 0;
}
