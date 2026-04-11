pragma circom 2.0.0;

include "../../node_modules/circomlib/circuits/sha256/sha256.circom";

// SHA-256 applied 50 times in sequence.
// Benchmark circuit: ~3M constraints, representative of real ZK workloads.
//
// Inputs:  in[512]    -- initial 512-bit input block (bits, big-endian)
// Outputs: out[256]   -- final hash after 50 rounds

template Sha256Chain(N) {
    signal input  in[512];
    signal output out[256];

    component h[N];
    for (var i = 0; i < N; i++) {
        h[i] = Sha256(512);
    }

    for (var j = 0; j < 512; j++) {
        h[0].in[j] <== in[j];
    }

    for (var i = 1; i < N; i++) {
        for (var j = 0; j < 256; j++) {
            h[i].in[j] <== h[i-1].out[j];
        }
        for (var j = 256; j < 512; j++) {
            h[i].in[j] <== 0;
        }
    }

    for (var j = 0; j < 256; j++) {
        out[j] <== h[N-1].out[j];
    }
}

component main { public [in] } = Sha256Chain(50);
