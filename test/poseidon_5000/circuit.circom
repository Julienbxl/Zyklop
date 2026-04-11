pragma circom 2.0.0;

include "../../node_modules/circomlib/circuits/poseidon.circom";

// Poseidon hash applied 5000 times in sequence.
// ~1.2M constraints (238 per Poseidon(2) call x 5000).
//
// Representative workload: 5000 Poseidon hashes correspond to
// building a Merkle tree of depth ~12 for 5000 leaves, or
// 5000 nullifier computations in a privacy protocol.
//
// Inputs:  in[2]  -- two Fr field elements
// Outputs: out    -- final hash after 5000 rounds

template PoseidonChain(N) {
    signal input  in[2];
    signal output out;

    component h[N];
    for (var i = 0; i < N; i++) {
        h[i] = Poseidon(2);
    }

    h[0].inputs[0] <== in[0];
    h[0].inputs[1] <== in[1];

    for (var i = 1; i < N; i++) {
        h[i].inputs[0] <== h[i-1].out;
        h[i].inputs[1] <== 0;
    }

    out <== h[N-1].out;
}

component main { public [in] } = PoseidonChain(5000);
