pragma circom 2.0.0;

include "../../third_party/benchmark-app/mopro-core/examples/circom/keccak256/keccak.circom";

template KeccakBatch(count) {
    signal input in[count][256];
    signal output checksum;

    component hashes[count];
    signal partial[count * 256 + 1];

    partial[0] <== 0;

    for (var i = 0; i < count; i++) {
        hashes[i] = Keccak(256, 256);

        for (var j = 0; j < 256; j++) {
            hashes[i].in[j] <== in[i][j];
        }

        for (var j = 0; j < 256; j++) {
            partial[i * 256 + j + 1] <== partial[i * 256 + j] + hashes[i].out[j];
        }
    }

    checksum <== partial[count * 256];
}

component main = KeccakBatch(40);
