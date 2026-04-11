pragma circom 2.0.2;

include "../../third_party/circom-ecdsa/circuits/ecdsa.circom";

template ECDSABatch(count) {
    signal input r[count][4];
    signal input s[count][4];
    signal input msghash[count][4];
    signal input pubkey[count][2][4];
    signal output valid_sum;

    component verifiers[count];
    signal partial[count + 1];

    partial[0] <== 0;

    for (var i = 0; i < count; i++) {
        verifiers[i] = ECDSAVerifyNoPubkeyCheck(64, 4);

        for (var j = 0; j < 4; j++) {
            verifiers[i].r[j] <== r[i][j];
            verifiers[i].s[j] <== s[i][j];
            verifiers[i].msghash[j] <== msghash[i][j];
            verifiers[i].pubkey[0][j] <== pubkey[i][0][j];
            verifiers[i].pubkey[1][j] <== pubkey[i][1][j];
        }

        partial[i + 1] <== partial[i] + verifiers[i].result;
    }

    valid_sum <== partial[count];
}

component main = ECDSABatch(4);
