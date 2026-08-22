pragma circom 2.1.6;

template AnchorSum(M, KMAX, SCALE, NBITS) {
    // Private
    signal input q[KMAX][M];
    signal input mask[KMAX];

    // Public
    signal input anchor_id;
    signal input round_idx;
    signal input rsu_id;
    signal input K_used;
    signal input root_poseidon_field;
    signal input pins_hash_field;
    signal input policy_id_field;
    signal input public_input_order_id_field;
    signal input r_chal;
    signal input agg_commit;

    // Row multiplier for range checks (rsu=1, global=NMAX)
    var row_mult = 1;

    // mask is boolean + compute K_used
    var ku = 0;
    for (var i = 0; i < KMAX; i++) {
        mask[i] * (mask[i] - 1) === 0;
        ku += mask[i];
    }
    signal ku_sig;
    ku_sig <== ku;
    ku_sig === K_used;

    // Q[d] = sum_i mask[i] * q[i][d]  (R1CS-safe: chained steps)
    signal Q[M];
    signal qsum[M][KMAX + 1];
    for (var d = 0; d < M; d++) {
        qsum[d][0] <== 0;
        for (var i = 0; i < KMAX; i++) {
            // qsum[d][i+1] = qsum[d][i] + (mask[i] * q[i][d])
            qsum[d][i + 1] <== qsum[d][i] + (mask[i] * q[i][d]);
        }
        Q[d] <== qsum[d][KMAX];
    }

    // poly = Horner(Q, r_chal)  (R1CS-safe: chained steps)
    signal poly_steps[M + 1];
    poly_steps[0] <== 0;
    for (var t = 0; t < M; t++) {
        poly_steps[t + 1] <== (poly_steps[t] * r_chal) + Q[M - 1 - t];
    }
    signal poly;
    poly <== poly_steps[M];

    signal mix;
    mix <== anchor_id + round_idx*1315423911 + rsu_id*2654435761 + K_used*97531 + root_poseidon_field*1580030173 + pins_hash_field*3266489917 + policy_id_field*668265263 + public_input_order_id_field*374761393;

    signal commit_check;
    commit_check <== poly + mix;
    commit_check === agg_commit;
}

component main { public [anchor_id, round_idx, rsu_id, K_used, root_poseidon_field, pins_hash_field, policy_id_field, public_input_order_id_field, r_chal, agg_commit] } = AnchorSum(64, 2, 100000, 32);
