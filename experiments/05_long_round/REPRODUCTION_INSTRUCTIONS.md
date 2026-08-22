# 20-Round Reproduction Instructions

This procedure executes the publication reconstruction of the documented
20-round CSE-CIC-IDS2018 supplementary run. It does **not** recreate a missing
historical output bundle or prove that the reconstructed source file was the
exact historical executable.

## Required repository context

The full FL-BC-IDS reproducibility repository should contain:

- `node_modules/` at the repository root;
- the environment described in the archive environment files;
- CSE-CIC-IDS2018 preprocessed train/validation/test CSVs under
  `data/preprocessed/CSE-CIC-IDS2018/`, or an override set with
  `FLBCIDS_CSE_PREPROC_DIR`.

## Run

From the repository root:

```text
python experiments/05_long_round/code/reconstructed_20_round_v10_stack/FL_DP_SSI_DualMerklePoseidon_RSU+Global_ZKVerify_V10_REPRO_20ROUND.py
```

If this ZIP is unpacked standalone rather than under `experiments/`, adjust the
path to the runner accordingly and set `FLBCIDS_REPO_ROOT` to the full
reproducibility-repository root.

Default outputs are isolated under:

```text
artifacts/05_long_round/rsu_outputs_dp
```

Override with `FLBCIDS_DP_OUTPUT_DIR` if required.

## Fixed scientific contract

- CSE-CIC-IDS2018
- 2 RSUs
- 2 vehicles per RSU
- 20 global rounds
- 10 local boosting rounds per participant per global round
- `tree_method=approxDP`
- `epsilon_tree=0.25`
- `subsample=0.2`
- `num_parallel_tree=1`

The learner-stage accounting evaluates to
`epsilon_round = 0.5525028433699525` and
`epsilon_total(20) = 11.05005686739905`, reported as `11.0501` at four
decimal places. This is conditional on the fixed preprocessing artifact and
must not be interpreted as end-to-end raw-record DP.
