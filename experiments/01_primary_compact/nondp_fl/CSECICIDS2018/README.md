# Legacy CSE-CIC-IDS2018 Non-DP FL Bundle

## Classification

This seven-file bundle is internally consistent historical evidence, but it is
**not the authoritative final non-DP baseline configuration**.

Its retained execution contract is:

```text
dataset             CSE-CIC-IDS2018
DP                  disabled
RSUs                2
vehicles / RSU      2
global rounds       2
local rounds        100
learning_rate       0.1
min_child_weight    200
subsample           0.9
tree_method         hist
seed                42
```

The two retained RSU models each contain 103 XGBoost trees, use
`num_parallel_tree=1`, identify XGBoost version 3.1.2, and expect 85 input
features. The model JSON files do not embed feature names, so feature semantics
come from the external CSE-CIC-IDS2018 preprocessing contract.

The ensemble test confusion matrix is:

```text
TN = 422587
FP = 56
FN = 88
TP = 27270
```

which exactly reproduces the retained ensemble accuracy, precision, recall,
and F1.

## Publication handling

Do **not** edit these historical JSON artifacts to change 100 local rounds into
10 or to replace their XGBoost parameters with the final baseline parameters.
That would destroy provenance.

For the public reproducibility archive, either exclude this bundle from the
authoritative results set or retain it only under an explicitly labelled
legacy/history directory. It must not be cited as the final Round-2 non-DP
baseline.

`iov_centralized_eval_summary.json` is value-equivalent to the
`per_rsu_test_metrics` mapping in the ensemble summary. Here, "centralized
evaluation" means evaluating the RSU models on a common central evaluation
split; it does not mean centralized model training.

Exact SHA-256 fingerprints are recorded in
`LEGACY_NONDP_CSE_100LOCAL_MANIFEST.json`.
