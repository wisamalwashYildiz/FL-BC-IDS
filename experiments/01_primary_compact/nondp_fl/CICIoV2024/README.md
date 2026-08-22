# Legacy CICIoV2024 Non-DP FL Bundle

## Classification

This seven-file bundle is internally consistent historical evidence, but it is
**not the authoritative final non-DP baseline configuration**.

Its retained execution contract is:

```text
dataset             CICIoV2024
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
`num_parallel_tree=1`, identify XGBoost version 3.1.2, and use nine input
features. The model JSON files do not embed feature names; the external
CICIoV2024 preprocessing contract supplies the ordered predictors:

```text
ID, DATA_0, DATA_1, DATA_2, DATA_3, DATA_4, DATA_5, DATA_6, DATA_7
```

The ensemble test confusion matrix is:

```text
TN = 183561
FP = 0
FN = 4
TP = 27668
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

`iov_centralized_eval_summary.json` is byte-for-value equivalent to the
`per_rsu_test_metrics` mapping in the ensemble summary. In this historical
bundle, "centralized evaluation" means evaluating the RSU models on the common
central evaluation split; it does not mean centralized model training.

Exact SHA-256 fingerprints are recorded in
`LEGACY_NONDP_100LOCAL_MANIFEST.json`.
