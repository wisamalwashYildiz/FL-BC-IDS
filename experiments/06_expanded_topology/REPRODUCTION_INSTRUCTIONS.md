# Expanded-Topology Reproduction Instructions

Use the publication-portable runner under:

```text
experiments/06_expanded_topology/code/portable_v10_stack/
```

The full reproducibility repository must provide `node_modules/` at its root and
the preprocessed dataset files under `data/preprocessed/`, unless the dataset
paths are overridden through the corresponding environment variables.

## CSE-CIC-IDS2018

Set:

```text
FLBCIDS_DATASET_NAME=CSECICIDS2018
```

Then run the portable V10 main runner.

## CICIoV2024

Set:

```text
FLBCIDS_DATASET_NAME=CICIoV2024
```

Then run the same portable V10 main runner.

The fixed scientific topology in both cases is:

- 10 RSUs
- 20 vehicles per RSU
- 200 vehicles total
- 2 global rounds
- 10 local boosting rounds per participant per round

The default publication output location is isolated under:

```text
artifacts/06_expanded_topology/<dataset-name>/
```

The retained V10 stack is historical source evidence and should not be edited.
