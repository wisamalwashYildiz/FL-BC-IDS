# Dataset Reconstruction

The experiments use **CSE-CIC-IDS2018** and **CICIoV2024**. Raw datasets are
not duplicated in this archive. Obtain them from their official distribution
sources and comply with the corresponding dataset terms.

All commands below are intended to be run from the repository root.

## CSE-CIC-IDS2018

Expected raw-file location:

```text
data/raw/CSE-CIC-IDS2018/CSECICIDS2018Dataset.csv
```

or set:

```text
FLBCIDS_CSE_RAW_CSV
```

Preprocessed-output location:

```text
data/preprocessed/CSE-CIC-IDS2018/
```

or set:

```text
FLBCIDS_CSE_PREPROC_DIR
```

Run:

```bash
python data_preparation/CSECICIDS2018/DatasetPreprocessing-V2.py
```

The retained preprocessing contract uses seed `42`, a stratified `70/15/15`
train/validation/test split, preprocessing fitted on TRAIN only, and
RandomOverSampler on TRAIN only. The CSE-CIC-IDS2018 workflow also retains the
historical source-wide Dask sampling rule toward approximately 3,000,000
modeling rows when the pinned source exceeds that size. Dask fractional
sampling is approximate in realized row count; this rule is part of the
reported execution contract and should not be silently removed when reproducing
the reported workflow.

The fitted transformer retained with the archive is:

```text
data_preparation/CSECICIDS2018/preproc_column_transformer.joblib
```

and its publication manifest is:

```text
data_preparation/CSECICIDS2018/preprocessing_manifest.json
```

## CICIoV2024

The selected representation is the six class-specific **decimal** CSV files.
Place them under:

```text
data/raw/CICIoV2024/
```

or set:

```text
FLBCIDS_CICIOV_RAW_DIR
```

Expected filenames:

```text
decimal_benign.csv
decimal_DoS.csv
decimal_spoofing-GAS.csv
decimal_spoofing-RPM.csv
decimal_spoofing-SPEED.csv
decimal_spoofing-STEERING_WHEEL.csv
```

Preprocessed-output location:

```text
data/preprocessed/CICIoV2024/
```

or set:

```text
FLBCIDS_CICIOV_PREPROC_DIR
```

Run:

```bash
python data_preparation/CICIoV2024/preprocess_ciciov2024_decimal.py
```

The retained contract uses seed `42`, a stratified `70/15/15` split,
TRAIN-only median imputation and standardization, and TRAIN-only balancing.
The CAN arbitration identifier `ID` is retained as a predictor together with
`DATA_0` through `DATA_7`.

The fitted transformer retained with the archive is:

```text
data_preparation/CICIoV2024/CICIoV2024_preprocessor.joblib
```

and its publication manifest is:

```text
data_preparation/CICIoV2024/CICIoV2024_manifest.json
```

## Multi-seed paired inputs

The Round-2 multi-seed evaluation uses seeds `42` through `51` with paired
dataset/partition contracts across the compared methods. The retained
orchestrator, per-seed manifests, and statistics are under:

```text
experiments/02_multiseed/
```

## Large derived artifacts

Large raw and derived CSV/NPZ files are intentionally not duplicated in the
public archive. The preprocessing code, retained fitted transformers,
manifests, hashes, seeds, and experiment contracts provide the reconstruction
boundary.

Do not replace a missing historical output bundle with a newly reconstructed
one without labelling it explicitly as a reproduction.
