# Dataset Placement

Large/raw datasets are not duplicated in this repository.

Place or reconstruct the preprocessed datasets at:

- `data/preprocessed/CSE-CIC-IDS2018/`
- `data/preprocessed/CICIoV2024/`

Alternatively set:

- `FLBCIDS_CSE_PREPROC_DIR`
- `FLBCIDS_CICIOV_PREPROC_DIR`

The preprocessing scripts under `data_preparation/` document the deterministic
preparation workflow. Run commands should be issued from the repository root so
the repository-relative defaults resolve consistently.
