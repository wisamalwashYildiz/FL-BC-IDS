# Reproduction Instructions

Run commands from the repository root.

## 1. Python environment

```bash
python -m venv .venv
```

Windows:

```text
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the pinned reproduction dependencies:

```bash
python -m pip install --upgrade pip
pip install -r environment/requirements-repro.txt
```

The fuller captured environment is retained in:

```text
environment/pip_freeze.txt
```

Persisted scikit-learn/joblib preprocessing artifacts should be loaded with the
repository-pinned package versions rather than being re-saved under an
arbitrary newer scikit-learn release.

## 2. Node / snarkjs dependencies

```bash
cd environment
npm ci
cd ..
```

The verification utilities can discover Node dependencies under
`environment/node_modules/`. An alternative installation may be supplied using:

```text
FLBCIDS_NODE_MODULES
ANCHOR_ZKP_NODE_MODULES
```

## 3. Reconstruct dataset artifacts

Follow `docs/DATASETS.md`.

CSE-CIC-IDS2018:

```bash
python data_preparation/CSECICIDS2018/DatasetPreprocessing-V2.py
```

CICIoV2024 decimal representation:

```bash
python data_preparation/CICIoV2024/preprocess_ciciov2024_decimal.py
```

Do not change the documented sampling, split, feature-order, or TRAIN-only
balancing rules when the goal is to reproduce the reported experiment family.

## 4. Repository integrity checks

```bash
python scripts/build_release_inventory.py --check
python scripts/release_self_check.py
python scripts/verify_checksums.py
```

`build_release_inventory.py --check` confirms that the frozen inventory matches
the exact repository tree without modifying it. These checks should be rerun after
any repository file is replaced or edited. If content is intentionally changed, run
`python scripts/build_release_inventory.py` once, review the resulting diff, and
rerun all release checks before publishing a new tag or archive.

## 5. Multi-seed statistical evidence

Retained statistics:

```text
experiments/02_multiseed/statistics/per_seed_metrics.csv
experiments/02_multiseed/statistics/summary_statistics.csv
experiments/02_multiseed/statistics/bootstrap_exact_audit.csv
experiments/02_multiseed/statistics/paired_permutation_tests_audit.csv
```

Retained orchestrator:

```bash
python experiments/02_multiseed/code/Reviewer1_Comment1_Run_MultiSeed.py --help
```

The documented seeds are `42` through `51`, using paired inputs across methods.

## 6. Controlled generalization and stress evaluations

Temporal/generalization artifacts:

```text
experiments/03_generalization/
```

Controlled non-IID / participation-stress artifacts:

```text
experiments/04_non_iid_stress/
```

## 7. Historical primary configuration

The documented main compact configuration is recorded in:

```text
experiments/01_primary_compact/PRIMARY_CONFIGURATION.json
docs/HISTORICAL_PRIMARY_UTILITY_STATUS.md
```

The exact historical main DP predictive-utility output bundle was not retained
in the recovered project tree. The repository does not silently substitute a
different run for it.

## 8. 20-round reproduction

See:

```text
experiments/05_long_round/README.md
```

The 20-round runner is explicitly labelled as a reconstructed reproduction
artifact because a dedicated historical executable snapshot for that run was
not retained.

## 9. Expanded topology

See:

```text
experiments/06_expanded_topology/README.md
```

The retained V10 lineage records 10 RSUs, 20 vehicles per RSU, 2 global rounds,
and 10 local boosting rounds.

## 10. Cryptographic and blockchain verification

Follow the executable commands in:

```text
docs/VERIFICATION.md
```

No private SSI signing key, blockchain wallet private key, mnemonic, or hosted
RPC credential is required for offline verification.
