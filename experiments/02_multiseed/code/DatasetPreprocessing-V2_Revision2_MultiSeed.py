from __future__ import annotations

import os
import sys
import json
import hashlib
import warnings
import random
import time
import platform
from typing import Dict, Any, Tuple, Optional
import inspect  # <-- add this
import argparse

import numpy as np
import pandas as pd

from scipy import sparse
import scipy

import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from imblearn.over_sampling import RandomOverSampler
import joblib
import xgboost as xgb
import sklearn


# =============================================================================
# Config
# =============================================================================
INPUT_CSV = os.getenv("FLBCIDS_CSE_RAW_CSV", 'data/raw/CSE-CIC-IDS2018/CSECICIDS2018Dataset.csv')
REVISION2_ROOT = os.getenv("FLBCIDS_MULTI_SEED_RESULTS_DIR", "experiments/02_multiseed/results")
OUT_DIR = ""  # Set at runtime to a seed-specific Revision-2 directory.

LABEL_COL = "Label"
RANDOM_STATE = 42  # Default only; overridden by --seed at runtime.

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Target fraction of attacks (1) in TRAIN AFTER oversampling
# NOTE: We now *balance* the TRAIN set to be ~50/50, regardless of this value.
#       Kept here only for logging/manifest purposes.
TARGET_POS_FRAC_TRAIN = 0.50  # balanced: ~50% attacks, 50% benign

# Small tag to embed in artifact filenames so their origin is obvious
DATASET_TAG = "CSECICIDS2018"

# XGBoost smoke test
# Disabled for the Reviewer-1 Round-2 multi-seed experiment because the
# preprocessing pipeline is already validated and the smoke test would add
# an unnecessary auxiliary XGBoost training run for every seed.
RUN_XGB_SMOKETEST = False

# Your current XGBoost build does NOT support GPU; keep this False unless you install a GPU-enabled build.
XGB_USE_GPU = False

# Memory guards (to avoid crashing the machine)
MAX_ROWS_FOR_PANDAS = 3_000_000         # limit rows loaded into a single pandas DataFrame
MAX_ROWS_FOR_OVERSAMPLING = 3_000_000   # skip oversampling if train > this many rows
MAX_ROWS_FOR_XGB_SMOKETEST = 1_000_000  # run XGB test on at most this many rows

# We DO want dense CSVs for this dataset (85 features is safe for your 64 GB machine)
SAVE_DENSE_CSV = True

# Exact source metadata for the CSE-CIC-IDS2018 CSV used by the completed
# Reviewer-1 Round-2 runs. When launched by the orchestrator, the raw file is
# SHA-256 verified once in preflight and these values are passed through the
# child environment. Standalone execution falls back to one local SHA-256
# verification, but never performs full-source row-count/label-count Dask scans.
EXPECTED_CSE_SOURCE_SHA256 = (
    "4335539845e880b1fb06703b5a68da0a03ed0682204bdda0863ddfc316782e3c"
)
EXPECTED_CSE_SOURCE_ROWS = 63_195_145
EXPECTED_CSE_LABEL0_COUNT = 59_353_486
EXPECTED_CSE_LABEL1_COUNT = 3_841_659

ENV_CSE_SOURCE_SHA256 = "REVIEWER1_CSE_SOURCE_SHA256"
ENV_CSE_SOURCE_ROWS = "REVIEWER1_CSE_SOURCE_ROWS"
ENV_CSE_LABEL0_COUNT = "REVIEWER1_CSE_LABEL0_COUNT"
ENV_CSE_LABEL1_COUNT = "REVIEWER1_CSE_LABEL1_COUNT"

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Utils
# =============================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_verified_source_metadata(input_csv: str) -> Dict[str, Any]:
    """
    Return immutable metadata for the exact CSE source used in this study.

    Preferred path:
      - the orchestrator has already SHA-256 verified the raw source once and
        exports the verified digest/row/label metadata in the child environment.

    Standalone fallback:
      - compute the raw SHA-256 once locally and require the exact expected
        digest before trusting the fixed row/label counts.

    This deliberately avoids the former per-seed Dask full-source
    value_counts() and shape[0].compute() scans.
    """
    env_sha = os.environ.get(ENV_CSE_SOURCE_SHA256, "").strip()
    env_rows = os.environ.get(ENV_CSE_SOURCE_ROWS, "").strip()
    env_label0 = os.environ.get(ENV_CSE_LABEL0_COUNT, "").strip()
    env_label1 = os.environ.get(ENV_CSE_LABEL1_COUNT, "").strip()

    env_complete = all(
        [env_sha, env_rows, env_label0, env_label1]
    )

    if env_complete:
        if env_sha.lower() != EXPECTED_CSE_SOURCE_SHA256.lower():
            raise RuntimeError(
                "Orchestrator-provided CSE source SHA-256 does not match the "
                "expected study source."
            )

        rows = int(env_rows)
        label0 = int(env_label0)
        label1 = int(env_label1)

        if rows != EXPECTED_CSE_SOURCE_ROWS:
            raise RuntimeError(
                f"Orchestrator-provided CSE row count mismatch: "
                f"expected {EXPECTED_CSE_SOURCE_ROWS}, got {rows}."
            )

        if (
            label0 != EXPECTED_CSE_LABEL0_COUNT
            or label1 != EXPECTED_CSE_LABEL1_COUNT
        ):
            raise RuntimeError(
                "Orchestrator-provided CSE label counts do not match the "
                "expected study source."
            )

        if label0 + label1 != rows:
            raise RuntimeError(
                "CSE source metadata is internally inconsistent: "
                "label counts do not sum to total rows."
            )

        return {
            "sha256": env_sha.lower(),
            "rows": rows,
            "label0_count": label0,
            "label1_count": label1,
            "verification_mode": "orchestrator_preflight_sha256",
        }

    # Standalone safety fallback. This is one sequential file hash, not a
    # repeated Dask parse/count scan.
    local_sha = sha256_file(input_csv)

    if local_sha.lower() != EXPECTED_CSE_SOURCE_SHA256.lower():
        raise RuntimeError(
            "CSE-CIC-IDS2018 raw source SHA-256 mismatch. "
            f"Expected {EXPECTED_CSE_SOURCE_SHA256}, got {local_sha}."
        )

    return {
        "sha256": local_sha.lower(),
        "rows": int(EXPECTED_CSE_SOURCE_ROWS),
        "label0_count": int(EXPECTED_CSE_LABEL0_COUNT),
        "label1_count": int(EXPECTED_CSE_LABEL1_COUNT),
        "verification_mode": "standalone_local_sha256",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def bytes_to_human(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(n_bytes)
    for u in units:
        if val < 1024.0:
            return f"{val:.2f} {u}"
        val /= 1024.0
    return f"{val:.2f} PB"


def save_sparse_split(
    basepath_no_ext: str,
    X,
    y: np.ndarray,
    label_name: str,
    max_dense_cols_for_csv: int = 300,
) -> Tuple[Optional[str], str]:
    """
    Save X (sparse or dense) and y to NPZ (always) + optionally CSV (if few columns).
    """
    # Ensure CSR.
    # scipy.sparse.csr_matrix safely accepts both dense and sparse inputs,
    # and avoids static-analysis ambiguity around SparseABC.tocsr().
    X_csr = sparse.csr_matrix(X)

    npz_path = basepath_no_ext + ".npz"
    np.savez_compressed(
        npz_path,
        X_data=X_csr.data,
        X_indices=X_csr.indices,
        X_indptr=X_csr.indptr,
        X_shape=X_csr.shape,
        y=np.asarray(y).astype(np.int8),
    )

    csv_path = None
    if max_dense_cols_for_csv > 0 and X_csr.shape[1] <= max_dense_cols_for_csv:
        # Only create dense CSV when feature dimension is small AND CSV is enabled.
        X_dense = X_csr.toarray()
        df = pd.DataFrame(X_dense)
        df.insert(0, label_name, y.astype(int))
        csv_path = basepath_no_ext + ".csv"
        df.to_csv(csv_path, index=False)

    return csv_path, npz_path


def load_sparse_split(npz_path: str) -> Tuple[sparse.csr_matrix, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    X = sparse.csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=z["X_shape"])
    y = z["y"].astype(int)
    return X, y


def get_versions() -> Dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "dask": dask.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
    }

# =============================================================================
# Loading + Cleaning with Dask
# =============================================================================
def load_and_clean_with_dask(
    input_csv: str,
    source_metadata: Dict[str, Any],
) -> dd.DataFrame:
    print("=" * 90)
    print("[Step 1/6] Loading CSE-CIC-IDS2018 dataset with Dask...")
    t0 = time.time()

    df = dd.read_csv(
        input_csv,
        assume_missing=True,
        blocksize="128MB",
    )

    print(f"  -> Loaded Dask DataFrame with {len(df.columns)} columns.")
    print("  dtypes (head):")
    print(df.dtypes.head())

    # Replace inf/-inf with NaN.
    print("\n[Step 2/6] Replacing infinite values with NaN...")
    df = df.map_partitions(
        lambda p: p.replace([np.inf, -np.inf], np.nan)
    )

    # Identify numerical columns.
    numerical_cols = df.select_dtypes(
        include=["float64", "float32", "int64", "int32"]
    ).columns.tolist()

    print(f"  Numerical columns detected: {len(numerical_cols)}")

    print(
        "\n[Step 3/6] Leaving numeric NaNs in place; they will be imputed "
        "in the sklearn pipeline (train-only)."
    )

    # Encode label: BENIGN -> 0, else 1.
    print("\n[Step 4/6] Encoding labels to binary (BENIGN=0, attack=1)...")

    if LABEL_COL not in df.columns:
        raise ValueError("Expected 'Label' column not found in dataset.")

    df[LABEL_COL] = df[LABEL_COL].map_partitions(
        lambda col: (
            col.astype(str)
            .str.strip()
            .str.upper()
            .map(lambda x: 0 if x == "BENIGN" else 1)
        )
    )

    # Drop obvious identifiers.
    print("\n[Step 5/6] Dropping identifier / non-feature columns...")

    drop_cols = ["id", "Flow ID", "Src IP", "Dst IP", "Timestamp"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    print(f"  Columns to drop: {drop_cols}")

    df = df.drop(columns=drop_cols)

    # The full-source distribution is seed-independent and was already
    # established for the exact SHA-256-verified raw source. Reuse it instead
    # of reparsing all 63M rows for every experimental seed.
    total = int(source_metadata["rows"])
    label0 = int(source_metadata["label0_count"])
    label1 = int(source_metadata["label1_count"])

    print("\n[Step 6/6] Using verified full-source label distribution...")
    print("  Label distribution (full dataset):")

    print(
        f"    Label=0: {label0} "
        f"({100.0 * label0 / float(total):.2f}%)"
    )

    print(
        f"    Label=1: {label1} "
        f"({100.0 * label1 / float(total):.2f}%)"
    )

    print(
        f"\nPrepared lazy Dask cleaning graph in "
        f"{time.time() - t0:.1f} seconds."
    )

    return df


# =============================================================================
# Convert to Pandas, split, preprocess, oversample
# =============================================================================
def convert_to_pandas(
    df: dd.DataFrame,
    source_rows: int,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    print("=" * 90)
    print("[Step 7] Converting cleaned Dask DataFrame to Pandas (with memory guard)...")

    t0 = time.time()

    approx_n = int(source_rows)

    if approx_n <= 0:
        raise ValueError(
            f"Verified source row count must be positive; got {approx_n}."
        )

    print(f"  Verified number of rows in source dataset: {approx_n:,}")

    if max_rows is not None and approx_n > max_rows:
        frac = max_rows / float(approx_n)

        print(
            f"  Dataset is large; sampling ~{max_rows:,} rows "
            f"(frac={frac:.4f}) to avoid exhausting RAM."
        )

        # IMPORTANT: preserve the exact seed-dependent Dask sampling rule used
        # by the completed runs. Only the redundant full-source count scans
        # have been removed.
        df = df.sample(
            frac=frac,
            random_state=RANDOM_STATE,
        )

    with ProgressBar():
        df_pd = df.compute()

    mem = int(
        df_pd.memory_usage(
            deep=True
        ).sum()
    )

    print(
        f"  -> Pandas DataFrame shape: {df_pd.shape}, "
        f"memory: {bytes_to_human(mem)}"
    )

    print(
        f"Conversion done in "
        f"{time.time() - t0:.1f} seconds."
    )

    return df_pd

def split_train_val_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("=" * 90)
    print("[Step 8] Splitting into train / val / test with stratification on Label...")
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' missing before split.")
    y = df[LABEL_COL].astype(int).values
    X = df.drop(columns=[LABEL_COL])

    # Initial train vs temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - TRAIN_SIZE),
        stratify=y,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    # Split temp into val/test
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_ratio),
        stratify=y_temp,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    def _dist(name: str, labels: np.ndarray) -> None:
        cnt = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)

        benign = int(cnt[0])
        attack = int(cnt[1])
        tot = benign + attack

        benign_pct = 100.0 * float(benign) / float(max(1, tot))
        attack_pct = 100.0 * float(attack) / float(max(1, tot))

        print(
            f"  [{name}] n={tot} | "
            f"benign={benign} ({benign_pct:.2f}%) | "
            f"attack={attack} ({attack_pct:.2f}%)"
        )

    _dist("TRAIN", y_train)
    _dist("VAL", y_val)
    _dist("TEST", y_test)

    train_df = X_train.copy()
    train_df[LABEL_COL] = y_train
    val_df = X_val.copy()
    val_df[LABEL_COL] = y_val
    test_df = X_test.copy()
    test_df[LABEL_COL] = y_test

    return train_df, val_df, test_df


def build_preprocessor(train_df: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    print("=" * 90)
    print("[Step 9] Building preprocessing pipeline (impute + scale + OHE)...")
    feature_cols = [c for c in train_df.columns if c != LABEL_COL]
    X_train = train_df[feature_cols]

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = sorted(set(feature_cols) - set(numeric_cols))

    print(f"  Numeric feature cols: {len(numeric_cols)}")
    print(f"  Categorical feature cols: {len(categorical_cols)}")

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ],
        memory=None,
    )

    cat_pipeline = None
    if categorical_cols:
        # Compatibility shim: figure out at runtime whether OneHotEncoder
        # supports 'sparse_output' (new sklearn) or 'sparse' (old sklearn),
        # and set exactly one of them via **kwargs so IDEs don't see
        # an "unexpected argument".
        ohe_kwargs: Dict[str, Any] = {"handle_unknown": "ignore"}

        ohe_init_params = inspect.signature(OneHotEncoder.__init__).parameters
        if "sparse_output" in ohe_init_params:
            ohe_kwargs["sparse_output"] = True
        elif "sparse" in ohe_init_params:
            ohe_kwargs["sparse"] = True
        # else: default encoder settings, still sparse by default in older versions

        ohe = OneHotEncoder(**ohe_kwargs)

        cat_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", ohe),
            ],
            memory=None,
        )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols and cat_pipeline is not None:
        transformers.append(("cat", cat_pipeline, categorical_cols))
    if not transformers:
        raise ValueError("No features selected (no numeric and no categorical).")

    ct = ColumnTransformer(transformers=transformers, remainder="drop")

    print("  Fitting ColumnTransformer on TRAIN features...")
    t0 = time.time()
    ct.fit(X_train)
    print(f"  Preprocessor fitted in {time.time()-t0:.1f} seconds.")

    return ct, numeric_cols, categorical_cols

def transform_splits(
    ct: ColumnTransformer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    print("=" * 90)
    print("[Step 10] Transforming TRAIN / VAL / TEST with fitted preprocessor...")
    feature_cols = [c for c in train_df.columns if c != LABEL_COL]

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = np.asarray(
        train_df[LABEL_COL].to_numpy(copy=True),
        dtype=np.int64,
    )
    y_val = np.asarray(
        val_df[LABEL_COL].to_numpy(copy=True),
        dtype=np.int64,
    )
    y_test = np.asarray(
        test_df[LABEL_COL].to_numpy(copy=True),
        dtype=np.int64,
    )

    t0 = time.time()
    X_train_t = ct.transform(X_train)
    X_val_t = ct.transform(X_val)
    X_test_t = ct.transform(X_test)
    print(f"  Transformed splits in {time.time()-t0:.1f} seconds.")

    # Ensure CSR.
    # csr_matrix accepts dense arrays, sparse arrays, and sparse matrices.
    X_train_csr = sparse.csr_matrix(X_train_t)
    X_val_csr = sparse.csr_matrix(X_val_t)
    X_test_csr = sparse.csr_matrix(X_test_t)

    print("  Shapes (rows x features):")
    print(f"    TRAIN: {X_train_csr.shape}")
    print(f"    VAL:   {X_val_csr.shape}")
    print(f"    TEST:  {X_test_csr.shape}")

    return X_train_csr, X_val_csr, X_test_csr, y_train, y_val, y_test

def oversample_train(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    target_pos_frac: float,
) -> Tuple[sparse.csr_matrix, np.ndarray, Dict[str, Any]]:
    print("=" * 90)
    print("[Step 11] Applying RandomOverSampler on TRAIN only (to mitigate imbalance)...")
    cnt = np.bincount(y_train, minlength=2)
    n0, n1 = int(cnt[0]), int(cnt[1])
    tot = n0 + n1
    print(f"  Original TRAIN: benign={n0} ({100*n0/tot:.2f}%), attack={n1} ({100*n1/tot:.2f}%)")

    # If dataset is huge, skip oversampling to avoid blowing up memory.
    n_samples = X_train.shape[0]
    if MAX_ROWS_FOR_OVERSAMPLING is not None and n_samples > MAX_ROWS_FOR_OVERSAMPLING:
        print(
            f"  TRAIN has {n_samples:,} rows > MAX_ROWS_FOR_OVERSAMPLING={MAX_ROWS_FOR_OVERSAMPLING:,}; "
            "skipping oversampling and relying on model-level imbalance handling (e.g., scale_pos_weight)."
        )
        return X_train, y_train, {
            "before": {"benign": n0, "attack": n1},
            "after": {"benign": n0, "attack": n1},
            "sampling_strategy": None,
            "skipped_due_to_size": True,
        }

    if n1 == 0 or n0 == 0:
        print("  WARNING: One of the classes is missing in TRAIN; skipping oversampling.")
        return X_train, y_train, {
            "before": {"benign": n0, "attack": n1},
            "after": {"benign": n0, "attack": n1},
            "sampling_strategy": None,
        }

    if n0 == n1:
        print("  TRAIN is already balanced; no oversampling needed.")
        return X_train, y_train, {
            "before": {"benign": n0, "attack": n1},
            "after": {"benign": n0, "attack": n1},
            "sampling_strategy": "already_balanced",
        }

    # We now *force* a balanced TRAIN set: minority count == majority count.
    # RandomOverSampler with sampling_strategy=1.0 does exactly that
    # (n_minority_after = 1.0 * n_majority_after).
    sampling_strategy = 1.0
    print(
        f"  Balancing TRAIN set with RandomOverSampler "
        f"(sampling_strategy={sampling_strategy}) — "
        f"minority class will be duplicated until counts are equal."
    )
    print(f"  (Requested TARGET_POS_FRAC_TRAIN={target_pos_frac:.3f} is kept for logging only.)")

    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    cnt_after = np.bincount(y_res, minlength=2)
    n0a, n1a = int(cnt_after[0]), int(cnt_after[1])
    tota = n0a + n1a
    print(f"  After oversampling: benign={n0a} ({100*n0a/tota:.2f}%), attack={n1a} ({100*n1a/tota:.2f}%)")

    X_res_csr = sparse.csr_matrix(X_res)
    y_res_array = np.asarray(y_res, dtype=np.int64)

    return X_res_csr, y_res_array, {
        "before": {"benign": n0, "attack": n1},
        "after": {"benign": n0a, "attack": n1a},
        "sampling_strategy": sampling_strategy,
    }

# =============================================================================
# Tests / sanity checks
# =============================================================================
def basic_checks_on_npz(train_npz: str, val_npz: str, test_npz: str) -> None:
    print("=" * 90)
    print("[Step 13] Running basic consistency checks on saved NPZ splits...")
    Xtr, ytr = load_sparse_split(train_npz)
    Xv, yv = load_sparse_split(val_npz)
    Xt, yt = load_sparse_split(test_npz)

    print(f"  TRAIN shape: {Xtr.shape}, VAL: {Xv.shape}, TEST: {Xt.shape}")
    assert Xtr.shape[1] == Xv.shape[1] == Xt.shape[1], "Feature dimensions mismatch across splits!"
    print("  ✅ Feature dimensions consistent across splits.")

    for name, X in [("TRAIN", Xtr), ("VAL", Xv), ("TEST", Xt)]:
        data = X.data
        if not np.isfinite(data).all():
            raise ValueError(f"Non-finite values found in {name} features.")
        print(f"  ✅ {name}: all feature values are finite.")

    for name, y in [("TRAIN", ytr), ("VAL", yv), ("TEST", yt)]:
        uniq = np.unique(y)
        if not set(uniq).issubset({0, 1}):
            raise ValueError(f"{name}: labels contain values outside {{0,1}}: {uniq}")
        print(f"  ✅ {name}: labels are binary with values {uniq}.")

    def _dist(name: str, labels: np.ndarray) -> None:
        cnt = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)

        benign = int(cnt[0])
        attack = int(cnt[1])
        tot = benign + attack

        benign_pct = 100.0 * float(benign) / float(max(1, tot))
        attack_pct = 100.0 * float(attack) / float(max(1, tot))

        print(
            f"  [{name}] n={tot} | "
            f"benign={benign} ({benign_pct:.2f}%) | "
            f"attack={attack} ({attack_pct:.2f}%)"
        )

    _dist("TRAIN", ytr)
    _dist("VAL", yv)
    _dist("TEST", yt)
    print("  ✅ Basic checks passed.")


def xgb_smoketest(
    train_npz: str,
    val_npz: str,
    test_npz: str,
    use_gpu: bool = False,
    max_rows: Optional[int] = None,
) -> None:
    print("=" * 90)
    print("[Step 14] Running a small XGBoost smoke test (just to verify pipeline)...")

    Xtr, ytr = load_sparse_split(train_npz)
    Xv, yv = load_sparse_split(val_npz)
    Xt, yt = load_sparse_split(test_npz)

    # Optionally sub-sample rows to keep the smoke test light.
    if max_rows is not None and Xtr.shape[0] > max_rows:
        print(
            f"  TRAIN has {Xtr.shape[0]:,} rows; sub-sampling to {max_rows:,} rows "
            "for the XGBoost smoke test."
        )
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(Xtr.shape[0], size=max_rows, replace=False)
        Xtr = Xtr[idx]
        ytr = ytr[idx]

    if max_rows is not None and Xv.shape[0] > max_rows // 4:
        rng = np.random.default_rng(RANDOM_STATE + 1)
        val_rows = min(max_rows // 4, Xv.shape[0])
        idx_v = rng.choice(Xv.shape[0], size=val_rows, replace=False)
        Xv = Xv[idx_v]
        yv = yv[idx_v]

    if max_rows is not None and Xt.shape[0] > max_rows // 4:
        rng = np.random.default_rng(RANDOM_STATE + 2)
        test_rows = min(max_rows // 4, Xt.shape[0])
        idx_t = rng.choice(Xt.shape[0], size=test_rows, replace=False)
        Xt = Xt[idx_t]
        yt = yt[idx_t]

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dval = xgb.DMatrix(Xv, label=yv)
    dtest = xgb.DMatrix(Xt, label=yt)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr", "logloss"],
        "tree_method": "gpu_hist" if use_gpu else "hist",
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(len(ytr) - ytr.sum()) / max(1.0, float(ytr.sum())),
        "verbosity": 1,
    }

    evals = [(dtrain, "train"), (dval, "val")]
    num_round = 80
    print(f"  Training XGBoost for {num_round} rounds (use_gpu={use_gpu})...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_round,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10,
    )

    print("  Evaluating on TEST split...")
    y_prob = booster.predict(dtest)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n  Classification report (TEST):")
    print(classification_report(yt, y_pred, digits=4))

    auc = roc_auc_score(yt, y_prob)
    ap = average_precision_score(yt, y_prob)
    cm = confusion_matrix(yt, y_pred)
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  PR-AUC:  {ap:.4f}")
    print("  Confusion matrix (TEST):")
    print(cm)
    print("  ✅ XGBoost smoke test finished.")


# =============================================================================
# Round-2 command-line configuration
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CSE-CIC-IDS2018 preprocessing for Reviewer 1 Round-2 multi-seed "
            "evaluation. Each seed is written to an isolated Revision-2 folder."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Experimental seed controlling sampling, splitting, oversampling, and smoke-test sampling.",
    )
    parser.add_argument(
        "--revision-root",
        type=str,
        default=REVISION2_ROOT,
        help="Root directory for all Reviewer-1 Round-2 multi-seed artifacts.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=INPUT_CSV,
        help="Path to the original CSE-CIC-IDS2018 CSV.",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    global RANDOM_STATE, OUT_DIR, INPUT_CSV

    args = parse_args()
    RANDOM_STATE = int(args.seed)
    INPUT_CSV = args.input_csv
    OUT_DIR = os.path.join(
        args.revision_root,
        "CSECICIDS2018",
        f"seed_{RANDOM_STATE}",
        "preprocessing",
    )
    seed_everything(RANDOM_STATE)
    ensure_dir(OUT_DIR)

    print("=" * 90)
    print("REVIEWER 1 ROUND-2 MULTI-SEED PREPROCESSING")
    print(f"Experimental seed: {RANDOM_STATE}")
    print(f"Revision-2 output directory: {OUT_DIR}")
    print("=" * 90)

    print("=" * 90)
    print("CSE-CIC-IDS2018 PREPROCESSING FOR FL (binary BENIGN vs ATTACK)")
    print("=" * 90)

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Input CSV not found: {INPUT_CSV}")
        sys.exit(1)

    source_metadata = get_verified_source_metadata(
        INPUT_CSV
    )

    raw_sha = str(
        source_metadata["sha256"]
    )

    print(f"Input CSV SHA-256: {raw_sha}")

    print(
        "Source metadata verification: "
        f"{source_metadata['verification_mode']}"
    )

    # 1) Load & clean with Dask.
    df_dask = load_and_clean_with_dask(
        INPUT_CSV,
        source_metadata,
    )

    # 2) Convert to Pandas using the exact verified source row count.
    # This preserves the original seed-dependent
    # df.sample(frac=..., random_state=seed) rule while avoiding a
    # separate full-source df.shape[0].compute() scan.
    df_pd = convert_to_pandas(
        df_dask,
        source_rows=int(
            source_metadata["rows"]
        ),
        max_rows=MAX_ROWS_FOR_PANDAS,
    )

    # 3) Train/Val/Test split
    train_df, val_df, test_df = split_train_val_test(df_pd)

    # 4) Build preprocessor
    ct, numeric_cols, categorical_cols = build_preprocessor(train_df)

    # 5) Transform splits
    Xtr, Xv, Xt, ytr, yv, yt = transform_splits(ct, train_df, val_df, test_df)

    # 6) Oversample TRAIN only
    Xtr_bal, ytr_bal, os_info = oversample_train(Xtr, ytr, TARGET_POS_FRAC_TRAIN)

    # 7) Save artifacts
    print("=" * 90)
    print("[Step 12] Saving preprocessed splits and metadata to disk...")

    # Include a dataset tag in filenames so their origin is obvious.
    # Example: CSECICIDS2018_train_preprocessed.csv / .npz
    base_train = os.path.join(OUT_DIR, f"{DATASET_TAG}_train_preprocessed")
    base_val = os.path.join(OUT_DIR, f"{DATASET_TAG}_val_preprocessed")
    base_test = os.path.join(OUT_DIR, f"{DATASET_TAG}_test_preprocessed")

    # If SAVE_DENSE_CSV is True, allow dense CSV creation up to 300 features.
    # (We have only 85 features, so CSVs are safe.)
    dense_cols_limit = 300 if SAVE_DENSE_CSV else 0

    train_csv, train_npz = save_sparse_split(
        base_train, Xtr_bal, ytr_bal, LABEL_COL, max_dense_cols_for_csv=dense_cols_limit
    )
    val_csv, val_npz = save_sparse_split(
        base_val, Xv, yv, LABEL_COL, max_dense_cols_for_csv=dense_cols_limit
    )
    test_csv, test_npz = save_sparse_split(
        base_test, Xt, yt, LABEL_COL, max_dense_cols_for_csv=dense_cols_limit
    )

    preproc_path = os.path.join(OUT_DIR, "preproc_column_transformer.joblib")
    joblib.dump(ct, preproc_path)

    # Try to record transformed feature names, if supported by this sklearn version.
    transformed_feature_names = None
    if hasattr(ct, "get_feature_names_out"):
        try:
            transformed_feature_names = ct.get_feature_names_out().tolist()
        except Exception:
            transformed_feature_names = None

    # Manifest
    manifest = {
        "input_csv": INPUT_CSV,
        "input_csv_sha256": raw_sha,
        "source_metadata": {
            "rows": int(
                source_metadata["rows"]
            ),
            "label0_count": int(
                source_metadata["label0_count"]
            ),
            "label1_count": int(
                source_metadata["label1_count"]
            ),
            "verification_mode": str(
                source_metadata["verification_mode"]
            ),
        },
        "out_dir": OUT_DIR,
        "label_col": LABEL_COL,
        "split_sizes": {"train": TRAIN_SIZE, "val": VAL_SIZE, "test": TEST_SIZE},
        "random_state": RANDOM_STATE,
        "revision_context": {
            "round": 2,
            "reviewer": 1,
            "comment": 1,
            "purpose": "multi-seed and repartitioned statistical evaluation",
            "revision_root": args.revision_root,
        },
        "features": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "transformed": transformed_feature_names,
        },
        "oversampling": {
            "enabled": os_info["sampling_strategy"] is not None,
            "target_pos_frac": TARGET_POS_FRAC_TRAIN,
            "before": os_info["before"],
            "after": os_info["after"],
            "sampling_strategy": os_info["sampling_strategy"],
        },
        "artifacts": {
            "train_npz": train_npz,
            "val_npz": val_npz,
            "test_npz": test_npz,
            "train_csv": train_csv,
            "val_csv": val_csv,
            "test_csv": test_csv,
            "preproc_column_transformer": preproc_path,
        },
        "library_versions": get_versions(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(OUT_DIR, "preprocessing_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest: {manifest_path}")
    print("  Artifacts:")
    for k, v in manifest["artifacts"].items():
        print(f"    {k}: {v}")

    # 8) Basic tests on NPZ splits
    basic_checks_on_npz(train_npz, val_npz, test_npz)

    # 9) Optional XGBoost smoke test
    if RUN_XGB_SMOKETEST:
        xgb_smoketest(
            train_npz,
            val_npz,
            test_npz,
            use_gpu=XGB_USE_GPU,
            max_rows=MAX_ROWS_FOR_XGB_SMOKETEST,
        )

    print("=" * 90)
    print("All done. Preprocessed CSE-CIC-IDS2018 is ready for FL experiments.")
    print("=" * 90)


if __name__ == "__main__":
    main()
