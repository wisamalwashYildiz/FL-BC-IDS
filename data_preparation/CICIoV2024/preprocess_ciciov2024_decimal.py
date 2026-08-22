#!/usr/bin/env python3
"""
Deterministic CICIoV2024 decimal preprocessing for the FL-BC-IDS archive.

The data-processing algorithm is intentionally frozen to the reported workflow:
merge the six class-specific decimal CSV files, perform a stratified 70/15/15
split with seed 42, fit median-imputation/standardization on TRAIN only, and
balance TRAIN only with RandomOverSampler.

Publication manifests record repository-relative logical locations rather than
machine-specific absolute filesystem paths. Exact input/output content is pinned
by SHA-256.
"""
from __future__ import annotations
import os

import json
import random
import hashlib
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import imblearn
import numpy as np
import pandas as pd
import sklearn

from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIG
# =============================================================================
DECIMAL_DIR = Path(os.getenv("FLBCIDS_CICIOV_RAW_DIR", "data/raw/CICIoV2024"))
OUT_DIR = Path(os.getenv("FLBCIDS_CICIOV_PREPROC_DIR", "data/preprocessed/CICIoV2024"))

EXPECTED_FILES = [
    "decimal_benign.csv",
    "decimal_DoS.csv",
    "decimal_spoofing-GAS.csv",
    "decimal_spoofing-RPM.csv",
    "decimal_spoofing-SPEED.csv",
    "decimal_spoofing-STEERING_WHEEL.csv",
]

EXPECTED_COLUMNS = [
    "ID",
    "DATA_0", "DATA_1", "DATA_2", "DATA_3",
    "DATA_4", "DATA_5", "DATA_6", "DATA_7",
    "label",
    "category",
    "specific_class",
]

FEATURE_COLS = [
    "ID",
    "DATA_0", "DATA_1", "DATA_2", "DATA_3",
    "DATA_4", "DATA_5", "DATA_6", "DATA_7",
]

RAW_LABEL_COL = "label"
CATEGORY_COL = "category"
SPECIFIC_CLASS_COL = "specific_class"
BINARY_LABEL_COL = "Label"

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

TRAIN_TARGET_POS_FRAC = 0.50
MAX_ROWS_FOR_OVERSAMPLING = 3_000_000
SANITY_CHUNK_SIZE = 250_000

DATASET_NAME = "CICIoV2024"

# Logical locations written to the public manifest. Actual execution paths can
# be overridden through the FLBCIDS_* environment variables above, but local
# machine paths are intentionally not persisted in the publication artifact.
PUBLIC_RAW_LOCATOR = "data/raw/CICIoV2024"
PUBLIC_PREPROC_LOCATOR = "data/preprocessed/CICIoV2024"


# =============================================================================
# HELPERS
# =============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def normalize_label_text(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )


def normalize_meta_text(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def expected_binary_from_filename(filename: str) -> int:
    return 0 if filename.lower() == "decimal_benign.csv" else 1


def bincount_binary(y: np.ndarray) -> Dict[str, int]:
    cnt = np.bincount(y.astype(int), minlength=2)
    return {"benign": int(cnt[0]), "attack": int(cnt[1])}


def describe_binary_distribution(name: str, y: np.ndarray) -> Dict[str, Any]:
    counts = bincount_binary(y)
    total = counts["benign"] + counts["attack"]
    benign_pct = 100.0 * counts["benign"] / max(total, 1)
    attack_pct = 100.0 * counts["attack"] / max(total, 1)

    print(
        f"[{name}] n={total} | benign={counts['benign']} ({benign_pct:.2f}%) | "
        f"attack={counts['attack']} ({attack_pct:.2f}%)"
    )

    return {
        "rows": int(total),
        "benign": int(counts["benign"]),
        "attack": int(counts["attack"]),
        "benign_pct": benign_pct,
        "attack_pct": attack_pct,
    }


def get_versions() -> Dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "imbalanced_learn": imblearn.__version__,
    }


def validate_binary_target(y: np.ndarray, context: str) -> None:
    uniq = np.unique(y.astype(int))
    if not np.array_equal(uniq, np.array([0, 1])):
        raise ValueError(f"{context}: expected binary labels {{0,1}}, found {uniq.tolist()}.")


def validate_global_split_feasibility(y: np.ndarray) -> None:
    validate_binary_target(y, "Full dataset before split")

    counts = np.bincount(y.astype(int), minlength=2)
    min_class_count = int(counts.min())

    # Conservative safeguard for a 3-way stratified split.
    if min_class_count < 4:
        raise ValueError(
            "Not enough minority-class samples to safely create stratified train/val/test splits. "
            f"Class counts are {counts.tolist()}."
        )


# =============================================================================
# LOAD + VALIDATE + MERGE
# =============================================================================
def read_and_validate_one_decimal_file(csv_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print("-" * 100)
    print(f"Reading: {csv_path.name}")
    print("-" * 100)

    df = pd.read_csv(csv_path)

    actual_cols = list(df.columns)
    if set(actual_cols) != set(EXPECTED_COLUMNS):
        missing = [c for c in EXPECTED_COLUMNS if c not in actual_cols]
        extra = [c for c in actual_cols if c not in EXPECTED_COLUMNS]
        raise ValueError(
            f"{csv_path.name}: schema mismatch.\n"
            f"Missing columns: {missing}\n"
            f"Extra columns: {extra}\n"
            f"Actual columns: {actual_cols}"
        )

    df = df[EXPECTED_COLUMNS].copy()

    df[RAW_LABEL_COL] = normalize_label_text(df[RAW_LABEL_COL])
    df[CATEGORY_COL] = normalize_meta_text(df[CATEGORY_COL])
    df[SPECIFIC_CLASS_COL] = normalize_meta_text(df[SPECIFIC_CLASS_COL])

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isna().any():
            raise ValueError(f"{csv_path.name}: numeric column '{col}' contains NaN after coercion.")
        df[col] = df[col].astype(np.int32)

    if int(df.isna().sum().sum()) != 0:
        raise ValueError(f"{csv_path.name}: missing values found.")

    if (df["ID"] < 0).any():
        raise ValueError(f"{csv_path.name}: negative ID values found.")

    for col in FEATURE_COLS[1:]:
        if (df[col] < 0).any() or (df[col] > 255).any():
            raise ValueError(f"{csv_path.name}: {col} contains values outside [0, 255].")

    label_values = sorted(df[RAW_LABEL_COL].dropna().unique().tolist())
    if not set(label_values).issubset({"BENIGN", "ATTACK"}):
        raise ValueError(f"{csv_path.name}: unexpected label values: {label_values}")

    df[BINARY_LABEL_COL] = (df[RAW_LABEL_COL] != "BENIGN").astype(np.int8)

    expected_binary = expected_binary_from_filename(csv_path.name)
    if not (df[BINARY_LABEL_COL] == expected_binary).all():
        raise ValueError(f"{csv_path.name}: filename-implied class disagrees with label column.")

    info = {
        "file": csv_path.name,
        "rows": int(len(df)),
        "sha256": sha256_file(csv_path),
        "binary_counts": bincount_binary(df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False)),
        "category_values": sorted(df[CATEGORY_COL].dropna().unique().tolist()),
        "specific_class_values": sorted(df[SPECIFIC_CLASS_COL].dropna().unique().tolist()),
        "id_min": int(df["ID"].min()),
        "id_max": int(df["ID"].max()),
    }

    print(f"Rows       : {info['rows']:,}")
    print(f"SHA-256    : {info['sha256']}")
    print(f"Categories : {info['category_values']}")
    print(f"Classes    : {info['specific_class_values']}")

    return df, info


def load_and_merge_decimal_files(decimal_dir: Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    if not decimal_dir.exists():
        raise FileNotFoundError(f"Decimal directory not found: {decimal_dir}")

    found_csvs = sorted(
        [p.name for p in decimal_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    )

    missing = [f for f in EXPECTED_FILES if f not in found_csvs]
    extra = [f for f in found_csvs if f not in EXPECTED_FILES]

    if missing:
        raise FileNotFoundError(f"Missing expected decimal CSV files: {missing}")
    if extra:
        raise ValueError(f"Unexpected extra CSV files in decimal directory: {extra}")

    frames: List[pd.DataFrame] = []
    infos: List[Dict[str, Any]] = []

    for fname in EXPECTED_FILES:
        csv_path = decimal_dir / fname
        df_one, info_one = read_and_validate_one_decimal_file(csv_path)
        frames.append(df_one)
        infos.append(info_one)

    merged_df: pd.DataFrame = pd.concat(frames, axis=0, ignore_index=True, copy=False)
    merged_y: np.ndarray = merged_df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False)

    print("=" * 100)
    print("Merged raw decimal dataset")
    print("=" * 100)
    print(f"Shape: {merged_df.shape}")

    merged_summary = describe_binary_distribution("MERGED_RAW", merged_y)
    validate_global_split_feasibility(merged_y)

    return merged_df, infos, merged_summary


# =============================================================================
# SPLIT + PREPROCESS
# =============================================================================
def split_train_val_test(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    print("=" * 100)
    print("Splitting into train / val / test with stratification")
    print("=" * 100)

    y = df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False)
    validate_global_split_feasibility(y)

    idx = np.arange(len(df), dtype=np.int64)

    try:
        idx_train, idx_temp = train_test_split(
            idx,
            test_size=(1.0 - TRAIN_SIZE),
            stratify=y,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        val_ratio_within_temp = VAL_SIZE / (VAL_SIZE + TEST_SIZE)

        idx_val, idx_test = train_test_split(
            idx_temp,
            test_size=(1.0 - val_ratio_within_temp),
            stratify=y[idx_temp],
            random_state=RANDOM_STATE,
            shuffle=True,
        )
    except ValueError as exc:
        raise ValueError(f"Stratified split failed: {exc}") from exc

    train_df = df.iloc[idx_train].reset_index(drop=True)
    val_df = df.iloc[idx_val].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)

    train_y = train_df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False)
    val_y = val_df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False)
    test_y = test_df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False)

    validate_binary_target(train_y, "TRAIN split before balancing")
    validate_binary_target(val_y, "VAL split")
    validate_binary_target(test_y, "TEST split")

    summary = {
        "train_before_balance": describe_binary_distribution("TRAIN_BEFORE_BALANCE", train_y),
        "val": describe_binary_distribution("VAL", val_y),
        "test": describe_binary_distribution("TEST", test_y),
    }

    return train_df, val_df, test_df, summary


def build_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    print("=" * 100)
    print("Building preprocessing pipeline (median impute + standardize on TRAIN only)")
    print("=" * 100)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, FEATURE_COLS),
        ],
        remainder="drop",
    )

    preprocessor.fit(train_df[FEATURE_COLS])
    return preprocessor


def transform_to_dataframe(preprocessor: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    X = preprocessor.transform(df[FEATURE_COLS])

    if hasattr(X, "toarray"):
        X = X.toarray()

    out_df = pd.DataFrame(X, columns=FEATURE_COLS)
    out_df.insert(0, BINARY_LABEL_COL, df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False))
    return out_df


def balance_train_only(train_pre_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print("=" * 100)
    print("Balancing TRAIN only")
    print("=" * 100)

    X_train = train_pre_df[FEATURE_COLS].copy()
    y_train = train_pre_df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False)

    validate_binary_target(y_train, "TRAIN before oversampling")

    if not np.isfinite(X_train.to_numpy(dtype=float, copy=False)).all():
        raise ValueError("TRAIN before oversampling contains non-finite feature values.")

    if len(train_pre_df) == 0:
        raise ValueError("TRAIN before oversampling is empty.")

    if len(train_pre_df) > MAX_ROWS_FOR_OVERSAMPLING:
        raise ValueError(
            f"TRAIN has {len(train_pre_df):,} rows, which exceeds MAX_ROWS_FOR_OVERSAMPLING="
            f"{MAX_ROWS_FOR_OVERSAMPLING:,}."
        )

    before = bincount_binary(y_train)
    n0 = before["benign"]
    n1 = before["attack"]

    if n0 == 0 or n1 == 0:
        raise ValueError("TRAIN before oversampling contains only one class; balancing is impossible.")

    if n0 == n1:
        print("TRAIN is already balanced; oversampling is skipped.")
        balanced_df = train_pre_df.copy().reset_index(drop=True)
        after = before
        sampling_strategy: Any = "already_balanced"
    else:
        print(
            f"Applying RandomOverSampler(sampling_strategy=1.0) to TRAIN. "
            f"Requested TRAIN_TARGET_POS_FRAC={TRAIN_TARGET_POS_FRAC:.2f} is for logging."
        )
        ros = RandomOverSampler(sampling_strategy=1.0, random_state=RANDOM_STATE)
        X_res, y_res = ros.fit_resample(X_train, y_train)

        balanced_df = pd.DataFrame(X_res, columns=FEATURE_COLS)
        balanced_df.insert(0, BINARY_LABEL_COL, y_res.astype(np.int8))
        after = bincount_binary(y_res)
        sampling_strategy = 1.0

        # Final shuffle after oversampling to avoid grouped duplicates at the end.
        rng = np.random.default_rng(RANDOM_STATE)
        perm = rng.permutation(len(balanced_df))
        balanced_df = balanced_df.iloc[perm].reset_index(drop=True)

    describe_binary_distribution(
        "TRAIN_AFTER_BALANCE",
        balanced_df[BINARY_LABEL_COL].to_numpy(dtype=np.int8, copy=False),
    )

    if after["benign"] != after["attack"]:
        raise ValueError(
            f"Balanced TRAIN is not 50/50. Got benign={after['benign']}, attack={after['attack']}."
        )

    if not np.isfinite(balanced_df[FEATURE_COLS].to_numpy(dtype=float, copy=False)).all():
        raise ValueError("Balanced TRAIN contains non-finite feature values.")

    return balanced_df, {
        "before": before,
        "after": after,
        "sampling_strategy": sampling_strategy,
        "target_pos_frac_train_logged_only": TRAIN_TARGET_POS_FRAC,
    }


# =============================================================================
# POST-SAVE SANITY CHECKS
# =============================================================================
def sanity_check_saved_csv(
    csv_path: Path,
    split_name: str,
    expect_train_balanced: bool = False,
) -> Dict[str, Any]:
    expected_cols = [BINARY_LABEL_COL] + FEATURE_COLS

    total_rows = 0
    label_counts = np.array([0, 0], dtype=np.int64)
    first_chunk = True

    for chunk in pd.read_csv(csv_path, chunksize=SANITY_CHUNK_SIZE):
        if first_chunk:
            if list(chunk.columns) != expected_cols:
                raise ValueError(
                    f"{split_name}: saved CSV columns mismatch.\n"
                    f"Expected: {expected_cols}\n"
                    f"Actual:   {list(chunk.columns)}"
                )
            first_chunk = False

        labels = pd.to_numeric(chunk[BINARY_LABEL_COL], errors="raise")
        if labels.isna().any():
            raise ValueError(f"{split_name}: label column contains NaN after reload.")

        label_values = labels.to_numpy(dtype=np.int64, copy=False)
        uniq = np.unique(label_values)
        if not set(uniq).issubset({0, 1}):
            raise ValueError(f"{split_name}: labels contain values outside {{0,1}}: {uniq.tolist()}")

        label_counts += np.bincount(label_values, minlength=2)

        for col in FEATURE_COLS:
            vals = pd.to_numeric(chunk[col], errors="raise")
            if vals.isna().any():
                raise ValueError(f"{split_name}: feature column '{col}' contains NaN after reload.")

            arr = vals.to_numpy(dtype=float, copy=False)
            if not np.isfinite(arr).all():
                raise ValueError(f"{split_name}: feature column '{col}' contains non-finite values.")

        total_rows += len(chunk)

    if total_rows == 0:
        raise ValueError(f"{split_name}: saved CSV is empty.")

    if expect_train_balanced and int(label_counts[0]) != int(label_counts[1]):
        raise ValueError(
            f"{split_name}: expected balanced TRAIN, got benign={int(label_counts[0])}, "
            f"attack={int(label_counts[1])}."
        )

    summary = {
        "rows": int(total_rows),
        "benign": int(label_counts[0]),
        "attack": int(label_counts[1]),
        "balanced_50_50": bool(int(label_counts[0]) == int(label_counts[1])),
    }

    print(f"✅ {split_name} sanity check passed: {summary}")
    return summary


def run_post_save_sanity_checks(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
) -> Dict[str, Any]:
    print("=" * 100)
    print("Running final sanity checks on saved CSV outputs")
    print("=" * 100)

    return {
        "train": sanity_check_saved_csv(train_csv, "TRAIN", expect_train_balanced=True),
        "val": sanity_check_saved_csv(val_csv, "VAL", expect_train_balanced=False),
        "test": sanity_check_saved_csv(test_csv, "TEST", expect_train_balanced=False),
    }


# =============================================================================
# SAVE
# =============================================================================
def save_outputs_and_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    input_infos: List[Dict[str, Any]],
    merged_summary: Dict[str, Any],
    split_summary: Dict[str, Any],
    balance_info: Dict[str, Any],
) -> None:
    ensure_dir(OUT_DIR)

    train_csv = OUT_DIR / f"{DATASET_NAME}_train_preprocessed.csv"
    val_csv = OUT_DIR / f"{DATASET_NAME}_val_preprocessed.csv"
    test_csv = OUT_DIR / f"{DATASET_NAME}_test_preprocessed.csv"
    preproc_path = OUT_DIR / f"{DATASET_NAME}_preprocessor.joblib"
    manifest_path = OUT_DIR / f"{DATASET_NAME}_manifest.json"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    joblib.dump(preprocessor, preproc_path)

    post_save_checks = run_post_save_sanity_checks(train_csv, val_csv, test_csv)

    manifest = {
        "dataset_name": DATASET_NAME,
        "source_representation": "decimal",
        "source_directory": PUBLIC_RAW_LOCATOR,
        "source_directory_env": "FLBCIDS_CICIOV_RAW_DIR",
        "output_directory": PUBLIC_PREPROC_LOCATOR,
        "output_directory_env": "FLBCIDS_CICIOV_PREPROC_DIR",
        "path_policy": (
            "Publication manifest stores logical repository-relative locations; "
            "machine-local absolute execution paths are intentionally omitted."
        ),
        "merge_policy": "Concatenate the six raw decimal class-specific CSV files, then split.",
        "expected_files": EXPECTED_FILES,
        "expected_columns": EXPECTED_COLUMNS,
        "model_feature_columns": FEATURE_COLS,
        "output_label_column": BINARY_LABEL_COL,
        "split_sizes": {
            "train": TRAIN_SIZE,
            "val": VAL_SIZE,
            "test": TEST_SIZE,
        },
        "train_balance_policy": {
            "train_only_balanced": True,
            "val_unchanged": True,
            "test_unchanged": True,
            **balance_info,
        },
        "merged_summary": merged_summary,
        "input_files": input_infos,
        "split_summary": split_summary,
        "post_save_sanity_checks": post_save_checks,
        "output_files": {
            "train_csv": f"{PUBLIC_PREPROC_LOCATOR}/{train_csv.name}",
            "val_csv": f"{PUBLIC_PREPROC_LOCATOR}/{val_csv.name}",
            "test_csv": f"{PUBLIC_PREPROC_LOCATOR}/{test_csv.name}",
            "preprocessor_joblib": f"{PUBLIC_PREPROC_LOCATOR}/{preproc_path.name}",
        },
        "file_hashes": {
            "train_csv_sha256": sha256_file(train_csv),
            "val_csv_sha256": sha256_file(val_csv),
            "test_csv_sha256": sha256_file(test_csv),
            "preprocessor_joblib_sha256": sha256_file(preproc_path),
        },
        "library_versions": get_versions(),
        "random_state": RANDOM_STATE,
        "preprocessing_fit_scope": "TRAIN_ONLY",
        "oversampling_scope": "TRAIN_ONLY",
        "artifact_integrity": "SHA256",
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "This final pipeline uses the merged raw decimal CICIoV2024 files.",
            "The final outputs are only the three preprocessed CSV splits, the fitted preprocessor, and this manifest.",
            "TRAIN is balanced after preprocessing using RandomOverSampler.",
            "VAL and TEST keep their natural post-split distribution.",
            "Post-save sanity checks verify finite feature values and binary labels in the saved CSV files.",
        ],
    }

    save_json(manifest_path, manifest)

    print("=" * 100)
    print("SAVED FINAL OUTPUTS")
    print("=" * 100)
    print(train_csv)
    print(val_csv)
    print(test_csv)
    print(preproc_path)
    print(manifest_path)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    seed_everything(RANDOM_STATE)
    ensure_dir(OUT_DIR)

    print("=" * 100)
    print("FINAL-FINAL CICIoV2024 DECIMAL PREPROCESSING")
    print("=" * 100)
    print(f"Decimal dir : {DECIMAL_DIR}")
    print(f"Output dir  : {OUT_DIR}")

    merged_df, input_infos, merged_summary = load_and_merge_decimal_files(DECIMAL_DIR)

    train_raw_df, val_raw_df, test_raw_df, split_summary = split_train_val_test(merged_df)

    preprocessor = build_preprocessor(train_raw_df)

    train_pre_df = transform_to_dataframe(preprocessor, train_raw_df)
    val_pre_df = transform_to_dataframe(preprocessor, val_raw_df)
    test_pre_df = transform_to_dataframe(preprocessor, test_raw_df)

    train_balanced_df, balance_info = balance_train_only(train_pre_df)

    save_outputs_and_manifest(
        train_df=train_balanced_df,
        val_df=val_pre_df,
        test_df=test_pre_df,
        preprocessor=preprocessor,
        input_infos=input_infos,
        merged_summary=merged_summary,
        split_summary=split_summary,
        balance_info=balance_info,
    )

    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()