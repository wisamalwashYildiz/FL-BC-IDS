#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Reviewer Concern #2 — Rolling-Origin Temporal Validation for FL-BC-IDS
======================================================================

PURPOSE
-------
This is a SEPARATE, CSE-CIC-IDS2018-only reviewer experiment created for
one narrow question:

    Does the IDS retain detection ability under genuine chronological shift
    when the attack type has already been observed in the training history?

It does NOT modify or rerun the principal FL-BC-IDS, DP, SSI, Groth16,
blockchain, multi-seed, CICIoV2024, or strict unseen-attack experiments.

WHY THIS SCRIPT EXISTS
----------------------
The earlier terminal 70/15/15 chronological split was discovered to contain
only attack types absent from its training interval. Therefore, its very low
attack recall is a combined temporal + complete attack-family novelty result;
it cannot estimate seen-attack temporal generalization.

This script replaces that missing endpoint with a PREDECLARED rolling-origin
protocol over chronologically ordered CSE-CIC-IDS2018 observations.

PRIMARY PROTOCOL (fixed before inspecting results)
--------------------------------------------------
1. Use the exact SHA-256-verified CSE-CIC-IDS2018 source used in the previous
   reviewer experiments.
2. Apply the same deterministic ~3,000,000-row source-wide Dask sampling rule
   used by the reviewer-validation / Comment-1 preprocessing family.
3. Parse Timestamp and sort strictly chronologically.
4. Divide the ordered sample into 10 approximately equal-row contiguous blocks,
   while NEVER splitting identical timestamps between adjacent blocks.
5. Run exactly five expanding-window rolling-origin folds:

       Fold 1: train B1-B4, calibrate B5, test B6
       Fold 2: train B1-B5, calibrate B6, test B7
       Fold 3: train B1-B6, calibrate B7, test B8
       Fold 4: train B1-B7, calibrate B8, test B9
       Fold 5: train B1-B8, calibrate B9, test B10

6. Fit preprocessing and XGBoost ONLY on each fold's training history.
7. Fix the PRIMARY operating threshold ONLY from benign scores in the
   immediately preceding calibration block, targeting empirical benign FPR <=1%.
8. Evaluate the subsequent test block without using test labels/scores for
   model fitting or threshold selection.
9. The PRIMARY temporal endpoint is restricted to attack types already present
   in that fold's training history ("seen-attack temporal" endpoint).
10. Attack types absent from the training history are reported separately as
    temporal + novelty diagnostics and do not contaminate the primary endpoint.

ANTI-CHERRY-PICKING RULES
-------------------------
- All five folds are predefined and are reported.
- A fold is excluded from the pooled seen-attack endpoint ONLY if it contains
  zero seen-attack test rows or cannot support the predefined calibration rule;
  the exclusion reason is recorded explicitly.
- The primary FPR target is fixed at 1% before test evaluation.
- A predefined FPR sensitivity grid (0.1%, 0.5%, 1%, 2%, 5%) is reported, but
  none of those alternatives may be selected post hoc as the "best" result.
- The script also reports a validation-F1 threshold only as a diagnostic
  comparator when the calibration block contains both classes.
- Weak folds and weak attack families are never hidden.

DIAGNOSTICS GENERATED
---------------------
For every fold, the script records:
- chronological train/calibration/test time ranges,
- class counts,
- attack types seen in training,
- test attack types split into seen vs novel,
- primary seen-attack temporal metrics,
- combined-period metrics,
- novel-attack diagnostic metrics,
- test benign FPR drift relative to the 1% development target,
- predictor-hash recurrence between training and test,
- seen-attack recall separately for predictor hashes that were and were not
  previously present in training,
- per-attack recall,
- score-distribution quantiles,
- predefined threshold sensitivity.

OUTPUT LOCATION (different from all prior Comment-2 outputs)
-------------------------------------------------------------
artifacts/generalization/rolling_origin_temporal_validation

Expected artifacts:
  run_manifest.json
  cse_source_metadata.json
  temporal_blocks.csv
  temporal_attack_coverage.csv
  rolling_origin_fold_metrics.csv
  rolling_origin_per_attack.csv
  rolling_origin_threshold_sensitivity.csv
  rolling_origin_score_diagnostics.csv
  rolling_origin_fold_details.json
  reviewer_ready_temporal_summary.json
  Reviewer_Concern2_RollingOrigin_Temporal_Report.md

IMPORTANT INTERPRETATION RULE
-----------------------------
This script diagnoses and measures temporal generalization; it is not designed
to manufacture a favorable value. If seen-attack temporal recall is weak, that
result must remain visible and should motivate a real temporal-adaptation method
rather than post-hoc test tuning.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from scipy import sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# Fixed reviewer-facing configuration
# =============================================================================
DEFAULT_CSE_RAW_CSV = Path(
    os.getenv("FLBCIDS_CSE_RAW_CSV", 'data/raw/CSE-CIC-IDS2018/CSECICIDS2018Dataset.csv')
)
DEFAULT_OUTPUT_ROOT = Path(
    os.getenv("FLBCIDS_ROLLING_TEMPORAL_RESULTS_DIR", "artifacts/generalization/rolling_origin_temporal_validation")
)

# Exact source identity already used by the completed reviewer experiments.
EXPECTED_CSE_SOURCE_SHA256 = (
    "4335539845e880b1fb06703b5a68da0a03ed0682204bdda0863ddfc316782e3c"
)
EXPECTED_CSE_SOURCE_ROWS = 63_195_145

CSE_LABEL_COL = "Label"
CSE_IDENTIFIER_COLS = ["id", "Flow ID", "Src IP", "Dst IP", "Timestamp"]

RANDOM_STATE = 42
CSE_MAX_MODEL_ROWS: Optional[int] = 3_000_000

# Fixed temporal protocol. Do not tune after seeing results.
N_TEMPORAL_BLOCKS = 10
FIRST_TEST_BLOCK = 6       # 1-based block number
LAST_TEST_BLOCK = 10       # 1-based block number
PRIMARY_TARGET_FPR = 0.01
FPR_SENSITIVITY = (0.001, 0.005, 0.01, 0.02, 0.05)

# Same XGBoost family as the prior Comment-2 corrective harness and Comment-1
# centralized baseline family.
XGB_COMMON: Dict[str, Any] = {
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.2,
    "min_child_weight": 500,
    "subsample": 0.2,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 1.0,
    "max_delta_step": 1.0,
    "seed": RANDOM_STATE,
    "verbosity": 0,
    "nthread": max(1, (os.cpu_count() or 2) - 1),
}
NUM_BOOST_ROUND = 100
BALANCE_BINARY_TRAIN = True
MAX_ROWS_FOR_OVERSAMPLING = 3_000_000

# Quantiles used only for diagnosis of score drift.
SCORE_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


# =============================================================================
# Generic helpers
# =============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def versions() -> Dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "joblib": joblib.__version__,
    }

def format_optional_float(
    value: Any,
    digits: int = 6,
) -> str:
    if value is None:
        return "NA"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"

    if not np.isfinite(numeric):
        return "NA"

    return f"{numeric:.{digits}f}"

def normalize_label_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )


def parse_cse_timestamp(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)


def hash_frame_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    return pd.util.hash_pandas_object(
        df[list(feature_cols)], index=False
    ).to_numpy(dtype=np.uint64, copy=False)


def make_onehot_encoder() -> OneHotEncoder:
    # Current project environment is sklearn 1.7.x, where sparse_output exists.
    return OneHotEncoder(handle_unknown="ignore", sparse_output=True)


def to_float32_matrix(X: Any) -> Any:
    if sp.issparse(X):
        return sp.csr_matrix(X, dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def build_preprocessor(train_x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = train_x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in train_x.columns if c not in numeric_cols]

    transformers: List[Tuple[str, Any, List[str]]] = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ],
                    memory=None,
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_encoder()),
                    ],
                    memory=None,
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable predictor columns found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def maybe_balance_train(X: Any, y: np.ndarray) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    y = np.asarray(y, dtype=np.int8)
    counts = np.bincount(y, minlength=2)
    info: Dict[str, Any] = {
        "enabled": BALANCE_BINARY_TRAIN,
        "before": {"benign": int(counts[0]), "attack": int(counts[1])},
    }

    if not BALANCE_BINARY_TRAIN:
        info.update({"performed": False, "reason": "disabled"})
        return X, y, info
    if len(y) > MAX_ROWS_FOR_OVERSAMPLING:
        info.update({"performed": False, "reason": "row_cap"})
        return X, y, info
    if counts.min() == 0 or counts[0] == counts[1]:
        info.update({"performed": False, "reason": "single_class_or_balanced"})
        return X, y, info

    ros = RandomOverSampler(sampling_strategy=1.0, random_state=RANDOM_STATE)
    X2, y2 = ros.fit_resample(X, y)
    y2 = np.asarray(y2, dtype=np.int8)
    after = np.bincount(y2.astype(int), minlength=2)
    info.update(
        {
            "performed": True,
            "after": {"benign": int(after[0]), "attack": int(after[1])},
        }
    )
    return X2, y2, info


# =============================================================================
# Model fitting and metric helpers
# =============================================================================
class DetectorBundle:
    def __init__(
        self,
        preprocessor: ColumnTransformer,
        booster: xgb.Booster,
        balance_info: Dict[str, Any],
        fit_seconds: float,
    ) -> None:
        self.preprocessor = preprocessor
        self.booster = booster
        self.balance_info = balance_info
        self.fit_seconds = fit_seconds


def fit_detector(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "LabelBinary",
) -> DetectorBundle:
    y_train = train_df[label_col].to_numpy(dtype=np.int8, copy=False)
    if np.unique(y_train).size != 2:
        raise ValueError("Training history must contain both benign and attack rows.")

    pre = build_preprocessor(train_df[list(feature_cols)])
    X_train = to_float32_matrix(pre.fit_transform(train_df[list(feature_cols)]))

    X_bal, y_bal, balance_info = maybe_balance_train(X_train, y_train)
    X_bal = to_float32_matrix(X_bal)
    counts = np.bincount(y_bal.astype(int), minlength=2)

    params = dict(XGB_COMMON)
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": (
                1.0
                if balance_info.get("performed")
                else float(counts[0] / max(1, counts[1]))
            ),
            "base_score": float(np.mean(y_bal)),
        }
    )

    t0 = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=xgb.DMatrix(X_bal, label=y_bal),
        num_boost_round=NUM_BOOST_ROUND,
        verbose_eval=False,
    )
    fit_seconds = time.perf_counter() - t0

    return DetectorBundle(pre, booster, balance_info, fit_seconds)


def transform(bundle: DetectorBundle, df: pd.DataFrame, feature_cols: Sequence[str]) -> Any:
    return to_float32_matrix(bundle.preprocessor.transform(df[list(feature_cols)]))


def supervised_scores(bundle: DetectorBundle, X: Any) -> np.ndarray:
    return np.asarray(bundle.booster.predict(xgb.DMatrix(X)), dtype=np.float64)


def threshold_for_empirical_fpr(benign_scores: np.ndarray, target_fpr: float) -> float:
    """Threshold chosen ONLY from benign calibration scores.

    Classification convention: score >= threshold => ATTACK.
    The returned threshold targets empirical benign FPR <= target_fpr.
    """
    x = np.asarray(benign_scores, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite benign calibration scores available.")
    if not (0.0 < target_fpr < 1.0):
        raise ValueError("target_fpr must lie strictly between 0 and 1.")

    desc = np.sort(x)[::-1]
    allowed_fp = int(math.floor(target_fpr * len(desc)))

    if allowed_fp <= 0:
        return float(np.nextafter(desc[0], np.inf))
    if allowed_fp >= len(desc):
        return float(np.nextafter(desc[-1], -np.inf))

    return float(np.nextafter(desc[allowed_fp], np.inf))


def validation_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    y = np.asarray(y_true, dtype=np.int8)
    if np.unique(y).size < 2:
        return None
    thresholds = np.linspace(0.01, 0.99, 199)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        pred = (scores >= thr).astype(np.int8)
        value = float(f1_score(y, pred, zero_division=0))
        if value > best_f1:
            best_f1 = value
            best_thr = float(thr)
    return best_thr


def safe_auc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=np.int8)
    if len(y) == 0 or np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, np.asarray(score, dtype=np.float64)))


def metrics_from_predictions(
    y_true: np.ndarray,
    pred: np.ndarray,
    score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(pred, dtype=np.int8)

    if len(y) == 0:
        return {
            "n": 0,
            "benign_n": 0,
            "attack_n": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "balanced_accuracy": None,
            "mcc": None,
            "fpr": None,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "roc_auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    benign_den = fp + tn

    result: Dict[str, Any] = {
        "n": int(len(y)),
        "benign_n": int((y == 0).sum()),
        "attack_n": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, p)) if np.unique(y).size == 2 else None
        ),
        "mcc": float(matthews_corrcoef(y, p)) if np.unique(y).size == 2 else None,
        "fpr": float(fp / benign_den) if benign_den > 0 else None,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": cm.tolist(),
        "roc_auc": safe_auc(y, score) if score is not None else None,
    }
    return result


def metrics_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, Any]:
    score = np.asarray(score, dtype=np.float64)
    pred = (score >= threshold).astype(np.int8)
    out = metrics_from_predictions(y, pred, score)
    out["threshold"] = float(threshold)
    return out


def recall_on_attack_only(y: np.ndarray, score: np.ndarray, threshold: float) -> Optional[float]:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    idx = np.flatnonzero(y == 1)
    if idx.size == 0:
        return None
    return float(np.mean(score[idx] >= threshold))


def score_quantiles(values: np.ndarray) -> Dict[str, Optional[float]]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(q * 100):02d}": None for q in SCORE_QUANTILES}
    return {
        f"q{int(q * 100):02d}": float(np.quantile(x, q))
        for q in SCORE_QUANTILES
    }


# =============================================================================
# CSE loading — aligned with prior reviewer / Comment-1 source contract
# =============================================================================
def load_cse_temporal_sample(
    path: Path,
    row_cap: Optional[int],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"  [CSE] Source file: {path}", flush=True)
    print("  [CSE] Verifying SHA-256 against the reviewer-run source...", flush=True)
    source_sha = sha256_file(path)
    if source_sha.lower() != EXPECTED_CSE_SOURCE_SHA256.lower():
        raise RuntimeError(
            "CSE source SHA-256 does not match the source used by the reviewer experiments.\n"
            f"Expected: {EXPECTED_CSE_SOURCE_SHA256}\n"
            f"Observed: {source_sha}\n"
            "Stop rather than silently evaluating a different dataset file."
        )
    print(f"  [CSE] SHA-256 verified: {source_sha}", flush=True)

    if row_cap is None:
        print("  [CSE] Loading the complete source into pandas...", flush=True)
        df = pd.read_csv(path, low_memory=False)
        sampling_fraction = None
    else:
        import dask.dataframe as dd

        sampling_fraction = min(1.0, float(row_cap) / float(EXPECTED_CSE_SOURCE_ROWS))
        print("  [CSE] Opening source with Dask...", flush=True)
        ddf = dd.read_csv(path, assume_missing=True, blocksize="128MB")

        if EXPECTED_CSE_SOURCE_ROWS > row_cap:
            print(
                f"  [CSE] Deterministic source-wide sampling toward {row_cap:,} rows "
                f"(fraction={sampling_fraction:.9f}, seed={RANDOM_STATE})...",
                flush=True,
            )
            ddf = ddf.sample(frac=sampling_fraction, random_state=RANDOM_STATE)

        print("  [CSE] Materializing sampled rows...", flush=True)
        df = ddf.compute()
        print(f"  [CSE] Materialized {len(df):,} rows.", flush=True)

        # Preserve the same deterministic cap behavior used by the prior harness
        # if Dask's Bernoulli sample lands slightly above the target.
        if len(df) > row_cap:
            print(
                f"  [CSE] Trimming {len(df):,} sampled rows to exactly {row_cap:,} "
                f"with pandas sample(seed={RANDOM_STATE})...",
                flush=True,
            )
            df = df.sample(n=row_cap, random_state=RANDOM_STATE, replace=False)

    df = df.reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    required = {CSE_LABEL_COL, "Timestamp"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required CSE columns: {missing}")

    df["AttackType"] = normalize_label_series(df[CSE_LABEL_COL])
    df["LabelBinary"] = (df["AttackType"] != "BENIGN").astype(np.int8)

    ts = parse_cse_timestamp(df["Timestamp"])

    valid_ts_mask = ts.notna().to_numpy(
        dtype=bool,
        copy=False,
    )

    nat_count = int(
        np.count_nonzero(~valid_ts_mask)
    )

    df = df.loc[valid_ts_mask].copy()

    df["_timestamp_parsed"] = (
        ts.to_numpy()[valid_ts_mask]
    )

    # Predictor contract matches the previous corrective script: remove obvious
    # identifiers and preserve the 85 model predictors.
    excluded = set(CSE_IDENTIFIER_COLS + [CSE_LABEL_COL, "AttackType", "LabelBinary", "_timestamp_parsed"])
    feature_cols = [c for c in df.columns if c not in excluded]

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    print(f"  [CSE] Sorting {len(df):,} timestamp-valid rows chronologically...", flush=True)
    df = df.sort_values("_timestamp_parsed", kind="mergesort").reset_index(drop=True)

    print(f"  [CSE] Computing predictor hashes for {len(df):,} rows...", flush=True)
    df["feature_hash"] = hash_frame_features(df, feature_cols)

    label_values = df["LabelBinary"].to_numpy(
        dtype=np.int8,
        copy=False,
    )
    meta = {
        "path": str(path),
        "sha256": source_sha,
        "expected_source_rows": EXPECTED_CSE_SOURCE_ROWS,
        "row_cap": row_cap,
        "sampling_fraction": sampling_fraction,
        "sampled_rows_after_cap": int(len(df) + nat_count),
        "timestamp_valid_rows": int(len(df)),
        "timestamp_nat_rows_removed": nat_count,
        "feature_count": int(len(feature_cols)),
        "first_timestamp": str(df["_timestamp_parsed"].min()),
        "last_timestamp": str(df["_timestamp_parsed"].max()),
        "benign_rows": int(
            np.count_nonzero(label_values == 0)
        ),
        "attack_rows": int(
            np.count_nonzero(label_values == 1)
        ),
        "attack_types": sorted(
            df.loc[df["LabelBinary"] == 1, "AttackType"].astype(str).unique().tolist()
        ),
    }
    return df, feature_cols, meta


# =============================================================================
# Timestamp-safe block construction
# =============================================================================
def make_timestamp_safe_blocks(
    sorted_df: pd.DataFrame,
    n_blocks: int = N_TEMPORAL_BLOCKS,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Assign approximately equal-row contiguous blocks without splitting a timestamp."""
    if n_blocks < 3:
        raise ValueError("At least three temporal blocks are required.")
    if len(sorted_df) < n_blocks:
        raise ValueError("Not enough rows to form temporal blocks.")

    ts = np.asarray(
        sorted_df["_timestamp_parsed"].to_numpy(),
        dtype="datetime64[ns]",
    ).reshape(-1)
    if len(ts) > 1 and np.any(ts[1:] < ts[:-1]):
        raise AssertionError("Input must be sorted chronologically before block construction.")

    # Valid cut positions are starts of new timestamp groups.
    group_change = np.flatnonzero(ts[1:] != ts[:-1]) + 1
    valid_cuts = np.concatenate(([0], group_change, [len(sorted_df)])).astype(np.int64)

    # Choose the closest valid timestamp boundary to each row-count target while
    # enforcing strictly increasing cuts. This uses no labels or model results.
    internal_cuts: List[int] = []
    previous = 0
    for k in range(1, n_blocks):
        target = int(round(k * len(sorted_df) / n_blocks))
        pos = int(np.searchsorted(valid_cuts, target, side="left"))
        candidates: List[int] = []
        if pos < len(valid_cuts):
            candidates.append(int(valid_cuts[pos]))
        if pos > 0:
            candidates.append(int(valid_cuts[pos - 1]))
        candidates = [c for c in candidates if previous < c < len(sorted_df)]
        if not candidates:
            future = valid_cuts[(valid_cuts > previous) & (valid_cuts < len(sorted_df))]
            if future.size == 0:
                raise ValueError("Unable to form the requested timestamp-safe blocks.")
            chosen = int(future[0])
        else:
            chosen = min(
                candidates,
                key=lambda c, target=target: (
                    abs(c - target),
                    c,
                ),
            )

        # Prevent duplicate cuts if one very large timestamp group spans targets.
        if chosen <= previous:
            future = valid_cuts[(valid_cuts > previous) & (valid_cuts < len(sorted_df))]
            if future.size == 0:
                raise ValueError("Too few distinct timestamp groups for requested blocks.")
            chosen = int(future[0])

        internal_cuts.append(chosen)
        previous = chosen

    cuts = np.asarray([0] + internal_cuts + [len(sorted_df)], dtype=np.int64)
    if np.any(np.diff(cuts) <= 0):
        raise AssertionError(f"Non-increasing temporal cuts: {cuts.tolist()}")
    if len(cuts) != n_blocks + 1:
        raise AssertionError("Unexpected number of temporal cuts.")

    block_ids = np.empty(len(sorted_df), dtype=np.int16)
    rows: List[Dict[str, Any]] = []

    for block_idx in range(n_blocks):
        start = int(cuts[block_idx])
        stop = int(cuts[block_idx + 1])
        block_id = block_idx + 1
        block_ids[start:stop] = block_id

        b = sorted_df.iloc[start:stop]

        block_labels = b["LabelBinary"].to_numpy(
            dtype=np.int8,
            copy=False,
        )

        attack = b.loc[
            block_labels == 1,
            "AttackType",
        ].astype(str)

        rows.append(
            {
                "block": block_id,
                "start_row": start,
                "stop_row_exclusive": stop,
                "rows": int(len(b)),
                "benign_rows": int(
                    np.count_nonzero(block_labels == 0)
                ),
                "attack_rows": int(
                    np.count_nonzero(block_labels == 1)
                ),
                "attack_type_count": int(attack.nunique()),
                "attack_types": " | ".join(sorted(attack.unique().tolist())),
                "start_timestamp": str(b["_timestamp_parsed"].min()),
                "end_timestamp": str(b["_timestamp_parsed"].max()),
                "distinct_timestamps": int(b["_timestamp_parsed"].nunique()),
            }
        )

    # Strong invariant: adjacent blocks cannot share any identical timestamp.
    for block_id in range(1, n_blocks):
        left = ts[
            int(cuts[block_id] - 1)
        ]
        right = ts[
            int(cuts[block_id])
        ]

        if left >= right:
            raise AssertionError(
                f"Timestamp-safe boundary violated between "
                f"B{block_id} and B{block_id + 1}: "
                f"{left!r} vs {right!r}"
            )

    return block_ids, pd.DataFrame(rows)


def build_attack_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    attack_df = df.loc[df["LabelBinary"] == 1].copy()
    rows: List[Dict[str, Any]] = []
    for attack_type, g in attack_df.groupby("AttackType", sort=True):
        block_counts = g["_temporal_block"].value_counts().sort_index()
        row: Dict[str, Any] = {
            "attack_type": str(attack_type),
            "total_rows": int(len(g)),
            "first_timestamp": str(g["_timestamp_parsed"].min()),
            "last_timestamp": str(g["_timestamp_parsed"].max()),
            "first_block": int(g["_temporal_block"].min()),
            "last_block": int(g["_temporal_block"].max()),
            "blocks_present": int(g["_temporal_block"].nunique()),
        }
        for b in range(1, N_TEMPORAL_BLOCKS + 1):
            row[f"B{b}_rows"] = int(block_counts.get(b, 0))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Fold evaluation
# =============================================================================
def _subset_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    return metrics_at_threshold(y, scores, threshold)


def _per_attack_rows(
    fold_id: int,
    test: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    train_attack_types: set[str],
    train_hashes: set[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    attacks = test.loc[test["LabelBinary"] == 1].copy()
    attack_positions = np.flatnonzero(test["LabelBinary"].to_numpy(dtype=np.int8) == 1)
    attacks["_score"] = scores[attack_positions]

    for attack_type, g in attacks.groupby("AttackType", sort=True):
        s = np.asarray(
            g["_score"].to_numpy(),
            dtype=np.float64,
        ).reshape(-1)

        hashes = g["feature_hash"].to_numpy(
            dtype=np.uint64,
            copy=False,
        )

        seen_in_training = (
                str(attack_type) in train_attack_types
        )

        hash_seen_mask = np.asarray(
            [
                int(h) in train_hashes
                for h in hashes
            ],
            dtype=bool,
        )

        quantiles = score_quantiles(s)

        rows.append(
            {
                "fold": fold_id,
                "attack_type": str(attack_type),
                "endpoint": "seen_temporal" if seen_in_training else "novel_temporal",
                "rows": int(len(g)),
                "recall": float(np.mean(s >= threshold)) if len(s) else None,
                "hash_seen_in_train_rows": int(hash_seen_mask.sum()),
                "hash_novel_to_train_rows": int((~hash_seen_mask).sum()),
                "hash_seen_recall": (
                    float(np.mean(s[hash_seen_mask] >= threshold)) if hash_seen_mask.any() else None
                ),
                "hash_novel_recall": (
                    float(np.mean(s[~hash_seen_mask] >= threshold)) if (~hash_seen_mask).any() else None
                ),
                "score_q10": quantiles["q10"],
                "score_median": quantiles["q50"],
                "score_q90": quantiles["q90"],
            }
        )
    return rows


def evaluate_fold(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    test_block: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    fold_id = test_block - FIRST_TEST_BLOCK + 1
    calibration_block = test_block - 1
    train_last_block = calibration_block - 1

    train = df.loc[df["_temporal_block"] <= train_last_block].copy()
    cal = df.loc[df["_temporal_block"] == calibration_block].copy()
    test = df.loc[df["_temporal_block"] == test_block].copy()

    print(
        f"\n[FOLD {fold_id}/5] train B1-B{train_last_block} | "
        f"calibrate B{calibration_block} | test B{test_block}",
        flush=True,
    )
    print(
        f"  rows: train={len(train):,}, calibration={len(cal):,}, test={len(test):,}",
        flush=True,
    )

    detail: Dict[str, Any] = {
        "fold": fold_id,
        "train_blocks": f"B1-B{train_last_block}",
        "calibration_block": f"B{calibration_block}",
        "test_block": f"B{test_block}",
        "train_rows": int(len(train)),
        "calibration_rows": int(len(cal)),
        "test_rows": int(len(test)),
        "train_time_range": [
            str(train["_timestamp_parsed"].min()),
            str(train["_timestamp_parsed"].max()),
        ],
        "calibration_time_range": [
            str(cal["_timestamp_parsed"].min()),
            str(cal["_timestamp_parsed"].max()),
        ],
        "test_time_range": [
            str(test["_timestamp_parsed"].min()),
            str(test["_timestamp_parsed"].max()),
        ],
    }

    # Strict chronological ordering invariant.
    if not (
        train["_timestamp_parsed"].max()
        < cal["_timestamp_parsed"].min()
        <= cal["_timestamp_parsed"].max()
        < test["_timestamp_parsed"].min()
    ):
        raise AssertionError(f"Fold {fold_id}: chronological ordering invariant failed.")

    train_y = train["LabelBinary"].to_numpy(dtype=np.int8, copy=False)
    cal_y = cal["LabelBinary"].to_numpy(dtype=np.int8, copy=False)
    test_y = test["LabelBinary"].to_numpy(dtype=np.int8, copy=False)

    detail["train_class_counts"] = {
        "benign": int((train_y == 0).sum()),
        "attack": int((train_y == 1).sum()),
    }
    detail["calibration_class_counts"] = {
        "benign": int((cal_y == 0).sum()),
        "attack": int((cal_y == 1).sum()),
    }
    detail["test_class_counts"] = {
        "benign": int((test_y == 0).sum()),
        "attack": int((test_y == 1).sum()),
    }

    if np.unique(train_y).size != 2:
        reason = "training_history_single_class"
        detail.update({"status": "ineligible", "reason": reason})
        ineligible_row = {
            "fold": fold_id,
            "train_blocks": f"B1-B{train_last_block}",
            "calibration_block": calibration_block,
            "test_block": test_block,
            "train_rows": int(len(train)),
            "calibration_rows": int(len(cal)),
            "test_rows": int(len(test)),
            "status": "ineligible",
            "exclusion_reason": reason,
        }
        return detail, [ineligible_row], [], []
    if int((cal_y == 0).sum()) == 0:
        reason = "calibration_block_has_no_benign_rows"
        detail.update({"status": "ineligible", "reason": reason})
        ineligible_row = {
            "fold": fold_id,
            "train_blocks": f"B1-B{train_last_block}",
            "calibration_block": calibration_block,
            "test_block": test_block,
            "train_rows": int(len(train)),
            "calibration_rows": int(len(cal)),
            "test_rows": int(len(test)),
            "status": "ineligible",
            "exclusion_reason": reason,
        }
        return detail, [ineligible_row], [], []

    train_attack_types: set[str] = set(
        train.loc[train["LabelBinary"] == 1, "AttackType"].astype(str).unique().tolist()
    )
    test_attack_types: set[str] = set(
        test.loc[test["LabelBinary"] == 1, "AttackType"].astype(str).unique().tolist()
    )
    seen_test_types = sorted(test_attack_types & train_attack_types)
    novel_test_types = sorted(test_attack_types - train_attack_types)

    test_attack_type_series = test["AttackType"].astype(str)
    seen_attack_mask = (
        (test["LabelBinary"].to_numpy(dtype=np.int8) == 1)
        & test_attack_type_series.isin(train_attack_types).to_numpy(dtype=bool)
    )
    novel_attack_mask = (
        (test["LabelBinary"].to_numpy(dtype=np.int8) == 1)
        & ~test_attack_type_series.isin(train_attack_types).to_numpy(dtype=bool)
    )
    benign_mask = test["LabelBinary"].to_numpy(dtype=np.int8) == 0

    detail["train_attack_types"] = sorted(train_attack_types)
    detail["test_attack_types"] = sorted(test_attack_types)
    detail["seen_test_attack_types"] = seen_test_types
    detail["novel_test_attack_types"] = novel_test_types
    detail["seen_attack_rows"] = int(seen_attack_mask.sum())
    detail["novel_attack_rows"] = int(novel_attack_mask.sum())

    print(
        f"  attack coverage: seen_test_rows={int(seen_attack_mask.sum()):,} "
        f"({len(seen_test_types)} types), novel_test_rows={int(novel_attack_mask.sum()):,} "
        f"({len(novel_test_types)} types)",
        flush=True,
    )

    print("  fitting train-only preprocessing + XGBoost...", flush=True)
    bundle = fit_detector(train, feature_cols, "LabelBinary")
    print(f"  model fit completed in {bundle.fit_seconds:.3f} s", flush=True)

    X_cal = transform(bundle, cal, feature_cols)
    X_test = transform(bundle, test, feature_cols)
    cal_scores = supervised_scores(bundle, X_cal)
    test_scores = supervised_scores(bundle, X_test)

    primary_thr = threshold_for_empirical_fpr(
        cal_scores[cal_y == 0], PRIMARY_TARGET_FPR
    )
    f1_thr = validation_f1_threshold(cal_y, cal_scores)

    detail["status"] = "evaluated"
    detail["primary_threshold"] = float(primary_thr)
    detail["validation_f1_threshold"] = float(f1_thr) if f1_thr is not None else None
    detail["fit_seconds"] = float(bundle.fit_seconds)
    detail["balance"] = bundle.balance_info

    # Primary seen-temporal endpoint = all benign test rows + seen attack rows.
    seen_eval_mask = benign_mask | seen_attack_mask
    novel_eval_mask = benign_mask | novel_attack_mask

    primary_seen = _subset_metrics(
        test_y[seen_eval_mask], test_scores[seen_eval_mask], primary_thr
    )
    combined = _subset_metrics(test_y, test_scores, primary_thr)
    novel_diag = _subset_metrics(
        test_y[novel_eval_mask], test_scores[novel_eval_mask], primary_thr
    )

    if f1_thr is not None:
        f1_seen = _subset_metrics(
            test_y[seen_eval_mask], test_scores[seen_eval_mask], f1_thr
        )
    else:
        f1_seen = None

    # Predictor recurrence diagnostics. This does not change primary membership.
    train_hashes: set[int] = set(map(int, train["feature_hash"].unique()))
    test_hashes = test["feature_hash"].to_numpy(dtype=np.uint64, copy=False)
    test_hash_seen = np.asarray([int(h) in train_hashes for h in test_hashes], dtype=bool)
    seen_hash_seen = seen_attack_mask & test_hash_seen
    seen_hash_novel = seen_attack_mask & ~test_hash_seen

    detail["predictor_hash_diagnostics"] = {
        "test_rows_hash_seen_in_train": int(test_hash_seen.sum()),
        "test_rows_hash_novel_to_train": int((~test_hash_seen).sum()),
        "seen_attack_rows_hash_seen_in_train": int(seen_hash_seen.sum()),
        "seen_attack_rows_hash_novel_to_train": int(seen_hash_novel.sum()),
        "seen_attack_recall_hash_seen_in_train": (
            float(np.mean(test_scores[seen_hash_seen] >= primary_thr))
            if seen_hash_seen.any()
            else None
        ),
        "seen_attack_recall_hash_novel_to_train": (
            float(np.mean(test_scores[seen_hash_novel] >= primary_thr))
            if seen_hash_novel.any()
            else None
        ),
    }

    detail["primary_seen_attack_temporal"] = primary_seen
    detail["combined_test_period"] = combined
    detail["novel_attack_with_benign_diagnostic"] = novel_diag
    detail["validation_f1_seen_attack_temporal_comparator"] = f1_seen

    # Score-distribution drift diagnostics.
    score_rows: List[Dict[str, Any]] = []
    categories = {
        "calibration_benign": cal_scores[cal_y == 0],
        "calibration_attack": cal_scores[cal_y == 1],
        "test_benign": test_scores[benign_mask],
        "test_seen_attack": test_scores[seen_attack_mask],
        "test_novel_attack": test_scores[novel_attack_mask],
    }
    for category, values in categories.items():
        q = score_quantiles(values)
        score_rows.append(
            {
                "fold": fold_id,
                "category": category,
                "n": int(len(values)),
                **q,
            }
        )

    # Predeclared threshold sensitivity. All target FPRs use calibration benign
    # scores only; no test-dependent threshold selection is performed.
    sensitivity_rows: List[Dict[str, Any]] = []
    for target in FPR_SENSITIVITY:
        thr = threshold_for_empirical_fpr(cal_scores[cal_y == 0], target)
        seen_metrics = _subset_metrics(
            test_y[seen_eval_mask], test_scores[seen_eval_mask], thr
        )
        combined_metrics = _subset_metrics(test_y, test_scores, thr)
        sensitivity_rows.append(
            {
                "fold": fold_id,
                "target_development_fpr": float(target),
                "threshold": float(thr),
                "seen_attack_rows": int(seen_attack_mask.sum()),
                "seen_temporal_recall": seen_metrics.get("recall"),
                "seen_temporal_precision": seen_metrics.get("precision"),
                "seen_temporal_f1": seen_metrics.get("f1"),
                "seen_temporal_test_fpr": seen_metrics.get("fpr"),
                "seen_temporal_auc": seen_metrics.get("roc_auc"),
                "combined_recall": combined_metrics.get("recall"),
                "combined_test_fpr": combined_metrics.get("fpr"),
            }
        )

    per_attack_rows = _per_attack_rows(
        fold_id,
        test,
        test_scores,
        primary_thr,
        train_attack_types,
        train_hashes,
    )

    fold_row = {
        "fold": fold_id,
        "train_blocks": f"B1-B{train_last_block}",
        "calibration_block": calibration_block,
        "test_block": test_block,
        "train_rows": int(len(train)),
        "calibration_rows": int(len(cal)),
        "test_rows": int(len(test)),
        "train_attack_type_count": int(len(train_attack_types)),
        "seen_test_attack_type_count": int(len(seen_test_types)),
        "novel_test_attack_type_count": int(len(novel_test_types)),
        "seen_attack_rows": int(seen_attack_mask.sum()),
        "novel_attack_rows": int(novel_attack_mask.sum()),
        "primary_threshold": float(primary_thr),
        "seen_temporal_accuracy": primary_seen.get("accuracy"),
        "seen_temporal_precision": primary_seen.get("precision"),
        "seen_temporal_recall": primary_seen.get("recall"),
        "seen_temporal_f1": primary_seen.get("f1"),
        "seen_temporal_balanced_accuracy": primary_seen.get("balanced_accuracy"),
        "seen_temporal_mcc": primary_seen.get("mcc"),
        "seen_temporal_test_fpr": primary_seen.get("fpr"),
        "seen_temporal_auc": primary_seen.get("roc_auc"),
        "seen_temporal_tn": primary_seen.get("tn"),
        "seen_temporal_fp": primary_seen.get("fp"),
        "seen_temporal_fn": primary_seen.get("fn"),
        "seen_temporal_tp": primary_seen.get("tp"),
        "combined_recall": combined.get("recall"),
        "combined_test_fpr": combined.get("fpr"),
        "combined_auc": combined.get("roc_auc"),
        "novel_diagnostic_recall": novel_diag.get("recall"),
        "novel_diagnostic_test_fpr": novel_diag.get("fpr"),
        "seen_attack_rows_hash_seen_in_train": int(seen_hash_seen.sum()),
        "seen_attack_rows_hash_novel_to_train": int(seen_hash_novel.sum()),
        "seen_attack_recall_hash_seen_in_train": detail["predictor_hash_diagnostics"][
            "seen_attack_recall_hash_seen_in_train"
        ],
        "seen_attack_recall_hash_novel_to_train": detail["predictor_hash_diagnostics"][
            "seen_attack_recall_hash_novel_to_train"
        ],
        "fit_seconds": float(bundle.fit_seconds),
        "status": "eligible" if int(seen_attack_mask.sum()) > 0 else "no_seen_attack_rows",
    }

    # We preserve the fold result even when the primary seen endpoint has zero
    # attacks. Such a fold is simply not included in the pooled seen recall.
    detail["primary_seen_endpoint_eligible"] = bool(int(seen_attack_mask.sum()) > 0)
    if int(seen_attack_mask.sum()) == 0:
        detail["primary_seen_endpoint_exclusion_reason"] = "zero_seen_attack_rows_in_test_block"

    del train, cal, test, X_cal, X_test, bundle
    gc.collect()

    return detail, [fold_row], per_attack_rows, sensitivity_rows + score_rows


# =============================================================================
# Aggregation helpers
# =============================================================================
def pooled_binary_metrics_from_fold_rows(fold_df: pd.DataFrame) -> Dict[str, Any]:
    status_values = (
        fold_df["status"]
        .astype(str)
        .to_numpy()
    )

    seen_attack_rows = np.asarray(
        pd.to_numeric(
            fold_df["seen_attack_rows"],
            errors="coerce",
        ),
        dtype=np.float64,
    ).reshape(-1)

    seen_attack_rows = np.nan_to_num(
        seen_attack_rows,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    eligible_mask = (
            (status_values == "eligible")
            & (seen_attack_rows > 0)
    )

    eligible = fold_df.loc[
        eligible_mask
    ].copy()

    if eligible.empty:
        return {
            "eligible_folds": 0,
            "status": "no_eligible_seen_attack_temporal_folds",
        }

    tn = int(pd.to_numeric(eligible["seen_temporal_tn"]).sum())
    fp = int(pd.to_numeric(eligible["seen_temporal_fp"]).sum())
    fn = int(pd.to_numeric(eligible["seen_temporal_fn"]).sum())
    tp = int(pd.to_numeric(eligible["seen_temporal_tp"]).sum())

    total = tn + fp + fn + tp
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else None
    accuracy = float((tp + tn) / total) if total > 0 else None
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    balanced = (
        float((recall + specificity) / 2.0)
        if recall is not None and specificity is not None
        else None
    )

    recalls = np.asarray(
        pd.to_numeric(
            eligible["seen_temporal_recall"],
            errors="coerce",
        ),
        dtype=np.float64,
    ).reshape(-1)

    recalls = recalls[
        np.isfinite(recalls)
    ]

    f1s = np.asarray(
        pd.to_numeric(
            eligible["seen_temporal_f1"],
            errors="coerce",
        ),
        dtype=np.float64,
    ).reshape(-1)

    f1s = f1s[
        np.isfinite(f1s)
    ]

    fprs = np.asarray(
        pd.to_numeric(
            eligible["seen_temporal_test_fpr"],
            errors="coerce",
        ),
        dtype=np.float64,
    ).reshape(-1)

    fprs = fprs[
        np.isfinite(fprs)
    ]

    aucs = np.asarray(
        pd.to_numeric(
            eligible["seen_temporal_auc"],
            errors="coerce",
        ),
        dtype=np.float64,
    ).reshape(-1)

    aucs = aucs[
        np.isfinite(aucs)
    ]

    def stat(x: np.ndarray) -> Dict[str, Optional[float]]:
        if x.size == 0:
            return {"mean": None, "sd": None, "median": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
            "median": float(np.median(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    eligible_fold_ids = [
        int(value)
        for value in eligible["fold"].tolist()
    ]

    eligible_fold_count = len(
        eligible_fold_ids
    )
    return {
        "status": "ok",
        "eligible_folds": eligible_fold_count,
        "eligible_fold_ids": eligible_fold_ids,
        "pooled_rows": int(total),
        "pooled_benign_rows": int(tn + fp),
        "pooled_seen_attack_rows": int(tp + fn),
        "pooled_confusion_matrix": [[tn, fp], [fn, tp]],
        "pooled_accuracy": accuracy,
        "pooled_precision": precision,
        "pooled_recall": recall,
        "pooled_f1": f1,
        "pooled_fpr": fpr,
        "pooled_balanced_accuracy": balanced,
        "fold_recall": stat(recalls),
        "fold_f1": stat(f1s),
        "fold_test_fpr": stat(fprs),
        "fold_auc": stat(aucs),
    }


def aggregate_per_attack(per_attack_df: pd.DataFrame) -> pd.DataFrame:
    if per_attack_df.empty:
        return pd.DataFrame()

    seen = per_attack_df.loc[per_attack_df["endpoint"] == "seen_temporal"].copy()
    if seen.empty:
        return pd.DataFrame()

    # Row-weighted pooled recall can be reconstructed from recall * rows because
    # each per-fold attack row is either detected or missed.
    rows: List[Dict[str, Any]] = []
    for attack_type, g in seen.groupby("attack_type", sort=True):
        n = np.asarray(
            pd.to_numeric(
                g["rows"],
                errors="coerce",
            ),
            dtype=np.float64,
        ).reshape(-1)

        n = np.nan_to_num(
            n,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        r = np.asarray(
            pd.to_numeric(
                g["recall"],
                errors="coerce",
            ),
            dtype=np.float64,
        ).reshape(-1)
        valid = np.isfinite(r) & (n > 0)
        total_rows = int(n[valid].sum()) if valid.any() else 0
        pooled_recall = (
            float(np.sum(n[valid] * r[valid]) / np.sum(n[valid]))
            if valid.any() and np.sum(n[valid]) > 0
            else None
        )
        rows.append(
            {
                "attack_type": attack_type,
                "folds_present": int(g["fold"].nunique()),
                "total_seen_temporal_rows": total_rows,
                "pooled_recall": pooled_recall,
                "mean_fold_recall": float(np.nanmean(r)) if np.isfinite(r).any() else None,
                "min_fold_recall": float(np.nanmin(r)) if np.isfinite(r).any() else None,
                "max_fold_recall": float(np.nanmax(r)) if np.isfinite(r).any() else None,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    out_path: Path,
    source_meta: Dict[str, Any],
    fold_df: pd.DataFrame,
    pooled: Dict[str, Any],
    per_attack_agg: pd.DataFrame,
) -> None:
    def fmt(v: Any, digits: int = 6) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "NA"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{digits}f}"
        return str(v)

    lines: List[str] = [
        "# Reviewer Concern #2 — Rolling-Origin Temporal Validation",
        "",
        "## Why this experiment is separate",
        "",
        "The earlier terminal chronological split contained only attack types absent from its training interval, so it could not estimate seen-attack temporal generalization. This experiment creates a separate rolling-origin endpoint in which each test attack is classified as seen or novel strictly from the preceding training history.",
        "",
        "## Protocol integrity",
        "",
        f"- Exact CSE source SHA-256 verified: `{source_meta['sha256']}`.",
        f"- Deterministic source-wide capped sample: {source_meta['timestamp_valid_rows']:,} timestamp-valid rows.",
        "- Ten contiguous chronological blocks; identical timestamps never cross a block boundary.",
        "- Five predefined expanding-window folds: B1-B4/B5/B6 through B1-B8/B9/B10.",
        "- Preprocessing and XGBoost are fitted from training history only.",
        "- Primary operating threshold uses only benign scores from the immediately preceding calibration block and targets empirical development FPR <= 1%.",
        "- Test labels/scores never affect training, fold construction, threshold choice, or endpoint eligibility.",
        "- Seen-attack temporal rows and novel-attack rows are reported separately.",
        "",
        "## Primary seen-attack temporal endpoint",
        "",
    ]

    if pooled.get("status") == "ok":
        lines.extend(
            [
                f"- Eligible folds: {pooled.get('eligible_folds')} ({pooled.get('eligible_fold_ids')}).",
                f"- Pooled seen-attack rows: "
                f"{int(pooled.get('pooled_seen_attack_rows') or 0):,}.",
                f"- Pooled recall: **{fmt(pooled.get('pooled_recall'))}**.",
                f"- Pooled precision: {fmt(pooled.get('pooled_precision'))}.",
                f"- Pooled F1: {fmt(pooled.get('pooled_f1'))}.",
                f"- Pooled test FPR: {fmt(pooled.get('pooled_fpr'))}.",
                f"- Pooled balanced accuracy: {fmt(pooled.get('pooled_balanced_accuracy'))}.",
                f"- Fold-level recall mean ± SD: {fmt(pooled.get('fold_recall', {}).get('mean'))} ± {fmt(pooled.get('fold_recall', {}).get('sd'))}.",
                f"- Worst eligible-fold recall: {fmt(pooled.get('fold_recall', {}).get('min'))}.",
            ]
        )
    else:
        lines.append("No eligible seen-attack temporal fold was available under the predefined protocol.")

    lines.extend(["", "## Fold-level primary results", ""])
    if fold_df.empty:
        lines.append("No fold metrics were produced.")
    else:
        cols = [
            "fold", "train_blocks", "calibration_block", "test_block",
            "seen_test_attack_type_count", "seen_attack_rows", "novel_attack_rows",
            "seen_temporal_recall", "seen_temporal_precision", "seen_temporal_f1",
            "seen_temporal_test_fpr", "seen_temporal_auc", "status",
        ]
        view = fold_df[[c for c in cols if c in fold_df.columns]].copy()
        try:
            lines.append(view.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + view.to_csv(index=False) + "```")

    lines.extend(["", "## Seen attack-family temporal recall", ""])
    if per_attack_agg.empty:
        lines.append("No seen attack family was available in the rolling test folds.")
    else:
        try:
            lines.append(per_attack_agg.to_markdown(index=False))
        except Exception:
            lines.append("```csv\n" + per_attack_agg.to_csv(index=False) + "```")

    lines.extend(
        [
            "",
            "## Interpretation constraint",
            "",
            "The pooled seen-attack temporal result is the reviewer-facing temporal endpoint. Novel attacks remain a separate open-set problem. The previously reported terminal all-unseen interval must not be relabeled as seen-attack temporal generalization. Sensitivity rows are diagnostic only; the 1% development-FPR operating point remains primary regardless of which sensitivity threshold performs best on test data.",
        ]
    )

    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Self-test (no dataset required)
# =============================================================================
def run_self_test() -> int:
    print("Running synthetic self-test...", flush=True)

    # 1) Threshold invariant.
    benign = np.asarray(
        np.linspace(
            0.0,
            1.0,
            1000,
            dtype=np.float64,
        ),
        dtype=np.float64,
    ).reshape(-1)

    thr = threshold_for_empirical_fpr(
        benign,
        0.01,
    )
    empirical = float(np.mean(benign >= thr))
    assert empirical <= 0.01 + 1e-12, (thr, empirical)

    # 2) Timestamp-safe block invariant, including repeated timestamps near cuts.
    n = 10_000
    base = pd.Timestamp("2026-01-01")
    # Repeat each timestamp 7 times to force grouped boundaries.
    ticks = np.repeat(np.arange(math.ceil(n / 7)), 7)[:n]
    ts = pd.Series(base + pd.to_timedelta(ticks, unit="s"))
    labels = np.where(np.arange(n) % 11 == 0, "ATTACK-X", "BENIGN")
    synthetic = pd.DataFrame(
        {
            "Timestamp": ts.astype(str),
            "_timestamp_parsed": ts,
            "AttackType": labels,
            "LabelBinary": (labels != "BENIGN").astype(np.int8),
            "f1": np.arange(n, dtype=np.float64),
            "feature_hash": np.arange(n, dtype=np.uint64),
        }
    )
    block_ids, block_df = make_timestamp_safe_blocks(synthetic, 10)
    synthetic["_temporal_block"] = block_ids
    assert len(block_df) == 10
    for b in range(1, 10):
        left = synthetic.loc[synthetic["_temporal_block"] == b, "_timestamp_parsed"].max()
        right = synthetic.loc[synthetic["_temporal_block"] == b + 1, "_timestamp_parsed"].min()
        assert left < right

    # 3) Pooled confusion reconstruction.
    fake = pd.DataFrame(
        [
            {
                "fold": 1, "status": "eligible", "seen_attack_rows": 10,
                "seen_temporal_tn": 90, "seen_temporal_fp": 10,
                "seen_temporal_fn": 2, "seen_temporal_tp": 8,
                "seen_temporal_recall": 0.8, "seen_temporal_f1": 0.57142857,
                "seen_temporal_test_fpr": 0.1, "seen_temporal_auc": 0.9,
            },
            {
                "fold": 2, "status": "eligible", "seen_attack_rows": 10,
                "seen_temporal_tn": 95, "seen_temporal_fp": 5,
                "seen_temporal_fn": 1, "seen_temporal_tp": 9,
                "seen_temporal_recall": 0.9, "seen_temporal_f1": 0.75,
                "seen_temporal_test_fpr": 0.05, "seen_temporal_auc": 0.95,
            },
        ]
    )
    pooled = pooled_binary_metrics_from_fold_rows(fake)
    assert abs(float(pooled["pooled_recall"]) - 0.85) < 1e-12

    print("Synthetic self-test: PASS", flush=True)
    return 0


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reviewer Concern #2 rolling-origin seen-attack temporal validation"
    )
    parser.add_argument("--cse", default=str(DEFAULT_CSE_RAW_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--cse-row-cap",
        type=int,
        default=CSE_MAX_MODEL_ROWS,
        help="Reviewer run default is 3000000. Set 0 only for an explicitly planned full-source run.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic protocol checks without loading the dataset.",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    seed_everything(RANDOM_STATE)
    out_root = Path(args.out)
    ensure_dir(out_root)
    row_cap = None if int(args.cse_row_cap) == 0 else int(args.cse_row_cap)

    manifest: Dict[str, Any] = {
        "script": Path(__file__).name,
        "purpose": "Reviewer Concern #2 genuine seen-attack rolling-origin temporal validation",
        "versions": versions(),
        "random_state": RANDOM_STATE,
        "source_contract": {
            "expected_cse_sha256": EXPECTED_CSE_SOURCE_SHA256,
            "expected_cse_source_rows": EXPECTED_CSE_SOURCE_ROWS,
        },
        "sampling": {
            "cse_row_cap": row_cap,
            "rule": "source-wide Dask sample with fixed seed, then timestamp sort",
        },
        "temporal_protocol": {
            "n_blocks": N_TEMPORAL_BLOCKS,
            "block_rule": "approximately equal-row contiguous blocks; identical timestamps never split",
            "test_blocks": list(range(FIRST_TEST_BLOCK, LAST_TEST_BLOCK + 1)),
            "fold_rule": "expanding train history; immediately preceding block calibration; next block test",
            "primary_endpoint": "test benign rows plus attacks whose attack type already occurred in training history",
            "novel_attack_rule": "attack types absent from training history are reported separately",
        },
        "threshold_policy": {
            "primary_target_benign_development_fpr": PRIMARY_TARGET_FPR,
            "sensitivity_grid": list(FPR_SENSITIVITY),
            "selection_data": "immediately preceding calibration block only",
            "test_leakage": "none: no test label/score used for fitting or threshold selection",
        },
        "model": {
            "xgboost_params": XGB_COMMON,
            "num_boost_round": NUM_BOOST_ROUND,
            "balance_binary_train": BALANCE_BINARY_TRAIN,
        },
        "paths": {
            "cse": str(args.cse),
            "output": str(out_root),
        },
        "anti_cherry_picking": (
            "All five folds and all predefined FPR sensitivity targets are retained. "
            "The 1% development-FPR operating point is primary regardless of test performance."
        ),
    }
    save_json(out_root / "run_manifest.json", manifest)

    print("\n" + "=" * 108, flush=True)
    print("Reviewer Concern #2 — rolling-origin temporal validation started", flush=True)
    print(f"CSE source: {args.cse}", flush=True)
    print(f"Output root: {out_root}", flush=True)
    print(f"CSE row cap: {row_cap}", flush=True)
    print("Primary endpoint: seen-attack temporal recall at development-calibrated 1% benign FPR", flush=True)
    print("=" * 108, flush=True)

    print("\n[1/5] Loading, verifying, and chronologically ordering CSE-CIC-IDS2018...", flush=True)
    df, feature_cols, source_meta = load_cse_temporal_sample(Path(args.cse), row_cap)
    save_json(out_root / "cse_source_metadata.json", source_meta)
    print(
        f"[1/5] Prepared {len(df):,} timestamp-valid rows with {len(feature_cols)} predictors.",
        flush=True,
    )

    print("\n[2/5] Constructing 10 timestamp-safe temporal blocks...", flush=True)
    block_ids, block_df = make_timestamp_safe_blocks(df, N_TEMPORAL_BLOCKS)
    df["_temporal_block"] = block_ids
    block_df.to_csv(out_root / "temporal_blocks.csv", index=False)
    print(block_df[["block", "rows", "benign_rows", "attack_rows", "attack_type_count"]].to_string(index=False), flush=True)

    attack_coverage = build_attack_coverage_table(df)
    attack_coverage.to_csv(out_root / "temporal_attack_coverage.csv", index=False)

    print("\n[3/5] Executing the five predefined rolling-origin folds...", flush=True)
    fold_details: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    per_attack_rows: List[Dict[str, Any]] = []
    sensitivity_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []

    for test_block in range(FIRST_TEST_BLOCK, LAST_TEST_BLOCK + 1):
        detail, fold_part, attack_part, mixed_diag = evaluate_fold(
            df, feature_cols, test_block
        )
        fold_details.append(detail)
        fold_rows.extend(fold_part)
        per_attack_rows.extend(attack_part)

        # evaluate_fold returns sensitivity and score rows together to keep its
        # public signature compact; separate by the presence of the category key.
        for row in mixed_diag:
            if "category" in row:
                score_rows.append(row)
            else:
                sensitivity_rows.append(row)

    fold_df = pd.DataFrame.from_records(fold_rows)
    per_attack_df = pd.DataFrame.from_records(per_attack_rows)
    sensitivity_df = pd.DataFrame.from_records(sensitivity_rows)
    score_df = pd.DataFrame.from_records(score_rows)

    fold_df.to_csv(out_root / "rolling_origin_fold_metrics.csv", index=False)
    per_attack_df.to_csv(out_root / "rolling_origin_per_attack.csv", index=False)
    sensitivity_df.to_csv(out_root / "rolling_origin_threshold_sensitivity.csv", index=False)
    score_df.to_csv(out_root / "rolling_origin_score_diagnostics.csv", index=False)
    save_json(out_root / "rolling_origin_fold_details.json", fold_details)

    print("\n[4/5] Aggregating the primary seen-attack temporal endpoint...", flush=True)
    pooled = pooled_binary_metrics_from_fold_rows(fold_df)
    per_attack_agg = aggregate_per_attack(per_attack_df)
    per_attack_agg.to_csv(out_root / "rolling_origin_seen_attack_summary.csv", index=False)

    summary = {
        "protocol": manifest["temporal_protocol"],
        "threshold_policy": manifest["threshold_policy"],
        "source_metadata": source_meta,
        "primary_seen_attack_temporal": pooled,
        "fold_metrics": fold_df.to_dict(orient="records"),
        "seen_attack_family_summary": per_attack_agg.to_dict(orient="records"),
        "novel_attack_note": (
            "Novel attacks are retained in per-fold diagnostics but are excluded from the primary seen-attack temporal endpoint."
        ),
    }
    save_json(out_root / "reviewer_ready_temporal_summary.json", summary)

    write_report(
        out_root / "Reviewer_Concern2_RollingOrigin_Temporal_Report.md",
        source_meta,
        fold_df,
        pooled,
        per_attack_agg,
    )

    print("\n[5/5] Validation artifacts written.", flush=True)
    print("\n" + "=" * 108, flush=True)
    print("Reviewer Concern #2 — rolling-origin temporal validation completed", flush=True)
    print(f"Output root: {out_root}", flush=True)
    print("Primary artifacts:", flush=True)
    for rel in [
        "run_manifest.json",
        "cse_source_metadata.json",
        "temporal_blocks.csv",
        "temporal_attack_coverage.csv",
        "rolling_origin_fold_metrics.csv",
        "rolling_origin_per_attack.csv",
        "rolling_origin_seen_attack_summary.csv",
        "rolling_origin_threshold_sensitivity.csv",
        "rolling_origin_score_diagnostics.csv",
        "rolling_origin_fold_details.json",
        "reviewer_ready_temporal_summary.json",
        "Reviewer_Concern2_RollingOrigin_Temporal_Report.md",
    ]:
        print(f"  - {out_root / rel}", flush=True)

    if pooled.get("status") == "ok":
        print(
            "\nPRIMARY SEEN-ATTACK TEMPORAL RESULT",
            flush=True,
        )

        eligible_folds = int(
            pooled.get("eligible_folds") or 0
        )

        pooled_seen_attack_rows = int(
            pooled.get("pooled_seen_attack_rows") or 0
        )

        fold_recall_stats = pooled.get(
            "fold_recall"
        )

        if not isinstance(
                fold_recall_stats,
                dict,
        ):
            fold_recall_stats = {}

        print(
            f"  eligible folds: {eligible_folds}",
            flush=True,
        )

        print(
            f"  pooled seen attack rows: "
            f"{pooled_seen_attack_rows:,}",
            flush=True,
        )

        print(
            f"  pooled recall: "
            f"{format_optional_float(pooled.get('pooled_recall'))}",
            flush=True,
        )

        print(
            f"  pooled precision: "
            f"{format_optional_float(pooled.get('pooled_precision'))}",
            flush=True,
        )

        print(
            f"  pooled F1: "
            f"{format_optional_float(pooled.get('pooled_f1'))}",
            flush=True,
        )

        print(
            f"  pooled test FPR: "
            f"{format_optional_float(pooled.get('pooled_fpr'))}",
            flush=True,
        )

        print(
            f"  fold recall mean ± SD: "
            f"{format_optional_float(fold_recall_stats.get('mean'))} ± "
            f"{format_optional_float(fold_recall_stats.get('sd'))}",
            flush=True,
        )

        print(
            f"  worst eligible-fold recall: "
            f"{format_optional_float(fold_recall_stats.get('min'))}",
            flush=True,
        )
    else:
        print("\nNo eligible seen-attack temporal fold was available under the predefined protocol.", flush=True)

    print("=" * 108, flush=True)

    del df
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
