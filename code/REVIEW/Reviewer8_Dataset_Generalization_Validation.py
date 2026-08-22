from __future__ import annotations

"""
Reviewer 8 dataset-validation / generalization stress-test harness for FL-BC-IDS.

Purpose
-------
This script does NOT replace the main FL-BC-IDS experiments and does NOT alter the
blockchain / evidence / Groth16 workflow.  It provides a deliberately separate,
reviewer-facing validation layer that answers the dataset-generalization questions:

1) Were the published main splits duplicate-free at the predictor-vector level?
2) Are the main splits chronological, session-disjoint, or source-vehicle-disjoint?
3) How does the learner behave under duplicate-group-disjoint splitting?
4) How does CSE-CIC-IDS2018 behave under chronological and Flow-ID-disjoint splits?
5) Can attacks that are COMPLETELY absent from training be detected?
6) Can the same benchmark representations support multiclass attack recognition?

The script uses the same dataset paths and XGBoost family already used in the project.
All stress-test preprocessing is fit on TRAIN only.  Duplicate grouping is performed
on the raw predictor representation BEFORE scaling / imputation.

Outputs
-------
artifacts/generalization/reviewer8_validation

Important interpretation
------------------------
- The "main split audit" examines the ACTUAL saved preprocessed CSVs used by the
  existing project, so its overlap counts describe the published split artifacts.
- The harder stress tests are classifier-level validation diagnostics using the same
  XGBoost model family.  They isolate dataset generalization from cryptographic and
  federated-protocol mechanics; they should be reported as additional robustness
  validation, not silently substituted for the main FL-BC-IDS results.
- "Vehicle-separated" in the reviewer's sense means source-vehicle-disjoint raw-data
  evaluation.  The current benchmark representations do not provide a suitable raw
  source-vehicle identifier.  Simulated FL clients are NOT treated as source vehicles.
"""

import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 0. PROJECT PATHS -- copied from the existing FL-BC-IDS code
# =============================================================================
CSE_RAW_CSV = Path(
    os.getenv("FLBCIDS_CSE_RAW_CSV", 'data/raw/CSE-CIC-IDS2018/CSECICIDS2018Dataset.csv')
)
CSE_PREPROC_DIR = Path(os.getenv("FLBCIDS_CSE_PREPROC_DIR", "data/preprocessed/CSE-CIC-IDS2018"))

CIC_DECIMAL_DIR = Path(
    os.getenv("FLBCIDS_CICIOV_RAW_DIR", "data/raw/CICIoV2024")
)
CIC_PREPROC_DIR = Path(
    os.getenv("FLBCIDS_CICIOV_PREPROC_DIR", "data/preprocessed/CICIoV2024")
)

OUTPUT_ROOT = Path(
    os.getenv("FLBCIDS_GENERALIZATION_RESULTS_DIR", "artifacts/generalization/reviewer8_validation")
)

CSE_MAIN_SPLITS = {
    "train": CSE_PREPROC_DIR / "CSECICIDS2018_train_preprocessed.csv",
    "val": CSE_PREPROC_DIR / "CSECICIDS2018_val_preprocessed.csv",
    "test": CSE_PREPROC_DIR / "CSECICIDS2018_test_preprocessed.csv",
}
CIC_MAIN_SPLITS = {
    "train": CIC_PREPROC_DIR / "CICIoV2024_train_preprocessed.csv",
    "val": CIC_PREPROC_DIR / "CICIoV2024_val_preprocessed.csv",
    "test": CIC_PREPROC_DIR / "CICIoV2024_test_preprocessed.csv",
}

CIC_FILES = [
    "decimal_benign.csv",
    "decimal_DoS.csv",
    "decimal_spoofing-GAS.csv",
    "decimal_spoofing-RPM.csv",
    "decimal_spoofing-SPEED.csv",
    "decimal_spoofing-STEERING_WHEEL.csv",
]
CIC_FEATURE_COLS = [
    "ID",
    "DATA_0",
    "DATA_1",
    "DATA_2",
    "DATA_3",
    "DATA_4",
    "DATA_5",
    "DATA_6",
    "DATA_7",
]


# =============================================================================
# 1. EXECUTION SWITCHES
# =============================================================================
RUN_MAIN_SPLIT_DUPLICATE_AUDIT = True
RUN_CIC_UNSEEN_ATTACK = True
RUN_CIC_MULTICLASS = True
RUN_CSE_DUPLICATE_GROUP_DISJOINT = True
RUN_CSE_SESSION_DISJOINT = True
RUN_CSE_CHRONOLOGICAL = True
RUN_CSE_UNSEEN_ATTACK = True
RUN_CSE_MULTICLASS = True

# Reproducibility
RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Chunk size for exact hash audits of the already-saved main split CSVs.
HASH_CHUNK_SIZE = 200_000

# CSE model-side memory guard.
# The original preprocessing code materializes at most 3,000,000 CSE rows.  This
# validation harness therefore uses the same default cap, but samples ACROSS THE
# FULL RAW FILE before any chronological ordering.  The chronological test is thus
# time-ordered on a deterministic whole-file sample rather than on the first N rows.
# Set None only if you intentionally want to materialize the complete raw dataset.
CSE_MAX_MODEL_ROWS: Optional[int] = 3_000_000

# Leave-one-attack-out controls for CSE.  Only attack classes with at least this many
# rows are evaluated.  None means test every qualifying attack type.
CSE_MIN_HOLDOUT_ATTACK_ROWS = 2_000
CSE_MAX_HOLDOUT_ATTACKS: Optional[int] = None

# Multiclass: drop extremely tiny CSE classes that cannot support a meaningful
# train/validation/test split.  CICIoV2024 keeps all six classes.
CSE_MIN_MULTICLASS_ROWS = 1_000

# Optional class-wise cap used ONLY in CSE multiclass to avoid one huge class
# dominating memory and training.  None disables the cap.
CSE_MULTICLASS_MAX_ROWS_PER_CLASS: Optional[int] = 300_000

# XGBoost validation family.  These mirror the existing centralized baseline.
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
EARLY_STOPPING_ROUNDS = 10

# Current preprocessing balances TRAIN only.  Keep that convention for binary
# stress tests when the resulting train size is not excessive.
BALANCE_BINARY_TRAIN = True
MAX_ROWS_FOR_OVERSAMPLING = 3_000_000


# =============================================================================
# 2. GENERIC HELPERS
# =============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)


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
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (z * z / n)
    center = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))
        / denom
    )
    return (max(0.0, center - half), min(1.0, center + half))


def normalize_label_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )


def hash_frame_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    """Stable pandas uint64 row hashes over predictor values only."""
    return pd.util.hash_pandas_object(df[list(feature_cols)], index=False).to_numpy(
        dtype=np.uint64, copy=False
    )


def make_onehot_encoder() -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )


def build_preprocessor(
    train_x: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = train_x.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_cols = [
        column
        for column in train_x.columns
        if column not in numeric_cols
    ]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ],
        memory=None,
    )

    transformers: List[Tuple[str, Any, List[str]]] = []

    if numeric_cols:
        transformers.append(
            ("num", numeric_pipe, numeric_cols)
        )

    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                ("onehot", make_onehot_encoder()),
            ],
            memory=None,
        )
        transformers.append(
            ("cat", categorical_pipe, categorical_cols)
        )

    if not transformers:
        raise ValueError(
            "No usable predictor columns found."
        )

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return transformer, numeric_cols, categorical_cols


def fit_transform_three(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[Any, Any, Any, ColumnTransformer, Dict[str, Any]]:
    pre, numeric_cols, categorical_cols = build_preprocessor(train_df[list(feature_cols)])
    X_train = pre.fit_transform(train_df[list(feature_cols)])
    X_val = pre.transform(val_df[list(feature_cols)])
    X_test = pre.transform(test_df[list(feature_cols)])
    info = {
        "feature_count_before_transform": len(feature_cols),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "output_feature_count": int(X_train.shape[1]),
    }
    return X_train, X_val, X_test, pre, info


def find_best_binary_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Same principle as the existing baseline: maximize validation F1.
    thresholds = np.linspace(0.01, 0.99, 199)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        pred = (y_prob >= thr).astype(np.int8)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)
    return best_thr


def binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= threshold).astype(np.int8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    acc_lo, acc_hi = wilson_interval(tn + tp, len(y_true))
    rec_lo, rec_hi = wilson_interval(tp, tp + fn)

    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_wilson95_low": float(acc_lo),
        "accuracy_wilson95_high": float(acc_hi),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "recall_wilson95_low": float(rec_lo),
        "recall_wilson95_high": float(rec_hi),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if np.unique(y_true).size > 1 else None,
        "roc_auc": safe_auc(y_true, y_prob),
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": cm.tolist(),
    }

def binary_unique_pattern_metrics(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    hash_col: str = "feature_hash",
) -> Dict[str, Any]:
    """
    Evaluate binary detection after collapsing repeated predictor vectors.

    Each distinct predictor hash contributes at most one observation.
    Hashes associated with conflicting binary labels inside the test set
    are counted explicitly as ambiguous and excluded from the
    duplicate-collapsed performance calculation.

    For an unambiguous hash, the probability assigned to the pattern is
    the mean model probability across all rows carrying that hash.
    Because those rows have identical predictor vectors, their predictions
    should normally be numerically identical; averaging is used only as a
    robust aggregation rule.
    """
    if hash_col not in test_df.columns:
        return {
            "status": "not_available",
            "reason": f"{hash_col!r} is absent from the test dataframe.",
        }

    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=np.float64,
    )

    if len(test_df) != len(y_true) or len(y_true) != len(y_prob):
        raise ValueError(
            "test_df, y_true, and y_prob must have identical lengths."
        )

    pattern_df = pd.DataFrame(
        {
            "feature_hash": test_df[hash_col].to_numpy(),
            "label": y_true,
            "probability": y_prob,
        }
    )

    grouped = (
        pattern_df.groupby(
            "feature_hash",
            sort=False,
        )
        .agg(
            label_nunique=("label", "nunique"),
            label=("label", "first"),
            probability=("probability", "mean"),
            row_count=("label", "size"),
        )
        .reset_index()
    )

    ambiguous_mask = (
        grouped["label_nunique"] > 1
    )

    ambiguous_groups = grouped.loc[
        ambiguous_mask
    ]

    unambiguous_groups = grouped.loc[
        ~ambiguous_mask
    ].copy()

    total_unique_hashes = int(
        len(grouped)
    )

    ambiguous_unique_hashes = int(
        ambiguous_mask.sum()
    )

    ambiguous_rows = int(
        ambiguous_groups["row_count"].sum()
    )

    unambiguous_unique_hashes = int(
        len(unambiguous_groups)
    )

    if unambiguous_unique_hashes == 0:
        return {
            "status": "not_evaluable",
            "test_unique_predictor_hashes": total_unique_hashes,
            "test_ambiguous_predictor_hashes": ambiguous_unique_hashes,
            "test_rows_on_ambiguous_predictor_hashes": ambiguous_rows,
            "test_unambiguous_unique_predictor_hashes": 0,
            "metrics": None,
        }

    pattern_y_true = unambiguous_groups[
        "label"
    ].to_numpy(
        dtype=np.int8,
        copy=False,
    )

    pattern_y_prob = unambiguous_groups[
        "probability"
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    represented_classes = set(
        np.unique(
            pattern_y_true
        ).tolist()
    )

    expected_classes = {
        0,
        1,
    }

    if represented_classes != expected_classes:
        return {
            "status": (
                "class_incomplete_after_ambiguity_exclusion"
            ),
            "definition": (
                "One equal-weight observation per distinct unambiguous "
                "predictor hash; repeated rows are collapsed before scoring."
            ),
            "test_rows": int(
                len(test_df)
            ),
            "test_unique_predictor_hashes": (
                total_unique_hashes
            ),
            "test_ambiguous_predictor_hashes": (
                ambiguous_unique_hashes
            ),
            "test_rows_on_ambiguous_predictor_hashes": (
                ambiguous_rows
            ),
            "test_unambiguous_unique_predictor_hashes": (
                unambiguous_unique_hashes
            ),
            "classes_present_after_collapse": sorted(
                represented_classes
            ),
            "metrics": None,
        }

    metrics = binary_metrics(
        pattern_y_true,
        pattern_y_prob,
        threshold,
    )

    return {
        "status": "evaluated",
        "definition": (
            "One equal-weight observation per distinct unambiguous "
            "predictor hash; repeated rows are collapsed before scoring."
        ),
        "test_rows": int(
            len(test_df)
        ),
        "test_unique_predictor_hashes": (
            total_unique_hashes
        ),
        "test_ambiguous_predictor_hashes": (
            ambiguous_unique_hashes
        ),
        "test_rows_on_ambiguous_predictor_hashes": (
            ambiguous_rows
        ),
        "test_unambiguous_unique_predictor_hashes": (
            unambiguous_unique_hashes
        ),
        "duplicate_collapse_ratio": float(
            unambiguous_unique_hashes
            / max(
                1,
                len(test_df) - ambiguous_rows,
            )
        ),
        "classes_present_after_collapse": sorted(
            represented_classes
        ),
        "metrics": metrics,
    }

def maybe_balance_binary_train(
    X: Any, y: np.ndarray
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    y = np.asarray(y, dtype=np.int8)
    counts = np.bincount(y, minlength=2)
    info: Dict[str, Any] = {
        "enabled": bool(BALANCE_BINARY_TRAIN),
        "before": {"benign": int(counts[0]), "attack": int(counts[1])},
    }

    if not BALANCE_BINARY_TRAIN:
        info["performed"] = False
        info["reason"] = "disabled"
        return X, y, info
    if len(y) > MAX_ROWS_FOR_OVERSAMPLING:
        info["performed"] = False
        info["reason"] = f"rows>{MAX_ROWS_FOR_OVERSAMPLING}"
        return X, y, info
    if counts.min() == 0 or counts[0] == counts[1]:
        info["performed"] = False
        info["reason"] = "single_class_or_already_balanced"
        return X, y, info

    ros = RandomOverSampler(sampling_strategy=1.0, random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X, y)
    after = np.bincount(y_res.astype(int), minlength=2)
    info["performed"] = True
    info["after"] = {"benign": int(after[0]), "attack": int(after[1])}
    return X_res, y_res.astype(np.int8), info


def train_binary_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    out_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    y_train = train_df[label_col].to_numpy(
        dtype=np.int8,
        copy=False,
    )
    y_val = val_df[label_col].to_numpy(
        dtype=np.int8,
        copy=False,
    )
    y_test = test_df[label_col].to_numpy(
        dtype=np.int8,
        copy=False,
    )

    if np.unique(y_train).size != 2:
        raise ValueError(
            f"{tag}: training split does not contain both binary classes."
        )

    if np.unique(y_val).size != 2:
        raise ValueError(
            f"{tag}: validation split does not contain both binary classes."
        )

    X_train, X_val, X_test, pre, pre_info = fit_transform_three(
        train_df,
        val_df,
        test_df,
        feature_cols,
    )

    X_train_bal, y_train_bal, balance_info = maybe_balance_binary_train(X_train, y_train)

    params = dict(XGB_COMMON)
    params.update({"objective": "binary:logistic", "eval_metric": "logloss"})

    if not balance_info.get("performed", False):
        counts = np.bincount(y_train_bal.astype(int), minlength=2)
        params["scale_pos_weight"] = float(counts[0] / max(1, counts[1]))
    else:
        params["scale_pos_weight"] = 1.0
    params["base_score"] = float(np.mean(y_train_bal))

    dtrain = xgb.DMatrix(X_train_bal, label=y_train_bal)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    t0 = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    train_sec = time.perf_counter() - t0

    best_iteration = int(
        getattr(booster, "best_iteration", -1)
    )

    if best_iteration >= 0:
        prediction_iteration_range = (
            0,
            best_iteration + 1,
        )

        val_prob = booster.predict(
            dval,
            iteration_range=prediction_iteration_range,
        )

        test_prob = booster.predict(
            dtest,
            iteration_range=prediction_iteration_range,
        )
    else:
        prediction_iteration_range = None

        val_prob = booster.predict(dval)
        test_prob = booster.predict(dtest)

    threshold = find_best_binary_threshold(
        y_val,
        val_prob,
    )

    test_unique_pattern_audit = (
        binary_unique_pattern_metrics(
            test_df,
            y_test,
            test_prob,
            threshold,
        )
        if "feature_hash" in test_df.columns
        else {
            "status": "not_available",
            "reason": (
                "The test dataframe does not contain feature_hash."
            ),
        }
    )

    result = {
        "tag": tag,
        "train_rows_before_balance": int(len(y_train)),
        "train_rows_after_balance": int(len(y_train_bal)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "preprocessing": pre_info,
        "balance": balance_info,
        "xgboost_params": params,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": best_iteration,
        "prediction_iteration_range": (
            list(prediction_iteration_range)
            if prediction_iteration_range is not None
            else None
        ),
        "best_score": float(getattr(booster, "best_score", float("nan"))),
        "train_time_sec": float(train_sec),
        "validation_metrics": binary_metrics(
            y_val,
            val_prob,
            threshold,
        ),
        "test_metrics": binary_metrics(
            y_test,
            test_prob,
            threshold,
        ),
        "test_unique_pattern_audit": (
            test_unique_pattern_audit
        ),
    }

    booster.save_model(out_dir / f"{tag}_xgboost.json")
    joblib.dump(pre, out_dir / f"{tag}_preprocessor.joblib")
    save_json(out_dir / f"{tag}_summary.json", result)
    return result


def multiclass_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    y_true = np.asarray(
        y_true,
        dtype=np.int32,
    )

    prob = np.asarray(
        prob,
        dtype=np.float64,
    )

    if prob.ndim != 2:
        raise ValueError(
            f"Expected a 2-D multiclass probability matrix, "
            f"got shape {prob.shape}."
        )

    if not np.all(np.isfinite(prob)):
        raise ValueError(
            "Multiclass probability matrix contains NaN or infinity."
        )

    raw_row_sums = prob.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(raw_row_sums <= 0.0):
        raise ValueError(
            "At least one multiclass probability row has a "
            "non-positive sum."
        )

    max_probability_sum_deviation = float(
        np.max(
            np.abs(
                raw_row_sums.ravel()
                - 1.0
            )
        )
    )

    # Normalize small floating-point deviations produced by the
    # multiclass predictor before log-loss/AUC evaluation.
    prob = prob / raw_row_sums

    pred = np.argmax(
        prob,
        axis=1,
    ).astype(np.int32)
    labels = np.arange(len(class_names), dtype=np.int32)
    cm = confusion_matrix(y_true, pred, labels=labels)

    p, r, f, support = precision_recall_fscore_support(
        y_true, pred, labels=labels, zero_division=0
    )
    per_class = []
    for idx, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": int(idx),
                "class_name": str(name),
                "precision": float(p[idx]),
                "recall": float(r[idx]),
                "f1": float(f[idx]),
                "support": int(support[idx]),
            }
        )

    result: Dict[str, Any] = {
        "n": int(len(y_true)),
        "max_raw_probability_sum_deviation": (
            max_probability_sum_deviation
        ),
        "probabilities_normalized_before_probabilistic_metrics": True,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "logloss": float(log_loss(y_true, prob, labels=labels)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }

    # OVR macro AUC requires all classes to be represented in y_true.
    try:
        result["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, prob, multi_class="ovr", average="macro")
        )
    except Exception:
        result["roc_auc_ovr_macro"] = None

    return result

def multiclass_unique_pattern_metrics(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    prob: np.ndarray,
    class_names: Sequence[str],
    hash_col: str = "feature_hash",
) -> Dict[str, Any]:
    """
    Evaluate multiclass recognition after collapsing repeated predictor
    vectors so that each distinct unambiguous predictor hash has equal
    weight.

    Predictor hashes associated with more than one class inside the test
    set are explicitly counted as ambiguous and excluded from the
    duplicate-collapsed multiclass score.
    """
    if hash_col not in test_df.columns:
        return {
            "status": "not_available",
            "reason": f"{hash_col!r} is absent from the test dataframe.",
        }

    y_true = np.asarray(
        y_true,
        dtype=np.int32,
    )

    prob = np.asarray(
        prob,
        dtype=np.float64,
    )

    if prob.ndim != 2:
        raise ValueError(
            "Multiclass probability input must be two-dimensional."
        )

    if len(test_df) != len(y_true) or len(y_true) != prob.shape[0]:
        raise ValueError(
            "test_df, y_true, and probability rows must have "
            "identical lengths."
        )

    if prob.shape[1] != len(class_names):
        raise ValueError(
            "Probability-column count does not match class_names."
        )

    base = pd.DataFrame(
        {
            "feature_hash": test_df[
                hash_col
            ].to_numpy(),
            "label": y_true,
        }
    )

    for class_id in range(
        len(class_names)
    ):
        base[
            f"prob_{class_id}"
        ] = prob[:, class_id]

    aggregation: Dict[str, Any] = {
        "label_nunique": (
            "label",
            "nunique",
        ),
        "label": (
            "label",
            "first",
        ),
        "row_count": (
            "label",
            "size",
        ),
    }

    for class_id in range(
        len(class_names)
    ):
        aggregation[
            f"prob_{class_id}"
        ] = (
            f"prob_{class_id}",
            "mean",
        )

    grouped = (
        base.groupby(
            "feature_hash",
            sort=False,
        )
        .agg(
            **aggregation
        )
        .reset_index()
    )

    ambiguous_mask = (
        grouped["label_nunique"] > 1
    )

    ambiguous_groups = grouped.loc[
        ambiguous_mask
    ]

    clean = grouped.loc[
        ~ambiguous_mask
    ].copy()

    total_unique_hashes = int(
        len(grouped)
    )

    ambiguous_unique_hashes = int(
        ambiguous_mask.sum()
    )

    ambiguous_rows = int(
        ambiguous_groups[
            "row_count"
        ].sum()
    )

    if clean.empty:
        return {
            "status": "not_evaluable",
            "test_unique_predictor_hashes": total_unique_hashes,
            "test_ambiguous_predictor_hashes": ambiguous_unique_hashes,
            "test_rows_on_ambiguous_predictor_hashes": ambiguous_rows,
            "test_unambiguous_unique_predictor_hashes": 0,
            "metrics": None,
        }

    clean_y = clean[
        "label"
    ].to_numpy(
        dtype=np.int32,
        copy=False,
    )

    clean_prob = clean[
        [
            f"prob_{class_id}"
            for class_id in range(
                len(class_names)
            )
        ]
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    represented_classes = set(
        np.unique(
            clean_y
        ).tolist()
    )

    expected_classes = set(
        range(
            len(class_names)
        )
    )

    metrics = None

    if represented_classes == expected_classes:
        metrics = multiclass_metrics(
            clean_y,
            clean_prob,
            class_names,
        )

    return {
        "status": (
            "evaluated"
            if metrics is not None
            else "class_incomplete_after_ambiguity_exclusion"
        ),
        "definition": (
            "One equal-weight observation per distinct unambiguous "
            "predictor hash; repeated rows are collapsed before scoring."
        ),
        "test_rows": int(
            len(test_df)
        ),
        "test_unique_predictor_hashes": (
            total_unique_hashes
        ),
        "test_ambiguous_predictor_hashes": (
            ambiguous_unique_hashes
        ),
        "test_rows_on_ambiguous_predictor_hashes": (
            ambiguous_rows
        ),
        "test_unambiguous_unique_predictor_hashes": int(
            len(clean)
        ),
        "classes_present_after_collapse": sorted(
            represented_classes
        ),
        "metrics": metrics,
    }

def train_multiclass_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    class_col: str,
    class_names: Sequence[str],
    out_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    y_train = train_df[class_col].to_numpy(
        dtype=np.int32,
        copy=False,
    )

    y_val = val_df[class_col].to_numpy(
        dtype=np.int32,
        copy=False,
    )

    y_test = test_df[class_col].to_numpy(
        dtype=np.int32,
        copy=False,
    )

    num_class = len(
        class_names
    )

    expected = set(
        range(
            num_class
        )
    )

    for split_name, arr in (
            ("train", y_train),
            ("val", y_val),
            ("test", y_test),
    ):
        found = set(
            np.unique(
                arr
            ).tolist()
        )

        if found != expected:
            raise ValueError(
                f"{tag}: {split_name} is missing classes. "
                f"Expected={sorted(expected)}, "
                f"found={sorted(found)}"
            )

    X_train, X_val, X_test, pre, pre_info = fit_transform_three(
        train_df,
        val_df,
        test_df,
        feature_cols,
    )

    counts = np.bincount(
        y_train,
        minlength=num_class,
    ).astype(float)
    inv = len(y_train) / (num_class * np.maximum(counts, 1.0))
    sample_weight = inv[y_train]

    params = dict(XGB_COMMON)
    params.update(
        {
            "objective": "multi:softprob",
            "num_class": num_class,
            "eval_metric": "mlogloss",
        }
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    t0 = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    train_sec = time.perf_counter() - t0

    best_iteration = int(
        getattr(booster, "best_iteration", -1)
    )

    if best_iteration >= 0:
        prediction_iteration_range = (
            0,
            best_iteration + 1,
        )

        val_prob = booster.predict(
            dval,
            iteration_range=prediction_iteration_range,
        )

        test_prob = booster.predict(
            dtest,
            iteration_range=prediction_iteration_range,
        )
    else:
        prediction_iteration_range = None

        val_prob = booster.predict(dval)
        test_prob = booster.predict(dtest)

    test_unique_pattern_audit = (
        multiclass_unique_pattern_metrics(
            test_df,
            y_test,
            test_prob,
            class_names,
        )
        if "feature_hash" in test_df.columns
        else {
            "status": "not_available",
            "reason": (
                "The test dataframe does not contain feature_hash."
            ),
        }
    )

    result = {
        "tag": tag,
        "class_names": list(class_names),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "train_class_counts": {
            str(class_names[i]): int(counts[i]) for i in range(num_class)
        },
        "preprocessing": pre_info,
        "xgboost_params": params,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": best_iteration,
        "prediction_iteration_range": (
            list(prediction_iteration_range)
            if prediction_iteration_range is not None
            else None
        ),
        "best_score": float(
            getattr(
                booster,
                "best_score",
                float("nan"),
            )
        ),
        "train_time_sec": float(train_sec),
        "validation_metrics": multiclass_metrics(
            y_val,
            val_prob,
            class_names,
        ),
        "test_metrics": multiclass_metrics(
            y_test,
            test_prob,
            class_names,
        ),
        "test_unique_pattern_audit": (
            test_unique_pattern_audit
        ),
    }

    booster.save_model(out_dir / f"{tag}_xgboost.json")
    joblib.dump(pre, out_dir / f"{tag}_preprocessor.joblib")
    save_json(out_dir / f"{tag}_summary.json", result)

    test_metrics: Dict[str, Any] = result["test_metrics"]

    per_class_records: List[Dict[str, Any]] = list(
        test_metrics["per_class"]
    )

    confusion_values = np.asarray(
        test_metrics["confusion_matrix"],
        dtype=np.int64,
    )

    class_labels: List[str] = [
        str(name) for name in class_names
    ]

    pd.DataFrame.from_records(
        per_class_records
    ).to_csv(
        out_dir / f"{tag}_test_per_class.csv",
        index=False,
    )

    pd.DataFrame(
        data=confusion_values,
        index=pd.Index(class_labels),
        columns=pd.Index(class_labels),
    ).to_csv(
        out_dir / f"{tag}_test_confusion_matrix.csv"
    )

    return result

def train_multiclass_xgboost_fixed_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    class_col: str,
    class_names: Sequence[str],
    out_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    """
    Train a multiclass XGBoost model with fixed hyperparameters and a fixed
    boosting-round count, without validation-based early stopping.

    This protocol is used when preserving a completely duplicate-group-disjoint
    held-out test set is more important than creating a third validation partition.
    The test set is never used for model selection, threshold selection, early
    stopping, or hyperparameter tuning.
    """
    ensure_dir(out_dir)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(
        train_df[list(feature_cols)]
    )

    x_train = preprocessor.fit_transform(
        train_df[list(feature_cols)]
    )

    x_test = preprocessor.transform(
        test_df[list(feature_cols)]
    )

    y_train = train_df[class_col].to_numpy(
        dtype=np.int32,
        copy=False,
    )

    y_test = test_df[class_col].to_numpy(
        dtype=np.int32,
        copy=False,
    )

    num_classes = len(class_names)
    expected_classes = set(
        range(num_classes)
    )

    train_classes = set(
        np.unique(y_train).tolist()
    )

    test_classes = set(
        np.unique(y_test).tolist()
    )

    if train_classes != expected_classes:
        raise ValueError(
            f"{tag}: training split is missing classes. "
            f"Expected={sorted(expected_classes)}, "
            f"found={sorted(train_classes)}"
        )

    if test_classes != expected_classes:
        raise ValueError(
            f"{tag}: test split is missing classes. "
            f"Expected={sorted(expected_classes)}, "
            f"found={sorted(test_classes)}"
        )

    train_counts = np.bincount(
        y_train,
        minlength=num_classes,
    ).astype(float)

    inverse_class_weights = (
        len(y_train)
        / (
            num_classes
            * np.maximum(
                train_counts,
                1.0,
            )
        )
    )

    sample_weight = inverse_class_weights[
        y_train
    ]

    params = dict(
        XGB_COMMON
    )

    params.update(
        {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
        }
    )

    dtrain = xgb.DMatrix(
        x_train,
        label=y_train,
        weight=sample_weight,
    )

    dtest = xgb.DMatrix(
        x_test,
        label=y_test,
    )

    start_time = time.perf_counter()

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        verbose_eval=False,
    )

    train_time_sec = (
        time.perf_counter()
        - start_time
    )

    test_prob = booster.predict(
        dtest
    )

    test_metrics = multiclass_metrics(
        y_test,
        test_prob,
        class_names,
    )
    test_unique_pattern_audit = (
        multiclass_unique_pattern_metrics(
            test_df,
            y_test,
            test_prob,
            class_names,
        )
        if "feature_hash" in test_df.columns
        else {
            "status": "not_available",
            "reason": (
                "The test dataframe does not contain feature_hash."
            ),
        }
    )

    preprocessing_info: Dict[str, Any] = {
        "feature_count_before_transform": len(
            feature_cols
        ),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "output_feature_count": int(
            x_train.shape[1]
        ),
    }

    train_distinct_hashes = (
        int(
            train_df["feature_hash"].nunique()
        )
        if "feature_hash" in train_df.columns
        else None
    )

    test_distinct_hashes = (
        int(
            test_df["feature_hash"].nunique()
        )
        if "feature_hash" in test_df.columns
        else None
    )

    result: Dict[str, Any] = {
        "tag": tag,
        "class_names": [
            str(name)
            for name in class_names
        ],
        "training_protocol": (
            "fixed-round multiclass XGBoost; no validation "
            "or test-driven early stopping"
        ),
        "train_rows": int(
            len(y_train)
        ),
        "test_rows": int(
            len(y_test)
        ),
        "train_distinct_predictor_hashes": (
            train_distinct_hashes
        ),
        "test_distinct_predictor_hashes": (
            test_distinct_hashes
        ),
        "train_class_counts": {
            str(class_names[class_id]): int(
                train_counts[class_id]
            )
            for class_id in range(
                num_classes
            )
        },
        "preprocessing": preprocessing_info,
        "xgboost_params": params,
        "num_boost_round": int(
            NUM_BOOST_ROUND
        ),
        "early_stopping_used": False,
        "validation_split_used": False,
        "test_used_for_model_selection": False,
        "train_time_sec": float(
            train_time_sec
        ),
        "test_metrics": test_metrics,
        "test_unique_pattern_audit": (
            test_unique_pattern_audit
        ),
    }

    booster.save_model(
        out_dir
        / f"{tag}_xgboost.json"
    )

    joblib.dump(
        preprocessor,
        out_dir
        / f"{tag}_preprocessor.joblib",
    )

    save_json(
        out_dir
        / f"{tag}_summary.json",
        result,
    )

    per_class_records: List[
        Dict[str, Any]
    ] = list(
        test_metrics["per_class"]
    )

    confusion_values = np.asarray(
        test_metrics["confusion_matrix"],
        dtype=np.int64,
    )

    class_labels: List[str] = [
        str(name)
        for name in class_names
    ]

    pd.DataFrame.from_records(
        per_class_records
    ).to_csv(
        out_dir
        / f"{tag}_test_per_class.csv",
        index=False,
    )

    pd.DataFrame(
        data=confusion_values,
        index=pd.Index(
            class_labels
        ),
        columns=pd.Index(
            class_labels
        ),
    ).to_csv(
        out_dir
        / f"{tag}_test_confusion_matrix.csv"
    )

    return result
# =============================================================================
# 3. EXACT AUDIT OF THE SAVED MAIN SPLITS
# =============================================================================
def read_csv_header(path: Path) -> List[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def collect_split_hashes(
    path: Path,
    label_col: str,
    chunksize: int = HASH_CHUNK_SIZE,
) -> Tuple[set[int], Dict[str, Any]]:
    cols = read_csv_header(path)
    if label_col not in cols:
        raise ValueError(f"{path}: expected label column {label_col!r} not found.")
    feature_cols = [c for c in cols if c != label_col]

    unique_hashes: set[int] = set()
    rows = 0
    duplicate_rows_inside_split = 0
    label_counts: Counter[int] = Counter()

    for chunk in pd.read_csv(path, chunksize=chunksize):
        rows += len(chunk)
        h = hash_frame_features(chunk, feature_cols)
        for value in h:
            iv = int(value)
            if iv in unique_hashes:
                duplicate_rows_inside_split += 1
            else:
                unique_hashes.add(iv)
        vc = chunk[label_col].value_counts(dropna=False)
        for k, v in vc.items():
            try:
                label_counts[int(k)] += int(v)
            except Exception:
                pass

    info = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(rows),
        "feature_count": len(feature_cols),
        "unique_predictor_hashes": int(len(unique_hashes)),
        "duplicate_rows_within_split": int(duplicate_rows_inside_split),
        "duplicate_row_fraction_within_split": float(
            duplicate_rows_inside_split / max(1, rows)
        ),
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
    }
    return unique_hashes, info


def count_rows_matching_hash_set(
    path: Path,
    label_col: str,
    reference_hashes: set[int],
    chunksize: int = HASH_CHUNK_SIZE,
) -> Dict[str, Any]:
    cols = read_csv_header(path)
    feature_cols = [c for c in cols if c != label_col]
    rows = 0
    matched_rows = 0
    matched_by_label: Counter[int] = Counter()
    for chunk in pd.read_csv(path, chunksize=chunksize):
        h = hash_frame_features(chunk, feature_cols)
        mask = np.fromiter(
            (int(v) in reference_hashes for v in h), dtype=bool, count=len(h)
        )
        rows += len(chunk)
        matched_rows += int(mask.sum())
        if mask.any():
            vc = chunk.loc[mask, label_col].value_counts(dropna=False)
            for k, v in vc.items():
                try:
                    matched_by_label[int(k)] += int(v)
                except Exception:
                    pass
    return {
        "rows": int(rows),
        "matching_rows": int(matched_rows),
        "matching_fraction": float(matched_rows / max(1, rows)),
        "matching_rows_by_label": {
            str(k): int(v) for k, v in sorted(matched_by_label.items())
        },
    }


def audit_main_split_duplicates(
    dataset_name: str,
    split_paths: Dict[str, Path],
    label_col: str,
    out_dir: Path,
) -> Dict[str, Any]:
    print(f"\n[MAIN SPLIT AUDIT] {dataset_name}")
    for p in split_paths.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing saved main split: {p}")

    train_hashes, train_info = collect_split_hashes(split_paths["train"], label_col)
    val_hashes, val_info = collect_split_hashes(split_paths["val"], label_col)
    test_hashes, test_info = collect_split_hashes(split_paths["test"], label_col)

    val_vs_train_rows = count_rows_matching_hash_set(
        split_paths["val"], label_col, train_hashes
    )
    test_vs_train_rows = count_rows_matching_hash_set(
        split_paths["test"], label_col, train_hashes
    )
    test_vs_val_rows = count_rows_matching_hash_set(
        split_paths["test"], label_col, val_hashes
    )

    result = {
        "dataset": dataset_name,
        "hash_definition": "exact saved predictor vector; label excluded",
        "splits": {"train": train_info, "val": val_info, "test": test_info},
        "unique_hash_intersections": {
            "train_val": int(len(train_hashes & val_hashes)),
            "train_test": int(len(train_hashes & test_hashes)),
            "val_test": int(len(val_hashes & test_hashes)),
            "all_three": int(len(train_hashes & val_hashes & test_hashes)),
        },
        "row_level_overlap": {
            "val_rows_matching_any_train_predictor": val_vs_train_rows,
            "test_rows_matching_any_train_predictor": test_vs_train_rows,
            "test_rows_matching_any_val_predictor": test_vs_val_rows,
        },
    }
    result["duplicate_group_disjoint_main_splits"] = bool(
        result["unique_hash_intersections"]["train_val"] == 0
        and result["unique_hash_intersections"]["train_test"] == 0
        and result["unique_hash_intersections"]["val_test"] == 0
    )

    ensure_dir(out_dir)
    save_json(out_dir / f"{dataset_name}_main_split_duplicate_audit.json", result)
    return result


# =============================================================================
# 4. GROUP-DISJOINT SPLITTING
# =============================================================================
def group_disjoint_split_indices(
    groups: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 approximate split with no group appearing in multiple partitions."""
    n = len(groups)
    indices = np.arange(n, dtype=np.int64)
    gss1 = GroupShuffleSplit(
        n_splits=1, train_size=TRAIN_FRAC, random_state=random_state
    )
    train_idx, temp_idx = next(gss1.split(indices, groups=groups))

    # 15/15 from the remaining 30% => half of temp to validation.
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_state)
    rel_val, rel_test = next(
        gss2.split(temp_idx, groups=groups[temp_idx])
    )
    val_idx = temp_idx[rel_val]
    test_idx = temp_idx[rel_test]
    return train_idx, val_idx, test_idx


def assert_group_disjoint(
    groups: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, int]:
    gt = set(map(int, groups[train_idx]))
    gv = set(map(int, groups[val_idx]))
    ge = set(map(int, groups[test_idx]))
    out = {
        "train_val_group_overlap": len(gt & gv),
        "train_test_group_overlap": len(gt & ge),
        "val_test_group_overlap": len(gv & ge),
    }
    if any(out.values()):
        raise AssertionError(f"Group-disjoint split failed: {out}")
    return out

def class_complete_group_disjoint_train_test_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Construct a strict duplicate-group-disjoint multiclass train/test split.

    This two-way protocol is used for CICIoV2024 because some attack classes
    contain fewer than three distinct ID + DATA_0..DATA_7 predictor patterns.
    A class-complete three-way train/validation/test split would therefore be
    mathematically impossible without placing an identical predictor pattern
    into more than one partition.

    Requirements enforced here:
      1) every class must occur in both train and test;
      2) no predictor-hash group may occur in both train and test;
      3) the selected split should remain as close as feasible to the target
         test fraction and to the original class proportions.

    Model hyperparameters are fixed before this split is evaluated, so no
    validation or test-driven hyperparameter selection is performed.
    """
    labels = np.asarray(labels, dtype=np.int32)
    groups = np.asarray(groups)

    if labels.ndim != 1 or groups.ndim != 1:
        raise ValueError(
            "labels and groups must both be one-dimensional."
        )

    if len(labels) != len(groups):
        raise ValueError(
            "labels and groups must have the same length."
        )

    if len(labels) == 0:
        raise ValueError(
            "Cannot split an empty dataset."
        )

    expected_classes = np.unique(labels)

    if not np.array_equal(
        expected_classes,
        np.arange(len(expected_classes), dtype=np.int32),
    ):
        raise ValueError(
            "Class identifiers must be contiguous integers starting at zero."
        )

    num_classes = len(expected_classes)

    unique_groups, inverse_group_index = np.unique(
        groups,
        return_inverse=True,
    )

    num_groups = len(unique_groups)

    group_class_row_counts = np.zeros(
        (num_groups, num_classes),
        dtype=np.int64,
    )

    np.add.at(
        group_class_row_counts,
        (inverse_group_index, labels),
        1,
    )

    distinct_groups_per_class: Dict[str, int] = {
        str(class_id): int(
            np.count_nonzero(
                group_class_row_counts[:, class_id]
            )
        )
        for class_id in range(num_classes)
    }

    minimum_groups = min(
        distinct_groups_per_class.values()
    )

    if minimum_groups < 2:
        raise ValueError(
            "A strict duplicate-group-disjoint train/test multiclass "
            "split is impossible because at least one class occurs in "
            "fewer than two distinct predictor-hash groups. "
            f"Distinct groups per class: "
            f"{distinct_groups_per_class}"
        )

    total_class_rows = group_class_row_counts.sum(
        axis=0
    )

    # Start from the manuscript's 15% test target, but allow larger
    # candidate group fractions when necessary for rare-pattern classes.
    candidate_test_group_fractions = (
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
    )

    best_candidate: Optional[
        Tuple[np.ndarray, float, float, np.ndarray, int]
    ] = None

    best_score = float("inf")

    group_indices = np.arange(
        num_groups,
        dtype=np.int64,
    )

    for candidate_fraction in candidate_test_group_fractions:
        number_test_groups = int(
            round(candidate_fraction * num_groups)
        )

        number_test_groups = max(
            1,
            min(
                num_groups - 1,
                number_test_groups,
            ),
        )

        for seed_offset in range(1000):
            seed = (
                random_state
                + seed_offset
                + int(candidate_fraction * 10000)
            )

            rng = np.random.default_rng(seed)

            shuffled_group_indices = rng.permutation(
                group_indices
            )

            test_group_indices = shuffled_group_indices[
                :number_test_groups
            ]

            test_class_rows = (
                group_class_row_counts[
                    test_group_indices
                ].sum(axis=0)
            )

            train_class_rows = (
                total_class_rows
                - test_class_rows
            )

            # Every class must appear on both sides.
            if np.any(test_class_rows == 0):
                continue

            if np.any(train_class_rows == 0):
                continue

            actual_test_row_fraction = float(
                test_class_rows.sum()
                / len(labels)
            )

            per_class_test_row_fractions = (
                test_class_rows
                / total_class_rows
            )

            # Prefer a row fraction close to TEST_FRAC and, secondarily,
            # similar test representation across classes.
            size_error = abs(
                actual_test_row_fraction
                - TEST_FRAC
            )

            class_distribution_error = float(
                np.mean(
                    np.abs(
                        per_class_test_row_fractions
                        - TEST_FRAC
                    )
                )
            )

            score = (
                size_error
                + class_distribution_error
            )

            if score < best_score:
                best_score = score
                best_candidate = (
                    test_group_indices.copy(),
                    float(candidate_fraction),
                    actual_test_row_fraction,
                    per_class_test_row_fractions.copy(),
                    int(seed),
                )

    if best_candidate is None:
        raise ValueError(
            "Could not construct a class-complete strict "
            "duplicate-group-disjoint CICIoV2024 multiclass "
            "train/test split after deterministic search. "
            f"Distinct groups per class: "
            f"{distinct_groups_per_class}"
        )

    (
        selected_test_group_indices,
        selected_group_fraction,
        actual_test_row_fraction,
        per_class_test_row_fractions,
        seed_used,
    ) = best_candidate

    test_group_mask = np.zeros(
        num_groups,
        dtype=bool,
    )

    test_group_mask[
        selected_test_group_indices
    ] = True

    row_is_test = test_group_mask[
        inverse_group_index
    ]

    train_idx = np.flatnonzero(
        ~row_is_test
    )

    test_idx = np.flatnonzero(
        row_is_test
    )

    train_groups = set(
        map(
            int,
            groups[train_idx],
        )
    )

    test_groups = set(
        map(
            int,
            groups[test_idx],
        )
    )

    group_overlap = len(
        train_groups & test_groups
    )

    if group_overlap != 0:
        raise AssertionError(
            "Strict train/test predictor-hash disjointness failed: "
            f"{group_overlap} groups overlap."
        )

    train_classes = set(
        np.unique(
            labels[train_idx]
        ).tolist()
    )

    test_classes = set(
        np.unique(
            labels[test_idx]
        ).tolist()
    )

    expected_class_set = set(
        expected_classes.tolist()
    )

    if train_classes != expected_class_set:
        raise AssertionError(
            "Training partition is missing one or more classes."
        )

    if test_classes != expected_class_set:
        raise AssertionError(
            "Test partition is missing one or more classes."
        )

    metadata: Dict[str, Any] = {
        "method": (
            "class-complete strict duplicate-group-disjoint "
            "multiclass train/test split"
        ),
        "reason_for_two_way_protocol": (
            "A strict three-way train/validation/test split is "
            "mathematically impossible for the selected CICIoV2024 "
            "decimal representation because at least one attack class "
            "contains fewer than three distinct predictor-hash groups."
        ),
        "seed_used": seed_used,
        "target_test_row_fraction": float(
            TEST_FRAC
        ),
        "selected_test_group_fraction": (
            selected_group_fraction
        ),
        "actual_train_row_fraction": float(
            len(train_idx) / len(labels)
        ),
        "actual_test_row_fraction": (
            actual_test_row_fraction
        ),
        "distinct_groups_per_class": (
            distinct_groups_per_class
        ),
        "per_class_test_row_fractions": {
            str(class_id): float(
                per_class_test_row_fractions[
                    class_id
                ]
            )
            for class_id in range(
                num_classes
            )
        },
        "train_unique_predictor_groups": int(
            len(train_groups)
        ),
        "test_unique_predictor_groups": int(
            len(test_groups)
        ),
        "train_test_group_overlap": int(
            group_overlap
        ),
        "fixed_round_training": True,
        "validation_split_used": False,
        "test_used_for_model_selection": False,
    }

    return train_idx, test_idx, metadata

# =============================================================================
# 5. CICIoV2024 RAW LOAD + TESTS
# =============================================================================
def cic_class_from_filename(filename: str) -> str:
    return filename.replace("decimal_", "").replace(".csv", "")


def load_cic_raw() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for fname in CIC_FILES:
        path = CIC_DECIMAL_DIR / fname
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        missing = [c for c in CIC_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{fname}: missing predictor columns {missing}")
        df = df.copy()
        df["attack_family"] = cic_class_from_filename(fname)
        df["Label"] = 0 if fname.lower() == "decimal_benign.csv" else 1
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, copy=False)
    merged["feature_hash"] = hash_frame_features(merged, CIC_FEATURE_COLS)
    return merged


def cic_protocol_metadata(
    cic_df: pd.DataFrame,
) -> Dict[str, Any]:
    cols = set(
        cic_df.columns
    )

    hash_class_counts = (
        cic_df.groupby(
            "feature_hash",
            sort=False,
        )["attack_family"]
        .nunique()
    )

    ambiguous_hashes = set(
        hash_class_counts[
            hash_class_counts > 1
        ].index.tolist()
    )

    ambiguous_mask = (
        cic_df["feature_hash"]
        .isin(
            ambiguous_hashes
        )
    )

    distinct_hashes_by_family = {
        str(family): int(
            group[
                "feature_hash"
            ].nunique()
        )
        for family, group
        in cic_df.groupby(
            "attack_family",
            sort=True,
        )
    }

    rows_by_family = {
        str(family): int(
            count
        )
        for family, count
        in cic_df[
            "attack_family"
        ]
        .value_counts()
        .sort_index()
        .items()
    }

    total_unique_hashes = int(
        cic_df[
            "feature_hash"
        ].nunique()
    )

    return {
        "dataset": "CICIoV2024",
        "evaluated_representation": (
            "decimal CAN ID + DATA_0..DATA_7"
        ),
        "predictor_hash_definition": (
            "exact ID + DATA_0..DATA_7 predictor vector"
        ),
        "has_timestamp_field": False,
        "chronological_split_supported_by_selected_representation": False,
        "has_session_identifier": False,
        "session_disjoint_split_supported_by_selected_representation": False,
        "has_source_vehicle_identifier": False,
        "source_vehicle_disjoint_split_supported_by_selected_representation": False,
        "raw_columns_seen": sorted(
            cols
        ),
        "attack_families": sorted(
            cic_df[
                "attack_family"
            ].unique().tolist()
        ),
        "rows_by_attack_family": (
            rows_by_family
        ),
        "total_rows": int(
            len(cic_df)
        ),
        "total_distinct_predictor_hashes": (
            total_unique_hashes
        ),
        "distinct_predictor_hashes_by_attack_family": (
            distinct_hashes_by_family
        ),
        "cross_class_ambiguous_predictor_hashes": int(
            len(
                ambiguous_hashes
            )
        ),
        "rows_on_cross_class_ambiguous_predictor_hashes": int(
            ambiguous_mask.sum()
        ),
        "fraction_of_rows_on_cross_class_ambiguous_predictor_hashes": float(
            ambiguous_mask.mean()
        ),
    }


def run_cic_unseen_attack(cic_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Leave-one-attack-family-out binary detection.

    For each attack family A:
      TRAIN/VAL: BENIGN + all attack families except A
      TEST: held-out BENIGN groups + attack family A only

    Predictor hashes are globally assigned so that no predictor vector used for train
    or validation can appear in test.  Any held-out attack predictor hash that is also
    present among seen training/validation families is removed from the test set; the
    number removed is reported explicitly rather than hidden.
    """
    out_rows: List[Dict[str, Any]] = []
    ensure_dir(out_dir)

    benign = cic_df[cic_df["attack_family"] == "benign"].copy()
    attacks = [x for x in sorted(cic_df["attack_family"].unique()) if x != "benign"]

    # Fix benign train/val/test once, group-disjoint by predictor hash.
    b_train_i, b_val_i, b_test_i = group_disjoint_split_indices(
        benign["feature_hash"].to_numpy(dtype=np.uint64, copy=False)
    )
    benign_train = benign.iloc[b_train_i].copy()
    benign_val = benign.iloc[b_val_i].copy()
    benign_test = benign.iloc[b_test_i].copy()

    for heldout in attacks:
        print(f"\n[CIC UNSEEN ATTACK] hold out = {heldout}")
        seen = cic_df[
            (cic_df["attack_family"] != "benign")
            & (cic_df["attack_family"] != heldout)
        ].copy()
        unseen = cic_df[cic_df["attack_family"] == heldout].copy()

        # Split seen attacks into train/val by group.  The temporary test portion is
        # intentionally not used because the target test condition is the unseen type.
        s_train_i, s_val_i, _ = group_disjoint_split_indices(
            seen["feature_hash"].to_numpy(dtype=np.uint64, copy=False)
        )
        seen_train = seen.iloc[s_train_i].copy()
        seen_val = seen.iloc[s_val_i].copy()

        train = pd.concat([benign_train, seen_train], ignore_index=True)
        val = pd.concat([benign_val, seen_val], ignore_index=True)

        train_val_hashes = set(
            map(int, pd.concat([train["feature_hash"], val["feature_hash"]]).unique())
        )
        unseen_hash = unseen["feature_hash"].astype("uint64")

        duplicate_mask = unseen_hash.isin(
            train_val_hashes
        ).to_numpy(dtype=bool)

        removed_unseen_rows = int(
            duplicate_mask.sum()
        )

        unseen_clean = unseen.loc[
            ~duplicate_mask
        ].copy()

        # Benign test was already group-disjoint from benign train/val, but it can in
        # principle share a predictor hash with an ATTACK row. Remove any such hash.
        bdup = benign_test["feature_hash"].isin(
            train_val_hashes
        ).to_numpy(dtype=bool)
        removed_benign_rows = int(bdup.sum())
        benign_test_clean = benign_test.loc[~bdup].copy()

        test = pd.concat([benign_test_clean, unseen_clean], ignore_index=True)
        test["Label"] = (test["attack_family"] != "benign").astype(np.int8)
        train["Label"] = (train["attack_family"] != "benign").astype(np.int8)
        val["Label"] = (val["attack_family"] != "benign").astype(np.int8)

        # Hard assertion of cross-split duplicate exclusion.
        train_h = set(map(int, train["feature_hash"].unique()))
        val_h = set(map(int, val["feature_hash"].unique()))
        test_h = set(map(int, test["feature_hash"].unique()))
        overlap = {
            "train_val": len(train_h & val_h),
            "train_test": len(train_h & test_h),
            "val_test": len(val_h & test_h),
        }
        if any(overlap.values()):
            raise AssertionError(f"CIC unseen-attack split overlap: {overlap}")

        tag = f"CICIoV2024_unseen_{heldout.replace('-', '_')}"
        result = train_binary_xgboost(
            train,
            val,
            test,
            CIC_FEATURE_COLS,
            "Label",
            out_dir / tag,
            tag,
        )
        m = result["test_metrics"]

        pattern_audit = result[
            "test_unique_pattern_audit"
        ]

        pattern_metrics = (
            pattern_audit.get("metrics")
            if pattern_audit.get("status") == "evaluated"
            else None
        )

        out_rows.append(
            {
                "held_out_attack": heldout,
                "train_rows": len(train),
                "val_rows": len(val),
                "test_benign_rows": int((test["Label"] == 0).sum()),
                "test_unseen_attack_rows": int((test["Label"] == 1).sum()),
                "heldout_rows_removed_due_to_predictor_overlap": removed_unseen_rows,
                "benign_test_rows_removed_due_to_predictor_overlap": removed_benign_rows,
                "train_test_unique_hash_overlap": overlap["train_test"],
                "val_test_unique_hash_overlap": overlap["val_test"],
                "test_unique_predictor_hashes": (
                    pattern_audit.get(
                        "test_unique_predictor_hashes"
                    )
                ),
                "test_ambiguous_predictor_hashes": (
                    pattern_audit.get(
                        "test_ambiguous_predictor_hashes"
                    )
                ),
                "test_unambiguous_unique_predictor_hashes": (
                    pattern_audit.get(
                        "test_unambiguous_unique_predictor_hashes"
                    )
                ),
                "unique_pattern_accuracy": (
                    pattern_metrics.get("accuracy")
                    if pattern_metrics is not None
                    else None
                ),
                "unique_pattern_balanced_accuracy": (
                    pattern_metrics.get(
                        "balanced_accuracy"
                    )
                    if pattern_metrics is not None
                    else None
                ),
                "unique_pattern_precision": (
                    pattern_metrics.get("precision")
                    if pattern_metrics is not None
                    else None
                ),
                "unique_pattern_unseen_attack_recall": (
                    pattern_metrics.get("recall")
                    if pattern_metrics is not None
                    else None
                ),
                "unique_pattern_f1": (
                    pattern_metrics.get("f1")
                    if pattern_metrics is not None
                    else None
                ),
                "unique_pattern_mcc": (
                    pattern_metrics.get("mcc")
                    if pattern_metrics is not None
                    else None
                ),
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "precision": m["precision"],
                "unseen_attack_recall": m["recall"],
                "unseen_attack_recall_wilson95_low": m["recall_wilson95_low"],
                "unseen_attack_recall_wilson95_high": m["recall_wilson95_high"],
                "f1": m["f1"],
                "mcc": m["mcc"],
                "roc_auc": m["roc_auc"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
        )

        del train, val, test, result
        gc.collect()

    out = pd.DataFrame(out_rows)
    out.to_csv(out_dir / "CICIoV2024_leave_one_attack_family_out.csv", index=False)
    return out


def run_cic_multiclass(
    cic_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, Any]:
    print(
        "\n[CIC MULTICLASS] "
        "strict duplicate-group-disjoint 6-class train/test evaluation"
    )

    ensure_dir(out_dir)

    class_names = [
        "benign",
        "DoS",
        "spoofing-GAS",
        "spoofing-RPM",
        "spoofing-SPEED",
        "spoofing-STEERING_WHEEL",
    ]

    class_to_id: Dict[str, int] = {
        name: class_id
        for class_id, name
        in enumerate(class_names)
    }

    df = cic_df[
        cic_df["attack_family"].isin(
            class_names
        )
    ].copy()

    df["ClassID"] = (
        df["attack_family"]
        .map(class_to_id)
        .astype(np.int32)
    )

    labels = df["ClassID"].to_numpy(
        dtype=np.int32,
        copy=False,
    )

    groups = df["feature_hash"].to_numpy(
        dtype=np.uint64,
        copy=False,
    )

    train_idx, test_idx, split_metadata = (
        class_complete_group_disjoint_train_test_indices(
            labels,
            groups,
            RANDOM_STATE,
        )
    )

    train = df.iloc[
        train_idx
    ].reset_index(
        drop=True
    )

    test = df.iloc[
        test_idx
    ].reset_index(
        drop=True
    )

    expected_classes = set(
        range(
            len(class_names)
        )
    )

    train_classes = set(
        train["ClassID"]
        .unique()
        .tolist()
    )

    test_classes = set(
        test["ClassID"]
        .unique()
        .tolist()
    )

    if train_classes != expected_classes:
        raise AssertionError(
            "CICIoV2024 multiclass training partition "
            "does not contain all six classes."
        )

    if test_classes != expected_classes:
        raise AssertionError(
            "CICIoV2024 multiclass test partition "
            "does not contain all six classes."
        )

    train_hashes = set(
        map(
            int,
            train["feature_hash"].unique(),
        )
    )

    test_hashes = set(
        map(
            int,
            test["feature_hash"].unique(),
        )
    )

    hash_overlap = len(
        train_hashes
        & test_hashes
    )

    if hash_overlap != 0:
        raise AssertionError(
            "CICIoV2024 multiclass train/test "
            f"predictor-hash overlap = {hash_overlap}; expected zero."
        )

    class_counts_by_split: Dict[
        str,
        Dict[str, int],
    ] = {}

    for split_name, split_df in (
        ("train", train),
        ("test", test),
    ):
        counts = (
            split_df[
                "attack_family"
            ].value_counts()
        )

        class_counts_by_split[
            split_name
        ] = {
            class_name: int(
                counts.get(
                    class_name,
                    0,
                )
            )
            for class_name
            in class_names
        }

    distinct_patterns_by_class: Dict[
        str,
        Dict[str, int],
    ] = {}

    for split_name, split_df in (
        ("train", train),
        ("test", test),
    ):
        distinct_patterns_by_class[
            split_name
        ] = {
            class_name: int(
                split_df.loc[
                    split_df[
                        "attack_family"
                    ] == class_name,
                    "feature_hash",
                ].nunique()
            )
            for class_name
            in class_names
        }

    result = (
        train_multiclass_xgboost_fixed_train_test(
            train,
            test,
            CIC_FEATURE_COLS,
            "ClassID",
            class_names,
            out_dir,
            "CICIoV2024_multiclass_group_disjoint",
        )
    )

    result["split_definition"] = (
        "Strict class-complete train/test split using exact raw "
        "ID + DATA_0..DATA_7 predictor hashes as indivisible groups. "
        "No predictor hash may occur in both partitions."
    )

    result["three_way_split_feasibility"] = (
        "Not feasible under strict duplicate-group disjointness "
        "because spoofing-GAS contains only two distinct predictor "
        "hash groups in the evaluated decimal representation."
    )

    result["split_metadata"] = (
        split_metadata
    )

    result["class_counts_by_split"] = (
        class_counts_by_split
    )

    result["distinct_predictor_patterns_by_class_and_split"] = (
        distinct_patterns_by_class
    )

    result["train_test_predictor_hash_overlap"] = int(
        hash_overlap
    )

    result["strict_predictor_hash_disjointness_verified"] = bool(
        hash_overlap == 0
    )

    result["all_six_classes_present_in_training"] = bool(
        train_classes
        == expected_classes
    )

    result["all_six_classes_present_in_test"] = bool(
        test_classes
        == expected_classes
    )

    result["model_selection_statement"] = (
        "The predefined XGBoost configuration and fixed "
        f"{NUM_BOOST_ROUND}-round training schedule were used without "
        "validation-based early stopping. The held-out test partition "
        "was not used for hyperparameter selection, threshold selection, "
        "or stopping decisions."
    )

    save_json(
        out_dir
        / "CICIoV2024_multiclass_group_disjoint_summary.json",
        result,
    )

    return result


# =============================================================================
# 6. CSE-CIC-IDS2018 RAW LOAD + TESTS
# =============================================================================
CSE_LABEL_COL = "Label"
CSE_IDENTIFIER_COLS = ["id", "Flow ID", "Src IP", "Dst IP", "Timestamp"]


def load_cse_raw() -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    if not CSE_RAW_CSV.exists():
        raise FileNotFoundError(CSE_RAW_CSV)

    print(f"\n[CSE LOAD] {CSE_RAW_CSV}")
    source_rows: Optional[int] = None
    sampling_fraction: Optional[float] = None

    if CSE_MAX_MODEL_ROWS is None:
        df = pd.read_csv(CSE_RAW_CSV, low_memory=False)
        source_rows = int(len(df))
    else:
        # Match the project's existing Dask-based memory strategy, but retain the raw
        # metadata columns needed for chronological / Flow-ID validation.  Sampling is
        # performed across the complete file, not by taking the first N rows.
        try:
            import dask.dataframe as dd

            ddf = dd.read_csv(
                CSE_RAW_CSV,
                assume_missing=True,
                blocksize="128MB",
            )
            source_rows = int(ddf.shape[0].compute())
            if source_rows > CSE_MAX_MODEL_ROWS:
                sampling_fraction = CSE_MAX_MODEL_ROWS / float(source_rows)
                ddf = ddf.sample(frac=sampling_fraction, random_state=RANDOM_STATE)
            df = ddf.compute()
            # Dask sampling is approximate.  If it returns slightly more than the cap,
            # trim deterministically without introducing first-N temporal bias.
            if len(df) > CSE_MAX_MODEL_ROWS:
                df = df.sample(
                    n=CSE_MAX_MODEL_ROWS,
                    random_state=RANDOM_STATE,
                    replace=False,
                )
        except Exception as exc:
            raise RuntimeError(
                "CSE capped loading requires Dask (already used by the project). "
                "Set CSE_MAX_MODEL_ROWS=None for direct pandas loading or fix the Dask read. "
                f"Original error: {exc}"
            ) from exc

    df = df.reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    if CSE_LABEL_COL not in df.columns:
        raise ValueError(f"CSE raw file has no {CSE_LABEL_COL!r} column.")

    # Preserve original attack identity before binary mapping.
    df["AttackType"] = normalize_label_series(df[CSE_LABEL_COL])
    df["LabelBinary"] = (df["AttackType"] != "BENIGN").astype(np.int8)

    feature_cols = [
        c
        for c in df.columns
        if c
        not in set(CSE_IDENTIFIER_COLS + [CSE_LABEL_COL, "AttackType", "LabelBinary"])
    ]

    # Normalize numeric infinities to NaN; categorical columns are left as text.
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    # Hash raw predictor representation before preprocessing.
    df["feature_hash"] = hash_frame_features(df, feature_cols)

    meta = {
        "raw_path": str(CSE_RAW_CSV),
        "raw_sha256": sha256_file(CSE_RAW_CSV),
        "source_rows_before_sampling": source_rows,
        "rows_loaded": int(len(df)),
        "row_cap": CSE_MAX_MODEL_ROWS,
        "sampling_fraction_across_full_file": sampling_fraction,
        "feature_count": len(feature_cols),
        "has_timestamp": "Timestamp" in df.columns,
        "has_flow_id": "Flow ID" in df.columns,
        "has_source_vehicle_identifier": False,
        "attack_type_counts": {
            str(k): int(v) for k, v in df["AttackType"].value_counts().items()
        },
    }
    return df, feature_cols, meta


def cse_duplicate_group_disjoint(
    df: pd.DataFrame, feature_cols: List[str], out_dir: Path
) -> Dict[str, Any]:
    print("\n[CSE DUPLICATE-GROUP-DISJOINT] binary")
    groups = df["feature_hash"].to_numpy(dtype=np.uint64, copy=False)
    tr, va, te = group_disjoint_split_indices(groups)
    overlap = assert_group_disjoint(groups, tr, va, te)

    train = df.iloc[tr].copy()
    val = df.iloc[va].copy()
    test = df.iloc[te].copy()
    for part in (train, val, test):
        part["Label"] = part["LabelBinary"].astype(np.int8)

    result = train_binary_xgboost(
        train,
        val,
        test,
        feature_cols,
        "Label",
        out_dir,
        "CSECICIDS2018_binary_duplicate_group_disjoint",
    )
    result["group_overlap"] = overlap
    result["split_definition"] = "70/15/15 GroupShuffleSplit on exact raw predictor hashes"
    save_json(out_dir / "CSECICIDS2018_binary_duplicate_group_disjoint_summary.json", result)
    return result


def cse_session_disjoint(
    df: pd.DataFrame, feature_cols: List[str], out_dir: Path
) -> Dict[str, Any]:
    print(
        "\n[CSE FLOW-ID-DISJOINT] "
        "dataset-provided grouping proxy"
    )
    if "Flow ID" not in df.columns:
        result = {
            "status": "not_supported",
            "reason": "Flow ID column is absent from the raw CSE file.",
        }
        save_json(out_dir / "CSECICIDS2018_session_disjoint_summary.json", result)
        return result

    valid = df["Flow ID"].notna()
    d = df.loc[valid].copy()
    d["FlowGroup"] = pd.factorize(d["Flow ID"].astype("string"), sort=True)[0].astype(
        np.int64
    )
    groups = d["FlowGroup"].to_numpy(dtype=np.int64, copy=False)
    tr, va, te = group_disjoint_split_indices(groups)
    overlap = assert_group_disjoint(groups, tr, va, te)

    train = d.iloc[tr].copy()
    val = d.iloc[va].copy()
    test = d.iloc[te].copy()
    for part in (train, val, test):
        part["Label"] = part["LabelBinary"].astype(np.int8)

    # Report predictor duplicate overlap separately.  Session separation alone does not
    # guarantee duplicate-vector separation, which is exactly the reviewer's concern.
    train_h = set(map(int, train["feature_hash"].unique()))
    val_h = set(map(int, val["feature_hash"].unique()))
    test_h = set(map(int, test["feature_hash"].unique()))
    predictor_overlap = {
        "train_val": len(train_h & val_h),
        "train_test": len(train_h & test_h),
        "val_test": len(val_h & test_h),
    }

    result = train_binary_xgboost(
        train,
        val,
        test,
        feature_cols,
        "Label",
        out_dir,
        "CSECICIDS2018_binary_FlowID_disjoint",
    )
    result["flow_group_overlap"] = overlap
    result["predictor_hash_overlap"] = predictor_overlap
    result["split_definition"] = (
        "Approximately 70/15/15 GroupShuffleSplit by raw Flow ID. "
        "Flow ID is used as the dataset-provided grouping proxy and is "
        "not interpreted as proof of separation by a higher-level "
        "application or network session identity."
    )
    save_json(out_dir / "CSECICIDS2018_session_disjoint_summary.json", result)
    return result


def parse_cse_timestamp(s: pd.Series) -> pd.Series:
    # The improved CSE file has historically appeared with day-first timestamp text.
    # pandas' mixed parser makes the script robust to seconds / fractional seconds.
    try:
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return parsed


def cse_chronological(
    df: pd.DataFrame, feature_cols: List[str], out_dir: Path
) -> Dict[str, Any]:
    print("\n[CSE CHRONOLOGICAL] earliest 70% / next 15% / latest 15%")
    if "Timestamp" not in df.columns:
        result = {
            "status": "not_supported",
            "reason": "Timestamp column is absent from the raw CSE file.",
        }
        save_json(out_dir / "CSECICIDS2018_chronological_summary.json", result)
        return result

    ts = parse_cse_timestamp(df["Timestamp"])
    valid = ts.notna()
    d = df.loc[valid].copy()
    d["_timestamp_parsed"] = ts.loc[valid]
    d = d.sort_values("_timestamp_parsed", kind="mergesort").reset_index(drop=True)

    n = len(d)
    i1 = int(math.floor(TRAIN_FRAC * n))
    i2 = int(math.floor((TRAIN_FRAC + VAL_FRAC) * n))
    train = d.iloc[:i1].copy()
    val = d.iloc[i1:i2].copy()
    test = d.iloc[i2:].copy()

    for part in (train, val, test):
        part["Label"] = part["LabelBinary"].astype(np.int8)

    # Chronology can legitimately create a one-class validation period.  If so, use a
    # deterministic time-preserving adjustment: extend validation backward until both
    # classes are present, without moving any test row into training.
    if np.unique(val["Label"]).size < 2:
        found = False
        for back_frac in (0.05, 0.10, 0.15, 0.20):
            j = max(0, i1 - int(back_frac * n))
            candidate = d.iloc[j:i2].copy()
            candidate["Label"] = candidate["LabelBinary"].astype(np.int8)
            if np.unique(candidate["Label"]).size == 2:
                val = candidate
                train = d.iloc[:j].copy()
                train["Label"] = train["LabelBinary"].astype(np.int8)
                found = True
                break
        if not found:
            result = {
                "status": "not_trainable",
                "reason": "Chronological validation interval contains a single class even after backward extension.",
                "train_time_range": [
                    str(train["_timestamp_parsed"].min()),
                    str(train["_timestamp_parsed"].max()),
                ],
                "val_time_range": [
                    str(val["_timestamp_parsed"].min()),
                    str(val["_timestamp_parsed"].max()),
                ],
                "test_time_range": [
                    str(test["_timestamp_parsed"].min()),
                    str(test["_timestamp_parsed"].max()),
                ],
            }
            save_json(out_dir / "CSECICIDS2018_chronological_summary.json", result)
            return result
    train_hashes = set(
        map(
            int,
            train[
                "feature_hash"
            ].unique(),
        )
    )

    val_hashes = set(
        map(
            int,
            val[
                "feature_hash"
            ].unique(),
        )
    )

    test_hashes = set(
        map(
            int,
            test[
                "feature_hash"
            ].unique(),
        )
    )

    predictor_hash_overlap = {
        "train_val": int(
            len(
                train_hashes
                & val_hashes
            )
        ),
        "train_test": int(
            len(
                train_hashes
                & test_hashes
            )
        ),
        "val_test": int(
            len(
                val_hashes
                & test_hashes
            )
        ),
    }

    earlier_hashes = (
            train_hashes
            | val_hashes
    )

    test_seen_pattern_mask = (
        test[
            "feature_hash"
        ].isin(
            earlier_hashes
        )
    )

    test_rows_matching_earlier_predictor = int(
        test_seen_pattern_mask.sum()
    )

    test_rows_matching_earlier_predictor_fraction = float(
        test_seen_pattern_mask.mean()
    )

    result = train_binary_xgboost(
        train,
        val,
        test,
        feature_cols,
        "Label",
        out_dir,
        "CSECICIDS2018_binary_chronological",
    )
    chronological_scope = (
        "complete raw dataset"
        if CSE_MAX_MODEL_ROWS is None
        else (
            "deterministic whole-file sample capped at "
            f"{CSE_MAX_MODEL_ROWS:,} rows"
        )
    )

    result["split_definition"] = (
        "Chronologically ordered earliest ~70%, next ~15%, and latest "
        f"~15% on the {chronological_scope}; no shuffle is applied after "
        "timestamp ordering."
    )

    result["chronological_evaluation_scope"] = (
        chronological_scope
    )
    result["predictor_hash_overlap"] = (
        predictor_hash_overlap
    )

    result[
        "test_rows_matching_train_or_validation_predictor"
    ] = (
        test_rows_matching_earlier_predictor
    )

    result[
        "test_fraction_matching_train_or_validation_predictor"
    ] = (
        test_rows_matching_earlier_predictor_fraction
    )

    result[
        "chronological_test_predictor_disjoint_from_earlier_splits"
    ] = bool(
        predictor_hash_overlap[
            "train_test"
        ] == 0
        and predictor_hash_overlap[
            "val_test"
        ] == 0
    )
    result["train_time_range"] = [
        str(train["_timestamp_parsed"].min()),
        str(train["_timestamp_parsed"].max()),
    ]
    result["val_time_range"] = [
        str(val["_timestamp_parsed"].min()),
        str(val["_timestamp_parsed"].max()),
    ]
    result["test_time_range"] = [
        str(test["_timestamp_parsed"].min()),
        str(test["_timestamp_parsed"].max()),
    ]
    result["attack_types_train"] = sorted(train["AttackType"].unique().tolist())
    result["attack_types_val"] = sorted(val["AttackType"].unique().tolist())
    result["attack_types_test"] = sorted(test["AttackType"].unique().tolist())
    result["attack_types_unseen_in_train_but_present_in_test"] = sorted(
        set(result["attack_types_test"]) - set(result["attack_types_train"])
    )
    save_json(out_dir / "CSECICIDS2018_chronological_summary.json", result)
    return result


def cse_attack_holdouts(df: pd.DataFrame) -> List[str]:
    counts = df.loc[df["AttackType"] != "BENIGN", "AttackType"].value_counts()
    qualified = counts[counts >= CSE_MIN_HOLDOUT_ATTACK_ROWS]
    names = qualified.index.tolist()
    if CSE_MAX_HOLDOUT_ATTACKS is not None:
        names = names[:CSE_MAX_HOLDOUT_ATTACKS]
    return [str(x) for x in names]


def run_cse_unseen_attack(
    df: pd.DataFrame, feature_cols: List[str], out_dir: Path
) -> pd.DataFrame:
    print("\n[CSE UNSEEN ATTACK] leave one attack type completely out of training")
    ensure_dir(out_dir)
    holdouts = cse_attack_holdouts(df)
    if not holdouts:
        raise ValueError("No CSE attack types satisfy CSE_MIN_HOLDOUT_ATTACK_ROWS.")

    # Fixed benign split, predictor-hash-disjoint.
    benign = df[df["AttackType"] == "BENIGN"].copy()
    bt, bv, be = group_disjoint_split_indices(
        benign["feature_hash"].to_numpy(dtype=np.uint64, copy=False)
    )
    benign_train = benign.iloc[bt].copy()
    benign_val = benign.iloc[bv].copy()
    benign_test = benign.iloc[be].copy()

    rows: List[Dict[str, Any]] = []
    for heldout in holdouts:
        print(f"  hold out: {heldout}")
        seen = df[(df["AttackType"] != "BENIGN") & (df["AttackType"] != heldout)].copy()
        unseen = df[df["AttackType"] == heldout].copy()

        st, sv, _ = group_disjoint_split_indices(
            seen["feature_hash"].to_numpy(dtype=np.uint64, copy=False)
        )
        train = pd.concat([benign_train, seen.iloc[st]], ignore_index=True)
        val = pd.concat([benign_val, seen.iloc[sv]], ignore_index=True)

        train_val_hashes = set(
            map(int, pd.concat([train["feature_hash"], val["feature_hash"]]).unique())
        )
        udup = unseen["feature_hash"].isin(
            train_val_hashes
        ).to_numpy(dtype=bool)

        bdup = benign_test["feature_hash"].isin(
            train_val_hashes
        ).to_numpy(dtype=bool)

        unseen_clean = unseen.loc[
            ~udup
        ].copy()

        benign_clean = benign_test.loc[
            ~bdup
        ].copy()

        train["Label"] = (train["AttackType"] != "BENIGN").astype(np.int8)
        val["Label"] = (val["AttackType"] != "BENIGN").astype(np.int8)
        test = pd.concat([benign_clean, unseen_clean], ignore_index=True)
        test["Label"] = (test["AttackType"] != "BENIGN").astype(np.int8)

        trh = set(map(int, train["feature_hash"].unique()))
        vah = set(map(int, val["feature_hash"].unique()))
        teh = set(map(int, test["feature_hash"].unique()))
        overlap = {
            "train_val": len(trh & vah),
            "train_test": len(trh & teh),
            "val_test": len(vah & teh),
        }
        if any(overlap.values()):
            raise AssertionError(f"CSE unseen-attack hash overlap for {heldout}: {overlap}")

        safe_name = "".join(ch if ch.isalnum() else "_" for ch in heldout)[:80]
        tag = f"CSECICIDS2018_unseen_{safe_name}"
        result = train_binary_xgboost(
            train,
            val,
            test,
            feature_cols,
            "Label",
            out_dir / safe_name,
            tag,
        )
        m = result["test_metrics"]

        pattern_audit = result[
            "test_unique_pattern_audit"
        ]

        pattern_metrics = (
            pattern_audit.get("metrics")
            if pattern_audit.get("status") == "evaluated"
            else None
        )

        rows.append(
            {
                "held_out_attack": heldout,
                "raw_heldout_rows": int(len(unseen)),
                "heldout_rows_removed_due_to_predictor_overlap": int(udup.sum()),
                "test_unseen_attack_rows": int((test["Label"] == 1).sum()),
                "test_benign_rows": int((test["Label"] == 0).sum()),
                "test_unique_predictor_hashes": (
                    pattern_audit.get(
                        "test_unique_predictor_hashes"
                    )
                ),
                "test_ambiguous_predictor_hashes": (
                    pattern_audit.get(
                        "test_ambiguous_predictor_hashes"
                    )
                ),
                "test_unambiguous_unique_predictor_hashes": (
                    pattern_audit.get(
                        "test_unambiguous_unique_predictor_hashes"
                    )
                ),
                "unique_pattern_accuracy": (
                    pattern_metrics.get("accuracy")
                    if pattern_metrics is not None
                    else None
                ),
                "unique_pattern_balanced_accuracy": (
                    pattern_metrics.get(
                        "balanced_accuracy"
                    )
                    if pattern_metrics is not None
                    else None
                ),
                "unique_pattern_precision": (
                    pattern_metrics.get("precision")
                    if pattern_metrics is not None
                    else None
                ),
                "unique_pattern_unseen_attack_recall": (
                    pattern_metrics.get("recall")
                    if pattern_metrics is not None
                    else None
                ),
                "unique_pattern_f1": (
                    pattern_metrics.get("f1")
                    if pattern_metrics is not None
                    else None
                ),
                "unique_pattern_mcc": (
                    pattern_metrics.get("mcc")
                    if pattern_metrics is not None
                    else None
                ),
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "precision": m["precision"],
                "unseen_attack_recall": m["recall"],
                "unseen_attack_recall_wilson95_low": m["recall_wilson95_low"],
                "unseen_attack_recall_wilson95_high": m["recall_wilson95_high"],
                "f1": m["f1"],
                "mcc": m["mcc"],
                "roc_auc": m["roc_auc"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
        )
        del train, val, test, result
        gc.collect()

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "CSECICIDS2018_leave_one_attack_type_out.csv", index=False)
    return out


def run_cse_multiclass(
    df: pd.DataFrame, feature_cols: List[str], out_dir: Path
) -> Dict[str, Any]:
    print("\n[CSE MULTICLASS] attack-type recognition with duplicate-group-disjoint split")
    ensure_dir(out_dir)

    counts = df["AttackType"].value_counts()
    keep_classes = counts[counts >= CSE_MIN_MULTICLASS_ROWS].index.tolist()
    d = df[df["AttackType"].isin(keep_classes)].copy()

    # Optional deterministic cap per class to keep the test executable while retaining
    # every sufficiently represented attack class.
    if CSE_MULTICLASS_MAX_ROWS_PER_CLASS is not None:
        parts = []
        for name, sub in d.groupby("AttackType", sort=True):
            if len(sub) > CSE_MULTICLASS_MAX_ROWS_PER_CLASS:
                sub = sub.sample(
                    n=CSE_MULTICLASS_MAX_ROWS_PER_CLASS,
                    random_state=RANDOM_STATE,
                    replace=False,
                )
            parts.append(sub)
        d = pd.concat(parts, ignore_index=True)

    class_names: List[str] = sorted(
        str(name)
        for name in d["AttackType"].dropna().unique().tolist()
    )

    class_to_id: Dict[str, int] = {
        name: class_id
        for class_id, name in enumerate(class_names)
    }

    d["ClassID"] = (
        d["AttackType"]
        .astype(str)
        .map(class_to_id)
        .astype(np.int32)
    )

    groups = d["feature_hash"].to_numpy(dtype=np.uint64, copy=False)
    labels = d["ClassID"].to_numpy(
        dtype=np.int32,
        copy=False,
    )

    distinct_groups_per_class: Dict[str, int] = {
        class_names[class_id]: int(
            np.unique(
                groups[labels == class_id]
            ).size
        )
        for class_id in range(
            len(class_names)
        )
    }

    minimum_distinct_groups = min(
        distinct_groups_per_class.values()
    )

    if minimum_distinct_groups < 3:
        result = {
            "status": "not_evaluable_as_strict_three_way_split",
            "reason": (
                "At least one retained CSE-CIC-IDS2018 class contains "
                "fewer than three distinct predictor-hash groups, so a "
                "class-complete train/validation/test split cannot be "
                "constructed while preserving strict predictor-hash "
                "disjointness."
            ),
            "distinct_predictor_groups_per_class": (
                distinct_groups_per_class
            ),
            "class_selection_min_rows": (
                CSE_MIN_MULTICLASS_ROWS
            ),
        }

        save_json(
            out_dir
            / "CSECICIDS2018_multiclass_group_disjoint_summary.json",
            result,
        )

        return result
    tr, va, te = group_disjoint_split_indices(groups)
    overlap = assert_group_disjoint(groups, tr, va, te)
    train, val, test = d.iloc[tr].copy(), d.iloc[va].copy(), d.iloc[te].copy()

    # Rare classes can accidentally disappear from one group-split partition.  Retry a
    # finite deterministic set of seeds; never silently drop a class after split.
    expected = set(range(len(class_names)))
    if any(
        set(np.unique(part["ClassID"])) != expected for part in (train, val, test)
    ):
        success = False
        for seed in range(RANDOM_STATE + 1, RANDOM_STATE + 101):
            tr2, va2, te2 = group_disjoint_split_indices(groups, seed)
            candidates = [d.iloc[tr2], d.iloc[va2], d.iloc[te2]]
            if all(set(np.unique(p["ClassID"])) == expected for p in candidates):
                tr, va, te = tr2, va2, te2
                train, val, test = [p.copy() for p in candidates]
                overlap = assert_group_disjoint(groups, tr, va, te)
                success = True
                break
        if not success:
            result = {
                "status": "no_class_complete_three_way_split_found",
                "reason": (
                    "No class-complete three-way predictor-hash-disjoint "
                    "partition was found in the deterministic seed search."
                ),
                "distinct_predictor_groups_per_class": (
                    distinct_groups_per_class
                ),
                "seeds_examined": 100,
            }

            save_json(
                out_dir
                / "CSECICIDS2018_multiclass_group_disjoint_summary.json",
                result,
            )

            return result

    result = train_multiclass_xgboost(
        train,
        val,
        test,
        feature_cols,
        "ClassID",
        class_names,
        out_dir,
        "CSECICIDS2018_multiclass_group_disjoint",
    )
    result["group_overlap"] = overlap
    result["class_selection_min_rows"] = CSE_MIN_MULTICLASS_ROWS
    result["max_rows_per_class"] = CSE_MULTICLASS_MAX_ROWS_PER_CLASS
    result["raw_counts_for_kept_classes"] = {
        class_name: int(
            counts.get(class_name, 0)
        )
        for class_name in class_names
    }
    save_json(out_dir / "CSECICIDS2018_multiclass_group_disjoint_summary.json", result)
    return result


# =============================================================================
# 7. REVIEWER-FACING SUMMARY
# =============================================================================
def build_protocol_matrix(
    main_audits: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    cse_dup = main_audits.get("CSECICIDS2018", {}).get(
        "duplicate_group_disjoint_main_splits"
    )
    cic_dup = main_audits.get("CICIoV2024", {}).get(
        "duplicate_group_disjoint_main_splits"
    )

    rows.append(
        {
            "dataset": "CSE-CIC-IDS2018",
            "published_main_split": "random row-level stratified 70/15/15",
            "chronological_main_split": "No",
            "session_disjoint_main_split": (
            "No; the additional stress test uses Flow ID only as the "
            "dataset-provided grouping proxy"
            ),
            "source_vehicle_disjoint_main_split": "No / no source-vehicle identity used",
            "duplicate_group_disjoint_main_split": (
                "Yes" if cse_dup is True else "No" if cse_dup is False else "Not audited"
            ),
            "additional_stress_tests_in_this_script": (
            "duplicate-group-disjoint; Flow-ID-disjoint using the "
            "dataset-provided grouping proxy; chronological validation on "
            "the deterministic whole-file sample; leave-one-attack-type-out; "
            "multiclass"
            ),
        }
    )
    rows.append(
        {
            "dataset": "CICIoV2024",
            "published_main_split": "random row-level stratified 70/15/15",
            "chronological_main_split": "No; selected decimal representation has no timestamp field",
            "session_disjoint_main_split": "No; selected decimal representation has no session identifier",
            "source_vehicle_disjoint_main_split": "No; selected decimal representation has no source-vehicle identifier",
            "duplicate_group_disjoint_main_split": (
                "Yes" if cic_dup is True else "No" if cic_dup is False else "Not audited"
            ),
            "additional_stress_tests_in_this_script": "duplicate-group-disjoint leave-one-attack-family-out; duplicate-group-disjoint 6-class detection",
        }
    )
    return pd.DataFrame(rows)


def aggregate_key_results(
    main_audits: Dict[str, Dict[str, Any]],
    cic_meta: Optional[Dict[str, Any]],
    cse_meta: Optional[Dict[str, Any]],
    cic_unseen: Optional[pd.DataFrame],
    cic_multi: Optional[Dict[str, Any]],
    cse_dup: Optional[Dict[str, Any]],
    cse_session: Optional[Dict[str, Any]],
    cse_chrono: Optional[Dict[str, Any]],
    cse_unseen: Optional[pd.DataFrame],
    cse_multi: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"main_split_audits": {}}
    if cic_meta is not None:
        summary["CICIoV2024_protocol_metadata"] = (
            cic_meta
        )

    if cse_meta is not None:
        summary["CSECICIDS2018_protocol_metadata"] = (
            cse_meta
        )
    for ds, audit in main_audits.items():
        summary["main_split_audits"][ds] = {
            "duplicate_group_disjoint": audit.get("duplicate_group_disjoint_main_splits"),
            "unique_hash_intersections": audit.get("unique_hash_intersections"),
            "test_rows_matching_any_train_predictor": audit.get("row_level_overlap", {}).get(
                "test_rows_matching_any_train_predictor"
            ),
        }

    if cic_unseen is not None and not cic_unseen.empty:
        summary["CICIoV2024_unseen_attack"] = {
            "attack_families_evaluated": int(
                len(cic_unseen)
            ),
            "mean_unseen_attack_recall": float(
                cic_unseen[
                    "unseen_attack_recall"
                ].mean()
            ),
            "minimum_unseen_attack_recall": float(
                cic_unseen[
                    "unseen_attack_recall"
                ].min()
            ),
            "maximum_unseen_attack_recall": float(
                cic_unseen[
                    "unseen_attack_recall"
                ].max()
            ),
            "per_attack_family": (
                cic_unseen.to_dict(
                    orient="records"
                )
            ),
        }
    if cic_multi is not None:
        summary["CICIoV2024_multiclass"] = (
            cic_multi
        )
    if cse_dup is not None:
        summary[
            "CSECICIDS2018_duplicate_group_disjoint"
        ] = cse_dup
    if cse_session is not None:
        summary[
            "CSECICIDS2018_FlowID_disjoint"
        ] = cse_session
    if cse_chrono is not None:
        summary[
            "CSECICIDS2018_chronological"
        ] = cse_chrono
    if cse_unseen is not None and not cse_unseen.empty:
        summary["CSECICIDS2018_unseen_attack"] = {
            "attack_types_evaluated": int(
                len(cse_unseen)
            ),
            "mean_unseen_attack_recall": float(
                cse_unseen[
                    "unseen_attack_recall"
                ].mean()
            ),
            "minimum_unseen_attack_recall": float(
                cse_unseen[
                    "unseen_attack_recall"
                ].min()
            ),
            "maximum_unseen_attack_recall": float(
                cse_unseen[
                    "unseen_attack_recall"
                ].max()
            ),
            "per_attack_type": (
                cse_unseen.to_dict(
                    orient="records"
                )
            ),
        }
    if cse_multi is not None:
        summary[
            "CSECICIDS2018_multiclass"
        ] = cse_multi
    return summary


def write_reviewer_report(
    protocol_matrix: pd.DataFrame,
    aggregate: Dict[str, Any],
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Reviewer 8 — Dataset Validation and Generalization Audit")
    lines.append("")
    lines.append("This report is generated from the FL-BC-IDS reviewer-validation script.")
    lines.append("")
    lines.append("## Published split protocol")
    lines.append("")
    try:
        lines.append(protocol_matrix.to_markdown(index=False))
    except Exception:
        lines.append(protocol_matrix.to_csv(index=False))
    lines.append("")
    lines.append("## Key machine-readable results")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(aggregate, indent=2, ensure_ascii=False, default=json_default))
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation guardrails")
    lines.append("")
    lines.append(
        "- The original full-system experiments remain controlled binary FL-BC-IDS benchmark runs; the script does not relabel them as chronological, session-disjoint, source-vehicle-disjoint, or duplicate-group-disjoint unless the audit proves that property."
    )
    lines.append(
        "- Source-vehicle separation is not inferred from simulated FL client shards. A source-vehicle-disjoint claim requires an actual source-vehicle identifier in the evaluated raw representation."
    )
    lines.append(
        "- Leave-one-attack-out tests remove the held-out attack type from both training and validation and exclude test predictor hashes that also occur in training/validation."
    )
    lines.append(
        "- Multiclass tests use the same XGBoost model family as an additional dataset-generalization diagnostic; they do not change the DP/Groth16 claims of the main binary FL-BC-IDS implementation."
    )
    lines.append(
        "- Flow-ID-disjoint CSE-CIC-IDS2018 evaluation uses Flow ID as "
        "the dataset-provided grouping proxy; it is not presented as proof "
        "of separation by a higher-level session identity."
    )

    lines.append(
        "- The CSE-CIC-IDS2018 chronological stress test is performed on "
        "the deterministic whole-file sample materialized by this harness "
        "under its configured row cap, unless that cap is explicitly disabled."
    )

    lines.append(
        "- Where feature_hash is available, the report distinguishes "
        "row-weighted metrics from duplicate-collapsed unique-pattern metrics. "
        "Each unambiguous predictor pattern contributes one equal-weight "
        "observation to the latter."
    )

    lines.append(
        "- Predictor hashes carrying conflicting labels are counted "
        "explicitly as ambiguous and are not silently treated as independent "
        "clean observations in duplicate-collapsed metrics."
    )
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 8. MAIN
# =============================================================================
def main() -> None:
    seed_everything()
    ensure_dir(OUTPUT_ROOT)

    run_manifest: Dict[str, Any] = {
        "script": Path(__file__).name,
        "versions": versions(),
        "random_state": RANDOM_STATE,
        "paths": {
            "CSE_RAW_CSV": str(CSE_RAW_CSV),
            "CSE_PREPROC_DIR": str(CSE_PREPROC_DIR),
            "CIC_DECIMAL_DIR": str(CIC_DECIMAL_DIR),
            "CIC_PREPROC_DIR": str(CIC_PREPROC_DIR),
            "OUTPUT_ROOT": str(OUTPUT_ROOT),
        },
        "switches": {
            "RUN_MAIN_SPLIT_DUPLICATE_AUDIT": RUN_MAIN_SPLIT_DUPLICATE_AUDIT,
            "RUN_CIC_UNSEEN_ATTACK": RUN_CIC_UNSEEN_ATTACK,
            "RUN_CIC_MULTICLASS": RUN_CIC_MULTICLASS,
            "RUN_CSE_DUPLICATE_GROUP_DISJOINT": RUN_CSE_DUPLICATE_GROUP_DISJOINT,
            "RUN_CSE_SESSION_DISJOINT": RUN_CSE_SESSION_DISJOINT,
            "RUN_CSE_CHRONOLOGICAL": RUN_CSE_CHRONOLOGICAL,
            "RUN_CSE_UNSEEN_ATTACK": RUN_CSE_UNSEEN_ATTACK,
            "RUN_CSE_MULTICLASS": RUN_CSE_MULTICLASS,
        },
    }
    save_json(OUTPUT_ROOT / "run_manifest.json", run_manifest)

    main_audits: Dict[str, Dict[str, Any]] = {}
    cic_unseen_df: Optional[pd.DataFrame] = None
    cic_multi_result: Optional[Dict[str, Any]] = None
    cse_dup_result: Optional[Dict[str, Any]] = None
    cse_session_result: Optional[Dict[str, Any]] = None
    cse_chrono_result: Optional[Dict[str, Any]] = None
    cse_unseen_df: Optional[pd.DataFrame] = None
    cse_multi_result: Optional[Dict[str, Any]] = None
    cic_meta: Optional[Dict[str, Any]] = None
    cse_meta: Optional[Dict[str, Any]] = None

    if RUN_MAIN_SPLIT_DUPLICATE_AUDIT:
        audit_dir = OUTPUT_ROOT / "01_main_split_duplicate_audit"
        main_audits["CSECICIDS2018"] = audit_main_split_duplicates(
            "CSECICIDS2018", CSE_MAIN_SPLITS, "Label", audit_dir
        )
        main_audits["CICIoV2024"] = audit_main_split_duplicates(
            "CICIoV2024", CIC_MAIN_SPLITS, "Label", audit_dir
        )

    # CIC is loaded once and reused.
    if RUN_CIC_UNSEEN_ATTACK or RUN_CIC_MULTICLASS:
        cic_df = load_cic_raw()
        cic_meta = cic_protocol_metadata(cic_df)
        save_json(OUTPUT_ROOT / "02_CICIoV2024" / "protocol_metadata.json", cic_meta)

        if RUN_CIC_UNSEEN_ATTACK:
            cic_unseen_df = run_cic_unseen_attack(
                cic_df, OUTPUT_ROOT / "02_CICIoV2024" / "unseen_attack"
            )
        if RUN_CIC_MULTICLASS:
            cic_multi_result = run_cic_multiclass(
                cic_df, OUTPUT_ROOT / "02_CICIoV2024" / "multiclass"
            )
        del cic_df
        gc.collect()

    # CSE is loaded once because several stress tests require raw labels/metadata.
    if any(
        [
            RUN_CSE_DUPLICATE_GROUP_DISJOINT,
            RUN_CSE_SESSION_DISJOINT,
            RUN_CSE_CHRONOLOGICAL,
            RUN_CSE_UNSEEN_ATTACK,
            RUN_CSE_MULTICLASS,
        ]
    ):
        cse_df, cse_feature_cols, cse_meta = load_cse_raw()
        save_json(OUTPUT_ROOT / "03_CSECICIDS2018" / "protocol_metadata.json", cse_meta)

        if RUN_CSE_DUPLICATE_GROUP_DISJOINT:
            cse_dup_result = cse_duplicate_group_disjoint(
                cse_df,
                cse_feature_cols,
                OUTPUT_ROOT / "03_CSECICIDS2018" / "duplicate_group_disjoint",
            )
        if RUN_CSE_SESSION_DISJOINT:
            cse_session_result = cse_session_disjoint(
                cse_df,
                cse_feature_cols,
                OUTPUT_ROOT / "03_CSECICIDS2018" / "session_disjoint",
            )
        if RUN_CSE_CHRONOLOGICAL:
            cse_chrono_result = cse_chronological(
                cse_df,
                cse_feature_cols,
                OUTPUT_ROOT / "03_CSECICIDS2018" / "chronological",
            )
        if RUN_CSE_UNSEEN_ATTACK:
            cse_unseen_df = run_cse_unseen_attack(
                cse_df,
                cse_feature_cols,
                OUTPUT_ROOT / "03_CSECICIDS2018" / "unseen_attack",
            )
        if RUN_CSE_MULTICLASS:
            cse_multi_result = run_cse_multiclass(
                cse_df,
                cse_feature_cols,
                OUTPUT_ROOT / "03_CSECICIDS2018" / "multiclass",
            )

        del cse_df
        gc.collect()

    protocol = build_protocol_matrix(main_audits)
    protocol.to_csv(OUTPUT_ROOT / "validation_protocol_matrix.csv", index=False)

    aggregate = aggregate_key_results(
        main_audits,
        cic_meta,
        cse_meta,
        cic_unseen_df,
        cic_multi_result,
        cse_dup_result,
        cse_session_result,
        cse_chrono_result,
        cse_unseen_df,
        cse_multi_result,
    )
    save_json(OUTPUT_ROOT / "reviewer8_key_results.json", aggregate)
    write_reviewer_report(
        protocol,
        aggregate,
        OUTPUT_ROOT / "Reviewer8_Dataset_Validation_Report.md",
    )

    print("\n" + "=" * 100)
    print("Reviewer 8 validation suite completed.")
    print(f"Outputs: {OUTPUT_ROOT}")
    print("Primary files:")
    print(f"  - {OUTPUT_ROOT / 'validation_protocol_matrix.csv'}")
    print(f"  - {OUTPUT_ROOT / 'reviewer8_key_results.json'}")
    print(f"  - {OUTPUT_ROOT / 'Reviewer8_Dataset_Validation_Report.md'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
