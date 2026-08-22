#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Reviewer Concern #2 — Complete Temporal Validation for FL-BC-IDS
================================================================

PURPOSE
-------
This is a SEPARATE, CSE-CIC-IDS2018-only reviewer experiment for the temporal
generalization concern. It does not modify the principal FL-BC-IDS, DP, SSI,
Groth16, blockchain, multi-seed, CICIoV2024, or strict unseen-attack experiments.

WHY THIS UNIFIED SCRIPT EXISTS
------------------------------
A strict rolling-origin audit was first performed with whole preceding blocks
reserved for calibration. That audit produced zero eligible "seen-attack"
temporal test rows in every predefined fold because the CSE-CIC-IDS2018 attack
campaigns are strongly scheduled in time: attack families often first appear in
the immediately preceding block, which the strict protocol had reserved
entirely for calibration rather than training.

This unified script keeps that strict result instead of hiding it, then adds a
clearly separated supervised periodic-refresh evaluation that can actually
measure temporal transfer after an attack family has entered training history.

THE SCRIPT GENERATES FOUR EVIDENCE LAYERS
-----------------------------------------
A. Chronology / campaign-structure audit
   - exact source verification
   - deterministic source-wide 3,000,000-row sample
   - 10 timestamp-safe chronological blocks
   - attack-family coverage over time

B. Strict non-adaptive rolling-origin diagnostic (retained)
   - exactly five folds:
       train B1-B4, cal B5, test B6
       ...
       train B1-B8, cal B9, test B10
   - no attack examples from the calibration block enter training
   - primary seen-attack temporal endpoint is evaluated only when eligible
   - novel attacks remain separate

C. Supervised adaptive rolling-origin evaluation
   - nine predefined transitions: test B2 through B10
   - immediately preceding block is split chronologically and timestamp-safely:
       first 70%  -> supervised refresh/adaptation
       final 30%  -> threshold calibration only
   - training = all earlier blocks + adaptation segment
   - threshold = benign calibration scores only, fixed at <=1% empirical FPR
   - test = complete next block
   - seen attack types are defined only from the resulting training history
   - novel attack types remain a separate open-set diagnostic
   - all nine transitions are retained; no transition is selected by performance

D. Timestamp-safe terminal 70/15/15 stress diagnostic
   - earliest ~70% train
   - next ~15% calibration
   - latest ~15% final test
   - used only to characterize the original combined temporal + attack-novelty
     failure mode
   - never relabeled as seen-attack temporal generalization when no seen attacks
     are present

PRIMARY OPERATING-POINT POLICY
------------------------------
For every model:
- preprocessing is fitted on training rows only;
- XGBoost is fitted on training rows only;
- primary threshold is fixed only from benign calibration scores;
- target empirical calibration benign FPR = 1%;
- no test labels or scores are used for threshold selection;
- a fixed sensitivity grid (0.1%, 0.5%, 1%, 2%, 5%) is diagnostic only.

IMPORTANT INTERPRETATION
------------------------
The adaptive protocol is a supervised periodic-refresh scenario. It assumes
labels for the first 70% of the immediately preceding block become available
before the next block is evaluated. The script reports that assumption
explicitly; it does not claim unsupervised online adaptation.

The adaptive protocol was introduced after the strict chronology audit exposed
a structural eligibility problem. Its 70/30 split, 1% FPR primary threshold,
nine transitions, model family, and reporting rules are fixed here before any
adaptive model-performance result is observed. Weak transitions and weak attack
families are retained.

OUTPUT LOCATION
---------------
artifacts/generalization/complete_temporal_validation
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
    os.getenv("FLBCIDS_TEMPORAL_RESULTS_DIR", "artifacts/generalization/complete_temporal_validation")
)

EXPECTED_CSE_SOURCE_SHA256 = (
    "4335539845e880b1fb06703b5a68da0a03ed0682204bdda0863ddfc316782e3c"
)
EXPECTED_CSE_SOURCE_ROWS = 63_195_145

CSE_LABEL_COL = "Label"
CSE_IDENTIFIER_COLS = ["id", "Flow ID", "Src IP", "Dst IP", "Timestamp"]

RANDOM_STATE = 42
CSE_MAX_MODEL_ROWS: Optional[int] = 3_000_000

N_TEMPORAL_BLOCKS = 10

# Strict diagnostic: same five folds as the already-completed strict run.
STRICT_FIRST_TEST_BLOCK = 6
STRICT_LAST_TEST_BLOCK = 10

# Adaptive evaluation: all adjacent transitions B1->B2 through B9->B10.
ADAPTIVE_FIRST_TEST_BLOCK = 2
ADAPTIVE_LAST_TEST_BLOCK = 10
ADAPTATION_FRACTION = 0.70

PRIMARY_TARGET_FPR = 0.01
FPR_SENSITIVITY = (0.001, 0.005, 0.01, 0.02, 0.05)

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

SCORE_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


# =============================================================================
# Generic helpers
# =============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
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
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            obj,
            handle,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def format_optional_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def numeric_array(
    values: Any,
    *,
    dtype: Any = np.float64,
    finite_only: bool = False,
) -> np.ndarray:
    array = np.asarray(
        pd.to_numeric(values, errors="coerce"),
        dtype=dtype,
    ).reshape(-1)

    if finite_only:
        array = array[np.isfinite(array)]

    return array


def normalize_label_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )


def parse_cse_timestamp(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=True,
            format="mixed",
        )
    except TypeError:
        return pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=True,
        )


def hash_frame_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> np.ndarray:
    return pd.util.hash_pandas_object(
        df[list(feature_cols)],
        index=False,
    ).to_numpy(
        dtype=np.uint64,
        copy=False,
    )


def make_onehot_encoder() -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )


def to_float32_matrix(matrix: Any) -> Any:
    if sp.issparse(matrix):
        return sp.csr_matrix(
            matrix,
            dtype=np.float32,
        )
    return np.asarray(
        matrix,
        dtype=np.float32,
    )


def dataframe_row_count(df: pd.DataFrame) -> int:
    return int(len(df.index))


# =============================================================================
# Preprocessing and model
# =============================================================================
def build_preprocessor(train_x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = (
        train_x
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )
    categorical_cols = [
        column
        for column in train_x.columns
        if column not in numeric_cols
    ]

    transformers: List[Tuple[str, Any, List[str]]] = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
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
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                            ),
                        ),
                        (
                            "onehot",
                            make_onehot_encoder(),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable predictor columns found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def maybe_balance_train(
    matrix: Any,
    labels: np.ndarray,
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
    y = np.asarray(
        labels,
        dtype=np.int8,
    ).reshape(-1)

    counts = np.bincount(
        y,
        minlength=2,
    )

    info: Dict[str, Any] = {
        "enabled": BALANCE_BINARY_TRAIN,
        "before": {
            "benign": int(counts[0]),
            "attack": int(counts[1]),
        },
    }

    if not BALANCE_BINARY_TRAIN:
        info.update(
            {
                "performed": False,
                "reason": "disabled",
            }
        )
        return matrix, y, info

    if y.size > MAX_ROWS_FOR_OVERSAMPLING:
        info.update(
            {
                "performed": False,
                "reason": "row_cap",
            }
        )
        return matrix, y, info

    if int(counts.min()) == 0 or int(counts[0]) == int(counts[1]):
        info.update(
            {
                "performed": False,
                "reason": "single_class_or_balanced",
            }
        )
        return matrix, y, info

    sampler = RandomOverSampler(
        sampling_strategy=1.0,
        random_state=RANDOM_STATE,
    )

    matrix_balanced, y_balanced = sampler.fit_resample(
        matrix,
        y,
    )

    y_balanced = np.asarray(
        y_balanced,
        dtype=np.int8,
    ).reshape(-1)

    after = np.bincount(
        y_balanced,
        minlength=2,
    )

    info.update(
        {
            "performed": True,
            "after": {
                "benign": int(after[0]),
                "attack": int(after[1]),
            },
        }
    )

    return (
        matrix_balanced,
        y_balanced,
        info,
    )


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
    y_train = train_df[label_col].to_numpy(
        dtype=np.int8,
        copy=False,
    )

    if np.unique(y_train).size != 2:
        raise ValueError(
            "Training history must contain both benign and attack rows."
        )

    preprocessor = build_preprocessor(
        train_df[list(feature_cols)]
    )

    x_train = to_float32_matrix(
        preprocessor.fit_transform(
            train_df[list(feature_cols)]
        )
    )

    x_balanced, y_balanced, balance_info = maybe_balance_train(
        x_train,
        y_train,
    )

    x_balanced = to_float32_matrix(
        x_balanced
    )

    counts = np.bincount(
        y_balanced.astype(int),
        minlength=2,
    )

    params = dict(
        XGB_COMMON
    )
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": (
                1.0
                if balance_info.get("performed")
                else float(
                    counts[0]
                    / max(1, int(counts[1]))
                )
            ),
            "base_score": float(
                np.mean(y_balanced)
            ),
        }
    )

    start = time.perf_counter()

    booster = xgb.train(
        params=params,
        dtrain=xgb.DMatrix(
            x_balanced,
            label=y_balanced,
        ),
        num_boost_round=NUM_BOOST_ROUND,
        verbose_eval=False,
    )

    elapsed = time.perf_counter() - start

    return DetectorBundle(
        preprocessor,
        booster,
        balance_info,
        elapsed,
    )


def transform(
    bundle: DetectorBundle,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Any:
    return to_float32_matrix(
        bundle.preprocessor.transform(
            df[list(feature_cols)]
        )
    )


def supervised_scores(
    bundle: DetectorBundle,
    matrix: Any,
) -> np.ndarray:
    return np.asarray(
        bundle.booster.predict(
            xgb.DMatrix(matrix)
        ),
        dtype=np.float64,
    ).reshape(-1)


# =============================================================================
# Thresholds and metrics
# =============================================================================
def threshold_for_empirical_fpr(
    benign_scores: np.ndarray,
    target_fpr: float,
) -> float:
    scores = np.asarray(
        benign_scores,
        dtype=np.float64,
    ).reshape(-1)

    scores = scores[
        np.isfinite(scores)
    ]

    if scores.size == 0:
        raise ValueError(
            "No finite benign calibration scores available."
        )

    if not (0.0 < target_fpr < 1.0):
        raise ValueError(
            "target_fpr must lie strictly between 0 and 1."
        )

    descending = np.sort(scores)[::-1]

    allowed_fp = int(
        math.floor(
            target_fpr
            * descending.size
        )
    )

    if allowed_fp <= 0:
        return float(
            np.nextafter(
                descending[0],
                np.inf,
            )
        )

    if allowed_fp >= descending.size:
        return float(
            np.nextafter(
                descending[-1],
                -np.inf,
            )
        )

    return float(
        np.nextafter(
            descending[allowed_fp],
            np.inf,
        )
    )


def validation_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Optional[float]:
    y = np.asarray(
        y_true,
        dtype=np.int8,
    ).reshape(-1)

    score = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    if np.unique(y).size < 2:
        return None

    thresholds = np.linspace(
        0.01,
        0.99,
        199,
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        prediction = (
            score >= threshold
        ).astype(np.int8)

        value = float(
            f1_score(
                y,
                prediction,
                zero_division=0,
            )
        )

        if value > best_f1:
            best_f1 = value
            best_threshold = float(
                threshold
            )

    return best_threshold


def safe_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Optional[float]:
    y = np.asarray(
        y_true,
        dtype=np.int8,
    ).reshape(-1)

    score = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    if y.size == 0:
        return None

    if np.unique(y).size < 2:
        return None

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def metrics_from_predictions(
    y_true: np.ndarray,
    prediction: np.ndarray,
    scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y = np.asarray(
        y_true,
        dtype=np.int8,
    ).reshape(-1)

    pred = np.asarray(
        prediction,
        dtype=np.int8,
    ).reshape(-1)

    if y.size == 0:
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

    matrix = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = [
        int(value)
        for value in matrix.ravel()
    ]

    benign_denominator = (
        tn + fp
    )

    score_array: Optional[np.ndarray]
    if scores is None:
        score_array = None
    else:
        score_array = np.asarray(
            scores,
            dtype=np.float64,
        ).reshape(-1)

    return {
        "n": int(y.size),
        "benign_n": int(
            np.count_nonzero(y == 0)
        ),
        "attack_n": int(
            np.count_nonzero(y == 1)
        ),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "balanced_accuracy": (
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            )
            if np.unique(y).size == 2
            else None
        ),
        "mcc": (
            float(
                matthews_corrcoef(
                    y,
                    pred,
                )
            )
            if np.unique(y).size == 2
            else None
        ),
        "fpr": (
            float(
                fp / benign_denominator
            )
            if benign_denominator > 0
            else None
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": matrix.tolist(),
        "roc_auc": (
            safe_auc(
                y,
                score_array,
            )
            if score_array is not None
            else None
        ),
    }


def metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    score = np.asarray(
        scores,
        dtype=np.float64,
    ).reshape(-1)

    prediction = (
        score >= threshold
    ).astype(np.int8)

    result = metrics_from_predictions(
        y_true,
        prediction,
        score,
    )

    result["threshold"] = float(
        threshold
    )

    return result


def score_quantiles(
    values: np.ndarray,
) -> Dict[str, Optional[float]]:
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return {
            f"q{int(q * 100):02d}": None
            for q in SCORE_QUANTILES
        }

    return {
        f"q{int(q * 100):02d}": float(
            np.quantile(
                array,
                q,
            )
        )
        for q in SCORE_QUANTILES
    }


def finite_stats(
    values: np.ndarray,
) -> Dict[str, Optional[float]]:
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "n": int(array.size),
        "mean": float(
            np.mean(array)
        ),
        "sd": (
            float(
                np.std(
                    array,
                    ddof=1,
                )
            )
            if array.size > 1
            else 0.0
        ),
        "median": float(
            np.median(array)
        ),
        "min": float(
            np.min(array)
        ),
        "max": float(
            np.max(array)
        ),
    }


# =============================================================================
# Dataset loading
# =============================================================================
def load_cse_temporal_sample(
    path: Path,
    row_cap: Optional[int],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    print(
        f"  [CSE] Source file: {path}",
        flush=True,
    )
    print(
        "  [CSE] Verifying SHA-256...",
        flush=True,
    )

    source_sha = sha256_file(
        path
    )

    if (
        source_sha.lower()
        != EXPECTED_CSE_SOURCE_SHA256.lower()
    ):
        raise RuntimeError(
            "CSE source SHA-256 does not match the source used by "
            "the reviewer experiments.\n"
            f"Expected: {EXPECTED_CSE_SOURCE_SHA256}\n"
            f"Observed: {source_sha}\n"
            "Stop rather than silently evaluate a different source."
        )

    print(
        f"  [CSE] SHA-256 verified: {source_sha}",
        flush=True,
    )

    sampling_fraction: Optional[float]

    if row_cap is None:
        print(
            "  [CSE] Loading complete source into pandas...",
            flush=True,
        )

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        sampling_fraction = None
    else:
        import dask.dataframe as dd

        sampling_fraction = min(
            1.0,
            float(row_cap)
            / float(EXPECTED_CSE_SOURCE_ROWS),
        )

        print(
            "  [CSE] Opening source with Dask...",
            flush=True,
        )

        ddf = dd.read_csv(
            path,
            assume_missing=True,
            blocksize="128MB",
        )

        if (
            EXPECTED_CSE_SOURCE_ROWS
            > row_cap
        ):
            print(
                f"  [CSE] Deterministic source-wide sampling toward "
                f"{row_cap:,} rows "
                f"(fraction={sampling_fraction:.9f}, "
                f"seed={RANDOM_STATE})...",
                flush=True,
            )

            ddf = ddf.sample(
                frac=sampling_fraction,
                random_state=RANDOM_STATE,
            )

        print(
            "  [CSE] Materializing sampled rows...",
            flush=True,
        )

        df = ddf.compute()

        print(
            f"  [CSE] Materialized {len(df):,} rows.",
            flush=True,
        )

        if len(df) > row_cap:
            print(
                f"  [CSE] Trimming {len(df):,} sampled rows to "
                f"exactly {row_cap:,} with pandas sample(seed={RANDOM_STATE})...",
                flush=True,
            )

            df = df.sample(
                n=row_cap,
                random_state=RANDOM_STATE,
                replace=False,
            )

    df = df.reset_index(
        drop=True
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    required = {
        CSE_LABEL_COL,
        "Timestamp",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required CSE columns: {missing}"
        )

    df["AttackType"] = normalize_label_series(
        df[CSE_LABEL_COL]
    )

    df["LabelBinary"] = (
        df["AttackType"]
        != "BENIGN"
    ).astype(np.int8)

    parsed = parse_cse_timestamp(
        df["Timestamp"]
    )

    valid_mask = parsed.notna().to_numpy(
        dtype=bool,
        copy=False,
    )

    nat_count = int(
        np.count_nonzero(
            ~valid_mask
        )
    )

    sampled_before_nat = int(
        len(df)
    )

    df = df.loc[
        valid_mask
    ].copy()

    parsed_values = np.asarray(
        parsed.to_numpy()
    )

    df["_timestamp_parsed"] = (
        parsed_values[valid_mask]
    )

    excluded = set(
        CSE_IDENTIFIER_COLS
        + [
            CSE_LABEL_COL,
            "AttackType",
            "LabelBinary",
            "_timestamp_parsed",
        ]
    )

    feature_cols = [
        column
        for column in df.columns
        if column not in excluded
    ]

    for column in feature_cols:
        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            df[column] = df[column].replace(
                [np.inf, -np.inf],
                np.nan,
            )

    print(
        f"  [CSE] Sorting {len(df):,} timestamp-valid rows chronologically...",
        flush=True,
    )

    df = df.sort_values(
        "_timestamp_parsed",
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    print(
        f"  [CSE] Computing predictor hashes for {len(df):,} rows...",
        flush=True,
    )

    df["feature_hash"] = hash_frame_features(
        df,
        feature_cols,
    )

    labels = df["LabelBinary"].to_numpy(
        dtype=np.int8,
        copy=False,
    )

    attack_mask = (
        labels == 1
    )

    attack_types = (
        df.loc[
            attack_mask,
            "AttackType",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    meta = {
        "path": str(path),
        "sha256": source_sha,
        "expected_source_rows": EXPECTED_CSE_SOURCE_ROWS,
        "row_cap": row_cap,
        "sampling_fraction": sampling_fraction,
        "sampled_rows_before_timestamp_filter": sampled_before_nat,
        "timestamp_valid_rows": int(
            len(df)
        ),
        "timestamp_nat_rows_removed": nat_count,
        "feature_count": int(
            len(feature_cols)
        ),
        "first_timestamp": str(
            df["_timestamp_parsed"].min()
        ),
        "last_timestamp": str(
            df["_timestamp_parsed"].max()
        ),
        "benign_rows": int(
            np.count_nonzero(
                labels == 0
            )
        ),
        "attack_rows": int(
            np.count_nonzero(
                labels == 1
            )
        ),
        "attack_types": sorted(
            str(value)
            for value in attack_types
        ),
    }

    return (
        df,
        feature_cols,
        meta,
    )


# =============================================================================
# Timestamp-safe cuts and blocks
# =============================================================================
def timestamp_array(
    df: pd.DataFrame,
) -> np.ndarray:
    return np.asarray(
        df["_timestamp_parsed"].to_numpy(),
        dtype="datetime64[ns]",
    ).reshape(-1)


def nearest_timestamp_safe_cut(
    timestamps: np.ndarray,
    target_index: int,
    *,
    lower_exclusive: int = 0,
    upper_exclusive: Optional[int] = None,
) -> int:
    ts = np.asarray(
        timestamps,
        dtype="datetime64[ns]",
    ).reshape(-1)

    n = int(
        ts.size
    )

    upper = (
        n
        if upper_exclusive is None
        else int(upper_exclusive)
    )

    if n < 2:
        raise ValueError(
            "At least two rows are required for a timestamp-safe cut."
        )

    changes = (
        np.flatnonzero(
            ts[1:] != ts[:-1]
        )
        + 1
    )

    valid = changes[
        (changes > int(lower_exclusive))
        & (changes < upper)
    ]

    if valid.size == 0:
        raise ValueError(
            "No timestamp-safe cut exists in the requested range."
        )

    target = int(
        target_index
    )

    distances = np.abs(
        valid.astype(np.int64)
        - target
    )

    minimum = np.min(
        distances
    )

    candidate_positions = np.flatnonzero(
        distances == minimum
    )

    # Deterministic tie-break: earlier cut.
    selected_position = int(
        candidate_positions[0]
    )

    return int(
        valid[selected_position]
    )


def make_timestamp_safe_blocks(
    sorted_df: pd.DataFrame,
    n_blocks: int = N_TEMPORAL_BLOCKS,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if n_blocks < 3:
        raise ValueError(
            "At least three temporal blocks are required."
        )

    if len(sorted_df) < n_blocks:
        raise ValueError(
            "Not enough rows to form temporal blocks."
        )

    ts = timestamp_array(
        sorted_df
    )

    if (
        ts.size > 1
        and np.any(
            ts[1:] < ts[:-1]
        )
    ):
        raise AssertionError(
            "Input must be chronologically sorted."
        )

    internal_cuts: List[int] = []
    previous = 0

    for block_index in range(
        1,
        n_blocks,
    ):
        target = int(
            round(
                block_index
                * len(sorted_df)
                / n_blocks
            )
        )

        cut = nearest_timestamp_safe_cut(
            ts,
            target,
            lower_exclusive=previous,
            upper_exclusive=len(sorted_df),
        )

        internal_cuts.append(
            cut
        )

        previous = cut

    cuts = np.asarray(
        [0]
        + internal_cuts
        + [len(sorted_df)],
        dtype=np.int64,
    )

    if np.any(
        np.diff(cuts) <= 0
    ):
        raise AssertionError(
            f"Non-increasing temporal cuts: {cuts.tolist()}"
        )

    if cuts.size != n_blocks + 1:
        raise AssertionError(
            "Unexpected number of temporal cuts."
        )

    block_ids = np.empty(
        len(sorted_df),
        dtype=np.int16,
    )

    rows: List[Dict[str, Any]] = []

    for block_index in range(
        n_blocks
    ):
        start = int(
            cuts[block_index]
        )
        stop = int(
            cuts[block_index + 1]
        )
        block_id = (
            block_index + 1
        )

        block_ids[
            start:stop
        ] = block_id

        block = sorted_df.iloc[
            start:stop
        ]

        labels = block["LabelBinary"].to_numpy(
            dtype=np.int8,
            copy=False,
        )

        attack_series = block.loc[
            labels == 1,
            "AttackType",
        ].astype(str)

        rows.append(
            {
                "block": block_id,
                "start_row": start,
                "stop_row_exclusive": stop,
                "rows": int(
                    len(block)
                ),
                "benign_rows": int(
                    np.count_nonzero(
                        labels == 0
                    )
                ),
                "attack_rows": int(
                    np.count_nonzero(
                        labels == 1
                    )
                ),
                "attack_type_count": int(
                    attack_series.nunique()
                ),
                "attack_types": " | ".join(
                    sorted(
                        attack_series
                        .unique()
                        .tolist()
                    )
                ),
                "start_timestamp": str(
                    block["_timestamp_parsed"].min()
                ),
                "end_timestamp": str(
                    block["_timestamp_parsed"].max()
                ),
                "distinct_timestamps": int(
                    block["_timestamp_parsed"].nunique()
                ),
            }
        )

    for boundary in range(
        1,
        n_blocks,
    ):
        left = ts[
            int(cuts[boundary] - 1)
        ]
        right = ts[
            int(cuts[boundary])
        ]

        if left >= right:
            raise AssertionError(
                f"Timestamp-safe boundary violated between "
                f"B{boundary} and B{boundary + 1}: "
                f"{left!r} vs {right!r}"
            )

    return (
        block_ids,
        pd.DataFrame.from_records(
            rows
        ),
    )


def split_block_timestamp_safe(
    block_df: pd.DataFrame,
    first_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if not (
        0.0
        < first_fraction
        < 1.0
    ):
        raise ValueError(
            "first_fraction must be between 0 and 1."
        )

    if len(block_df) < 2:
        raise ValueError(
            "Block is too small to split."
        )

    ts = timestamp_array(
        block_df
    )

    target = int(
        round(
            first_fraction
            * len(block_df)
        )
    )

    cut = nearest_timestamp_safe_cut(
        ts,
        target,
        lower_exclusive=0,
        upper_exclusive=len(block_df),
    )

    first = block_df.iloc[
        :cut
    ].copy()

    second = block_df.iloc[
        cut:
    ].copy()

    if first.empty or second.empty:
        raise AssertionError(
            "Timestamp-safe split produced an empty segment."
        )

    first_end = np.asarray(
        first["_timestamp_parsed"].to_numpy(),
        dtype="datetime64[ns]",
    ).reshape(-1)[-1]

    second_start = np.asarray(
        second["_timestamp_parsed"].to_numpy(),
        dtype="datetime64[ns]",
    ).reshape(-1)[0]

    if first_end >= second_start:
        raise AssertionError(
            "Adaptation/calibration split shares or reverses a timestamp."
        )

    meta = {
        "requested_first_fraction": float(
            first_fraction
        ),
        "actual_first_fraction": float(
            len(first)
            / len(block_df)
        ),
        "split_row": int(
            cut
        ),
        "first_rows": int(
            len(first)
        ),
        "second_rows": int(
            len(second)
        ),
        "first_start_timestamp": str(
            first["_timestamp_parsed"].min()
        ),
        "first_end_timestamp": str(
            first["_timestamp_parsed"].max()
        ),
        "second_start_timestamp": str(
            second["_timestamp_parsed"].min()
        ),
        "second_end_timestamp": str(
            second["_timestamp_parsed"].max()
        ),
    }

    return (
        first,
        second,
        meta,
    )


def make_terminal_70_15_15(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    ts = timestamp_array(
        df
    )

    cut70 = nearest_timestamp_safe_cut(
        ts,
        int(
            round(
                0.70
                * len(df)
            )
        ),
        lower_exclusive=0,
        upper_exclusive=len(df),
    )

    cut85 = nearest_timestamp_safe_cut(
        ts,
        int(
            round(
                0.85
                * len(df)
            )
        ),
        lower_exclusive=cut70,
        upper_exclusive=len(df),
    )

    train = df.iloc[
        :cut70
    ].copy()

    calibration = df.iloc[
        cut70:cut85
    ].copy()

    test = df.iloc[
        cut85:
    ].copy()

    meta = {
        "cut70_row": int(
            cut70
        ),
        "cut85_row": int(
            cut85
        ),
        "train_rows": int(
            len(train)
        ),
        "calibration_rows": int(
            len(calibration)
        ),
        "test_rows": int(
            len(test)
        ),
        "train_fraction": float(
            len(train)
            / len(df)
        ),
        "calibration_fraction": float(
            len(calibration)
            / len(df)
        ),
        "test_fraction": float(
            len(test)
            / len(df)
        ),
        "timestamp_safe": True,
    }

    return (
        train,
        calibration,
        test,
        meta,
    )


# =============================================================================
# Coverage audit
# =============================================================================
def build_attack_coverage_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    labels = df["LabelBinary"].to_numpy(
        dtype=np.int8,
        copy=False,
    )

    attack_df = df.loc[
        labels == 1
    ].copy()

    rows: List[Dict[str, Any]] = []

    for attack_type, group in attack_df.groupby(
        "AttackType",
        sort=True,
    ):
        counts = (
            group["_temporal_block"]
            .value_counts()
            .sort_index()
        )

        row: Dict[str, Any] = {
            "attack_type": str(
                attack_type
            ),
            "total_rows": int(
                len(group)
            ),
            "first_timestamp": str(
                group["_timestamp_parsed"].min()
            ),
            "last_timestamp": str(
                group["_timestamp_parsed"].max()
            ),
            "first_block": int(
                group["_temporal_block"].min()
            ),
            "last_block": int(
                group["_temporal_block"].max()
            ),
            "blocks_present": int(
                group["_temporal_block"].nunique()
            ),
        }

        for block_id in range(
            1,
            N_TEMPORAL_BLOCKS + 1,
        ):
            row[
                f"B{block_id}_rows"
            ] = int(
                counts.get(
                    block_id,
                    0,
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame.from_records(
        rows
    )


# =============================================================================
# Unified partition evaluation
# =============================================================================
def attack_type_set(
    df: pd.DataFrame,
) -> set[str]:
    labels = df["LabelBinary"].to_numpy(
        dtype=np.int8,
        copy=False,
    )

    return set(
        df.loc[
            labels == 1,
            "AttackType",
        ]
        .astype(str)
        .unique()
        .tolist()
    )


def class_counts(
    df: pd.DataFrame,
) -> Dict[str, int]:
    labels = df["LabelBinary"].to_numpy(
        dtype=np.int8,
        copy=False,
    )

    return {
        "benign": int(
            np.count_nonzero(
                labels == 0
            )
        ),
        "attack": int(
            np.count_nonzero(
                labels == 1
            )
        ),
    }


def chronological_invariant(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    train_max = np.asarray(
        train["_timestamp_parsed"].to_numpy(),
        dtype="datetime64[ns]",
    ).reshape(-1)[-1]

    calibration_min = np.asarray(
        calibration["_timestamp_parsed"].to_numpy(),
        dtype="datetime64[ns]",
    ).reshape(-1)[0]

    calibration_max = np.asarray(
        calibration["_timestamp_parsed"].to_numpy(),
        dtype="datetime64[ns]",
    ).reshape(-1)[-1]

    test_min = np.asarray(
        test["_timestamp_parsed"].to_numpy(),
        dtype="datetime64[ns]",
    ).reshape(-1)[0]

    if not (
        train_max
        < calibration_min
        <= calibration_max
        < test_min
    ):
        raise AssertionError(
            "Chronological train/calibration/test invariant failed."
        )


def evaluate_partition(
    *,
    protocol: str,
    evaluation_id: str,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
    metadata: Dict[str, Any],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    print(
        f"\n[{protocol}] {evaluation_id}",
        flush=True,
    )
    print(
        f"  rows: train={len(train):,}, "
        f"cal={len(calibration):,}, "
        f"test={len(test):,}",
        flush=True,
    )

    detail: Dict[str, Any] = {
        "protocol": protocol,
        "evaluation_id": evaluation_id,
        **metadata,
        "train_rows": int(
            len(train)
        ),
        "calibration_rows": int(
            len(calibration)
        ),
        "test_rows": int(
            len(test)
        ),
        "train_time_range": [
            str(
                train["_timestamp_parsed"].min()
            ),
            str(
                train["_timestamp_parsed"].max()
            ),
        ],
        "calibration_time_range": [
            str(
                calibration["_timestamp_parsed"].min()
            ),
            str(
                calibration["_timestamp_parsed"].max()
            ),
        ],
        "test_time_range": [
            str(
                test["_timestamp_parsed"].min()
            ),
            str(
                test["_timestamp_parsed"].max()
            ),
        ],
        "train_class_counts": class_counts(
            train
        ),
        "calibration_class_counts": class_counts(
            calibration
        ),
        "test_class_counts": class_counts(
            test
        ),
    }

    if train.empty or calibration.empty or test.empty:
        reason = "empty_partition"

        detail.update(
            {
                "status": "ineligible",
                "reason": reason,
            }
        )

        row = {
            "protocol": protocol,
            "evaluation_id": evaluation_id,
            "status": "ineligible",
            "exclusion_reason": reason,
            **metadata,
        }

        return (
            detail,
            row,
            [],
            [],
            [],
        )

    chronological_invariant(
        train,
        calibration,
        test,
    )

    train_y = train["LabelBinary"].to_numpy(
        dtype=np.int8,
        copy=False,
    )

    calibration_y = calibration["LabelBinary"].to_numpy(
        dtype=np.int8,
        copy=False,
    )

    test_y = test["LabelBinary"].to_numpy(
        dtype=np.int8,
        copy=False,
    )

    if np.unique(train_y).size != 2:
        reason = "training_history_single_class"

        detail.update(
            {
                "status": "ineligible",
                "reason": reason,
            }
        )

        row = {
            "protocol": protocol,
            "evaluation_id": evaluation_id,
            "status": "ineligible",
            "exclusion_reason": reason,
            **metadata,
        }

        return (
            detail,
            row,
            [],
            [],
            [],
        )

    benign_calibration_mask = (
        calibration_y == 0
    )

    if not np.any(
        benign_calibration_mask
    ):
        reason = "calibration_has_no_benign_rows"

        detail.update(
            {
                "status": "ineligible",
                "reason": reason,
            }
        )

        row = {
            "protocol": protocol,
            "evaluation_id": evaluation_id,
            "status": "ineligible",
            "exclusion_reason": reason,
            **metadata,
        }

        return (
            detail,
            row,
            [],
            [],
            [],
        )

    train_attack_types = attack_type_set(
        train
    )

    test_attack_types = attack_type_set(
        test
    )

    seen_test_types = sorted(
        test_attack_types
        & train_attack_types
    )

    novel_test_types = sorted(
        test_attack_types
        - train_attack_types
    )

    test_attack_type_values = (
        test["AttackType"]
        .astype(str)
        .to_numpy()
    )

    attack_mask = (
        test_y == 1
    )

    benign_test_mask = (
        test_y == 0
    )

    seen_attack_mask = (
        attack_mask
        & np.isin(
            test_attack_type_values,
            list(train_attack_types),
        )
    )

    novel_attack_mask = (
        attack_mask
        & ~np.isin(
            test_attack_type_values,
            list(train_attack_types),
        )
    )

    seen_attack_rows = int(
        np.count_nonzero(
            seen_attack_mask
        )
    )

    novel_attack_rows = int(
        np.count_nonzero(
            novel_attack_mask
        )
    )

    print(
        f"  attack coverage: seen={seen_attack_rows:,} "
        f"({len(seen_test_types)} types), "
        f"novel={novel_attack_rows:,} "
        f"({len(novel_test_types)} types)",
        flush=True,
    )

    detail.update(
        {
            "train_attack_types": sorted(
                train_attack_types
            ),
            "test_attack_types": sorted(
                test_attack_types
            ),
            "seen_test_attack_types": seen_test_types,
            "novel_test_attack_types": novel_test_types,
            "seen_attack_rows": seen_attack_rows,
            "novel_attack_rows": novel_attack_rows,
        }
    )

    print(
        "  fitting train-only preprocessing + XGBoost...",
        flush=True,
    )

    bundle = fit_detector(
        train,
        feature_cols,
        "LabelBinary",
    )

    print(
        f"  model fit completed in "
        f"{bundle.fit_seconds:.3f} s",
        flush=True,
    )

    x_calibration = transform(
        bundle,
        calibration,
        feature_cols,
    )

    x_test = transform(
        bundle,
        test,
        feature_cols,
    )

    calibration_scores = supervised_scores(
        bundle,
        x_calibration,
    )

    test_scores = supervised_scores(
        bundle,
        x_test,
    )

    primary_threshold = threshold_for_empirical_fpr(
        calibration_scores[
            benign_calibration_mask
        ],
        PRIMARY_TARGET_FPR,
    )

    diagnostic_f1_threshold = validation_f1_threshold(
        calibration_y,
        calibration_scores,
    )

    primary_calibration_metrics = metrics_at_threshold(
        calibration_y,
        calibration_scores,
        primary_threshold,
    )

    combined_metrics = metrics_at_threshold(
        test_y,
        test_scores,
        primary_threshold,
    )

    benign_only_prediction = (
        test_scores[
            benign_test_mask
        ]
        >= primary_threshold
    )

    benign_test_fpr = (
        float(
            np.mean(
                benign_only_prediction
            )
        )
        if np.any(
            benign_test_mask
        )
        else None
    )

    seen_metrics: Optional[Dict[str, Any]]

    if seen_attack_rows > 0:
        seen_eval_mask = (
            benign_test_mask
            | seen_attack_mask
        )

        seen_metrics = metrics_at_threshold(
            test_y[
                seen_eval_mask
            ],
            test_scores[
                seen_eval_mask
            ],
            primary_threshold,
        )
    else:
        seen_eval_mask = (
            benign_test_mask.copy()
        )
        seen_metrics = None

    novel_metrics: Optional[Dict[str, Any]]

    if novel_attack_rows > 0:
        novel_eval_mask = (
            benign_test_mask
            | novel_attack_mask
        )

        novel_metrics = metrics_at_threshold(
            test_y[
                novel_eval_mask
            ],
            test_scores[
                novel_eval_mask
            ],
            primary_threshold,
        )
    else:
        novel_metrics = None

    train_hashes = set(
        map(
            int,
            train["feature_hash"].unique(),
        )
    )

    test_hashes = test["feature_hash"].to_numpy(
        dtype=np.uint64,
        copy=False,
    )

    test_hash_seen = np.fromiter(
        (
            int(value)
            in train_hashes
            for value in test_hashes
        ),
        dtype=bool,
        count=int(
            len(test_hashes)
        ),
    )

    seen_hash_seen = (
        seen_attack_mask
        & test_hash_seen
    )

    seen_hash_novel = (
        seen_attack_mask
        & ~test_hash_seen
    )

    hash_seen_recall = (
        float(
            np.mean(
                test_scores[
                    seen_hash_seen
                ]
                >= primary_threshold
            )
        )
        if np.any(
            seen_hash_seen
        )
        else None
    )

    hash_novel_recall = (
        float(
            np.mean(
                test_scores[
                    seen_hash_novel
                ]
                >= primary_threshold
            )
        )
        if np.any(
            seen_hash_novel
        )
        else None
    )

    detail.update(
        {
            "status": "evaluated",
            "primary_seen_endpoint_eligible": (
                seen_attack_rows > 0
            ),
            "primary_threshold": float(
                primary_threshold
            ),
            "diagnostic_validation_f1_threshold": (
                float(
                    diagnostic_f1_threshold
                )
                if diagnostic_f1_threshold is not None
                else None
            ),
            "primary_calibration_metrics": primary_calibration_metrics,
            "combined_test_period": combined_metrics,
            "seen_attack_temporal": seen_metrics,
            "novel_attack_with_benign_diagnostic": novel_metrics,
            "test_benign_fpr": benign_test_fpr,
            "fit_seconds": float(
                bundle.fit_seconds
            ),
            "balance": bundle.balance_info,
            "predictor_hash_diagnostics": {
                "test_rows_hash_seen_in_train": int(
                    np.count_nonzero(
                        test_hash_seen
                    )
                ),
                "test_rows_hash_novel_to_train": int(
                    np.count_nonzero(
                        ~test_hash_seen
                    )
                ),
                "seen_attack_rows_hash_seen_in_train": int(
                    np.count_nonzero(
                        seen_hash_seen
                    )
                ),
                "seen_attack_rows_hash_novel_to_train": int(
                    np.count_nonzero(
                        seen_hash_novel
                    )
                ),
                "seen_attack_recall_hash_seen_in_train": hash_seen_recall,
                "seen_attack_recall_hash_novel_to_train": hash_novel_recall,
            },
        }
    )

    if seen_attack_rows == 0:
        detail[
            "primary_seen_endpoint_exclusion_reason"
        ] = "zero_seen_attack_rows_in_test"

    per_attack_rows: List[Dict[str, Any]] = []

    attack_positions = np.flatnonzero(
        attack_mask
    )

    attack_test = test.iloc[
        attack_positions
    ].copy()

    attack_test["_score"] = test_scores[
        attack_positions
    ]

    for attack_type, group in attack_test.groupby(
        "AttackType",
        sort=True,
    ):
        scores = np.asarray(
            group["_score"].to_numpy(),
            dtype=np.float64,
        ).reshape(-1)

        hashes = group["feature_hash"].to_numpy(
            dtype=np.uint64,
            copy=False,
        )

        hash_seen_mask = np.fromiter(
            (
                int(value)
                in train_hashes
                for value in hashes
            ),
            dtype=bool,
            count=int(
                len(hashes)
            ),
        )

        quantiles = score_quantiles(
            scores
        )

        seen_in_training = (
            str(attack_type)
            in train_attack_types
        )

        per_attack_rows.append(
            {
                "protocol": protocol,
                "evaluation_id": evaluation_id,
                **metadata,
                "attack_type": str(
                    attack_type
                ),
                "endpoint": (
                    "seen_temporal"
                    if seen_in_training
                    else "novel_temporal"
                ),
                "rows": int(
                    len(group)
                ),
                "recall": (
                    float(
                        np.mean(
                            scores
                            >= primary_threshold
                        )
                    )
                    if scores.size > 0
                    else None
                ),
                "hash_seen_in_train_rows": int(
                    np.count_nonzero(
                        hash_seen_mask
                    )
                ),
                "hash_novel_to_train_rows": int(
                    np.count_nonzero(
                        ~hash_seen_mask
                    )
                ),
                "hash_seen_recall": (
                    float(
                        np.mean(
                            scores[
                                hash_seen_mask
                            ]
                            >= primary_threshold
                        )
                    )
                    if np.any(
                        hash_seen_mask
                    )
                    else None
                ),
                "hash_novel_recall": (
                    float(
                        np.mean(
                            scores[
                                ~hash_seen_mask
                            ]
                            >= primary_threshold
                        )
                    )
                    if np.any(
                        ~hash_seen_mask
                    )
                    else None
                ),
                "score_q10": quantiles[
                    "q10"
                ],
                "score_median": quantiles[
                    "q50"
                ],
                "score_q90": quantiles[
                    "q90"
                ],
            }
        )

    sensitivity_rows: List[Dict[str, Any]] = []

    for target_fpr in FPR_SENSITIVITY:
        threshold = threshold_for_empirical_fpr(
            calibration_scores[
                benign_calibration_mask
            ],
            target_fpr,
        )

        combined_target = metrics_at_threshold(
            test_y,
            test_scores,
            threshold,
        )

        if seen_attack_rows > 0:
            seen_target = metrics_at_threshold(
                test_y[
                    seen_eval_mask
                ],
                test_scores[
                    seen_eval_mask
                ],
                threshold,
            )
        else:
            seen_target = None

        sensitivity_rows.append(
            {
                "protocol": protocol,
                "evaluation_id": evaluation_id,
                **metadata,
                "target_development_fpr": float(
                    target_fpr
                ),
                "threshold": float(
                    threshold
                ),
                "seen_attack_rows": seen_attack_rows,
                "seen_temporal_recall": (
                    seen_target.get(
                        "recall"
                    )
                    if seen_target is not None
                    else None
                ),
                "seen_temporal_precision": (
                    seen_target.get(
                        "precision"
                    )
                    if seen_target is not None
                    else None
                ),
                "seen_temporal_f1": (
                    seen_target.get(
                        "f1"
                    )
                    if seen_target is not None
                    else None
                ),
                "seen_temporal_test_fpr": (
                    seen_target.get(
                        "fpr"
                    )
                    if seen_target is not None
                    else None
                ),
                "combined_recall": combined_target.get(
                    "recall"
                ),
                "combined_test_fpr": combined_target.get(
                    "fpr"
                ),
            }
        )

    score_rows: List[Dict[str, Any]] = []

    score_categories = {
        "calibration_benign": calibration_scores[
            calibration_y == 0
        ],
        "calibration_attack": calibration_scores[
            calibration_y == 1
        ],
        "test_benign": test_scores[
            benign_test_mask
        ],
        "test_seen_attack": test_scores[
            seen_attack_mask
        ],
        "test_novel_attack": test_scores[
            novel_attack_mask
        ],
    }

    for category, values in score_categories.items():
        quantiles = score_quantiles(
            values
        )

        score_rows.append(
            {
                "protocol": protocol,
                "evaluation_id": evaluation_id,
                **metadata,
                "category": category,
                "n": int(
                    len(values)
                ),
                **quantiles,
            }
        )

    if seen_metrics is not None:
        seen_fields = {
            "seen_temporal_accuracy": seen_metrics.get(
                "accuracy"
            ),
            "seen_temporal_precision": seen_metrics.get(
                "precision"
            ),
            "seen_temporal_recall": seen_metrics.get(
                "recall"
            ),
            "seen_temporal_f1": seen_metrics.get(
                "f1"
            ),
            "seen_temporal_balanced_accuracy": seen_metrics.get(
                "balanced_accuracy"
            ),
            "seen_temporal_mcc": seen_metrics.get(
                "mcc"
            ),
            "seen_temporal_test_fpr": seen_metrics.get(
                "fpr"
            ),
            "seen_temporal_auc": seen_metrics.get(
                "roc_auc"
            ),
            "seen_temporal_tn": seen_metrics.get(
                "tn"
            ),
            "seen_temporal_fp": seen_metrics.get(
                "fp"
            ),
            "seen_temporal_fn": seen_metrics.get(
                "fn"
            ),
            "seen_temporal_tp": seen_metrics.get(
                "tp"
            ),
        }

        status = "eligible"
        exclusion_reason = ""
    else:
        seen_fields = {
            "seen_temporal_accuracy": None,
            "seen_temporal_precision": None,
            "seen_temporal_recall": None,
            "seen_temporal_f1": None,
            "seen_temporal_balanced_accuracy": None,
            "seen_temporal_mcc": None,
            "seen_temporal_test_fpr": None,
            "seen_temporal_auc": None,
            "seen_temporal_tn": None,
            "seen_temporal_fp": None,
            "seen_temporal_fn": None,
            "seen_temporal_tp": None,
        }

        status = "no_seen_attack_rows"
        exclusion_reason = (
            "zero_seen_attack_rows_in_test"
        )

    row = {
        "protocol": protocol,
        "evaluation_id": evaluation_id,
        **metadata,
        "train_rows": int(
            len(train)
        ),
        "calibration_rows": int(
            len(calibration)
        ),
        "test_rows": int(
            len(test)
        ),
        "train_attack_type_count": int(
            len(train_attack_types)
        ),
        "seen_test_attack_type_count": int(
            len(seen_test_types)
        ),
        "novel_test_attack_type_count": int(
            len(novel_test_types)
        ),
        "seen_attack_rows": seen_attack_rows,
        "novel_attack_rows": novel_attack_rows,
        "primary_threshold": float(
            primary_threshold
        ),
        "calibration_empirical_fpr": primary_calibration_metrics.get(
            "fpr"
        ),
        "test_benign_fpr": benign_test_fpr,
        **seen_fields,
        "combined_recall": combined_metrics.get(
            "recall"
        ),
        "combined_precision": combined_metrics.get(
            "precision"
        ),
        "combined_f1": combined_metrics.get(
            "f1"
        ),
        "combined_test_fpr": combined_metrics.get(
            "fpr"
        ),
        "combined_auc": combined_metrics.get(
            "roc_auc"
        ),
        "novel_diagnostic_recall": (
            novel_metrics.get(
                "recall"
            )
            if novel_metrics is not None
            else None
        ),
        "seen_attack_rows_hash_seen_in_train": int(
            np.count_nonzero(
                seen_hash_seen
            )
        ),
        "seen_attack_rows_hash_novel_to_train": int(
            np.count_nonzero(
                seen_hash_novel
            )
        ),
        "seen_attack_recall_hash_seen_in_train": hash_seen_recall,
        "seen_attack_recall_hash_novel_to_train": hash_novel_recall,
        "fit_seconds": float(
            bundle.fit_seconds
        ),
        "status": status,
        "exclusion_reason": exclusion_reason,
    }

    del (
        x_calibration,
        x_test,
        bundle,
    )
    gc.collect()

    return (
        detail,
        row,
        per_attack_rows,
        sensitivity_rows,
        score_rows,
    )


# =============================================================================
# Protocol runners
# =============================================================================
def run_strict_protocol(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[
    List[Dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    print(
        "\n"
        + "=" * 108,
        flush=True,
    )
    print(
        "B. STRICT NON-ADAPTIVE ROLLING-ORIGIN DIAGNOSTIC",
        flush=True,
    )
    print(
        "=" * 108,
        flush=True,
    )

    details: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    per_attack: List[Dict[str, Any]] = []
    sensitivity: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []

    for test_block in range(
        STRICT_FIRST_TEST_BLOCK,
        STRICT_LAST_TEST_BLOCK + 1,
    ):
        calibration_block = (
            test_block - 1
        )

        train_last_block = (
            calibration_block - 1
        )

        train = df.loc[
            df["_temporal_block"]
            <= train_last_block
        ].copy()

        calibration = df.loc[
            df["_temporal_block"]
            == calibration_block
        ].copy()

        test = df.loc[
            df["_temporal_block"]
            == test_block
        ].copy()

        evaluation_id = (
            f"strict_test_B{test_block}"
        )

        metadata = {
            "fold": int(
                test_block
                - STRICT_FIRST_TEST_BLOCK
                + 1
            ),
            "train_blocks": (
                f"B1-B{train_last_block}"
            ),
            "calibration_block": int(
                calibration_block
            ),
            "test_block": int(
                test_block
            ),
        }

        (
            detail,
            row,
            attack_rows,
            sensitivity_rows,
            score_rows,
        ) = evaluate_partition(
            protocol="strict_nonadaptive",
            evaluation_id=evaluation_id,
            train=train,
            calibration=calibration,
            test=test,
            feature_cols=feature_cols,
            metadata=metadata,
        )

        details.append(
            detail
        )
        rows.append(
            row
        )
        per_attack.extend(
            attack_rows
        )
        sensitivity.extend(
            sensitivity_rows
        )
        scores.extend(
            score_rows
        )

        del (
            train,
            calibration,
            test,
        )
        gc.collect()

    return (
        details,
        pd.DataFrame.from_records(
            rows
        ),
        pd.DataFrame.from_records(
            per_attack
        ),
        pd.DataFrame.from_records(
            sensitivity
        ),
        pd.DataFrame.from_records(
            scores
        ),
    )


def run_adaptive_protocol(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[
    List[Dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    print(
        "\n"
        + "=" * 108,
        flush=True,
    )
    print(
        "C. SUPERVISED ADAPTIVE ROLLING-ORIGIN EVALUATION",
        flush=True,
    )
    print(
        "=" * 108,
        flush=True,
    )

    details: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    per_attack: List[Dict[str, Any]] = []
    sensitivity: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    for test_block in range(
        ADAPTIVE_FIRST_TEST_BLOCK,
        ADAPTIVE_LAST_TEST_BLOCK + 1,
    ):
        prior_block = (
            test_block - 1
        )

        prior = df.loc[
            df["_temporal_block"]
            == prior_block
        ].copy()

        adaptation, calibration, split_meta = split_block_timestamp_safe(
            prior,
            ADAPTATION_FRACTION,
        )

        earlier = df.loc[
            df["_temporal_block"]
            < prior_block
        ].copy()

        if earlier.empty:
            train = adaptation.copy()
            earlier_blocks = "none"
        else:
            train = pd.concat(
                [
                    earlier,
                    adaptation,
                ],
                ignore_index=True,
                copy=False,
            )
            earlier_blocks = (
                f"B1-B{prior_block - 1}"
            )

        test = df.loc[
            df["_temporal_block"]
            == test_block
        ].copy()

        evaluation_id = (
            f"adaptive_test_B{test_block}"
        )

        metadata = {
            "transition": (
                f"B{prior_block}->B{test_block}"
            ),
            "prior_block": int(
                prior_block
            ),
            "test_block": int(
                test_block
            ),
            "earlier_history_blocks": earlier_blocks,
            "adaptation_fraction_requested": float(
                ADAPTATION_FRACTION
            ),
            "adaptation_rows": int(
                len(adaptation)
            ),
            "calibration_rows_from_prior_block": int(
                len(calibration)
            ),
            "adaptation_actual_fraction": float(
                split_meta[
                    "actual_first_fraction"
                ]
            ),
        }

        split_rows.append(
            {
                "evaluation_id": evaluation_id,
                **metadata,
                **split_meta,
            }
        )

        (
            detail,
            row,
            attack_rows,
            sensitivity_rows,
            score_rows,
        ) = evaluate_partition(
            protocol="adaptive_supervised_refresh",
            evaluation_id=evaluation_id,
            train=train,
            calibration=calibration,
            test=test,
            feature_cols=feature_cols,
            metadata=metadata,
        )

        detail[
            "adaptation_split"
        ] = split_meta

        detail[
            "operational_assumption"
        ] = (
            "Labels for the first 70% of the immediately preceding block "
            "are assumed available for supervised periodic refresh before "
            "the next block is evaluated."
        )

        details.append(
            detail
        )
        rows.append(
            row
        )
        per_attack.extend(
            attack_rows
        )
        sensitivity.extend(
            sensitivity_rows
        )
        scores.extend(
            score_rows
        )

        del (
            prior,
            adaptation,
            calibration,
            earlier,
            train,
            test,
        )
        gc.collect()

    return (
        details,
        pd.DataFrame.from_records(
            rows
        ),
        pd.DataFrame.from_records(
            per_attack
        ),
        pd.DataFrame.from_records(
            sensitivity
        ),
        pd.DataFrame.from_records(
            scores
        ),
        pd.DataFrame.from_records(
            split_rows
        ),
    )


def run_terminal_protocol(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[
    Dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    print(
        "\n"
        + "=" * 108,
        flush=True,
    )
    print(
        "D. TIMESTAMP-SAFE TERMINAL 70/15/15 STRESS DIAGNOSTIC",
        flush=True,
    )
    print(
        "=" * 108,
        flush=True,
    )

    (
        train,
        calibration,
        test,
        split_meta,
    ) = make_terminal_70_15_15(
        df
    )

    (
        detail,
        row,
        per_attack,
        sensitivity,
        scores,
    ) = evaluate_partition(
        protocol="terminal_70_15_15",
        evaluation_id="terminal_70_15_15",
        train=train,
        calibration=calibration,
        test=test,
        feature_cols=feature_cols,
        metadata={
            "split_contract": "timestamp_safe_approximately_70_15_15",
            **split_meta,
        },
    )

    detail[
        "interpretation_constraint"
    ] = (
        "This terminal split is a combined temporal + attack-novelty stress "
        "condition whenever the final test contains no attack family "
        "represented in training. It must not be relabeled as seen-attack "
        "temporal recall in that case."
    )

    del (
        train,
        calibration,
        test,
    )
    gc.collect()

    return (
        detail,
        pd.DataFrame.from_records(
            [row]
        ),
        pd.DataFrame.from_records(
            per_attack
        ),
        pd.DataFrame.from_records(
            sensitivity + scores
        ),
    )


# =============================================================================
# Aggregation
# =============================================================================
def pooled_seen_metrics(
    results: pd.DataFrame,
) -> Dict[str, Any]:
    if results.empty:
        return {
            "status": "no_rows",
            "eligible_evaluations": 0,
        }

    status = (
        results["status"]
        .astype(str)
        .to_numpy()
    )

    seen_rows = numeric_array(
        results[
            "seen_attack_rows"
        ],
        dtype=np.float64,
    )

    seen_rows = np.nan_to_num(
        seen_rows,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mask = (
        (status == "eligible")
        & (seen_rows > 0)
    )

    eligible = results.loc[
        mask
    ].copy()

    if eligible.empty:
        return {
            "status": "no_eligible_seen_attack_temporal_evaluations",
            "eligible_evaluations": 0,
        }

    ids = [
        str(value)
        for value in eligible[
            "evaluation_id"
        ].tolist()
    ]

    tn_values = numeric_array(
        eligible[
            "seen_temporal_tn"
        ],
        dtype=np.float64,
    )
    fp_values = numeric_array(
        eligible[
            "seen_temporal_fp"
        ],
        dtype=np.float64,
    )
    fn_values = numeric_array(
        eligible[
            "seen_temporal_fn"
        ],
        dtype=np.float64,
    )
    tp_values = numeric_array(
        eligible[
            "seen_temporal_tp"
        ],
        dtype=np.float64,
    )

    tn = int(
        np.nansum(
            tn_values
        )
    )
    fp = int(
        np.nansum(
            fp_values
        )
    )
    fn = int(
        np.nansum(
            fn_values
        )
    )
    tp = int(
        np.nansum(
            tp_values
        )
    )

    total = (
        tn + fp + fn + tp
    )

    recall = (
        float(
            tp / (tp + fn)
        )
        if (tp + fn) > 0
        else None
    )

    precision = (
        float(
            tp / (tp + fp)
        )
        if (tp + fp) > 0
        else None
    )

    if (
        recall is not None
        and precision is not None
        and (recall + precision) > 0
    ):
        f1_value = float(
            2.0
            * recall
            * precision
            / (
                recall
                + precision
            )
        )
    else:
        f1_value = None

    fpr = (
        float(
            fp / (fp + tn)
        )
        if (fp + tn) > 0
        else None
    )

    accuracy = (
        float(
            (tp + tn)
            / total
        )
        if total > 0
        else None
    )

    specificity = (
        float(
            tn / (tn + fp)
        )
        if (tn + fp) > 0
        else None
    )

    balanced_accuracy = (
        float(
            (
                recall
                + specificity
            )
            / 2.0
        )
        if (
            recall is not None
            and specificity is not None
        )
        else None
    )

    recalls = numeric_array(
        eligible[
            "seen_temporal_recall"
        ],
        dtype=np.float64,
        finite_only=True,
    )

    f1s = numeric_array(
        eligible[
            "seen_temporal_f1"
        ],
        dtype=np.float64,
        finite_only=True,
    )

    fprs = numeric_array(
        eligible[
            "seen_temporal_test_fpr"
        ],
        dtype=np.float64,
        finite_only=True,
    )

    aucs = numeric_array(
        eligible[
            "seen_temporal_auc"
        ],
        dtype=np.float64,
        finite_only=True,
    )

    return {
        "status": "ok",
        "eligible_evaluations": int(
            len(ids)
        ),
        "eligible_evaluation_ids": ids,
        "pooled_rows": int(
            total
        ),
        "pooled_benign_rows": int(
            tn + fp
        ),
        "pooled_seen_attack_rows": int(
            tp + fn
        ),
        "pooled_confusion_matrix": [
            [tn, fp],
            [fn, tp],
        ],
        "pooled_accuracy": accuracy,
        "pooled_precision": precision,
        "pooled_recall": recall,
        "pooled_f1": f1_value,
        "pooled_fpr": fpr,
        "pooled_balanced_accuracy": balanced_accuracy,
        "evaluation_recall": finite_stats(
            recalls
        ),
        "evaluation_f1": finite_stats(
            f1s
        ),
        "evaluation_test_fpr": finite_stats(
            fprs
        ),
        "evaluation_auc": finite_stats(
            aucs
        ),
    }


def aggregate_seen_attacks(
    per_attack_df: pd.DataFrame,
) -> pd.DataFrame:
    if per_attack_df.empty:
        return pd.DataFrame()

    endpoint_values = (
        per_attack_df[
            "endpoint"
        ]
        .astype(str)
        .to_numpy()
    )

    seen = per_attack_df.loc[
        endpoint_values
        == "seen_temporal"
    ].copy()

    if seen.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for attack_type, group in seen.groupby(
        "attack_type",
        sort=True,
    ):
        counts = numeric_array(
            group[
                "rows"
            ],
            dtype=np.float64,
        )

        counts = np.nan_to_num(
            counts,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        recalls = numeric_array(
            group[
                "recall"
            ],
            dtype=np.float64,
        )

        valid = (
            np.isfinite(
                recalls
            )
            & (counts > 0)
        )

        if np.any(
            valid
        ):
            total_rows = int(
                np.sum(
                    counts[
                        valid
                    ]
                )
            )

            pooled_recall = float(
                np.sum(
                    counts[
                        valid
                    ]
                    * recalls[
                        valid
                    ]
                )
                / np.sum(
                    counts[
                        valid
                    ]
                )
            )
        else:
            total_rows = 0
            pooled_recall = None

        finite_recall = recalls[
            np.isfinite(
                recalls
            )
        ]

        rows.append(
            {
                "attack_type": str(
                    attack_type
                ),
                "evaluations_present": int(
                    group[
                        "evaluation_id"
                    ].nunique()
                ),
                "total_seen_temporal_rows": total_rows,
                "pooled_recall": pooled_recall,
                "mean_evaluation_recall": (
                    float(
                        np.mean(
                            finite_recall
                        )
                    )
                    if finite_recall.size > 0
                    else None
                ),
                "min_evaluation_recall": (
                    float(
                        np.min(
                            finite_recall
                        )
                    )
                    if finite_recall.size > 0
                    else None
                ),
                "max_evaluation_recall": (
                    float(
                        np.max(
                            finite_recall
                        )
                    )
                    if finite_recall.size > 0
                    else None
                ),
            }
        )

    return pd.DataFrame.from_records(
        rows
    )


def adaptive_vs_strict_summary(
    strict_pooled: Dict[str, Any],
    adaptive_pooled: Dict[str, Any],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "strict_status": strict_pooled.get(
            "status"
        ),
        "adaptive_status": adaptive_pooled.get(
            "status"
        ),
        "strict_eligible_evaluations": strict_pooled.get(
            "eligible_evaluations",
            0,
        ),
        "adaptive_eligible_evaluations": adaptive_pooled.get(
            "eligible_evaluations",
            0,
        ),
    }

    if adaptive_pooled.get(
        "status"
    ) == "ok":
        summary.update(
            {
                "adaptive_pooled_seen_attack_rows": adaptive_pooled.get(
                    "pooled_seen_attack_rows"
                ),
                "adaptive_pooled_recall": adaptive_pooled.get(
                    "pooled_recall"
                ),
                "adaptive_pooled_precision": adaptive_pooled.get(
                    "pooled_precision"
                ),
                "adaptive_pooled_f1": adaptive_pooled.get(
                    "pooled_f1"
                ),
                "adaptive_pooled_fpr": adaptive_pooled.get(
                    "pooled_fpr"
                ),
            }
        )

    return summary


# =============================================================================
# Reviewer-ready report
# =============================================================================
def dataframe_markdown(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    if df.empty:
        return "No rows."

    selected = [
        column
        for column in columns
        if column in df.columns
    ]

    view = df[
        selected
    ].copy()

    try:
        return view.to_markdown(
            index=False
        )
    except Exception:
        return (
            "```csv\n"
            + view.to_csv(
                index=False
            )
            + "```"
        )


def write_report(
    out_path: Path,
    source_meta: Dict[str, Any],
    strict_df: pd.DataFrame,
    adaptive_df: pd.DataFrame,
    terminal_df: pd.DataFrame,
    strict_pooled: Dict[str, Any],
    adaptive_pooled: Dict[str, Any],
    adaptive_attack_summary: pd.DataFrame,
) -> None:
    lines: List[str] = [
        "# Reviewer Concern #2 — Complete Temporal Validation",
        "",
        "## Source and chronology integrity",
        "",
        f"- CSE source SHA-256: `{source_meta['sha256']}`.",
        f"- Timestamp-valid deterministic sample rows: {int(source_meta['timestamp_valid_rows']):,}.",
        f"- Predictor count: {int(source_meta['feature_count'])}.",
        "- Ten chronological blocks were constructed without splitting identical timestamps.",
        "",
        "## Why the strict rolling audit had no eligible seen-attack folds",
        "",
        "The strict protocol reserves the immediately preceding block entirely for calibration. "
        "The dataset's attack campaigns are strongly scheduled in time, so attack families that "
        "recur in the next block commonly first appear in that calibration block rather than in "
        "the trained history. Therefore the strict protocol can legitimately contain zero attacks "
        "whose family has already been seen by the trained model. This is an eligibility property "
        "of the chronology, not a measured recall failure.",
        "",
        f"- Strict eligible seen-attack evaluations: {int(strict_pooled.get('eligible_evaluations') or 0)}.",
        "",
        "## Supervised adaptive rolling protocol",
        "",
        "- Test transitions: B1->B2 through B9->B10.",
        "- For every transition, the immediately preceding block is split chronologically and timestamp-safely.",
        "- First 70% of the preceding block is added to training as a supervised periodic refresh.",
        "- Final 30% of the preceding block is used only for operating-point calibration.",
        "- The primary threshold is fixed from benign calibration scores at <=1% empirical development FPR.",
        "- The complete next block is untouched until final evaluation.",
        "- Seen and novel attack families are reported separately.",
        "- Operational assumption: labels for the adaptation segment become available before the subsequent block.",
        "",
        "## Primary adaptive seen-attack temporal endpoint",
        "",
    ]

    if adaptive_pooled.get(
        "status"
    ) == "ok":
        recall_stats = adaptive_pooled.get(
            "evaluation_recall"
        )

        if not isinstance(
            recall_stats,
            dict,
        ):
            recall_stats = {}

        lines.extend(
            [
                f"- Eligible adaptive evaluations: {int(adaptive_pooled.get('eligible_evaluations') or 0)}.",
                f"- Pooled seen-attack rows: {int(adaptive_pooled.get('pooled_seen_attack_rows') or 0):,}.",
                f"- Pooled recall: **{format_optional_float(adaptive_pooled.get('pooled_recall'))}**.",
                f"- Pooled precision: {format_optional_float(adaptive_pooled.get('pooled_precision'))}.",
                f"- Pooled F1: {format_optional_float(adaptive_pooled.get('pooled_f1'))}.",
                f"- Pooled test FPR: {format_optional_float(adaptive_pooled.get('pooled_fpr'))}.",
                f"- Evaluation recall mean ± SD: "
                f"{format_optional_float(recall_stats.get('mean'))} ± "
                f"{format_optional_float(recall_stats.get('sd'))}.",
                f"- Worst eligible evaluation recall: "
                f"{format_optional_float(recall_stats.get('min'))}.",
            ]
        )
    else:
        lines.append(
            "No adaptive evaluation contained an eligible seen-attack temporal endpoint."
        )

    lines.extend(
        [
            "",
            "## Strict non-adaptive fold results",
            "",
            dataframe_markdown(
                strict_df,
                [
                    "evaluation_id",
                    "train_blocks",
                    "calibration_block",
                    "test_block",
                    "seen_attack_rows",
                    "novel_attack_rows",
                    "combined_recall",
                    "combined_test_fpr",
                    "status",
                ],
            ),
            "",
            "## Adaptive transition results",
            "",
            dataframe_markdown(
                adaptive_df,
                [
                    "evaluation_id",
                    "transition",
                    "adaptation_rows",
                    "calibration_rows_from_prior_block",
                    "seen_test_attack_type_count",
                    "seen_attack_rows",
                    "novel_attack_rows",
                    "seen_temporal_recall",
                    "seen_temporal_precision",
                    "seen_temporal_f1",
                    "seen_temporal_test_fpr",
                    "seen_temporal_auc",
                    "status",
                ],
            ),
            "",
            "## Adaptive seen attack-family summary",
            "",
            dataframe_markdown(
                adaptive_attack_summary,
                [
                    "attack_type",
                    "evaluations_present",
                    "total_seen_temporal_rows",
                    "pooled_recall",
                    "mean_evaluation_recall",
                    "min_evaluation_recall",
                    "max_evaluation_recall",
                ],
            ),
            "",
            "## Terminal 70/15/15 stress diagnostic",
            "",
            dataframe_markdown(
                terminal_df,
                [
                    "evaluation_id",
                    "seen_attack_rows",
                    "novel_attack_rows",
                    "seen_temporal_recall",
                    "combined_recall",
                    "combined_precision",
                    "combined_f1",
                    "combined_test_fpr",
                    "combined_auc",
                    "status",
                ],
            ),
            "",
            "## Interpretation constraints",
            "",
            "1. The strict protocol is retained even if it has zero eligible seen-attack rows; "
            "it documents the dataset chronology and prevents selective omission.",
            "",
            "2. The adaptive protocol is a supervised periodic-refresh scenario, not an "
            "unsupervised online detector. Its label-availability assumption must be stated.",
            "",
            "3. Novel attacks remain an open-set problem and are never merged into the primary "
            "seen-attack temporal endpoint.",
            "",
            "4. The terminal 70/15/15 result is a combined temporal plus attack-novelty stress "
            "diagnostic whenever its attack families are absent from training.",
            "",
            "5. The 1% development-FPR threshold is primary regardless of which sensitivity "
            "target later produces the best test result.",
        ]
    )

    ensure_dir(
        out_path.parent
    )

    out_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# =============================================================================
# Self-test
# =============================================================================
def run_self_test() -> int:
    print(
        "Running synthetic self-test...",
        flush=True,
    )

    # Threshold invariant.
    benign = np.asarray(
        np.linspace(
            0.0,
            1.0,
            1000,
            dtype=np.float64,
        ),
        dtype=np.float64,
    ).reshape(-1)

    threshold = threshold_for_empirical_fpr(
        benign,
        0.01,
    )

    empirical = float(
        np.mean(
            benign
            >= threshold
        )
    )

    assert (
        empirical
        <= 0.01
        + 1e-12
    )

    # Timestamp-safe block and within-block split invariants.
    n = 10_000
    base = pd.Timestamp(
        "2026-01-01"
    )

    ticks = np.repeat(
        np.arange(
            math.ceil(
                n / 7
            )
        ),
        7,
    )[:n]

    timestamps = pd.Series(
        base
        + pd.to_timedelta(
            ticks,
            unit="s",
        )
    )

    labels = np.where(
        np.arange(n)
        % 11
        == 0,
        "ATTACK-X",
        "BENIGN",
    )

    synthetic = pd.DataFrame(
        {
            "_timestamp_parsed": timestamps,
            "AttackType": labels,
            "LabelBinary": (
                labels
                != "BENIGN"
            ).astype(np.int8),
            "feature_hash": np.arange(
                n,
                dtype=np.uint64,
            ),
            "f1": np.arange(
                n,
                dtype=np.float64,
            ),
        }
    )

    block_ids, block_table = make_timestamp_safe_blocks(
        synthetic,
        10,
    )

    synthetic["_temporal_block"] = (
        block_ids
    )

    assert len(
        block_table
    ) == 10

    block_one = synthetic.loc[
        synthetic[
            "_temporal_block"
        ]
        == 1
    ].copy()

    first, second, split_meta = split_block_timestamp_safe(
        block_one,
        0.70,
    )

    first_end = timestamp_array(
        first
    )[-1]

    second_start = timestamp_array(
        second
    )[0]

    assert (
        first_end
        < second_start
    )

    assert (
        0.60
        < float(
            split_meta[
                "actual_first_fraction"
            ]
        )
        < 0.80
    )

    # Pooled metric reconstruction.
    fake = pd.DataFrame.from_records(
        [
            {
                "evaluation_id": "A",
                "status": "eligible",
                "seen_attack_rows": 10,
                "seen_temporal_tn": 90,
                "seen_temporal_fp": 10,
                "seen_temporal_fn": 2,
                "seen_temporal_tp": 8,
                "seen_temporal_recall": 0.8,
                "seen_temporal_f1": 0.57142857,
                "seen_temporal_test_fpr": 0.1,
                "seen_temporal_auc": 0.9,
            },
            {
                "evaluation_id": "B",
                "status": "eligible",
                "seen_attack_rows": 10,
                "seen_temporal_tn": 95,
                "seen_temporal_fp": 5,
                "seen_temporal_fn": 1,
                "seen_temporal_tp": 9,
                "seen_temporal_recall": 0.9,
                "seen_temporal_f1": 0.75,
                "seen_temporal_test_fpr": 0.05,
                "seen_temporal_auc": 0.95,
            },
        ]
    )

    pooled = pooled_seen_metrics(
        fake
    )

    assert abs(
        float(
            pooled[
                "pooled_recall"
            ]
        )
        - 0.85
    ) < 1e-12

    print(
        "Synthetic self-test: PASS",
        flush=True,
    )

    return 0


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reviewer Concern #2 complete strict + adaptive "
            "temporal validation"
        )
    )

    parser.add_argument(
        "--cse",
        default=str(
            DEFAULT_CSE_RAW_CSV
        ),
    )

    parser.add_argument(
        "--out",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--cse-row-cap",
        type=int,
        default=CSE_MAX_MODEL_ROWS,
        help=(
            "Default reviewer run: 3000000. "
            "Set 0 only for an explicitly planned full-source run."
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    seed_everything(
        RANDOM_STATE
    )

    out_root = Path(
        args.out
    )

    ensure_dir(
        out_root
    )

    row_cap = (
        None
        if int(
            args.cse_row_cap
        )
        == 0
        else int(
            args.cse_row_cap
        )
    )

    manifest: Dict[str, Any] = {
        "script": Path(
            __file__
        ).name,
        "purpose": (
            "Complete temporal validation for Reviewer Concern #2"
        ),
        "versions": versions(),
        "random_state": RANDOM_STATE,
        "source_contract": {
            "expected_cse_sha256": EXPECTED_CSE_SOURCE_SHA256,
            "expected_cse_source_rows": EXPECTED_CSE_SOURCE_ROWS,
        },
        "sampling": {
            "row_cap": row_cap,
            "rule": (
                "source-wide Dask sample with fixed seed, "
                "then strict chronological sort"
            ),
        },
        "block_protocol": {
            "n_blocks": N_TEMPORAL_BLOCKS,
            "rule": (
                "approximately equal-row contiguous blocks; "
                "identical timestamps never cross block boundaries"
            ),
        },
        "strict_nonadaptive_protocol": {
            "test_blocks": list(
                range(
                    STRICT_FIRST_TEST_BLOCK,
                    STRICT_LAST_TEST_BLOCK + 1,
                )
            ),
            "rule": (
                "train all blocks before calibration; "
                "whole immediately preceding block calibration; "
                "next block test"
            ),
            "purpose": (
                "retain and explain the original zero-eligibility "
                "rolling-origin audit"
            ),
        },
        "adaptive_protocol": {
            "test_blocks": list(
                range(
                    ADAPTIVE_FIRST_TEST_BLOCK,
                    ADAPTIVE_LAST_TEST_BLOCK + 1,
                )
            ),
            "adaptation_fraction": ADAPTATION_FRACTION,
            "calibration_fraction": (
                1.0
                - ADAPTATION_FRACTION
            ),
            "rule": (
                "all earlier blocks + first 70% of preceding block "
                "for supervised refresh; final 30% preceding block "
                "calibration only; complete next block test"
            ),
            "operational_assumption": (
                "labels for adaptation segment are available before "
                "subsequent test block"
            ),
            "provenance_note": (
                "Adaptive protocol was introduced after the strict chronology "
                "audit demonstrated zero eligible seen-attack test rows. "
                "Its split fraction, transitions, model, threshold policy, "
                "and reporting rules are fixed before adaptive performance "
                "is observed."
            ),
        },
        "threshold_policy": {
            "primary_target_benign_calibration_fpr": PRIMARY_TARGET_FPR,
            "sensitivity_grid": list(
                FPR_SENSITIVITY
            ),
            "test_leakage": (
                "no test row used for fitting or threshold selection"
            ),
        },
        "model": {
            "xgboost_params": XGB_COMMON,
            "num_boost_round": NUM_BOOST_ROUND,
            "balance_binary_train": BALANCE_BINARY_TRAIN,
        },
        "paths": {
            "cse": str(
                args.cse
            ),
            "output": str(
                out_root
            ),
        },
    }

    save_json(
        out_root
        / "run_manifest.json",
        manifest,
    )

    print(
        "\n"
        + "=" * 108,
        flush=True,
    )
    print(
        "Reviewer Concern #2 — COMPLETE TEMPORAL VALIDATION",
        flush=True,
    )
    print(
        f"CSE source: {args.cse}",
        flush=True,
    )
    print(
        f"Output root: {out_root}",
        flush=True,
    )
    print(
        f"CSE row cap: {row_cap}",
        flush=True,
    )
    print(
        "Primary adaptive endpoint: seen-attack temporal recall "
        "at development-calibrated 1% benign FPR",
        flush=True,
    )
    print(
        "=" * 108,
        flush=True,
    )

    print(
        "\nA. SOURCE + CHRONOLOGY AUDIT",
        flush=True,
    )

    df, feature_cols, source_meta = load_cse_temporal_sample(
        Path(
            args.cse
        ),
        row_cap,
    )

    save_json(
        out_root
        / "cse_source_metadata.json",
        source_meta,
    )

    block_ids, block_df = make_timestamp_safe_blocks(
        df,
        N_TEMPORAL_BLOCKS,
    )

    df["_temporal_block"] = (
        block_ids
    )

    block_df.to_csv(
        out_root
        / "temporal_blocks.csv",
        index=False,
    )

    coverage_df = build_attack_coverage_table(
        df
    )

    coverage_df.to_csv(
        out_root
        / "temporal_attack_coverage.csv",
        index=False,
    )

    print(
        block_df[
            [
                "block",
                "rows",
                "benign_rows",
                "attack_rows",
                "attack_type_count",
            ]
        ].to_string(
            index=False
        ),
        flush=True,
    )

    # B. Strict diagnostic.
    (
        strict_details,
        strict_df,
        strict_attack_df,
        strict_sensitivity_df,
        strict_score_df,
    ) = run_strict_protocol(
        df,
        feature_cols,
    )

    strict_df.to_csv(
        out_root
        / "strict_nonadaptive_fold_metrics.csv",
        index=False,
    )
    strict_attack_df.to_csv(
        out_root
        / "strict_nonadaptive_per_attack.csv",
        index=False,
    )
    strict_sensitivity_df.to_csv(
        out_root
        / "strict_nonadaptive_threshold_sensitivity.csv",
        index=False,
    )
    strict_score_df.to_csv(
        out_root
        / "strict_nonadaptive_score_diagnostics.csv",
        index=False,
    )
    save_json(
        out_root
        / "strict_nonadaptive_details.json",
        strict_details,
    )

    strict_pooled = pooled_seen_metrics(
        strict_df
    )

    save_json(
        out_root
        / "strict_nonadaptive_summary.json",
        strict_pooled,
    )

    # C. Adaptive evaluation.
    (
        adaptive_details,
        adaptive_df,
        adaptive_attack_df,
        adaptive_sensitivity_df,
        adaptive_score_df,
        adaptive_split_df,
    ) = run_adaptive_protocol(
        df,
        feature_cols,
    )

    adaptive_df.to_csv(
        out_root
        / "adaptive_rolling_metrics.csv",
        index=False,
    )
    adaptive_attack_df.to_csv(
        out_root
        / "adaptive_rolling_per_attack.csv",
        index=False,
    )
    adaptive_sensitivity_df.to_csv(
        out_root
        / "adaptive_rolling_threshold_sensitivity.csv",
        index=False,
    )
    adaptive_score_df.to_csv(
        out_root
        / "adaptive_rolling_score_diagnostics.csv",
        index=False,
    )
    adaptive_split_df.to_csv(
        out_root
        / "adaptive_rolling_split_audit.csv",
        index=False,
    )
    save_json(
        out_root
        / "adaptive_rolling_details.json",
        adaptive_details,
    )

    adaptive_pooled = pooled_seen_metrics(
        adaptive_df
    )

    adaptive_attack_summary = aggregate_seen_attacks(
        adaptive_attack_df
    )

    adaptive_attack_summary.to_csv(
        out_root
        / "adaptive_seen_attack_summary.csv",
        index=False,
    )

    save_json(
        out_root
        / "adaptive_rolling_summary.json",
        adaptive_pooled,
    )

    # D. Terminal diagnostic.
    (
        terminal_detail,
        terminal_df,
        terminal_attack_df,
        terminal_diag_df,
    ) = run_terminal_protocol(
        df,
        feature_cols,
    )

    terminal_df.to_csv(
        out_root
        / "terminal_70_15_15_metrics.csv",
        index=False,
    )
    terminal_attack_df.to_csv(
        out_root
        / "terminal_70_15_15_per_attack.csv",
        index=False,
    )
    terminal_diag_df.to_csv(
        out_root
        / "terminal_70_15_15_diagnostics.csv",
        index=False,
    )
    save_json(
        out_root
        / "terminal_70_15_15_details.json",
        terminal_detail,
    )

    comparison = adaptive_vs_strict_summary(
        strict_pooled,
        adaptive_pooled,
    )

    complete_summary = {
        "source_metadata": source_meta,
        "strict_nonadaptive": {
            "pooled_seen_temporal": strict_pooled,
            "fold_metrics": strict_df.to_dict(
                orient="records"
            ),
        },
        "adaptive_supervised_refresh": {
            "pooled_seen_temporal": adaptive_pooled,
            "transition_metrics": adaptive_df.to_dict(
                orient="records"
            ),
            "seen_attack_family_summary": adaptive_attack_summary.to_dict(
                orient="records"
            ),
        },
        "terminal_70_15_15": terminal_detail,
        "strict_vs_adaptive": comparison,
        "interpretation": {
            "strict_zero_eligibility": (
                "If strict eligible_evaluations=0, this is a chronology "
                "eligibility result, not a zero temporal recall estimate."
            ),
            "adaptive_scope": (
                "Adaptive results quantify supervised periodic-refresh "
                "temporal transfer under the stated label-availability assumption."
            ),
            "novel_attack_scope": (
                "Novel attacks are separate open-set diagnostics and are "
                "excluded from the primary seen-attack temporal endpoint."
            ),
        },
    }

    save_json(
        out_root
        / "reviewer_ready_complete_temporal_summary.json",
        complete_summary,
    )

    write_report(
        out_root
        / "Reviewer_Concern2_Complete_Temporal_Report.md",
        source_meta,
        strict_df,
        adaptive_df,
        terminal_df,
        strict_pooled,
        adaptive_pooled,
        adaptive_attack_summary,
    )

    print(
        "\n"
        + "=" * 108,
        flush=True,
    )
    print(
        "COMPLETE TEMPORAL VALIDATION FINISHED",
        flush=True,
    )
    print(
        f"Output root: {out_root}",
        flush=True,
    )

    print(
        "\nSTRICT NON-ADAPTIVE",
        flush=True,
    )
    print(
        f"  eligible seen-attack evaluations: "
        f"{int(strict_pooled.get('eligible_evaluations') or 0)}",
        flush=True,
    )

    print(
        "\nADAPTIVE SUPERVISED REFRESH",
        flush=True,
    )

    if adaptive_pooled.get(
        "status"
    ) == "ok":
        recall_stats = adaptive_pooled.get(
            "evaluation_recall"
        )

        if not isinstance(
            recall_stats,
            dict,
        ):
            recall_stats = {}

        print(
            f"  eligible evaluations: "
            f"{int(adaptive_pooled.get('eligible_evaluations') or 0)}",
            flush=True,
        )
        print(
            f"  pooled seen attack rows: "
            f"{int(adaptive_pooled.get('pooled_seen_attack_rows') or 0):,}",
            flush=True,
        )
        print(
            f"  pooled recall: "
            f"{format_optional_float(adaptive_pooled.get('pooled_recall'))}",
            flush=True,
        )
        print(
            f"  pooled precision: "
            f"{format_optional_float(adaptive_pooled.get('pooled_precision'))}",
            flush=True,
        )
        print(
            f"  pooled F1: "
            f"{format_optional_float(adaptive_pooled.get('pooled_f1'))}",
            flush=True,
        )
        print(
            f"  pooled test FPR: "
            f"{format_optional_float(adaptive_pooled.get('pooled_fpr'))}",
            flush=True,
        )
        print(
            f"  evaluation recall mean ± SD: "
            f"{format_optional_float(recall_stats.get('mean'))} ± "
            f"{format_optional_float(recall_stats.get('sd'))}",
            flush=True,
        )
        print(
            f"  worst eligible recall: "
            f"{format_optional_float(recall_stats.get('min'))}",
            flush=True,
        )
    else:
        print(
            "  no eligible adaptive seen-attack temporal evaluation",
            flush=True,
        )

    print(
        "\nTERMINAL 70/15/15",
        flush=True,
    )

    terminal_seen = int(
        terminal_df[
            "seen_attack_rows"
        ].iloc[0]
    ) if not terminal_df.empty else 0

    terminal_novel = int(
        terminal_df[
            "novel_attack_rows"
        ].iloc[0]
    ) if not terminal_df.empty else 0

    print(
        f"  seen attack rows: {terminal_seen:,}",
        flush=True,
    )
    print(
        f"  novel attack rows: {terminal_novel:,}",
        flush=True,
    )

    print(
        "\nPrimary reviewer-ready artifact:",
        flush=True,
    )
    print(
        f"  {out_root / 'reviewer_ready_complete_temporal_summary.json'}",
        flush=True,
    )
    print(
        f"  {out_root / 'Reviewer_Concern2_Complete_Temporal_Report.md'}",
        flush=True,
    )
    print(
        "=" * 108,
        flush=True,
    )

    del df
    gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
