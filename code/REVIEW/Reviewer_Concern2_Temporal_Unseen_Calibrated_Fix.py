#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Reviewer Concern #2 corrective validation for FL-BC-IDS
=======================================================

This is a SEPARATE reviewer-facing experiment. It does not modify the principal
FL-BC-IDS, DP, Groth16, SSI, or blockchain code.

It directly addresses three concerns:
  1) temporal generalization,
  2) completely unseen attacks, and
  3) threshold/calibration robustness under distribution shift.

Design principles
-----------------
A. No final-test leakage.
   - Model fitting uses training data only.
   - Thresholds are locked using development/calibration data only.
   - Held-out attack examples are never used for training or threshold selection.

B. The old "chronological" result is decomposed into:
   - PURE TEMPORAL: final-period rows whose attack type was already represented
     in training (temporal shift without complete attack-type novelty), and
   - COMBINED SHIFT: the complete final chronological period, including attack
     types absent from training.

C. A predeclared operational threshold is selected by an empirical benign-FPR
   constraint (default 1%), instead of transporting a validation-F1 optimum into
   a shifted test distribution.

D. An explicit open-set improvement is evaluated for unseen attacks:
   - supervised XGBoost detector, plus
   - benign-only Isolation Forest novelty detector,
   - combined using a predeclared OR rule with a Bonferroni-style FPR budget.
   Each component receives half of the primary benign-FPR budget, so the total
   empirical development FPR is targeted at <= PRIMARY_TARGET_FPR.

E. For transparency, the script ALSO reports the old validation-F1 threshold
   result beside the improved operating-point results. Weak outcomes are never
   hidden or overwritten.

Expected outputs
----------------
<OUTPUT_ROOT>/
  run_manifest.json
  chronological/
    chronological_summary.json
    chronological_threshold_sensitivity.csv
  unseen_cse/
    CSECICIDS2018_unseen_attack_corrective.csv
  unseen_cic/
    CICIoV2024_unseen_attack_corrective.csv
  reviewer_ready_summary.json
  Reviewer_Concern2_Corrective_Report.md

Important
---------
This code is designed to make the evaluation scientifically stronger. It cannot
and should not guarantee that every held-out attack will have high recall. If a
held-out attack is genuinely indistinguishable from benign/seen behavior in the
selected representation, the result should remain visible.
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
from sklearn.ensemble import IsolationForest
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# Default project paths (same project as the previous reviewer-validation code)
# -----------------------------------------------------------------------------
DEFAULT_CSE_RAW_CSV = Path(
    os.getenv("FLBCIDS_CSE_RAW_CSV", 'data/raw/CSE-CIC-IDS2018/CSECICIDS2018Dataset.csv')
)
DEFAULT_CIC_DECIMAL_DIR = Path(
    os.getenv("FLBCIDS_CICIOV_RAW_DIR", "data/raw/CICIoV2024")
)
DEFAULT_OUTPUT_ROOT = Path(
    os.getenv("FLBCIDS_CORRECTIVE_RESULTS_DIR", "artifacts/generalization/corrective_validation")
)

CIC_FILES = [
    "decimal_benign.csv",
    "decimal_DoS.csv",
    "decimal_spoofing-GAS.csv",
    "decimal_spoofing-RPM.csv",
    "decimal_spoofing-SPEED.csv",
    "decimal_spoofing-STEERING_WHEEL.csv",
]
CIC_FEATURE_COLS = [
    "ID", "DATA_0", "DATA_1", "DATA_2", "DATA_3",
    "DATA_4", "DATA_5", "DATA_6", "DATA_7",
]

CSE_IDENTIFIER_COLS = ["id", "Flow ID", "Src IP", "Dst IP", "Timestamp"]
CSE_LABEL_COL = "Label"

RANDOM_STATE = 42
CSE_MAX_MODEL_ROWS: Optional[int] = 3_000_000
CSE_MIN_HOLDOUT_ATTACK_ROWS = 2_000

# Model family retained from the prior reviewer-validation harness.
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

# Primary reviewer-facing operating point.
PRIMARY_TARGET_FPR = 0.01
# Prespecified sensitivity analysis; NONE of these is selected using final test data.
FPR_SENSITIVITY = (0.001, 0.005, 0.01, 0.02, 0.05)

# Hybrid open-set detector. Each detector receives half of the total FPR budget.
HYBRID_COMPONENT_FPR = PRIMARY_TARGET_FPR / 2.0
ISO_N_ESTIMATORS = 200
ISO_MAX_SAMPLES = 4096
ISO_BENIGN_FIT_CAP = 150_000


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


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


def normalize_label_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )


def hash_frame_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    return pd.util.hash_pandas_object(
        df[list(feature_cols)], index=False
    ).to_numpy(dtype=np.uint64, copy=False)


def make_onehot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=True)


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
        "before": {"benign": int(counts[0]), "attack": int(counts[1])},
        "enabled": BALANCE_BINARY_TRAIN,
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
    after = np.bincount(np.asarray(y2, dtype=int), minlength=2)
    info.update(
        {
            "performed": True,
            "after": {"benign": int(after[0]), "attack": int(after[1])},
        }
    )
    return X2, np.asarray(y2, dtype=np.int8), info


class DetectorBundle:
    def __init__(
        self,
        preprocessor: ColumnTransformer,
        booster: xgb.Booster,
        isolation_forest: Optional[IsolationForest],
        balance_info: Dict[str, Any],
        fit_seconds: float,
    ) -> None:
        self.preprocessor = preprocessor
        self.booster = booster
        self.isolation_forest = isolation_forest
        self.balance_info = balance_info
        self.fit_seconds = fit_seconds


def _to_iso_matrix(X: Any) -> Any:
    if sp.issparse(X):
        return sp.csc_matrix(X)
    return np.asarray(X)


def fit_detector(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    fit_open_set_detector: bool = True,
) -> DetectorBundle:
    y_train = train_df[label_col].to_numpy(dtype=np.int8, copy=False)
    if np.unique(y_train).size != 2:
        raise ValueError("Training data must contain benign and attack rows.")

    pre = build_preprocessor(train_df[list(feature_cols)])
    X_train = pre.fit_transform(train_df[list(feature_cols)])

    # Preserve original transformed benign rows for benign-only novelty fitting.
    benign_idx = np.flatnonzero(y_train == 0)
    X_benign_original = X_train[benign_idx]

    X_bal, y_bal, balance_info = maybe_balance_train(X_train, y_train)
    counts = np.bincount(y_bal.astype(int), minlength=2)

    params = dict(XGB_COMMON)
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": 1.0
            if balance_info.get("performed")
            else float(counts[0] / max(1, counts[1])),
            "base_score": float(np.mean(y_bal)),
        }
    )

    dtrain = xgb.DMatrix(X_bal, label=y_bal)
    t0 = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        verbose_eval=False,
    )

    iso: Optional[IsolationForest] = None
    if fit_open_set_detector and len(benign_idx) >= 100:
        rng = np.random.default_rng(RANDOM_STATE)
        n_fit = min(int(len(benign_idx)), ISO_BENIGN_FIT_CAP)
        if n_fit < len(benign_idx):
            chosen = rng.choice(len(benign_idx), size=n_fit, replace=False)
            X_iso_fit = X_benign_original[chosen]
        else:
            X_iso_fit = X_benign_original

        X_iso_fit = _to_iso_matrix(X_iso_fit)
        iso = IsolationForest(
            n_estimators=ISO_N_ESTIMATORS,
            max_samples=min(ISO_MAX_SAMPLES, n_fit),
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=max(1, (os.cpu_count() or 2) - 1),
        )
        iso.fit(X_iso_fit)

    fit_seconds = time.perf_counter() - t0
    return DetectorBundle(pre, booster, iso, balance_info, fit_seconds)


def transform(bundle: DetectorBundle, df: pd.DataFrame, feature_cols: Sequence[str]) -> Any:
    return bundle.preprocessor.transform(df[list(feature_cols)])


def supervised_scores(bundle: DetectorBundle, X: Any) -> np.ndarray:
    return np.asarray(bundle.booster.predict(xgb.DMatrix(X)), dtype=np.float64)


def novelty_scores(bundle: DetectorBundle, X: Any) -> Optional[np.ndarray]:
    if bundle.isolation_forest is None:
        return None
    # IsolationForest decision_function: larger = more normal. Invert it so larger = more anomalous.
    return -np.asarray(
        bundle.isolation_forest.decision_function(_to_iso_matrix(X)),
        dtype=np.float64,
    )


def old_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 199)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        pred = (scores >= thr).astype(np.int8)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr


def threshold_for_empirical_fpr(
    benign_scores: np.ndarray,
    target_fpr: float,
) -> float:
    """Return a threshold whose empirical benign FPR is <= target_fpr.

    Classification convention is score >= threshold => attack.
    The threshold is determined ONLY from benign development scores.
    """
    x = np.asarray(benign_scores, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite benign development scores available.")
    if not (0.0 < target_fpr < 1.0):
        raise ValueError("target_fpr must be between 0 and 1.")

    desc = np.sort(x)[::-1]
    allowed_fp = int(math.floor(target_fpr * len(desc)))

    if allowed_fp <= 0:
        return float(np.nextafter(desc[0], np.inf))
    if allowed_fp >= len(desc):
        return float(np.nextafter(desc[-1], -np.inf))

    # Just above the (allowed_fp + 1)-th largest benign score.
    return float(np.nextafter(desc[allowed_fp], np.inf))


def safe_auc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    if np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, score))


def metrics_from_predictions(
    y_true: np.ndarray,
    pred: np.ndarray,
    score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(pred, dtype=np.int8)
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    result: Dict[str, Any] = {
        "n": int(len(y)),
        "benign_n": int((y == 0).sum()),
        "attack_n": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y, p)) if len(y) else None,
        "precision": float(precision_score(y, p, zero_division=0)) if len(y) else None,
        "recall": float(recall_score(y, p, zero_division=0)) if len(y) else None,
        "f1": float(f1_score(y, p, zero_division=0)) if len(y) else None,
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, p)) if np.unique(y).size == 2 else None
        ),
        "mcc": float(matthews_corrcoef(y, p)) if np.unique(y).size == 2 else None,
        "fpr": float(fp / max(1, fp + tn)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": cm.tolist(),
    }
    if score is not None:
        result["roc_auc"] = safe_auc(y, np.asarray(score, dtype=np.float64))
    else:
        result["roc_auc"] = None
    return result


def metrics_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, Any]:
    pred = (np.asarray(score) >= threshold).astype(np.int8)
    out = metrics_from_predictions(y, pred, score)
    out["threshold"] = float(threshold)
    return out


def percentile_against_reference(
    values: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Empirical percentile score in [0,1]; larger means more attack/anomaly-like."""
    ref = np.sort(
        np.asarray(
            reference,
            dtype=np.float64,
        )
    )

    vals = np.asarray(
        values,
        dtype=np.float64,
    )

    if ref.size == 0:
        raise ValueError(
            "Empty reference distribution."
        )

    ranks = np.asarray(
        np.searchsorted(
            ref,
            vals,
            side="right",
        ),
        dtype=np.float64,
    )

    percentiles = np.empty_like(
        ranks,
        dtype=np.float64,
    )

    np.divide(
        ranks,
        np.float64(ref.size),
        out=percentiles,
    )

    return percentiles


def hybrid_predictions(
    supervised: np.ndarray,
    novelty: np.ndarray,
    supervised_threshold: float,
    novelty_threshold: float,
) -> np.ndarray:
    return (
        (np.asarray(supervised) >= supervised_threshold)
        | (np.asarray(novelty) >= novelty_threshold)
    ).astype(np.int8)


def split_groups_70_15_15(groups: np.ndarray, seed: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(groups), dtype=np.int64)
    g1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    tr, temp = next(g1.split(idx, groups=groups))
    g2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    va_rel, te_rel = next(g2.split(temp, groups=groups[temp]))
    return tr, temp[va_rel], temp[te_rel]


def parse_cse_timestamp(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)


def load_cse_raw(
    path: Path,
    row_cap: Optional[int],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    print(
        f"  [CSE] Source file found: {path}",
        flush=True,
    )

    source_rows: Optional[int] = None
    sampling_fraction: Optional[float] = None

    if row_cap is None:
        df = pd.read_csv(path, low_memory=False)
        source_rows = int(len(df))
    else:
        import dask.dataframe as dd

        print(
            "  [CSE] Opening dataset with Dask...",
            flush=True,
        )

        ddf = dd.read_csv(
            path,
            assume_missing=True,
            blocksize="128MB",
        )

        print(
            "  [CSE] Counting source rows...",
            flush=True,
        )

        source_rows_value = int(
            ddf.shape[0].compute()
        )

        print(
            f"  [CSE] Source rows: {source_rows_value:,}",
            flush=True,
        )

        source_rows = source_rows_value

        if source_rows_value > row_cap:
            sampling_fraction = (
                    float(row_cap)
                    / float(source_rows_value)
            )

            print(
                f"  [CSE] Sampling approximately {row_cap:,} rows "
                f"(fraction={sampling_fraction:.6f})...",
                flush=True,
            )

            ddf = ddf.sample(
                frac=sampling_fraction,
                random_state=RANDOM_STATE,
            )

        print(
            "  [CSE] Materializing selected rows into memory...",
            flush=True,
        )

        df = ddf.compute()

        print(
            f"  [CSE] Materialized {len(df):,} rows.",
            flush=True,
        )
        if len(df) > row_cap:
            df = df.sample(n=row_cap, random_state=RANDOM_STATE, replace=False)

    df = df.reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    if CSE_LABEL_COL not in df.columns:
        raise ValueError(f"Missing CSE label column {CSE_LABEL_COL!r}")

    df["AttackType"] = normalize_label_series(df[CSE_LABEL_COL])
    df["LabelBinary"] = (df["AttackType"] != "BENIGN").astype(np.int8)

    feature_cols = [
        c
        for c in df.columns
        if c not in set(CSE_IDENTIFIER_COLS + [CSE_LABEL_COL, "AttackType", "LabelBinary"])
    ]
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    print(
        f"  [CSE] Computing predictor hashes for {len(df):,} rows...",
        flush=True,
    )
    df["feature_hash"] = hash_frame_features(df, feature_cols)
    print(
        "  [CSE] Predictor hashing completed.",
        flush=True,
    )
    print(
        "  [CSE] Computing SHA-256 of the original dataset file...",
        flush=True,
    )

    source_sha256 = sha256_file(path)

    print(
        f"  [CSE] SHA-256 completed: {source_sha256}",
        flush=True,
    )

    meta = {
        "path": str(path),
        "sha256": source_sha256,
        "source_rows": source_rows,
        "rows_loaded": int(len(df)),
        "row_cap": row_cap,
        "sampling_fraction": sampling_fraction,
        "feature_count": len(feature_cols),
        "timestamp_available": "Timestamp" in df.columns,
    }
    return df, feature_cols, meta


def load_cic_raw(directory: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frames: List[pd.DataFrame] = []
    file_hashes: Dict[str, str] = {}
    for fname in CIC_FILES:
        path = directory / fname
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        missing = [c for c in CIC_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{fname}: missing {missing}")
        family = fname.replace("decimal_", "").replace(".csv", "")
        df = df.copy()
        df["attack_family"] = family
        df["Label"] = 0 if family == "benign" else 1
        frames.append(df)
        file_hashes[fname] = sha256_file(path)

    merged = pd.concat(frames, ignore_index=True, copy=False)
    merged["feature_hash"] = hash_frame_features(merged, CIC_FEATURE_COLS)
    meta = {
        "directory": str(directory),
        "files_sha256": file_hashes,
        "rows": int(len(merged)),
        "distinct_patterns": int(merged["feature_hash"].nunique()),
    }
    return merged, meta


def evaluate_score_family(
    y_cal: np.ndarray,
    cal_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> Dict[str, Any]:
    old_thr = old_f1_threshold(y_cal, cal_scores)
    primary_thr = threshold_for_empirical_fpr(
        cal_scores[np.asarray(y_cal) == 0], PRIMARY_TARGET_FPR
    )
    out: Dict[str, Any] = {
        "old_validation_f1_threshold": metrics_at_threshold(y_test, test_scores, old_thr),
        "primary_target_fpr_threshold": metrics_at_threshold(y_test, test_scores, primary_thr),
        "thresholds": {
            "old_validation_f1": float(old_thr),
            "primary_target_fpr": float(primary_thr),
        },
    }
    return out


def run_chronological_corrective(
    cse_df: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    print("\n[CORRECTIVE CHRONOLOGICAL] separating pure temporal from combined shift")
    ensure_dir(out_dir)
    if "Timestamp" not in cse_df.columns:
        raise ValueError("CSE Timestamp field is unavailable.")

    ts = parse_cse_timestamp(cse_df["Timestamp"])
    d = cse_df.loc[ts.notna()].copy()
    d["_timestamp_parsed"] = ts.loc[ts.notna()]
    d = d.sort_values("_timestamp_parsed", kind="mergesort").reset_index(drop=True)

    n = len(d)
    i1 = int(math.floor(0.70 * n))
    i2 = int(math.floor(0.85 * n))
    train = d.iloc[:i1].copy()
    cal = d.iloc[i1:i2].copy()
    test = d.iloc[i2:].copy()

    for part in (train, cal, test):
        part["Label"] = part["LabelBinary"].astype(np.int8)

    if np.unique(train["Label"]).size != 2:
        raise ValueError("Chronological training period is not binary-trainable.")
    if np.unique(cal["Label"]).size != 2:
        raise ValueError(
            "Chronological calibration period contains a single class. "
            "Do not silently move final-test rows backward; inspect the time periods instead."
        )

    bundle = fit_detector(train, feature_cols, "Label", fit_open_set_detector=True)
    X_cal = transform(bundle, cal, feature_cols)
    X_test = transform(bundle, test, feature_cols)

    cal_sup = supervised_scores(bundle, X_cal)
    test_sup = supervised_scores(bundle, X_test)
    cal_nov = novelty_scores(bundle, X_cal)
    test_nov = novelty_scores(bundle, X_test)

    y_cal = cal["Label"].to_numpy(dtype=np.int8, copy=False)
    y_test = test["Label"].to_numpy(dtype=np.int8, copy=False)

    # Old threshold, retained transparently as comparator.
    old_thr = old_f1_threshold(y_cal, cal_sup)

    # Primary supervised threshold: fixed benign-FPR operating point.
    sup_thr = threshold_for_empirical_fpr(
        cal_sup[y_cal == 0], PRIMARY_TARGET_FPR
    )

    # Hybrid open-set thresholds: split FPR budget equally between the two detectors.
    hybrid_available = (
            cal_nov is not None
            and test_nov is not None
    )

    if hybrid_available:
        cal_nov_arr = np.asarray(
            cal_nov,
            dtype=np.float64,
        )
        test_nov_arr = np.asarray(
            test_nov,
            dtype=np.float64,
        )

        sup_hybrid_thr = threshold_for_empirical_fpr(
            cal_sup[y_cal == 0],
            HYBRID_COMPONENT_FPR,
        )

        nov_hybrid_thr = threshold_for_empirical_fpr(
            cal_nov_arr[y_cal == 0],
            HYBRID_COMPONENT_FPR,
        )

        hybrid_pred_full = hybrid_predictions(
            test_sup,
            test_nov_arr,
            sup_hybrid_thr,
            nov_hybrid_thr,
        )

        cal_sup_pct = percentile_against_reference(
            cal_sup,
            cal_sup[y_cal == 0],
        )

        cal_nov_pct = percentile_against_reference(
            cal_nov_arr,
            cal_nov_arr[y_cal == 0],
        )

        test_sup_pct = percentile_against_reference(
            test_sup,
            cal_sup[y_cal == 0],
        )

        test_nov_pct = percentile_against_reference(
            test_nov_arr,
            cal_nov_arr[y_cal == 0],
        )

        hybrid_score_full = np.maximum(
            test_sup_pct,
            test_nov_pct,
        )

        hybrid_cal_pred = hybrid_predictions(
            cal_sup,
            cal_nov_arr,
            sup_hybrid_thr,
            nov_hybrid_thr,
        )

        hybrid_cal_metrics = metrics_from_predictions(
            y_cal,
            hybrid_cal_pred,
            np.maximum(
                cal_sup_pct,
                cal_nov_pct,
            ),
        )
    else:
        sup_hybrid_thr = None
        nov_hybrid_thr = None
        hybrid_pred_full = None
        hybrid_score_full = None
        hybrid_cal_metrics = None

    train_attack_types = set(train.loc[train["Label"] == 1, "AttackType"].astype(str))
    test_seen_mask = (test["Label"] == 0) | test["AttackType"].astype(str).isin(train_attack_types)
    test_unseen_mask = (test["Label"] == 1) & ~test["AttackType"].astype(str).isin(train_attack_types)

    def subset_metrics(mask: np.ndarray, mode: str) -> Dict[str, Any]:
        idx = np.flatnonzero(mask)
        y = y_test[idx]
        sup = test_sup[idx]
        result: Dict[str, Any] = {
            "rows": int(len(idx)),
            "attack_types": sorted(test.iloc[idx]["AttackType"].astype(str).unique().tolist()),
            "old_validation_f1_threshold": metrics_at_threshold(y, sup, old_thr),
            "calibrated_supervised_1pct_fpr": metrics_at_threshold(y, sup, sup_thr),
        }
        if (
                hybrid_available
                and hybrid_pred_full is not None
                and hybrid_score_full is not None
        ):
            hp = hybrid_pred_full[idx]
            hs = hybrid_score_full[idx]

            result[
                "hybrid_open_set_1pct_fpr_budget"
            ] = metrics_from_predictions(
                y,
                hp,
                hs,
            )
        result["interpretation"] = mode
        return result

    full_mask = np.ones(len(test), dtype=bool)
    seen_mask_np = test_seen_mask.to_numpy(dtype=bool)
    unseen_mask_np = test_unseen_mask.to_numpy(dtype=bool)

    sensitivity_rows: List[Dict[str, Any]] = []
    for target in FPR_SENSITIVITY:
        thr = threshold_for_empirical_fpr(cal_sup[y_cal == 0], target)
        m_full = metrics_at_threshold(y_test, test_sup, thr)
        m_seen = metrics_at_threshold(y_test[seen_mask_np], test_sup[seen_mask_np], thr)
        sensitivity_rows.append(
            {
                "target_development_fpr": target,
                "threshold": thr,
                "full_test_recall": m_full["recall"],
                "full_test_precision": m_full["precision"],
                "full_test_f1": m_full["f1"],
                "full_test_fpr": m_full["fpr"],
                "seen_attack_temporal_recall": m_seen["recall"],
                "seen_attack_temporal_precision": m_seen["precision"],
                "seen_attack_temporal_f1": m_seen["f1"],
                "seen_attack_temporal_fpr": m_seen["fpr"],
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(
        out_dir / "chronological_threshold_sensitivity.csv", index=False
    )

    result: Dict[str, Any] = {
        "protocol": {
            "definition": (
                "Earliest 70% train; next 15% development/calibration; latest 15% final test; "
                "no shuffling after timestamp ordering. The final test is reported twice: "
                "(i) all rows = combined temporal + attack-novelty shift, and "
                "(ii) rows whose attack types were already represented in training = pure temporal diagnostic."
            ),
            "threshold_selection": (
                "Primary threshold fixed from benign development scores to empirical FPR <= 1%. "
                "No final-test labels or scores are used for threshold selection."
            ),
            "hybrid_open_set": (
                "XGBoost OR benign-only IsolationForest; each component receives 0.5% empirical benign-FPR budget."
            ),
            "train_time_range": [
                str(train["_timestamp_parsed"].min()), str(train["_timestamp_parsed"].max())
            ],
            "calibration_time_range": [
                str(cal["_timestamp_parsed"].min()), str(cal["_timestamp_parsed"].max())
            ],
            "test_time_range": [
                str(test["_timestamp_parsed"].min()), str(test["_timestamp_parsed"].max())
            ],
            "train_rows": int(len(train)),
            "calibration_rows": int(len(cal)),
            "test_rows": int(len(test)),
            "train_attack_types": sorted(train_attack_types),
            "test_attack_types": sorted(test["AttackType"].astype(str).unique().tolist()),
            "unseen_attack_types_in_test": sorted(
                set(test.loc[test["Label"] == 1, "AttackType"].astype(str)) - train_attack_types
            ),
            "old_f1_threshold": float(old_thr),
            "primary_supervised_threshold": float(sup_thr),
            "hybrid_supervised_threshold": (
                float(sup_hybrid_thr) if sup_hybrid_thr is not None else None
            ),
            "hybrid_novelty_threshold": (
                float(nov_hybrid_thr) if nov_hybrid_thr is not None else None
            ),
            "development_hybrid_metrics": hybrid_cal_metrics,
            "fit_seconds": bundle.fit_seconds,
            "balance": bundle.balance_info,
        },
        "combined_final_period": subset_metrics(
            full_mask, "combined temporal shift plus complete attack-type novelty"
        ),
        "pure_temporal_seen_attack_types": subset_metrics(
            seen_mask_np, "temporal shift restricted to attack types represented in training"
        ),
        "unseen_attack_rows_only": subset_metrics(
            unseen_mask_np, "attack rows whose attack type is absent from training; no benign rows in this subset"
        )
        if unseen_mask_np.any()
        else {"rows": 0, "status": "no_unseen_attack_rows_in_final_period"},
        "sensitivity_file": str(out_dir / "chronological_threshold_sensitivity.csv"),
    }

    save_json(out_dir / "chronological_summary.json", result)
    return result


def _prepare_unseen_split(
    benign: pd.DataFrame,
    seen_attacks: pd.DataFrame,
    unseen: pd.DataFrame,
    family_col: str,
    label_col: str,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Create a strict predictor-hash-disjoint unseen-attack protocol.

    BENIGN + seen attacks are first combined, then split by predictor hash as a
    single population. This is stronger than splitting benign and attacks
    independently because it also prevents cross-class hash collisions between
    train/calibration/test.

    The held-out attack remains completely absent from train and calibration.
    Any held-out row whose predictor hash occurs in development is removed and
    counted explicitly.
    """
    dev = pd.concat([benign, seen_attacks], ignore_index=True).copy()
    dev[label_col] = (
        dev[family_col].astype(str).str.upper() != "BENIGN"
    ).astype(np.int8)

    groups = dev["feature_hash"].to_numpy(dtype=np.uint64, copy=False)

    selected: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = None
    for offset in range(500):
        candidate_seed = seed + offset
        tr, ca, be = split_groups_70_15_15(groups, candidate_seed)
        if (
            np.unique(dev.iloc[tr][label_col]).size == 2
            and np.unique(dev.iloc[ca][label_col]).size == 2
            and (dev.iloc[be][label_col] == 0).any()
        ):
            selected = (tr, ca, be, candidate_seed)
            break

    if selected is None:
        raise ValueError(
            "Could not construct a class-complete, predictor-hash-disjoint "
            "train/calibration split after 500 deterministic seeds."
        )

    tr, ca, be, seed_used = selected
    train = dev.iloc[tr].copy().reset_index(drop=True)
    cal = dev.iloc[ca].copy().reset_index(drop=True)

    # Only benign rows from the development-test partition are used as the
    # negative class in the final held-out-attack test.
    benign_test = dev.iloc[be].loc[
        dev.iloc[be][label_col].to_numpy(dtype=np.int8) == 0
    ].copy()

    earlier_hashes = set(
        map(int, pd.concat([train["feature_hash"], cal["feature_hash"]]).unique())
    )

    unseen_overlap = unseen["feature_hash"].isin(earlier_hashes).to_numpy(dtype=bool)
    benign_overlap = benign_test["feature_hash"].isin(earlier_hashes).to_numpy(dtype=bool)

    unseen_clean = unseen.loc[~unseen_overlap].copy()
    benign_clean = benign_test.loc[~benign_overlap].copy()
    test = pd.concat([benign_clean, unseen_clean], ignore_index=True)
    test[label_col] = (
        test[family_col].astype(str).str.upper() != "BENIGN"
    ).astype(np.int8)

    trh = set(map(int, train["feature_hash"].unique()))
    cah = set(map(int, cal["feature_hash"].unique()))
    teh = set(map(int, test["feature_hash"].unique()))
    overlaps = {
        "train_cal": len(trh & cah),
        "train_test": len(trh & teh),
        "cal_test": len(cah & teh),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Predictor-hash leakage in unseen split: {overlaps}")

    if np.unique(test[label_col]).size != 2:
        raise ValueError(
            "Final held-out-attack test must contain both benign and held-out attack rows."
        )

    meta = {
        "split_seed_used": int(seed_used),
        "heldout_rows_removed_due_to_overlap": int(unseen_overlap.sum()),
        "benign_test_rows_removed_due_to_overlap": int(benign_overlap.sum()),
        "hash_overlap": overlaps,
        "train_rows": int(len(train)),
        "calibration_rows": int(len(cal)),
        "test_rows": int(len(test)),
        "global_group_split": True,
        "heldout_attack_absent_from_train_and_calibration": True,
    }
    return train, cal, test, meta


def evaluate_unseen_case(
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
) -> Dict[str, Any]:
    bundle = fit_detector(train, feature_cols, label_col, fit_open_set_detector=True)
    X_cal = transform(bundle, cal, feature_cols)
    X_test = transform(bundle, test, feature_cols)

    y_cal = cal[label_col].to_numpy(dtype=np.int8, copy=False)
    y_test = test[label_col].to_numpy(dtype=np.int8, copy=False)

    cal_sup = supervised_scores(bundle, X_cal)
    test_sup = supervised_scores(bundle, X_test)
    cal_nov = novelty_scores(bundle, X_cal)
    test_nov = novelty_scores(bundle, X_test)

    old_thr = old_f1_threshold(y_cal, cal_sup)
    supervised_thr = threshold_for_empirical_fpr(
        cal_sup[y_cal == 0], PRIMARY_TARGET_FPR
    )

    out: Dict[str, Any] = {
        "old_validation_f1": metrics_at_threshold(y_test, test_sup, old_thr),
        "calibrated_supervised_1pct_fpr": metrics_at_threshold(
            y_test, test_sup, supervised_thr
        ),
        "old_threshold": float(old_thr),
        "calibrated_supervised_threshold": float(supervised_thr),
        "fit_seconds": bundle.fit_seconds,
        "balance": bundle.balance_info,
    }

    if cal_nov is not None and test_nov is not None:
        cal_nov_arr = np.asarray(
            cal_nov,
            dtype=np.float64,
        )
        test_nov_arr = np.asarray(
            test_nov,
            dtype=np.float64,
        )

        sup_hybrid_thr = threshold_for_empirical_fpr(
            cal_sup[y_cal == 0],
            HYBRID_COMPONENT_FPR,
        )

        nov_hybrid_thr = threshold_for_empirical_fpr(
            cal_nov_arr[y_cal == 0],
            HYBRID_COMPONENT_FPR,
        )

        pred = hybrid_predictions(
            test_sup,
            test_nov_arr,
            sup_hybrid_thr,
            nov_hybrid_thr,
        )

        sup_pct = percentile_against_reference(
            test_sup,
            cal_sup[y_cal == 0],
        )

        nov_pct = percentile_against_reference(
            test_nov_arr,
            cal_nov_arr[y_cal == 0],
        )
        hybrid_score = np.maximum(sup_pct, nov_pct)
        out["hybrid_open_set_1pct_fpr_budget"] = metrics_from_predictions(
            y_test, pred, hybrid_score
        )
        out["hybrid_supervised_threshold"] = float(sup_hybrid_thr)
        out["hybrid_novelty_threshold"] = float(nov_hybrid_thr)
        out["novelty_detector_auc"] = safe_auc(
            y_test,
            test_nov_arr,
        )
    else:
        out["hybrid_open_set_1pct_fpr_budget"] = None
        out["novelty_detector_auc"] = None

    return out


def run_cse_unseen_corrective(
    df: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
) -> pd.DataFrame:
    print("\n[CORRECTIVE CSE UNSEEN] strict held-out attacks + calibrated/open-set detector")
    ensure_dir(out_dir)
    counts = df.loc[df["AttackType"] != "BENIGN", "AttackType"].value_counts()
    holdouts = [str(x) for x in counts[counts >= CSE_MIN_HOLDOUT_ATTACK_ROWS].index]
    benign: pd.DataFrame = df.loc[
        df["AttackType"] == "BENIGN",
        :,
    ].copy()

    rows: List[Dict[str, Any]] = []
    for heldout in holdouts:
        print(f"  [CSE] hold out: {heldout}")
        seen: pd.DataFrame = df.loc[
            (df["AttackType"] != "BENIGN")
            & (df["AttackType"] != heldout),
            :,
        ].copy()

        unseen: pd.DataFrame = df.loc[
            df["AttackType"] == heldout,
            :,
        ].copy()
        train, cal, test, meta = _prepare_unseen_split(
            benign, seen, unseen, "AttackType", "Label", RANDOM_STATE
        )
        result = evaluate_unseen_case(train, cal, test, feature_cols, "Label")

        old = result["old_validation_f1"]
        sup = result["calibrated_supervised_1pct_fpr"]
        hyb = result.get("hybrid_open_set_1pct_fpr_budget") or {}
        rows.append(
            {
                "held_out_attack": heldout,
                "raw_heldout_rows": int(len(unseen)),
                **meta,
                "old_recall": old.get("recall"),
                "old_precision": old.get("precision"),
                "old_f1": old.get("f1"),
                "old_fpr": old.get("fpr"),
                "calibrated_supervised_recall": sup.get("recall"),
                "calibrated_supervised_precision": sup.get("precision"),
                "calibrated_supervised_f1": sup.get("f1"),
                "calibrated_supervised_fpr": sup.get("fpr"),
                "hybrid_recall": hyb.get("recall"),
                "hybrid_precision": hyb.get("precision"),
                "hybrid_f1": hyb.get("f1"),
                "hybrid_fpr": hyb.get("fpr"),
                "hybrid_auc": hyb.get("roc_auc"),
                "supervised_auc": sup.get("roc_auc"),
                "novelty_auc": result.get("novelty_detector_auc"),
                "old_threshold": result.get("old_threshold"),
                "calibrated_supervised_threshold": result.get(
                    "calibrated_supervised_threshold"
                ),
                "hybrid_supervised_threshold": result.get("hybrid_supervised_threshold"),
                "hybrid_novelty_threshold": result.get("hybrid_novelty_threshold"),
            }
        )
        del train, cal, test, result
        gc.collect()

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "CSECICIDS2018_unseen_attack_corrective.csv", index=False)
    return out


def run_cic_unseen_corrective(
    df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    print(
        "\n[CORRECTIVE CIC UNSEEN] "
        "strict held-out families + calibrated/open-set detector"
    )

    ensure_dir(out_dir)

    family_values = np.asarray(
        df["attack_family"].to_numpy(
            copy=False
        ),
        dtype=str,
    ).reshape(-1)

    benign_positions = np.flatnonzero(
        family_values == "benign"
    )

    benign: pd.DataFrame = df.iloc[
        benign_positions,
        :,
    ].copy()

    holdouts: List[str] = sorted(
        {
            str(value)
            for value in family_values.tolist()
            if str(value) != "benign"
        }
    )

    rows: List[Dict[str, Any]] = []

    for heldout in holdouts:
        print(
            f"  [CIC] hold out: {heldout}"
        )

        seen_mask = (
            (family_values != "benign")
            & (family_values != heldout)
        )

        unseen_mask = (
            family_values == heldout
        )

        seen_positions = np.flatnonzero(
            seen_mask
        )

        unseen_positions = np.flatnonzero(
            unseen_mask
        )

        seen: pd.DataFrame = df.iloc[
            seen_positions,
            :,
        ].copy()

        unseen: pd.DataFrame = df.iloc[
            unseen_positions,
            :,
        ].copy()

        train, cal, test, meta = _prepare_unseen_split(
            benign,
            seen,
            unseen,
            "attack_family",
            "Label",
            RANDOM_STATE,
        )

        result = evaluate_unseen_case(
            train,
            cal,
            test,
            CIC_FEATURE_COLS,
            "Label",
        )

        old: Dict[str, Any] = result[
            "old_validation_f1"
        ]

        sup: Dict[str, Any] = result[
            "calibrated_supervised_1pct_fpr"
        ]

        hybrid_result = result.get(
            "hybrid_open_set_1pct_fpr_budget"
        )

        hyb: Dict[str, Any] = (
            hybrid_result
            if isinstance(hybrid_result, dict)
            else {}
        )

        rows.append(
            {
                "held_out_attack": heldout,
                "raw_heldout_rows": int(
                    len(unseen)
                ),
                **meta,
                "old_recall": old.get(
                    "recall"
                ),
                "old_precision": old.get(
                    "precision"
                ),
                "old_f1": old.get(
                    "f1"
                ),
                "old_fpr": old.get(
                    "fpr"
                ),
                "calibrated_supervised_recall": sup.get(
                    "recall"
                ),
                "calibrated_supervised_precision": sup.get(
                    "precision"
                ),
                "calibrated_supervised_f1": sup.get(
                    "f1"
                ),
                "calibrated_supervised_fpr": sup.get(
                    "fpr"
                ),
                "hybrid_recall": hyb.get(
                    "recall"
                ),
                "hybrid_precision": hyb.get(
                    "precision"
                ),
                "hybrid_f1": hyb.get(
                    "f1"
                ),
                "hybrid_fpr": hyb.get(
                    "fpr"
                ),
                "hybrid_auc": hyb.get(
                    "roc_auc"
                ),
                "supervised_auc": sup.get(
                    "roc_auc"
                ),
                "novelty_auc": result.get(
                    "novelty_detector_auc"
                ),
                "old_threshold": result.get(
                    "old_threshold"
                ),
                "calibrated_supervised_threshold": result.get(
                    "calibrated_supervised_threshold"
                ),
                "hybrid_supervised_threshold": result.get(
                    "hybrid_supervised_threshold"
                ),
                "hybrid_novelty_threshold": result.get(
                    "hybrid_novelty_threshold"
                ),
            }
        )

        del train, cal, test, result
        gc.collect()

    out = pd.DataFrame.from_records(
        rows
    )

    out.to_csv(
        out_dir
        / "CICIoV2024_unseen_attack_corrective.csv",
        index=False,
    )

    return out


def summarize_unseen(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    if df.empty:
        return {
            "rows": 0
        }

    def desc(
        col: str,
    ) -> Dict[str, Any]:
        numeric_values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        values = np.asarray(
            numeric_values,
            dtype=np.float64,
        ).reshape(-1)

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            return {
                "n": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }

        return {
            "n": int(
                values.size
            ),
            "mean": float(
                np.mean(values)
            ),
            "median": float(
                np.median(values)
            ),
            "min": float(
                np.min(values)
            ),
            "max": float(
                np.max(values)
            ),
        }

    return {
        "attacks_evaluated": int(
            len(df)
        ),
        "old_recall": desc(
            "old_recall"
        ),
        "calibrated_supervised_recall": desc(
            "calibrated_supervised_recall"
        ),
        "hybrid_recall": desc(
            "hybrid_recall"
        ),
        "per_attack": df.to_dict(
            orient="records"
        ),
    }



def _df_to_report_text(df: pd.DataFrame, empty_message: str) -> str:
    if df.empty:
        return empty_message
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```csv\n" + df.to_csv(index=False) + "```"

def write_report(
    out_path: Path,
    chrono: Dict[str, Any],
    cse_unseen: pd.DataFrame,
    cic_unseen: pd.DataFrame,
) -> None:
    pure = chrono.get("pure_temporal_seen_attack_types", {})
    combined = chrono.get("combined_final_period", {})

    def metric_line(block: Dict[str, Any], key: str) -> str:
        m = block.get(key) or {}
        return (
            f"recall={m.get('recall')}, precision={m.get('precision')}, "
            f"F1={m.get('f1')}, FPR={m.get('fpr')}, AUC={m.get('roc_auc')}"
        )

    lines = [
        "# Reviewer Concern #2 — Corrective Generalization Validation",
        "",
        "## Protocol integrity",
        "",
        "- Final-test data are never used for model fitting or threshold selection.",
        "- The chronological evaluation separates pure temporal shift from complete attack-type novelty.",
        "- The primary operating point is predeclared at 1% empirical benign FPR on development data.",
        "- Unseen-attack tests remove each held-out attack from both training and development data.",
        "- The open-set improvement combines supervised XGBoost with a benign-only Isolation Forest using a split FPR budget.",
        "",
        "## Chronological evaluation",
        "",
        f"Pure temporal / old F1 threshold: {metric_line(pure, 'old_validation_f1_threshold')}",
        f"Pure temporal / calibrated 1% FPR: {metric_line(pure, 'calibrated_supervised_1pct_fpr')}",
        f"Pure temporal / hybrid open-set: {metric_line(pure, 'hybrid_open_set_1pct_fpr_budget')}",
        "",
        f"Combined final period / old F1 threshold: {metric_line(combined, 'old_validation_f1_threshold')}",
        f"Combined final period / calibrated 1% FPR: {metric_line(combined, 'calibrated_supervised_1pct_fpr')}",
        f"Combined final period / hybrid open-set: {metric_line(combined, 'hybrid_open_set_1pct_fpr_budget')}",
        "",
        "## Unseen-attack evaluation",
        "",
        "See the CSV tables for per-attack old, calibrated-supervised, and hybrid-open-set results.",
        "",
        "### CSE-CIC-IDS2018",
        "",
        _df_to_report_text(cse_unseen, "No CSE unseen results."),
        "",
        "### CICIoV2024",
        "",
        _df_to_report_text(cic_unseen, "No CIC unseen results."),
        "",
        "## Interpretation rule",
        "",
        "The corrected results should replace the earlier confounded chronological interpretation only if the protocol and generated artifacts are retained. The old numbers remain useful as a diagnostic comparator and should not be described as a typographical error.",
    ]
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cse", default=str(DEFAULT_CSE_RAW_CSV))
    parser.add_argument("--cic-dir", default=str(DEFAULT_CIC_DECIMAL_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--cse-row-cap",
        type=int,
        default=CSE_MAX_MODEL_ROWS,
        help="Set 0 to load the complete CSE file.",
    )
    parser.add_argument("--skip-cse-unseen", action="store_true")
    parser.add_argument("--skip-cic-unseen", action="store_true")
    args = parser.parse_args()

    seed_everything()
    out_root = Path(args.out)
    ensure_dir(out_root)
    row_cap = None if int(args.cse_row_cap) == 0 else int(args.cse_row_cap)

    manifest: Dict[str, Any] = {
        "script": Path(__file__).name,
        "versions": versions(),
        "random_state": RANDOM_STATE,
        "primary_target_fpr": PRIMARY_TARGET_FPR,
        "fpr_sensitivity": list(FPR_SENSITIVITY),
        "hybrid_component_fpr": HYBRID_COMPONENT_FPR,
        "xgboost_params": XGB_COMMON,
        "num_boost_round": NUM_BOOST_ROUND,
        "isolation_forest": {
            "n_estimators": ISO_N_ESTIMATORS,
            "max_samples": ISO_MAX_SAMPLES,
            "benign_fit_cap": ISO_BENIGN_FIT_CAP,
        },
        "paths": {
            "cse": args.cse,
            "cic_dir": args.cic_dir,
            "output": str(out_root),
        },
        "cse_row_cap": row_cap,
        "test_leakage_policy": "No final-test data used for fitting, detector choice, or threshold selection.",
    }
    save_json(out_root / "run_manifest.json", manifest)

    print("\n" + "=" * 100, flush=True)
    print("Reviewer Concern #2 corrective validation started.", flush=True)
    print(f"CSE source: {args.cse}", flush=True)
    print(f"CIC source: {args.cic_dir}", flush=True)
    print(f"Output root: {out_root}", flush=True)
    print(f"CSE model row cap: {row_cap}", flush=True)
    print("=" * 100, flush=True)

    print(
        "\n[1/4] Loading and preparing CSE-CIC-IDS2018...",
        flush=True,
    )

    cse_df, cse_features, cse_meta = load_cse_raw(
        Path(args.cse),
        row_cap,
    )

    print(
        f"[1/4] CSE preparation completed: "
        f"{len(cse_df):,} rows, {len(cse_features)} predictors.",
        flush=True,
    )
    save_json(out_root / "cse_source_metadata.json", cse_meta)

    chrono = run_chronological_corrective(
        cse_df, cse_features, out_root / "chronological"
    )

    if args.skip_cse_unseen:
        cse_unseen = pd.DataFrame()
    else:
        cse_unseen = run_cse_unseen_corrective(
            cse_df, cse_features, out_root / "unseen_cse"
        )

    del cse_df
    gc.collect()

    if args.skip_cic_unseen:
        cic_unseen = pd.DataFrame()
        cic_meta = {"status": "skipped"}
    else:
        cic_df, cic_meta = load_cic_raw(Path(args.cic_dir))
        save_json(out_root / "cic_source_metadata.json", cic_meta)
        cic_unseen = run_cic_unseen_corrective(
            cic_df, out_root / "unseen_cic"
        )
        del cic_df
        gc.collect()

    aggregate = {
        "chronological": chrono,
        "CSECICIDS2018_unseen": summarize_unseen(cse_unseen),
        "CICIoV2024_unseen": summarize_unseen(cic_unseen),
        "source_metadata": {"cse": cse_meta, "cic": cic_meta},
    }
    save_json(out_root / "reviewer_ready_summary.json", aggregate)
    write_report(
        out_root / "Reviewer_Concern2_Corrective_Report.md",
        chrono,
        cse_unseen,
        cic_unseen,
    )

    print("\n" + "=" * 100)
    print("Reviewer Concern #2 corrective validation completed.")
    print(f"Output root: {out_root}")
    print("Primary artifacts:")
    print(f"  - {out_root / 'chronological' / 'chronological_summary.json'}")
    print(f"  - {out_root / 'chronological' / 'chronological_threshold_sensitivity.csv'}")
    print(f"  - {out_root / 'unseen_cse' / 'CSECICIDS2018_unseen_attack_corrective.csv'}")
    print(f"  - {out_root / 'unseen_cic' / 'CICIoV2024_unseen_attack_corrective.csv'}")
    print(f"  - {out_root / 'reviewer_ready_summary.json'}")
    print(f"  - {out_root / 'Reviewer_Concern2_Corrective_Report.md'}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
