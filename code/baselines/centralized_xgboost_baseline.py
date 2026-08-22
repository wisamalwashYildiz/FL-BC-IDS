#!/usr/bin/env python3
"""
Simple centralized non-FL XGBoost baseline for the IoV thesis.

Purpose
-------
Train a plain centralized XGBoost model on the same preprocessed train/val/test
CSV splits used by the main IoV prototype, but without any FL, DP, SSI,
Merkle, ZK, blockchain, or custom project utilities.

What it does
------------
- Loads the preprocessed CSV splits for CSE-CIC-IDS2018 and/or CICIoV2024.
- Uses the same fixed train/validation/test split files.
- Trains a centralized non-DP XGBoost model.
- Chooses the classification threshold on the validation split by maximizing F1.
- Evaluates on validation and test.
- Saves the model, confusion matrix, and summary JSON.

Notes
-----
- This is intentionally simple and standalone.
- It assumes the preprocessing outputs encode the feature space and binary
  label mapping expected by the FL-BC-IDS pipeline.
- It validates that validation/test splits preserve the training feature schema.
- It is meant for utility comparison only; do not compare communication,
  RSU/GLOBAL proofing, or on-chain costs against this baseline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "CSECICIDS2018": {
        "preproc_dir": os.getenv(
            "FLBCIDS_CSE_PREPROC_DIR",
            "data/preprocessed/CSE-CIC-IDS2018",
        ),
        "train_csv": "CSECICIDS2018_train_preprocessed.csv",
        "val_csv": "CSECICIDS2018_val_preprocessed.csv",
        "test_csv": "CSECICIDS2018_test_preprocessed.csv",
        "label_col": "Label",
    },
    "CICIoV2024": {
        "preproc_dir": os.getenv(
            "FLBCIDS_CICIOV_PREPROC_DIR",
            "data/preprocessed/CICIoV2024",
        ),
        "train_csv": "CICIoV2024_train_preprocessed.csv",
        "val_csv": "CICIoV2024_val_preprocessed.csv",
        "test_csv": "CICIoV2024_test_preprocessed.csv",
        "label_col": "Label",
    },
}

# Run both one after the other by default.
DATASETS_TO_RUN: List[str] = ["CSECICIDS2018", "CICIoV2024"]

OUTPUT_ROOT = Path(
    os.getenv(
        "FLBCIDS_CENTRALIZED_OUTPUT_DIR",
        "artifacts/centralized_xgboost_outputs",
    )
)
RANDOM_STATE = 42

# Chosen to stay close to the thesis non-DP XGBoost family while remaining simple.
XGB_PARAMS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
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
    "verbosity": 1,
}
NUM_BOOST_ROUND = 100
EARLY_STOPPING_ROUNDS = 10


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_split_csvs(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_CONFIGS)}"
        )

    cfg = DATASET_CONFIGS[dataset_name]
    preproc_dir = Path(cfg["preproc_dir"])
    train_path = preproc_dir / cfg["train_csv"]
    val_path = preproc_dir / cfg["val_csv"]
    test_path = preproc_dir / cfg["test_csv"]
    label_col = cfg["label_col"]

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in {dataset_name} {name} split"
            )

    return df_train, df_val, df_test, label_col



def extract_numeric_xy(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    label_col: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    """Extract the numeric feature schema in deterministic TRAIN column order."""

    numeric_feature_cols = [
        c
        for c in df_train.columns
        if c != label_col and pd.api.types.is_numeric_dtype(df_train[c])
    ]
    if not numeric_feature_cols:
        raise ValueError(
            "No numeric feature columns were found after excluding the label column"
        )

    missing_val = [c for c in numeric_feature_cols if c not in df_val.columns]
    missing_test = [c for c in numeric_feature_cols if c not in df_test.columns]
    if missing_val or missing_test:
        raise ValueError(
            "Feature-schema mismatch across splits: "
            f"missing_in_val={missing_val}, missing_in_test={missing_test}"
        )

    nonnumeric_val = [
        c for c in numeric_feature_cols
        if not pd.api.types.is_numeric_dtype(df_val[c])
    ]
    nonnumeric_test = [
        c for c in numeric_feature_cols
        if not pd.api.types.is_numeric_dtype(df_test[c])
    ]
    if nonnumeric_val or nonnumeric_test:
        raise ValueError(
            "Numeric feature dtype mismatch across splits: "
            f"nonnumeric_in_val={nonnumeric_val}, "
            f"nonnumeric_in_test={nonnumeric_test}"
        )

    X_train = df_train.loc[:, numeric_feature_cols].to_numpy(dtype=np.float32)
    y_train = df_train[label_col].to_numpy(dtype=np.int32)
    X_val = df_val.loc[:, numeric_feature_cols].to_numpy(dtype=np.float32)
    y_val = df_val[label_col].to_numpy(dtype=np.int32)
    X_test = df_test.loc[:, numeric_feature_cols].to_numpy(dtype=np.float32)
    y_test = df_test[label_col].to_numpy(dtype=np.int32)

    for split_name, y in (
        ("train", y_train),
        ("validation", y_val),
        ("test", y_test),
    ):
        labels = set(np.unique(y).tolist())
        if not labels.issubset({0, 1}):
            raise ValueError(
                f"{split_name} labels are not binary 0/1: {sorted(labels)}"
            )
        if len(labels) < 2:
            raise ValueError(
                f"{split_name} split contains only one class: {sorted(labels)}"
            )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        numeric_feature_cols,
    )

def compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos <= 0 or neg <= 0:
        raise ValueError("Training split must contain both classes")
    return float(neg) / float(pos)



def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    # Simple and stable threshold search.
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(np.int32)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr



def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_proba = np.clip(np.asarray(y_proba, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y_pred = (y_proba >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred)

    metrics: Dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "logloss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier": float(np.mean((y_proba - y_true) ** 2)),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        "fp": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        "fn": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        "tp": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
    }
    return metrics



def train_and_evaluate_dataset(dataset_name: str) -> Dict[str, Any]:
    print(f"\n{'=' * 90}")
    print(f"Running centralized baseline for: {dataset_name}")
    print(f"{'=' * 90}")

    dataset_out_dir = OUTPUT_ROOT / dataset_name
    ensure_dir(dataset_out_dir)

    df_train, df_val, df_test, label_col = load_split_csvs(dataset_name)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = extract_numeric_xy(
        df_train, df_val, df_test, label_col
    )

    scale_pos_weight = compute_scale_pos_weight(y_train)
    base_score = float(np.mean(y_train))

    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight
    params["base_score"] = base_score

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    train_start = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    train_time_sec = time.perf_counter() - train_start

    val_start = time.perf_counter()
    y_val_proba = booster.predict(dval)
    val_inference_time_sec = time.perf_counter() - val_start

    threshold = find_best_threshold(y_val, y_val_proba)
    val_metrics = compute_metrics(y_val, y_val_proba, threshold)

    test_start = time.perf_counter()
    y_test_proba = booster.predict(dtest)
    test_inference_time_sec = time.perf_counter() - test_start
    test_metrics = compute_metrics(y_test, y_test_proba, threshold)

    model_path = dataset_out_dir / f"centralized_xgboost_{dataset_name}.json"
    booster.save_model(model_path)

    results: Dict[str, Any] = {
        "dataset": dataset_name,
        "label_col": label_col,
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "num_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "xgboost_params": params,
        "experiment_seed": int(RANDOM_STATE),
        "publication_role": "centralized_non_dp_design_reference",
        "threshold_selection": {
            "split": "validation",
            "criterion": "maximum_f1",
            "grid_min": 0.05,
            "grid_max": 0.95,
            "grid_points": 181,
        },
        "num_boost_round": int(NUM_BOOST_ROUND),
        "early_stopping_rounds": int(EARLY_STOPPING_ROUNDS),
        "best_iteration": int(getattr(booster, "best_iteration", -1)),
        "best_score": (
            float(getattr(booster, "best_score"))
            if getattr(booster, "best_score", None) is not None
            else None
        ),
        "train_time_sec": float(train_time_sec),
        "val_inference_time_sec": float(val_inference_time_sec),
        "test_inference_time_sec": float(test_inference_time_sec),
        "model_path": str(model_path),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    summary_path = dataset_out_dir / f"centralized_xgboost_{dataset_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved model:   {model_path}")
    print(f"Saved summary: {summary_path}")
    print(
        "TEST | "
        f"Acc={test_metrics['accuracy']:.6f} | "
        f"Prec={test_metrics['precision']:.6f} | "
        f"Rec={test_metrics['recall']:.6f} | "
        f"F1={test_metrics['f1']:.6f} | "
        f"AUC={test_metrics['auc']:.6f} | "
        f"LogLoss={test_metrics['logloss']:.6f} | "
        f"Brier={test_metrics['brier']:.6f} | "
        f"Thr={test_metrics['threshold']:.4f}"
    )

    return results



def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    all_results: Dict[str, Any] = {}

    for dataset_name in DATASETS_TO_RUN:
        result = train_and_evaluate_dataset(dataset_name)
        all_results[dataset_name] = result

    overall_path = OUTPUT_ROOT / "centralized_xgboost_all_datasets_summary.json"
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved overall summary: {overall_path}")


if __name__ == "__main__":
    main()
