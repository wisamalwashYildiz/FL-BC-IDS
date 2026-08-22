# FL_IoV_DPxgb_UtilsV10.py
import base64
import hashlib
import inspect
import json
import logging
from pathlib import Path
import os
import warnings
import shutil
import tempfile
import ast  # <-- NEW: to safely parse str(history)
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from decimal import Decimal, InvalidOperation
import numpy as np
import pandas as pd
import json as _json
import dp_xgboost as xgb  # DP-enabled XGBoost (Sarus)
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import FL_IoV_AnchorZKP_UtilsV10 as azkp          # AnchorSum ZK lane
import FL_IoV_CanonicalSpecV10 as canon           # ✅ Phase-1 canonical bytes authority
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,  # <-- NEW: for threshold tuning
    brier_score_loss,
    confusion_matrix,
    log_loss,
)
# ---------------------------------------------------------------------------
# CanonicalSpec contract imports (single source of truth for DP record bytes+hash)
# ---------------------------------------------------------------------------
try:
    from FL_IoV_CanonicalSpecV10 import (
        canon_json_bytes_v1,
        build_dp_record_v1_minimal,
        assert_sha256_hex_str_v1,
        sha256_hex,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing CanonicalSpec dependencies. Ensure FL_IoV_CanonicalSpecV10.py is available.\n"
        "Expected: canon_json_bytes_v1, build_dp_record_v1_minimal, assert_sha256_hex_str_v1, sha256_hex."
    ) from e
# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class IoVDatasetConfig:
    preproc_dir: str
    train_csv: str
    val_csv: str
    test_csv: str
    label_col: str = "Label"  # <-- IoV label (BENIGN vs ATTACK)
@dataclass
class RSUConfig:
    rsu_id: int
    num_rsus: int
    vehicles_per_rsu: int
    num_rounds: int
    num_local_rounds: int
    output_dir: str
@dataclass
class XGBoostConfig:
    params: Dict
    num_local_rounds: int
    early_stopping_rounds: int | None = None
# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    os.environ["FLWR_TELEMETRY_ENABLED"] = "0"
    # Silence Ray's local-mode / client-mode warnings (any warning type)
    warnings.filterwarnings(
        "ignore",
        message=".*local mode is an experimental feature.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*client mode.*",
    )
    # Filter out Flower deprecation logs we cannot yet fully refactor away
    class _FlowerDeprecationFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            msg = record.getMessage()
            # start_simulation deprecation
            if "DEPRECATED FEATURE: flwr.simulation.start_simulation()" in msg:
                return False
            # old-style client_fn(str) deprecation (for safety, although you fixed it)
            if "DEPRECATED FEATURE" in msg and "client_fn" in msg:
                return False
            # evaluate_metrics_aggregation_fn requirement (you already provide it)
            if "No evaluate_metrics_aggregation_fn" in msg:
                return False
            return True
    # Attach the filter to root *and* Flower loggers
    flt = _FlowerDeprecationFilter()
    logging.getLogger().addFilter(flt)                     # root logger
    logging.getLogger("flwr").addFilter(flt)               # all Flower logs
    logging.getLogger("flwr.simulation").addFilter(flt)
    logging.getLogger("flwr.simulation.legacy_app").addFilter(flt)
def extract_history_distributed_summary(
    history: object,
) -> Tuple[List, Dict]:
    """
    Best-effort, Flower-version-safe extraction of distributed loss and
    evaluation metrics from a History object.
    1) Try history.losses_distributed and history.metrics_distributed
       (the "official" attributes).
    2) If they are empty or missing, fall back to parsing str(history),
       which is what Flower uses for the [SUMMARY] block.
    This version also supports the newer multi-line History format, e.g.:
        History (loss, distributed):
            round 1: 0
            round 2: 0
        History (metrics, distributed, evaluate):
            {'accuracy': [(1, ...), (2, ...)], ...}
    """
    # -----------------------
    # 1) Direct attributes
    # -----------------------
    losses = getattr(history, "losses_distributed", None)
    metrics_raw = getattr(history, "metrics_distributed", None)
    if losses is None:
        losses = []
    if metrics_raw is None:
        metrics_raw = {}
    # If Flower already populated metrics_distributed in the standard
    # way, prefer that.
    if isinstance(metrics_raw, dict) and metrics_raw:
        # Newer layout: {"evaluate": {...}, "fit": {...}}
        if "evaluate" in metrics_raw and isinstance(metrics_raw["evaluate"], dict):
            metrics_eval = metrics_raw["evaluate"]
        else:
            # Older layout: metrics_distributed is already per-metric dict
            metrics_eval = metrics_raw
        return list(losses), metrics_eval
    # -----------------------
    # 2) Fallback: parse str(history) (multi-line aware)
    # -----------------------
    metrics_eval: Dict = {}
    parsed_losses: List = []
    try:
        history_str = str(history)
        lines = history_str.splitlines()
        in_loss_block = False
        in_metrics_block = False
        metrics_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            # Start of loss block
            if stripped.startswith("History (loss, distributed"):
                in_loss_block = True
                in_metrics_block = False
                continue
            # Start of metrics block
            if stripped.startswith("History (metrics, distributed, evaluate"):
                in_metrics_block = True
                in_loss_block = False
                continue
            # Inside loss block
            if in_loss_block:
                # End conditions: blank line or new History section
                if not stripped or stripped.startswith("History ("):
                    in_loss_block = False
                    continue
                # Two possible formats:
                # (a) Old list format on a single line:
                #     [(1, 0.01), (2, 0.005)]
                if "[" in stripped:
                    try:
                        tmp = ast.literal_eval(stripped)
                        if isinstance(tmp, list):
                            parsed_losses = tmp
                            in_loss_block = False
                            continue
                    except Exception:
                        # Fall through and try round-based parsing
                        pass
                # (b) New "round N: value" format
                #     round 1: 0
                #     round 2: 0
                if stripped.lower().startswith("round"):
                    try:
                        before, _, val_str = stripped.partition(":")
                        # "round 1" -> take last token as index
                        tokens = before.split()
                        rnd_idx = int(tokens[-1])
                        loss_val = float(val_str.strip())
                        parsed_losses.append((rnd_idx, loss_val))
                    except Exception:
                        # Ignore malformed lines
                        pass
            # Inside metrics block
            if in_metrics_block:
                # End conditions: blank line or new History section
                if not stripped or stripped.startswith("History ("):
                    in_metrics_block = False
                    continue
                # Collect all lines of the dict; we will parse them together
                metrics_lines.append(stripped)
        # Finalize losses
        if parsed_losses:
            losses = parsed_losses
        # Finalize metrics
        if metrics_lines:
            metrics_str = " ".join(metrics_lines)
            try:
                tmp = ast.literal_eval(metrics_str)
                if isinstance(tmp, dict):
                    metrics_eval = tmp
            except Exception:
                # Best-effort: leave metrics_eval as {}
                pass
    except Exception:
        # Completely best-effort; if anything goes wrong, we just return
        # whatever we have so far.
        pass
    return list(losses), metrics_eval
def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for binary classification.
    Equal-width bins in [0, 1]; each bin is weighted by its sample fraction.
    If only one class is present, we return 0.0 (ECE not meaningful).
    """
    try:
        y_true = np.asarray(y_true, dtype=np.int32)
        y_proba = np.asarray(y_proba, dtype=np.float32)
        if y_true.ndim != 1 or y_proba.ndim != 1 or len(y_true) != len(y_proba):
            return 0.0
        if len(np.unique(y_true)) < 2:
            return 0.0
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
        n = float(len(y_true))
        ece = 0.0
        for i in range(n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if not np.any(mask):
                continue
            frac_pos = float(np.mean(y_true[mask]))
            conf_avg = float(np.mean(y_proba[mask]))
            bin_prob = float(np.sum(mask)) / n
            ece += bin_prob * abs(frac_pos - conf_avg)
        return float(ece)
    except Exception:
        return 0.0
def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    calibration: bool = False,
) -> Dict[str, float]:
    """Compute standard binary classification metrics for IoV IDS.

    If `calibration` is True, also compute log-loss, Brier score, and ECE.
    """
    # Normalize inputs
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    # Stabilize probabilities for probabilistic metrics
    y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)
    y_proba = np.clip(y_proba, 0.0, 1.0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        auc = 0.0

    metrics: Dict[str, float] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
    }

    if calibration:
        # Use a strictly interior probability range for stable log-loss
        y_proba_log = np.clip(y_proba, 1e-7, 1.0 - 1e-7)

        try:
            logloss_val = log_loss(y_true, y_proba_log, labels=[0, 1])
        except Exception:
            logloss_val = 0.0

        try:
            brier = brier_score_loss(y_true, y_proba)
        except Exception:
            brier = 0.0

        ece = compute_ece(y_true, y_proba)

        metrics["logloss"] = float(logloss_val)
        metrics["brier"] = float(brier)
        metrics["ece"] = float(ece)

    return metrics
def robust_aggregate_probas(
    probas_list: List[np.ndarray],
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    """
    Robust aggregation across multiple probability vectors using
    coordinate-wise clipped mean.
    Parameters
    ----------
    probas_list : List[np.ndarray]
        List of probability arrays, all with the same shape (e.g., (N,)).
    clip_min : float, optional
        Lower clipping bound (default 0.0).
    clip_max : float, optional
        Upper clipping bound (default 1.0).
    Returns
    -------
    np.ndarray
        Aggregated probability array of the same shape as the inputs,
        obtained by:
        1) stacking all vectors,
        2) clipping each coordinate into [clip_min, clip_max],
        3) taking the mean along axis=0.
    Notes
    -----
    - Intentionally restricted to *clipped mean only*; we no longer support
      plain mean or trimmed/median modes.
    - Designed to be easy to port into a zk-SNARK circuit later:
      clip → sum → divide by the number of RSUs.
    """
    if not probas_list:
        raise ValueError("robust_aggregate_probas: probas_list is empty")
    # Convert all arrays to float64 and check shapes
    arrs = [np.asarray(p, dtype=np.float64) for p in probas_list]
    base_shape = arrs[0].shape
    for a in arrs[1:]:
        if a.shape != base_shape:
            raise ValueError(
                f"All probability arrays must have the same shape; "
                f"got {a.shape} vs {base_shape}"
            )
    # Stack into shape (M, ...) and apply clipping
    stacked = np.stack(arrs, axis=0)
    clipped = np.clip(stacked, clip_min, clip_max)
    # Coordinate-wise mean across RSUs
    return clipped.mean(axis=0)
def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    default_threshold: float = 0.5,
) -> float:
    """
    Choose a decision threshold in [0, 1] that maximizes F1 on (y_true, y_proba).
    We use precision_recall_curve over the validation set and fall back to
    default_threshold if the curve is degenerate.
    """
    # Basic shape guards
    if y_true.ndim != 1 or y_proba.ndim != 1 or len(y_true) != len(y_proba):
        return default_threshold
    # If only one class is present, threshold doesn't matter
    if len(np.unique(y_true)) < 2:
        return default_threshold
    # DP/regression outputs can slightly leave [0, 1]; clamp for stability
    y_proba = np.clip(y_proba, 0.0, 1.0)
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        if thresholds.size == 0:
            return default_threshold
        # precision_recall_curve returns len(thresholds) = len(precision) - 1
        prec = precision[:-1]
        rec = recall[:-1]
        denom = np.clip(prec + rec, a_min=1e-12, a_max=None)
        f1_scores = 2.0 * prec * rec / denom
        best_idx = int(np.nanargmax(f1_scores))
        best_thr = float(thresholds[best_idx])
        best_thr = max(0.0, min(1.0, best_thr))
        # If best F1 is still zero, don't overfit a useless threshold
        if f1_scores[best_idx] <= 0.0:
            return default_threshold
        return best_thr
    except Exception:
        return default_threshold
def log_metrics_pretty(
    prefix: str,
    metrics: Dict[str, float],
    cm: np.ndarray | None = None,
) -> None:
    msg = (
        f"{prefix} - Acc: {metrics['accuracy']:.6f}, "
        f"Prec: {metrics['precision']:.6f}, Rec: {metrics['recall']:.6f}, "
        f"F1: {metrics['f1']:.6f}, AUC: {metrics['auc']:.6f}"
    )

    if "logloss" in metrics:
        msg += f", LogLoss: {float(metrics['logloss']):.6f}"
    if "brier" in metrics:
        msg += f", Brier: {float(metrics['brier']):.6f}"
    if "ece" in metrics:
        msg += f", ECE: {float(metrics['ece']):.6f}"

    logging.info(msg)
    if cm is not None:
        logging.info(f"{prefix} - Confusion matrix:\n{cm}")
def load_iov_splits(cfg: IoVDatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed IoV IDS train/validation/test splits."""
    logging.info("Loading IoV IDS splits (train/val/test)...")
    df_train = pd.read_csv(cfg.train_csv)
    df_val = pd.read_csv(cfg.val_csv)
    df_test = pd.read_csv(cfg.test_csv)
    if cfg.label_col not in df_train.columns:
        raise ValueError(f"Label column '{cfg.label_col}' not found in train CSV")
    if cfg.label_col not in df_val.columns:
        raise ValueError(f"Label column '{cfg.label_col}' not found in val CSV")
    if cfg.label_col not in df_test.columns:
        raise ValueError(f"Label column '{cfg.label_col}' not found in test CSV")
    logging.info(
        "Loaded IoV splits: "
        f"Train={df_train.shape}, Val={df_val.shape}, Test={df_test.shape}"
    )
    return df_train, df_val, df_test
def extract_numeric_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    label_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Extract the shared numeric feature schema in deterministic TRAIN order.

    The earlier set-intersection implementation could reorder features across
    Python processes because set iteration order is hash-dependent.  This
    implementation preserves the preprocessing/training column order while
    still requiring every selected feature to be numeric in all three splits.
    """
    train_numeric = [
        c
        for c in df_train.select_dtypes(include=["number"]).columns
        if c != label_col
    ]
    val_numeric = set(df_val.select_dtypes(include=["number"]).columns)
    test_numeric = set(df_test.select_dtypes(include=["number"]).columns)

    numeric_cols = [
        c
        for c in train_numeric
        if c in val_numeric and c in test_numeric
    ]

    if not numeric_cols:
        raise ValueError("No shared numeric feature columns found across splits")

    missing_val = [c for c in numeric_cols if c not in df_val.columns]
    missing_test = [c for c in numeric_cols if c not in df_test.columns]
    if missing_val or missing_test:
        raise ValueError(
            "Feature-schema mismatch across splits: "
            f"missing_in_val={missing_val}, missing_in_test={missing_test}"
        )

    logging.info(
        "Number of IoV numeric features (ordered shared schema): %d",
        len(numeric_cols),
    )

    X_train = df_train.loc[:, numeric_cols]
    y_train = df_train[label_col].astype(int)
    X_val = df_val.loc[:, numeric_cols]
    y_val = df_val[label_col].astype(int)
    X_test = df_test.loc[:, numeric_cols]
    y_test = df_test[label_col].astype(int)

    for name, X_, y_ in [
        ("Train", X_train, y_train),
        ("Val", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        if X_.empty or y_.empty:
            raise ValueError(f"{name} split is empty or invalid")
        labels = set(np.unique(y_.to_numpy(dtype=np.int32)).tolist())
        if not labels.issubset({0, 1}):
            raise ValueError(
                f"{name} split contains non-binary labels: {sorted(labels)}"
            )

    return X_train, y_train, X_val, y_val, X_test, y_test

def partition_train_among_rsus_and_vehicles(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_rsus: int,
    vehicles_per_rsu: int,
    random_state: int = 42,
    k_min_pos: int = 1,
    k_min_neg: int = 1,
) -> List[List[Tuple[pd.DataFrame, pd.Series]]]:
    """
    Stratified partition of the training split among RSUs and vehicles.
    Stage 1: split positives and negatives across RSUs so each RSU sees
             (approximately) the same class distribution.
    Stage 2: inside each RSU, split its own data into balanced per-vehicle
             shards using `_allocate_counts`.
    If one of the classes is missing globally, we fall back to a simple
    equal-chunk partitioning.
    Returns: List[rsu_id] -> List[(X_vehicle, y_vehicle)].
    """
    total_vehicles = num_rsus * vehicles_per_rsu
    n = len(X_train)
    if total_vehicles > n:
        raise ValueError(
            f"Total vehicles ({total_vehicles}) exceeds number of train samples ({n})."
        )
    rng = np.random.default_rng(seed=random_state)
    indices = np.arange(n)
    rng.shuffle(indices)
    y_arr = y_train.to_numpy()
    # Indices of positives (ATTACK) and negatives (BENIGN), in shuffled order
    pos_indices_all = indices[y_arr[indices] == 1]
    neg_indices_all = indices[y_arr[indices] == 0]
    n_pos_total = len(pos_indices_all)
    n_neg_total = len(neg_indices_all)
    logging.info(
        "Global train label counts: total=%d, pos=%d, neg=%d, vehicles=%d, rsus=%d",
        n,
        n_pos_total,
        n_neg_total,
        total_vehicles,
        num_rsus,
    )
    # If one of the classes is missing globally, fall back to the old
    # equal-chunk behavior (cannot do meaningful stratification).
    if n_pos_total == 0 or n_neg_total == 0:
        logging.warning(
            "Global train split has only one class (pos=%d, neg=%d); "
            "falling back to simple equal-chunk partitioning.",
            n_pos_total,
            n_neg_total,
        )
        idx_chunks = np.array_split(indices, total_vehicles)
        rsu_partitions_fallback: List[List[Tuple[pd.DataFrame, pd.Series]]] = []
        chunk_idx = 0
        for _ in range(num_rsus):
            rsu_vehicles: List[Tuple[pd.DataFrame, pd.Series]] = []
            for _ in range(vehicles_per_rsu):
                idx = idx_chunks[chunk_idx]
                chunk_idx += 1
                X_v = X_train.iloc[idx].reset_index(drop=True)
                y_v = y_train.iloc[idx].reset_index(drop=True)
                rsu_vehicles.append((X_v, y_v))
            rsu_partitions_fallback.append(rsu_vehicles)
        logging.info(
            "Partitioned train data (fallback) into %d RSUs x %d vehicles "
            "(total vehicles = %d).",
            num_rsus,
            vehicles_per_rsu,
            total_vehicles,
        )
        return rsu_partitions_fallback
    def _allocate_counts(total: int, total_buckets: int, k_min: int) -> List[int]:
        """
        Allocate `total` items across `total_buckets` buckets.
        - If total >= k_min * total_buckets, give at least k_min to each
          bucket, then distribute the remainder as evenly as possible.
        - Otherwise, give 1 item to as many buckets as possible, others get 0.
        """
        if total <= 0:
            return [0] * total_buckets
        if total >= k_min * total_buckets:
            counts = [k_min] * total_buckets
            remaining = total - k_min * total_buckets
            extra_base = remaining // total_buckets
            extra_rem = remaining % total_buckets
            for i in range(total_buckets):
                counts[i] += extra_base
            for i in range(extra_rem):
                counts[i] += 1
        else:
            counts = [0] * total_buckets
            for i in range(total):
                counts[i] += 1
        if sum(counts) != total:
            raise RuntimeError(
                f"_allocate_counts internal error: sum(counts)={sum(counts)} != total={total}"
            )
        return counts
    # -------------------------
    # Stage 1: split per RSU
    # -------------------------
    # Split positives and negatives into `num_rsus` RSU sub-pools
    pos_chunks_rsu = np.array_split(pos_indices_all, num_rsus)
    neg_chunks_rsu = np.array_split(neg_indices_all, num_rsus)
    rsu_partitions: List[List[Tuple[pd.DataFrame, pd.Series]]] = []
    for rsu_idx in range(num_rsus):
        pos_indices_rsu = pos_chunks_rsu[rsu_idx]
        neg_indices_rsu = neg_chunks_rsu[rsu_idx]
        n_pos_rsu = len(pos_indices_rsu)
        n_neg_rsu = len(neg_indices_rsu)
        n_rsu_total = n_pos_rsu + n_neg_rsu
        logging.info(
            "RSU %d: train subset total=%d, pos=%d, neg=%d (pos_frac=%.4f)",
            rsu_idx + 1,
            n_rsu_total,
            n_pos_rsu,
            n_neg_rsu,
            float(n_pos_rsu) / n_rsu_total if n_rsu_total > 0 else 0.0,
        )
        # -------------------------
        # Stage 2: split per vehicle inside this RSU
        # -------------------------
        pos_counts = _allocate_counts(n_pos_rsu, vehicles_per_rsu, k_min_pos)
        neg_counts = _allocate_counts(n_neg_rsu, vehicles_per_rsu, k_min_neg)
        pos_cursor = 0
        neg_cursor = 0
        rsu_vehicles: List[Tuple[pd.DataFrame, pd.Series]] = []
        for veh_idx in range(vehicles_per_rsu):
            n_pos_v = pos_counts[veh_idx]
            n_neg_v = neg_counts[veh_idx]
            idx_pos_v = pos_indices_rsu[pos_cursor : pos_cursor + n_pos_v]
            idx_neg_v = neg_indices_rsu[neg_cursor : neg_cursor + n_neg_v]
            pos_cursor += n_pos_v
            neg_cursor += n_neg_v
            combined_idx = np.concatenate([idx_pos_v, idx_neg_v])
            rng.shuffle(combined_idx)
            X_v = X_train.iloc[combined_idx].reset_index(drop=True)
            y_v = y_train.iloc[combined_idx].reset_index(drop=True)
            # Sanity check: does this vehicle see both BENIGN and ATTACK?
            unique_labels = sorted(y_v.unique().tolist())
            if len(unique_labels) < 2:
                logging.info(
                    "RSU %d, vehicle %d: local shard has only labels %s "
                    "(pos=%d, neg=%d).",
                    rsu_idx + 1,
                    veh_idx + 1,
                    unique_labels,
                    int((y_v == 1).sum()),
                    int((y_v == 0).sum()),
                )
            rsu_vehicles.append((X_v, y_v))
        rsu_partitions.append(rsu_vehicles)
    logging.info(
        "Stratified partitioned train data into %d RSUs x %d vehicles "
        "(total vehicles = %d) with per-RSU label balancing.",
        num_rsus,
        vehicles_per_rsu,
        total_vehicles,
    )
    return rsu_partitions
def log_partition_stats(
    rsu_partitions: List[List[Tuple[pd.DataFrame, pd.Series]]]
) -> None:
    """
    Log detailed statistics about the RSU/vehicle partitions:
    - For each RSU and each vehicle:
      * number of samples,
      * number of positives,
      * number of negatives,
      * positive fraction.
    - Global summary:
      * min/max/mean positive fraction across all vehicles,
      * number of vehicles with only one class.
    """
    logging.info("===== Partition statistics (per RSU / vehicle) =====")
    total_vehicles = 0
    pos_fracs: List[float] = []
    one_class_vehicles = 0
    for rsu_idx, rsu_vehicles in enumerate(rsu_partitions, start=1):
        logging.info("RSU %d: %d vehicles", rsu_idx, len(rsu_vehicles))
        for veh_idx, (_, y_v) in enumerate(rsu_vehicles, start=1):
            n_samples = int(len(y_v))
            n_pos = int((y_v == 1).sum())
            n_neg = n_samples - n_pos
            frac_pos = float(n_pos) / n_samples if n_samples > 0 else 0.0
            logging.info(
                "RSU %d | Vehicle %d: samples=%d, pos=%d, neg=%d, pos_frac=%.4f",
                rsu_idx,
                veh_idx,
                n_samples,
                n_pos,
                n_neg,
                frac_pos,
            )
            total_vehicles += 1
            if n_samples > 0:
                pos_fracs.append(frac_pos)
            if (n_pos == 0 or n_neg == 0) and n_samples > 0:
                one_class_vehicles += 1
    if total_vehicles > 0 and pos_fracs:
        pos_fracs_arr = np.asarray(pos_fracs, dtype=float)
        logging.info(
            "Global partition stats: vehicles=%d, "
            "min_pos_frac=%.4f, max_pos_frac=%.4f, mean_pos_frac=%.4f, "
            "vehicles_with_one_class=%d",
            total_vehicles,
            float(pos_fracs_arr.min()),
            float(pos_fracs_arr.max()),
            float(pos_fracs_arr.mean()),
            one_class_vehicles,
        )
    else:
        logging.warning("Partition stats: no vehicles to summarize.")
def initialize_booster_with_dummy_data(
    params: Dict,
    feature_count: int,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
) -> xgb.Booster:
    """
    Create a tiny XGBoost booster with the correct feature dimensionality.
    For DP-XGBoost we must pass feature_min/feature_max so that the DP
    histogram/sketch mechanisms know the bounds (even for this dummy tree).
    """
    if feature_count <= 0:
        raise ValueError("Feature count must be positive.")
    dummy_data = np.zeros((1, feature_count), dtype=np.float32)
    dummy_labels = np.array([0], dtype=np.int32)
    dtrain = xgb.DMatrix(
        dummy_data,
        label=dummy_labels,
        feature_min=feature_min,
        feature_max=feature_max,
    )
    booster = xgb.train(params, dtrain, num_boost_round=1)
    return booster
def booster_to_json_bytes(booster: xgb.Booster) -> bytes:
    """
    Serialize an XGBoost (or dp-xgboost) Booster to JSON bytes compatible
    with FedXgbBagging.
    In addition to saving in JSON format, this function ensures that
    'iteration_indptr' exists under:
        learner.gradient_booster.model["iteration_indptr"]
    Some dp-xgboost builds omit this field, but Flower's FedXgbBagging
    aggregate() always expects it in the *previous* global model.
    """
    # Make static analyzers happy and provide a final fallback
    data: bytes = b""
    # Unique temporary JSON path
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(tmp_fd)  # close the low-level fd so XGBoost can reopen the file
    try:
        # 1) Save the Booster as JSON (due to .json suffix)
        booster.save_model(tmp_path)
        # 2) Read as text so we can inspect/patch the JSON
        with open(tmp_path, "r", encoding="utf-8") as f:
            model_str = f.read()
        try:
            model_json = _json.loads(model_str)
            learner = model_json.get("learner", {})
            gb = learner.get("gradient_booster", {})
            gbtree_model = gb.get("model", {})
            gbparams = gbtree_model.get("gbtree_model_param", {})
            trees_list = gbtree_model.get("trees", [])
            # --- Determine num_trees and num_parallel_tree robustly ---
            try:
                num_trees = int(gbparams.get("num_trees", len(trees_list)))
            except Exception:
                num_trees = len(trees_list)
            try:
                num_parallel = int(gbparams.get("num_parallel_tree", 1))
            except Exception:
                num_parallel = 1
            # 3) Inject iteration_indptr if missing
            if (
                isinstance(gbtree_model, dict)
                and "iteration_indptr" not in gbtree_model
                and num_trees >= 0
                and num_parallel > 0
            ):
                # Standard XGBoost meaning:
                # iteration_indptr = [0, P, 2P, ..., num_trees]
                # where P = num_parallel_tree
                iteration_indptr = list(range(0, num_trees + 1, num_parallel))
                gbtree_model["iteration_indptr"] = iteration_indptr
                # Write back to the nested structure
                gb["model"] = gbtree_model
                learner["gradient_booster"] = gb
                model_json["learner"] = learner
                # Re-serialize the patched JSON
                model_str = _json.dumps(model_json)
            # 4) Final bytes
            data = model_str.encode("utf-8")
        except Exception as e:
            # If *anything* goes wrong during JSON surgery, fall back to raw bytes
            logging.warning(
                "booster_to_json_bytes: failed to inject iteration_indptr (%s), "
                "falling back to raw JSON bytes.",
                e,
            )
            try:
                with open(tmp_path, "rb") as f:
                    data = f.read()
            except Exception as e_read:
                logging.warning(
                    "booster_to_json_bytes: also failed to read raw bytes (%s); "
                    "returning empty bytes.",
                    e_read,
                )
    except Exception as e_outer:
        # If saving or opening the model itself fails, we log and return empty bytes.
        logging.warning(
            "booster_to_json_bytes: failed to save/read model JSON (%s); "
            "returning empty bytes.",
            e_outer,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            logging.warning("Could not delete temporary model file %s", tmp_path)
    return data
def booster_from_json_bytes(model_bytes: bytes) -> xgb.Booster:
    """
    Deserialize JSON-encoded Booster bytes back into an XGBoost Booster.
    XGBoost can load JSON directly from a bytes-like object.
    """
    booster = xgb.Booster()
    booster.load_model(bytearray(model_bytes))
    return booster
def make_delta_booster(
    booster: xgb.Booster,
    prev_num_rounds: int,
) -> xgb.Booster:
    """
    Return a Booster containing only the trees added after `prev_num_rounds`.
    This uses XGBoost's native slicing (booster[start:end]) instead of
    manual JSON surgery, so it is robust to internal JSON layout changes.
    """
    try:
        total_rounds = booster.num_boosted_rounds()
    except Exception:
        # If we cannot inspect the booster, just return it unchanged
        return booster
    # No new rounds: return an empty (0-tree) booster
    if total_rounds <= max(prev_num_rounds, 0):
        try:
            return booster[0:0]
        except Exception as exc:
            logging.warning(
                "make_delta_booster: failed to create empty slice: %s; "
                "falling back to full booster",
                exc,
            )
            return booster
    start = max(prev_num_rounds, 0)
    end = total_rounds
    try:
        delta = booster[start:end]
    except Exception as exc:
        logging.warning(
            "make_delta_booster: failed to slice booster [%d:%d]: %s; "
            "falling back to full booster",
            start,
            end,
            exc,
        )
        return booster
    return delta
def _count_trees_in_booster(booster: xgb.Booster) -> int:
    """
    Return the number of trees in a booster.

    Uses booster.get_dump(), which works across CPU/GPU tree methods.
    """
    try:
        return len(booster.get_dump())
    except Exception:
        return -1
def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float):
            return int(x)
        if isinstance(x, str):
            s = x.strip()
            return int(s) if s else default
        return int(x)  # last resort if it supports __int__
    except Exception:
        return default
def _as_str(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default
def _decimal_str_canon_v1(x: Any, *, field_name: str) -> str:
    """
    Canonical decimal string encoder used for DP parameters (epsilon, delta, scales).
    Rules:
      - input may be int, float, Decimal, or string
      - output is a base-10 string with no exponent
      - non-negative only
      - no trailing zeros, no trailing decimal point
      - empty becomes "0"
    """
    if x is None:
        return "0"
    if isinstance(x, (int, np.integer)):
        xi = int(x)
        if xi < 0:
            raise ValueError(f"{field_name} must be non-negative, got {xi}")
        return str(xi)
    if isinstance(x, float):
        s0 = str(x).strip()
        if s0 == "":
            return "0"
        try:
            d0 = Decimal(s0)
        except Exception as e:
            raise ValueError(f"{field_name} invalid float value: {s0}") from e
        if d0.is_nan() or d0.is_infinite():
            raise ValueError(f"{field_name} must be finite, got {s0}")
        if d0 < 0:
            raise ValueError(f"{field_name} must be non-negative, got {s0}")
        s = format(d0, "f")
    elif isinstance(x, Decimal):
        if x.is_nan() or x.is_infinite():
            raise ValueError(f"{field_name} must be finite")
        if x < 0:
            raise ValueError(f"{field_name} must be non-negative")
        s = format(x, "f")
    else:
        s0 = str(x).strip()
        if s0 == "":
            return "0"
        try:
            d0 = Decimal(s0)
        except InvalidOperation as e:
            raise ValueError(f"{field_name} invalid decimal string: {s0}") from e
        if d0.is_nan() or d0.is_infinite():
            raise ValueError(f"{field_name} must be finite, got {s0}")
        if d0 < 0:
            raise ValueError(f"{field_name} must be non-negative, got {s0}")
        s = format(d0, "f")
    if "." in s:
        s = s.rstrip("0")
        if s.endswith("."):
            s = s[:-1]
    if s == "":
        s = "0"
    if s.startswith("+"):
        s = s[1:]
    if s.startswith("-"):
        raise ValueError(f"{field_name} must be non-negative, got {s}")
    return s
def _assert_canon_json_bytes_roundtrip_v1(record_bytes: bytes, *, field_name: str) -> None:
    """
    Enforce the CanonicalSpec contract:
    bytes -> json -> canon_json_bytes_v1(json) must equal the original bytes.
    """
    if not isinstance(record_bytes, (bytes, bytearray)) or len(record_bytes) == 0:
        raise ValueError(f"{field_name} must be non-empty bytes")
    try:
        obj = _json.loads(bytes(record_bytes).decode("utf-8"))
    except Exception as e:
        raise ValueError(f"{field_name} is not valid UTF-8 JSON bytes") from e
    b2 = canon_json_bytes_v1(obj)
    if b2 != bytes(record_bytes):
        raise ValueError(
            f"{field_name} is not canonical per canon_json_bytes_v1 "
            f"(round-trip bytes mismatch)"
        )
def build_dp_round_record_and_sha256_v1(
    *,
    round_idx: int,
    mechanism: str,
    clip_l1: int,
    epsilon: Any,
    delta: Any = "0",
    accountant: str = "RDP",
    extra: str = "",
) -> Tuple[bytes, str]:
    """
    Build canonical DPRecordV1 bytes and its SHA-256 hex digest.
    HARD CONTRACT (plan): the returned hash MUST equal sha256_hex(returned_bytes),
    and returned_bytes MUST be canonical under canon_json_bytes_v1.
    """
    r = int(round_idx)
    if r < 0:
        raise ValueError(f"round_idx must be >= 0, got {r}")
    mech = str(mechanism or "").strip()
    if mech == "":
        raise ValueError("mechanism must be a non-empty string")
    acc = str(accountant or "").strip()
    if acc == "":
        raise ValueError("accountant must be a non-empty string")
    c = int(clip_l1)
    if c <= 0:
        raise ValueError(f"clip_l1 must be > 0, got {c}")
    eps_str = _decimal_str_canon_v1(epsilon, field_name="epsilon")
    del_str = _decimal_str_canon_v1(delta, field_name="delta")
    b, h = build_dp_record_v1_minimal(
        round_idx=r,
        mechanism=mech,
        clip_l1=c,
        epsilon=str(eps_str),
        delta=str(del_str),
        accountant=acc,
        extra=str(extra or ""),
    )
    # 1) Enforce canonicality (bytes are the source of truth)
    _assert_canon_json_bytes_roundtrip_v1(b, field_name="dp_record_bytes")
    # 2) Enforce hash correctness (returned h must match actual bytes)
    h2 = sha256_hex(b)
    if str(h).strip().lower() != str(h2).strip().lower():
        raise ValueError(
            "DP record hash mismatch: build_dp_record_v1_minimal returned a hash "
            "that does not equal sha256_hex(dp_record_bytes)"
        )
    assert_sha256_hex_str_v1(h2, allow_empty=False, field_name="dp_round_json_sha256")
    return b, h2
def _atomic_write_bytes_v1(path: str, data: bytes) -> None:
    """
    Atomic write (Ray/multiprocess safe):
    write to a temp file in the same directory, then os.replace().
    """
    if not isinstance(path, str) or path.strip() == "":
        raise ValueError("path must be a non-empty string")
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        raise ValueError("data must be non-empty bytes")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}.{os.urandom(6).hex()}")
    with open(tmp, "wb") as f:
        f.write(bytes(data))
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))
def write_dp_record_bytes_v1(out_path: str, *, dp_record_bytes: bytes) -> None:
    """
    Write canonical DP record bytes to disk exactly as hashed.
    Never re-dump with indent or different key ordering.
    Atomic write prevents partial files under Ray/multiprocess.
    """
    if not isinstance(out_path, str) or out_path.strip() == "":
        raise ValueError("out_path must be a non-empty string")
    # Enforce canonicality before writing (prevents “write garbage then hash later” drift)
    _assert_canon_json_bytes_roundtrip_v1(dp_record_bytes, field_name="dp_record_bytes")
    _atomic_write_bytes_v1(out_path, dp_record_bytes)
def build_and_write_dp_round_record_and_sha256_v1(
    out_path: str,
    *,
    round_idx: int,
    mechanism: str,
    clip_l1: int,
    epsilon: Any,
    delta: Any = "0",
    accountant: str = "RDP",
    extra: str = "",
) -> str:
    """
    Vehicle-side convenience:
    - build canonical DP bytes + dp_round_json_sha256
    - write bytes exactly as hashed
    - return dp_round_json_sha256 (the value that must go into SSI)
    """
    b, h = build_dp_round_record_and_sha256_v1(
        round_idx=round_idx,
        mechanism=mechanism,
        clip_l1=clip_l1,
        epsilon=epsilon,
        delta=delta,
        accountant=accountant,
        extra=extra,
    )
    write_dp_record_bytes_v1(out_path, dp_record_bytes=b)
    return h
def dp_record_sha256_from_bytes_v1(dp_record_bytes: bytes) -> str:
    """
    Convenience: validate canonical DP bytes, then compute SHA-256 hex.
    """
    _assert_canon_json_bytes_roundtrip_v1(dp_record_bytes, field_name="dp_record_bytes")
    h = sha256_hex(dp_record_bytes)
    assert_sha256_hex_str_v1(h, allow_empty=False, field_name="dp_round_json_sha256")
    return h
def metrics_to_canon_json_v1(metrics: Dict[str, float], *, scale: int = 1_000_000) -> Dict[str, int]:
    """
    Convert float metrics to scaled integers so they can safely enter canonical JSON.
    Example: accuracy=0.912345 -> 912345 with scale=1_000_000.
    """
    out: Dict[str, int] = {}
    for k, v in metrics.items():
        try:
            d = Decimal(str(float(v)))
            if d.is_nan() or d.is_infinite():
                out[str(k)] = 0
            else:
                out[str(k)] = int(d * Decimal(int(scale)))
        except Exception:
            out[str(k)] = 0
    return out
def _safe_get_tree_nums(xgb_model_org: bytes) -> tuple[int, int]:
    """
    Replacement for flwr.server.strategy.fedxgb_bagging._get_tree_nums.
    In some DP-XGBoost builds, `gbtree_model_param` does not contain
    `num_parallel_tree`. Flower's aggregation assumes it exists, so we
    default it to 1 when missing and fall back to counting the number
    of trees if `num_trees` is also missing.
    """
    # Flower passes raw JSON bytes into _get_tree_nums
    xgb_model = _json.loads(bytearray(xgb_model_org))
    gbtree_model = xgb_model["learner"]["gradient_booster"]["model"]
    gbparams = gbtree_model["gbtree_model_param"]
    # Number of trees
    try:
        tree_num = int(gbparams["num_trees"])
    except KeyError:
        # Fallback: infer from the number of tree objects
        tree_num = len(gbtree_model.get("trees", []))
    # Number of parallel trees; dp_xgboost may omit this
    paral_tree_num = int(gbparams.get("num_parallel_tree", 1))
    return tree_num, paral_tree_num
def _sha256_to_nontrivial_field_v1(msg: str, prime: int) -> int:
    v = int.from_bytes(hashlib.sha256(msg.encode("utf-8")).digest(), "big") % int(prime)
    if v in (0, 1):
        v = 2
    return v
def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")
def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))
def _did_from_pubkey_raw_ed25519(pub_raw: bytes) -> str:
    # Simple, stable DID form for simulation evidence (still cryptographically bound)
    return "did:key:ed25519:" + pub_raw.hex()
def _assert_sha256_hex_maybe_empty_v1(x: object, *, field_name: str) -> str:
    """
    Accept empty string (""), otherwise require strict 64-hex SHA-256.
    Returns normalized lowercase hex or "".
    """
    s = str(x or "").strip().lower()
    if s == "":
        return ""
    assert_sha256_hex_str_v1(s, allow_empty=False, field_name=field_name)
    return s
def _ssi_vehicle_report_bytes_v1(
    rsu_id: int,
    vehicle_id: int,
    round_idx: int,
    model_delta_sha256: str,
    dp_round_json_sha256: str,
    q_anchor_sha256: str,
) -> bytes:
    # ✅ Strict hash validation (on-chain anchor safety)
    model_delta_sha256_norm = _assert_sha256_hex_maybe_empty_v1(
        model_delta_sha256, field_name="model_delta_sha256"
    )
    dp_round_json_sha256_norm = _assert_sha256_hex_maybe_empty_v1(
        dp_round_json_sha256, field_name="dp_round_json_sha256"
    )
    q_anchor_sha256_norm = _assert_sha256_hex_maybe_empty_v1(
        q_anchor_sha256, field_name="q_anchor_sha256"
    )
    # If DP evidence is required by the protocol, don't allow empty
    if dp_round_json_sha256_norm == "":
        raise ValueError("dp_round_json_sha256 must be non-empty for VehicleSSIReportV1")
    report = {
        "schema": "VehicleSSIReportV1",
        "v": 1,
        "rsu_id": int(rsu_id),
        "vehicle_id": int(vehicle_id),
        "round": int(round_idx),
        "model_delta_sha256": str(model_delta_sha256_norm),
        "dp_round_json_sha256": str(dp_round_json_sha256_norm),
        "q_anchor_sha256": str(q_anchor_sha256_norm),
    }
    return canon.canon_json_bytes_v1(report)
def _ssi_verify_vehicle_report_sig_v1(
    report_bytes: bytes,
    pubkey_b64: str,
    sig_b64: str,
) -> bool:
    pub_raw = _b64d(str(pubkey_b64))
    sig_raw = _b64d(str(sig_b64))
    pk = Ed25519PublicKey.from_public_bytes(pub_raw)
    pk.verify(sig_raw, report_bytes)
    return True
def prove_verify_rsu_anchor_sum_from_meta_compat(**kwargs):
    fn = getattr(azkp, "prove_verify_rsu_anchor_sum_from_meta", None)
    if fn is None:
        raise AttributeError("azkp.prove_verify_rsu_anchor_sum_from_meta not found")
    sig = inspect.signature(fn)
    # Drop unsupported kwargs (version tolerant)
    required = {"root_poseidon_field"}
    missing = [k for k in required if k not in sig.parameters]
    if missing:
        raise TypeError(f"azkp.prove_verify_rsu_anchor_sum_from_meta missing required params: {missing}")
    for k in tuple(kwargs):
        if k not in sig.parameters:
            kwargs.pop(k, None)
    return fn(**kwargs)
def weighted_average_eval_metrics(
    results: List[Tuple[int, Dict[str, float]]],
) -> Dict[str, float]:
    """Aggregate evaluation metrics from multiple vehicles by weighted average."""
    if not results:
        return {}
    total_examples = sum(num_examples for num_examples, _ in results)
    if total_examples == 0:
        return {}
    agg: Dict[str, float] = {}
    for num_examples, metrics in results:
        for k, v in metrics.items():
            # Only aggregate numeric metrics (skip strings like confusion_matrix)
            if isinstance(v, (int, float)):
                agg[k] = agg.get(k, 0.0) + float(v) * num_examples
    return {k: v / total_examples for k, v in agg.items()}
def _looks_like_sha256_hex(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if len(s) != 64:
        return False
    try:
        int(s, 16)
        return True
    except Exception:
        return False
def _extract_public_inputs_v15(proof_obj: Any) -> List[str]:
    """
    Best-effort extraction of public inputs from different artifact shapes.
    Returns a list[str]. Empty list means “not found”.
    """
    if isinstance(proof_obj, dict):
        # Common direct shapes
        x = (
            proof_obj.get("public_inputs")
            or proof_obj.get("publicSignals")
            or proof_obj.get("public_signals")
            or proof_obj.get("public")
            or []
        )
        # Some wrappers: {"public": {...}}
        if isinstance(x, dict):
            x = (
                x.get("public_inputs")
                or x.get("publicSignals")
                or x.get("public_signals")
                or x.get("public")
                or []
            )
        if isinstance(x, (str, int)):
            return [str(x)]
        if isinstance(x, list):
            return [str(v) for v in x]
        return []
    # Non-dict: nothing we can do
    return []
def _write_public_sidecar_v15(sidecar_path: str, public_inputs: List[str], *, who: str, round_idx: int) -> None:
    os.makedirs(os.path.dirname(sidecar_path), exist_ok=True)
    payload = {
        "schema": "AnchorZKPPublicSidecarV1",
        "who": str(who),
        "round": int(round_idx),
        "public_inputs": [str(x) for x in public_inputs],
        "num_public_inputs": int(len(public_inputs)),
    }
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
def _make_dmatrix_version_safe(
    X: np.ndarray,
    y: np.ndarray,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
) -> "xgb.DMatrix":
    """
    dp_xgboost patches xgb.DMatrix to accept feature_min/feature_max.
    Vanilla XGBoost may not accept these kwargs -> TypeError.
    """
    try:
        return xgb.DMatrix(X, label=y, feature_min=feature_min, feature_max=feature_max)
    except TypeError:
        return xgb.DMatrix(X, label=y)
def evaluate_saved_rsu_model_centralized(
    rsu_id: int,
    output_dir: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
) -> Dict[str, float]:
    """
    Load the saved RSU-global XGBoost model and evaluate it centrally
    on the IoV test split, using an F1-optimal threshold calibrated on VAL.
    """
    model_path = os.path.join(output_dir, f"iov_global_model_rsu_{rsu_id}.json")
    if not os.path.exists(model_path):
        logging.warning(
            f"[CENTRALIZED EVAL] RSU {rsu_id}: model file not found at {model_path}"
        )
        return {}

    logging.info(f"[CENTRALIZED EVAL] RSU {rsu_id}: loading model from {model_path}")
    booster = xgb.Booster()
    booster.load_model(model_path)

    # Validation split: threshold calibration + VAL metrics at chosen threshold
    X_val_np = np.clip(X_val.values.astype(np.float32), feature_min, feature_max)
    y_val_np = y_val.values.astype(np.int32)
    dval = _make_dmatrix_version_safe(X_val_np, y_val_np, feature_min, feature_max)

    y_val_proba = np.asarray(booster.predict(dval), dtype=np.float64)
    y_val_proba = np.nan_to_num(y_val_proba, nan=0.0, posinf=1.0, neginf=0.0)
    y_val_proba = np.clip(y_val_proba, 0.0, 1.0)

    best_thr = find_optimal_threshold(
        y_val_np,
        y_val_proba,
        default_threshold=0.5,
    )

    y_val_pred = (y_val_proba >= best_thr).astype(int)
    val_metrics = compute_binary_metrics(
        y_val_np, y_val_pred, y_val_proba, calibration=True
    )

    # Test split: final metrics
    X_test_np = np.clip(X_test.values.astype(np.float32), feature_min, feature_max)
    y_test_np = y_test.values.astype(np.int32)
    dtest = _make_dmatrix_version_safe(X_test_np, y_test_np, feature_min, feature_max)

    y_proba = np.asarray(booster.predict(dtest), dtype=np.float64)
    y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)
    y_proba = np.clip(y_proba, 0.0, 1.0)

    y_pred = (y_proba >= best_thr).astype(int)
    metrics = compute_binary_metrics(
        y_test_np, y_pred, y_proba, calibration=True
    )

    cm = confusion_matrix(y_test_np, y_pred)
    log_metrics_pretty(
        f"[CENTRALIZED EVAL] RSU {rsu_id} | TEST (thr={best_thr:.4f})",
        metrics,
        cm,
    )

    metrics["threshold"] = float(best_thr)

    # Validation-set metrics used for threshold tuning (for comparison script)
    metrics["val_accuracy"] = float(val_metrics["accuracy"])
    metrics["val_precision"] = float(val_metrics["precision"])
    metrics["val_recall"] = float(val_metrics["recall"])
    metrics["val_f1"] = float(val_metrics["f1"])
    metrics["val_auc"] = float(val_metrics["auc"])
    metrics["val_logloss"] = float(val_metrics.get("logloss", 0.0))
    metrics["val_brier"] = float(val_metrics.get("brier", 0.0))
    metrics["val_ece"] = float(val_metrics.get("ece", 0.0))
    metrics["val_threshold"] = float(best_thr)

    return metrics
def _resolve_circuit_path_and_stem_v15(cfg: Any, spec: Any, *, label: str) -> tuple[Path, str]:
    """
    CircuitSpec(frozen schema) doesn't expose a stable name. Derive it from the SAME naming logic
    used by azkp.precompile_anchorsum_groth16(): write_anchorsum_circuit(...).stem.
    """
    gen_dir = Path(cfg.generated_circuits_dir)
    circuit_path_str = azkp.write_anchorsum_circuit(spec, gen_dir, overwrite=False)
    circuit_path = Path(str(circuit_path_str))
    stem = circuit_path.stem.strip()
    if not stem:
        raise RuntimeError(f"[ZKP][V15] Empty circuit stem for {label}: circuit_path={str(circuit_path)!r}")
    return circuit_path, stem
def _assert_anchor_precompile_outputs_v15(cfg: Any, spec: Any, *, label: str) -> None:
    """
    Frozen circuit contract:
      - .circom source under cfg.generated_circuits_dir
      - circom outputs + groth16 artifacts under cfg.artifacts_dir/<stem>/
    """
    gen_dir = Path(cfg.generated_circuits_dir)
    art_root = Path(cfg.artifacts_dir)
    circuit_path, stem = _resolve_circuit_path_and_stem_v15(cfg, spec, label=label)
    pre_dir = art_root / stem
    # 1) Circuit source must exist
    has_circom = circuit_path.exists() or any(gen_dir.rglob(f"{stem}.circom"))
    # 2) Circom outputs live in pre_dir
    has_r1cs = (pre_dir / f"{stem}.r1cs").exists() or (any(pre_dir.rglob("*.r1cs")) if pre_dir.exists() else False)
    has_wasm = any(pre_dir.rglob("*.wasm")) if pre_dir.exists() else False
    has_sym = (pre_dir / f"{stem}.sym").exists() or (any(pre_dir.rglob("*.sym")) if pre_dir.exists() else False)
    # 3) Groth16 artifacts live in pre_dir (or below it)
    has_zkey = any(pre_dir.rglob("*.zkey")) if pre_dir.exists() else False
    has_vkey = any(
        (p.name.endswith("verification_key.json") or p.name.endswith("vkey.json"))
        for p in (pre_dir.rglob("*.json") if pre_dir.exists() else [])
    )
    if not (has_circom and has_r1cs and has_wasm and has_sym and has_zkey and has_vkey):
        raise RuntimeError(
            f"[ZKP][V15] Precompile incomplete for {label}: "
            f"gen_dir={str(gen_dir)} pre_dir={str(pre_dir)} "
            f"(circom={has_circom}, r1cs={has_r1cs}, wasm={has_wasm}, sym={has_sym}, zkey={has_zkey}, vkey={has_vkey})"
        )
    logging.info(
        "[ZKP][V15] %s OK | stem=%s | circuit=%s | pre_dir=%s",
        label, stem, str(circuit_path), str(pre_dir)
    )
def _hard_clean_circuit_v15(cfg: Any, spec: Any, *, label: str) -> None:
    """
    Clean ONLY this circuit’s outputs.
    Do NOT wipe cfg.artifacts_dir entirely, because RSU and GLOBAL have different stems.
    """
    gen_dir = Path(cfg.generated_circuits_dir)
    art_root = Path(cfg.artifacts_dir)
    circuit_path, stem = _resolve_circuit_path_and_stem_v15(cfg, spec, label=label)
    pre_dir = art_root / stem
    # Remove only the per-circuit artifacts folder
    if pre_dir.exists():
        shutil.rmtree(pre_dir, ignore_errors=True)
    pre_dir.mkdir(parents=True, exist_ok=True)
    # Remove only the specific .circom source for this spec (keeps rebuild deterministic)
    if circuit_path.exists():
        try:
            circuit_path.unlink()
        except OSError:
            pass
    # Optional: also remove any stale duplicate named circom under gen_dir
    # (safe even if absent)
    alt = gen_dir / f"{stem}.circom"
    if alt.exists() and alt != circuit_path:
        try:
            alt.unlink()
        except OSError:
            pass
def _ensure_precompile_ok_v15(cfg: Any, spec: Any, *, label: str) -> None:
    # 1) Normal precompile first
    azkp.precompile_anchorsum_groth16(
        cfg=cfg,
        spec=spec,
        overwrite_circuit=False,
        overwrite_precompile=False,
    )
    try:
        _assert_anchor_precompile_outputs_v15(cfg, spec, label=label)
        return
    except RuntimeError as exc:
        logging.warning("[ZKP][V15] %s -> forcing rebuild (%s)", label, exc)
    # 2) Per-circuit rebuild (safe when you have RSU + GLOBAL specs)
    _hard_clean_circuit_v15(cfg, spec, label=label)
    azkp.precompile_anchorsum_groth16(
        cfg=cfg,
        spec=spec,
        overwrite_circuit=True,
        overwrite_precompile=True,
    )
    # 3) Assert again
    _assert_anchor_precompile_outputs_v15(cfg, spec, label=label)
def _resolve_ssi_fp_v15(azkp_mod: Any) -> dict[str, Any]:
    fp_fn = getattr(azkp_mod, "ssi_preimage_fingerprint_v1", None)
    if not callable(fp_fn):
        raise RuntimeError("Missing azkp.ssi_preimage_fingerprint_v1 (required for SSI fingerprint audit).")
    fp_obj: Any = fp_fn()
    try:
        fp_dict = dict(fp_obj)  # accepts Mapping or iterable of (k, v) pairs
    except Exception as exc:
        raise TypeError(f"ssi_preimage_fingerprint_v1 returned non-dict-like value: {type(fp_obj)}") from exc
    return {str(k): v for k, v in fp_dict.items()}
def _canon_json_bytes(obj: dict[str, Any]) -> bytes:
    fn = getattr(canon, "canon_json_bytes_v1", None)
    if not callable(fn):
        raise RuntimeError("Missing canon.canon_json_bytes_v1 (required for V15 canonicalization).")
    out: Any = fn(obj)
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    raise TypeError(f"canon.canon_json_bytes_v1 returned non-bytes: {type(out)}")
FIELD_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617
def _sha256_to_field(b: bytes) -> int:
    return int.from_bytes(hashlib.sha256(b).digest(), "big") % FIELD_MODULUS
def _coerce_int_field(x: object, *, name: str) -> int:
    try:
        return int(x)  # type: ignore[arg-type]
    except Exception as exc:
        raise RuntimeError(f"{name} is not int-like: {x!r}") from exc
def _resolve_policy_id_field_v1() -> int:
    fn = getattr(canon, "ssi_policy_id_field_v1", None)
    if callable(fn):
        return _coerce_int_field(fn(), name="canon.ssi_policy_id_field_v1()")
    const = getattr(canon, "SSI_POLICY_ID_FIELD_V1", None)
    if const is not None:
        return _coerce_int_field(const, name="canon.SSI_POLICY_ID_FIELD_V1")
    return _sha256_to_field(b"SSI_POLICY_ID_FIELD_V1")
def _resolve_public_input_order_id_field_v1(spec_obj: object, *, cfg_obj: object) -> int:
    """
    V15 STRICT:
    - Public-input order ID must come from the spec, or be derived ONLY from spec-provided ordered names.
    - NO .sym parsing, NO nPublic inference, NO artifact-driven reconstruction.
    - cfg_obj is kept in the signature for backward compatibility with existing call sites.
    """
    v = getattr(spec_obj, "public_input_order_id_field", None)
    if callable(v):
        return _coerce_int_field(v(), name="spec.public_input_order_id_field()")
    if v is not None:
        return _coerce_int_field(v, name="spec.public_input_order_id_field")
    names = getattr(spec_obj, "public_inputs", None)
    if names is None:
        names = getattr(spec_obj, "public_input_names", None)
    if names is None:
        names = getattr(spec_obj, "public_signals", None)
    if isinstance(names, (list, tuple)) and len(names) > 0:
        payload = {
            "v": 1,
            "public_inputs": [str(x) for x in names],
        }
        return _sha256_to_field(_canon_json_bytes(payload))
    circuit_name = str(
        getattr(spec_obj, "circuit_name", "")
        or getattr(spec_obj, "name", "")
        or getattr(spec_obj, "circuit", "")
        or ""
    ).strip()
    raise RuntimeError(
        "Spec missing public_input_order_id_field and also missing ordered public input names. "
        "Update AnchorZKP spec to provide a stable public-input ordering ID (required in V15). "
        f"(circuit_name={circuit_name!r}, spec={spec_obj!r})"
    )
def _resolve_pins_hash_field_v1(spec_obj: object, cfg_obj: object, anchor_ctx_obj: Dict[str, Any], RMAX: int,
                                NMAX: int) -> int:
    circuit_name = str(
        getattr(spec_obj, "circuit_name", "")
        or getattr(spec_obj, "name", "")
        or getattr(spec_obj, "circuit", "")
    )
    payload = {
        "v": 1,
        "selection_mode": "pubmask",
        "circuit_name": circuit_name,
        "M": int(_as_int(anchor_ctx_obj.get("M"), 0)),
        "SCALE": int(_as_int(anchor_ctx_obj.get("SCALE"), 0)),
        "RMAX": int(RMAX),
        "NMAX": int(NMAX),
        "anchor_version": str(anchor_ctx_obj.get("anchor_version", "")),
        "anchor_id_field": str(anchor_ctx_obj.get("anchor_id_field", "")),
        "anchor_root_poseidon_field": int(_as_int(anchor_ctx_obj.get("anchor_root_poseidon_field"), 0)),
        "generated_circuits_dir": str(getattr(cfg_obj, "generated_circuits_dir", "")),
        "artifacts_dir": str(getattr(cfg_obj, "artifacts_dir", "")),
    }
    return _sha256_to_field(_canon_json_bytes(payload))
def _get_round_keyed(d: object, rnd: int, default: object) -> object:
    """Return d[rnd] or d[str(rnd)] when d is a dict; else default."""
    if not isinstance(d, dict):
        return default
    if rnd in d:
        return d[rnd]
    if str(rnd) in d:
        return d[str(rnd)]
    return default
def _read_int_field_from(obj: dict, key: str) -> int:
    v = obj.get(key, None)
    if v is None:
        return 0
    s = str(v).strip()
    if not s:
        return 0
    try:
        return int(s, 0)
    except Exception:
        return int(s)
def _norm_sha256_hex_v15(x: object) -> str:
    return str(x or "").strip().lower()
def _parse_intish_v15(x: object) -> int:
    s = str(x or "").strip()
    if not s:
        return 0
    try:
        return int(s, 0)
    except Exception:
        return int(s)
def _extract_ssi_def_fp_v15(art_obj: dict, art_meta_obj: dict) -> tuple[str, int]:
    fp1 = art_obj.get("ssi_preimage_fingerprint_v1")
    fp1 = fp1 if isinstance(fp1, dict) else {}
    fp2 = art_meta_obj.get("ssi_preimage_fingerprint_v1")
    fp2 = fp2 if isinstance(fp2, dict) else {}
    sha_raw = (
            art_obj.get("ssi_preimage_def_sha256_v1")
            or art_meta_obj.get("ssi_preimage_def_sha256_v1")
            or fp1.get("ssi_preimage_def_sha256_v1")
            or fp2.get("ssi_preimage_def_sha256_v1")
            or ""
    )
    field_raw = (
            art_obj.get("ssi_preimage_def_field_bn254_v1")
            or art_meta_obj.get("ssi_preimage_def_field_bn254_v1")
            or fp1.get("ssi_preimage_def_field_bn254_v1")
            or fp2.get("ssi_preimage_def_field_bn254_v1")
            or 0
    )
    return _norm_sha256_hex_v15(sha_raw), _parse_intish_v15(field_raw)