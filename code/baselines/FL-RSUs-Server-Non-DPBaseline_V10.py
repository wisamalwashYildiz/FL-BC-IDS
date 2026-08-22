#!/usr/bin/env python3
"""
Publication/reproducibility version of the matched non-DP hierarchical
FL baseline used with FL-BC-IDS.

Scientific role
---------------
This is the matched non-DP ablation: the hierarchical Flower/XGBoost workflow,
data splits, topology, local-round schedule, thresholding procedure, and learner
hyperparameters follow the Round-2 matched reference, while the DP-specific tree
mechanism/accounting is disabled.

Default compact configuration:
- 2 RSUs
- 2 vehicles per RSU
- 2 federated rounds
- 10 local boosting rounds per participating vehicle
- seed 42

The exact historical Round-2 multi-seed execution source/results remain retained
under experiments/02_multiseed/ in the reproducibility archive.  This copy is a
portable, reviewer-facing implementation with stricter failure handling and no
test-set use during federated training.
"""
import json
import logging
import os
import time
import warnings
import tempfile
import ast  # <-- NEW: to safely parse str(history)
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb  # Standard (non-DP) XGBoost

import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedXgbBagging
from flwr.server import ServerAppComponents, ServerConfig
# App-based runtime imports (Flower 1.23.0)
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,  # <-- NEW: for threshold tuning
    brier_score_loss,        # calibration quality
    log_loss,
)

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
    """Compute binary IDS metrics.

    The matched non-DP learner uses ``reg:squarederror`` so predictions can
    occasionally fall slightly outside [0, 1].  Probabilities are therefore
    normalized to finite values and clipped before probabilistic metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)
    y_proba = np.clip(y_proba, 1e-7, 1.0 - 1e-7)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = (
        roc_auc_score(y_true, y_proba)
        if len(np.unique(y_true)) > 1
        else 0.0
    )

    metrics: Dict[str, float] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
    }

    if calibration:
        metrics["brier"] = float(brier_score_loss(y_true, y_proba))
        metrics["ece"] = float(compute_ece(y_true, y_proba))
        metrics["logloss"] = float(
            log_loss(y_true, y_proba, labels=[0, 1])
        )

    return metrics

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

    # If only one class is present, threshold doesn't matter.
    if len(np.unique(y_true)) < 2:
        return default_threshold

    y_proba = np.asarray(y_proba, dtype=np.float64)
    y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)
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
    logging.info(msg)
    if cm is not None:
        logging.info(f"{prefix} - Confusion matrix:\n{cm}")


def load_iov_splits(cfg: IoVDatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed IoV IDS splits (train/val/test) from the selected dataset."""
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
    """Extract the shared numeric feature schema in TRAIN column order.

    Using TRAIN order avoids hash/set-order dependence while retaining exactly
    the columns that are numeric and present in all three splits.
    """
    train_numeric = [
        c
        for c in df_train.select_dtypes(include=["number"]).columns
        if c != label_col
    ]
    val_numeric = set(df_val.select_dtypes(include=["number"]).columns)
    test_numeric = set(df_test.select_dtypes(include=["number"]).columns)

    numeric_cols = [
        c for c in train_numeric
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

    - Try to ensure each vehicle gets at least k_min_pos positives and k_min_neg
      negatives (if the global label counts allow it).
    - If there are not enough positives/negatives to satisfy that for all
      vehicles, distribute them as evenly as possible and log any one-class shards.
    - Returns: List[rsu_id] -> List[(X_vehicle, y_vehicle)].
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
        "Global train label counts: total=%d, pos=%d, neg=%d, vehicles=%d",
        n,
        n_pos_total,
        n_neg_total,
        total_vehicles,
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
        for rsu_id in range(num_rsus):
            rsu_vehicles: List[Tuple[pd.DataFrame, pd.Series]] = []
            for local_idx in range(vehicles_per_rsu):
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

    def _allocate_counts(total: int, total_vehicles: int, k_min: int) -> List[int]:
        """
        Allocate `total` items across `total_vehicles` buckets.

        - If total >= k_min * total_vehicles, give at least k_min to each
          vehicle, then distribute the remainder as evenly as possible.
        - Otherwise, give 1 item to as many vehicles as possible, others get 0.
        """
        if total <= 0:
            return [0] * total_vehicles

        # Enough to satisfy k_min everywhere
        if total >= k_min * total_vehicles:
            counts = [k_min] * total_vehicles
            remaining = total - k_min * total_vehicles
            extra_base = remaining // total_vehicles
            extra_rem = remaining % total_vehicles

            for i in range(total_vehicles):
                counts[i] += extra_base
            for i in range(extra_rem):
                counts[i] += 1
        else:
            # Not enough total to give k_min to everyone: spread as evenly as possible
            counts = [0] * total_vehicles
            for i in range(total):
                counts[i] += 1

        # Sanity check
        if sum(counts) != total:
            raise RuntimeError(
                f"_allocate_counts internal error: sum(counts)={sum(counts)} != total={total}"
            )
        return counts

    # How many positives/negatives per vehicle?
    pos_counts = _allocate_counts(n_pos_total, total_vehicles, k_min_pos)
    neg_counts = _allocate_counts(n_neg_total, total_vehicles, k_min_neg)

    pos_cursor = 0
    neg_cursor = 0
    vehicle_indices: List[np.ndarray] = []

    for v in range(total_vehicles):
        n_pos_v = pos_counts[v]
        n_neg_v = neg_counts[v]

        idx_pos_v = pos_indices_all[pos_cursor : pos_cursor + n_pos_v]
        idx_neg_v = neg_indices_all[neg_cursor : neg_cursor + n_neg_v]
        pos_cursor += n_pos_v
        neg_cursor += n_neg_v

        combined = np.concatenate([idx_pos_v, idx_neg_v])
        rng.shuffle(combined)
        vehicle_indices.append(combined)

    # Final sanity checks
    if pos_cursor != n_pos_total or neg_cursor != n_neg_total:
        logging.warning(
            "partition_train_among_rsus_and_vehicles: cursor mismatch "
            "(pos_cursor=%d/%d, neg_cursor=%d/%d)",
            pos_cursor,
            n_pos_total,
            neg_cursor,
            n_neg_total,
        )

    # Build RSU -> vehicles structure
    rsu_partitions: List[List[Tuple[pd.DataFrame, pd.Series]]] = []
    vehicle_linear = 0
    for rsu_id in range(num_rsus):
        rsu_vehicles: List[Tuple[pd.DataFrame, pd.Series]] = []
        for local_idx in range(vehicles_per_rsu):
            idx = vehicle_indices[vehicle_linear]
            vehicle_linear += 1

            X_v = X_train.iloc[idx].reset_index(drop=True)
            y_v = y_train.iloc[idx].reset_index(drop=True)

            # Sanity check: does this vehicle see both BENIGN and ATTACK?
            unique_labels = sorted(y_v.unique().tolist())
            if len(unique_labels) < 2:
                logging.info(
                    "RSU %d, vehicle %d: local shard has only labels %s "
                    "(pos=%d, neg=%d).",
                    rsu_id + 1,
                    local_idx + 1,
                    unique_labels,
                    int((y_v == 1).sum()),
                    int((y_v == 0).sum()),
                )

            rsu_vehicles.append((X_v, y_v))
        rsu_partitions.append(rsu_vehicles)

    logging.info(
        "Stratified partitioned train data into %d RSUs x %d vehicles "
        "(total vehicles = %d).",
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
) -> xgb.Booster:
    """
    Create a tiny XGBoost booster with the correct feature dimensionality.

    Used to bootstrap a valid Booster for FedXgbBagging.
    """
    if feature_count <= 0:
        raise ValueError("Feature count must be positive.")

    dummy_data = np.zeros((1, feature_count), dtype=np.float32)
    dummy_labels = np.array([0], dtype=np.int32)

    dtrain = xgb.DMatrix(
        dummy_data,
        label=dummy_labels,
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
    import json as _json

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

# ---------------------------------------------------------------------------
# Flower Vehicle Client
# ---------------------------------------------------------------------------

class VehicleClient(fl.client.Client):
    """
    Vehicle acting as a Flower client in IoV.

    Each vehicle:
      - trains on its own local traffic flows (X_train_vehicle, y_train_vehicle),
      - evaluates on RSU-wide validation and test splits,
      - exchanges XGBoost models with the RSU (Flower server).
    """

    def __init__(
            self,
            vehicle_id: int,
            rsu_id: int,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            xgb_cfg: XGBoostConfig,
            feature_count: int,
    ):
        self.vehicle_id = vehicle_id
        self.rsu_id = rsu_id
        self.xgb_cfg = xgb_cfg
        self.feature_count = feature_count

        # FedXgbBagging uses num_parallel_tree to interpret tree groups
        self.num_parallel_tree = int(self.xgb_cfg.params.get("num_parallel_tree", 1))
        if self.num_parallel_tree < 1:
            self.num_parallel_tree = 1

        # Store raw arrays (picklable); DMatrix lives only inside the worker
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        # Create DMatrix objects locally in the worker process.
        self.train_dmatrix = xgb.DMatrix(
            self.X_train,
            label=self.y_train,
        )
        self.val_dmatrix = xgb.DMatrix(
            self.X_val,
            label=self.y_val,
        )
        self.test_dmatrix = xgb.DMatrix(
            self.X_test,
            label=self.y_test,
        )

        # Initialize a minimal booster (safe, created in worker)
        self.model = initialize_booster_with_dummy_data(
            self.xgb_cfg.params,
            feature_count=self.feature_count,
        )

        self.logger = logging.getLogger(f"RSU{self.rsu_id}-Vehicle{self.vehicle_id}")
        self.logger.info(
            f"Initialized Vehicle {self.vehicle_id} on RSU {self.rsu_id} "
            f"with {self.train_dmatrix.num_row()} local samples"
        )

    # ----- Parameter exchange -----

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Send current XGBoost model encoded as JSON bytes.

        If self.model is None (e.g., if we later start from scratch for dp_xgboost),
        we send empty parameters and let the strategy handle it.
        """
        try:
            if self.model is None:
                flwr_params = Parameters(tensor_type="xgboost_json", tensors=[])
            else:
                # Serialize the Booster to JSON bytes compatible with FedXgbBagging
                model_bytes = booster_to_json_bytes(self.model)
                flwr_params = Parameters(
                    tensor_type="xgboost_json",
                    tensors=[model_bytes],
                )

            return GetParametersRes(
                status=Status(
                    code=Code.OK,
                    message=f"Vehicle {self.vehicle_id} sending initial parameters",
                ),
                parameters=flwr_params,
            )
        except Exception:
            self.logger.exception("get_parameters failed")
            raise

    def set_parameters(self, parameters: Parameters) -> None:
        """Load global XGBoost model from JSON bytes, if provided."""
        try:
            if not parameters.tensors:
                # First round or empty parameters: keep local dummy model
                return

            model_bytes = parameters.tensors[0]

            # Interpret these bytes as JSON-encoded XGBoost model
            booster = booster_from_json_bytes(model_bytes)
            self.model = booster
            self.logger.debug(
                f"Vehicle {self.vehicle_id}: Loaded global model from RSU"
            )
        except Exception:
            self.logger.exception("set_parameters failed")
            raise

    # ----- Local training -----

    def fit(self, ins: FitIns) -> FitRes:
        """Train locally on vehicle's own traffic flows."""
        try:
            round_num = int(ins.config.get("round", 0))
            self.logger.info(
                f"Vehicle {self.vehicle_id} on RSU {self.rsu_id}: "
                f"starting local training for round {round_num}"
            )

            # Approximate communication from RSU -> vehicle in this round
            download_bytes = 0
            try:
                if ins.parameters and ins.parameters.tensors:
                    for t in ins.parameters.tensors:
                        if isinstance(t, (bytes, bytearray)):
                            download_bytes += len(t)
            except Exception:
                download_bytes = 0

            # Load current global model if present
            self.set_parameters(ins.parameters)

            # Remember how many boosting rounds the incoming global model has.
            # We will use this to slice out only the newly grown trees.
            try:
                prev_num_rounds = (
                    self.model.num_boosted_rounds() if self.model is not None else 0
                )
            except Exception:
                prev_num_rounds = 0

            start_time = time.time()
            evals = [(self.train_dmatrix, "train"), (self.val_dmatrix, "val")]

            # Decide early-stopping rounds: use None to disable
            xgb_es_rounds = (
                self.xgb_cfg.early_stopping_rounds
                if self.xgb_cfg.early_stopping_rounds is not None
                   and self.xgb_cfg.early_stopping_rounds > 0
                else None
            )

            # Continue boosting from the current global model
            self.model = xgb.train(
                self.xgb_cfg.params,
                self.train_dmatrix,
                num_boost_round=self.xgb_cfg.num_local_rounds,
                xgb_model=self.model,
                evals=evals,
                early_stopping_rounds=xgb_es_rounds,
            )

            train_time = time.time() - start_time
            self.logger.info(
                "[RSU %s | Vehicle %s | Round %s] Local XGBoost training time = %.3f sec",
                self.rsu_id,
                self.vehicle_id,
                round_num,
                train_time,
            )

            # Compute how many new rounds/trees were added in this call
            try:
                total_rounds = self.model.num_boosted_rounds()
            except Exception:
                total_rounds = prev_num_rounds
            delta_rounds = max(0, total_rounds - prev_num_rounds)
            num_new_trees = delta_rounds * self.num_parallel_tree

            self.logger.info(
                "[RSU %s | Vehicle %s | Round %s] "
                "global rounds before=%d, after=%d, newly_added_rounds=%d "
                "(≈%d new trees with num_parallel_tree=%d)",
                self.rsu_id,
                self.vehicle_id,
                round_num,
                prev_num_rounds,
                total_rounds,
                delta_rounds,
                num_new_trees,
                self.num_parallel_tree,
            )

            # Build a delta booster containing only the newly added trees
            delta_booster = make_delta_booster(
                booster=self.model,
                prev_num_rounds=prev_num_rounds,
            )

            # Early stopping diagnostics
            # ---------------------------
            es_enabled = (
                    self.xgb_cfg.early_stopping_rounds is not None
                    and self.xgb_cfg.early_stopping_rounds > 0
            )
            metric_name = self.xgb_cfg.params.get("eval_metric", "auc")

            best_iteration = getattr(self.model, "best_iteration", None)
            best_score_raw = getattr(self.model, "best_score", None)
            try:
                best_score = float(best_score_raw) if best_score_raw is not None else -1.0
            except (TypeError, ValueError):
                best_score = -1.0

            stopped_early = bool(
                es_enabled
                and delta_rounds < self.xgb_cfg.num_local_rounds
            )
            reason = "Early stopping disabled"
            if es_enabled:
                if stopped_early:
                    reason = (
                        f"Local training added {delta_rounds}/"
                        f"{self.xgb_cfg.num_local_rounds} configured rounds; "
                        f"validation early stopping was active for "
                        f"eval_metric '{metric_name}'."
                    )
                else:
                    reason = (
                        "Reached configured local boosting-round limit without "
                        "an early-stop reduction."
                    )

            if es_enabled:
                if stopped_early:
                    self.logger.info(
                        f"[RSU {self.rsu_id} | Vehicle {self.vehicle_id} | Round {round_num}] "
                        f"Early stopping TRIGGERED after {delta_rounds}/"
                        f"{self.xgb_cfg.num_local_rounds} local rounds, "
                        f"best {metric_name}={best_score:.6f}"
                    )
                else:
                    self.logger.info(
                        f"[RSU {self.rsu_id} | Vehicle {self.vehicle_id} | Round {round_num}] "
                        f"Early stopping ENABLED; local training added all "
                        f"{self.xgb_cfg.num_local_rounds} configured rounds, "
                        f"best {metric_name}={best_score:.6f}"
                    )
            else:
                self.logger.info(
                    f"[RSU {self.rsu_id} | Vehicle {self.vehicle_id} | Round {round_num}] "
                    "Early stopping DISABLED"
                )

            # Training metrics on vehicle's own traffic
            y_train_true: np.ndarray = np.asarray(
                self.train_dmatrix.get_label(), dtype=np.int32
            )
            y_train_proba: np.ndarray = np.asarray(
                self.model.predict(self.train_dmatrix), dtype=np.float32
            )
            y_train_pred: np.ndarray = (y_train_proba >= 0.5).astype(int)
            train_metrics = compute_binary_metrics(
                y_train_true, y_train_pred, y_train_proba
            )

            # Validation metrics (RSU-wide validation)
            y_val_true: np.ndarray = np.asarray(
                self.val_dmatrix.get_label(), dtype=np.int32
            )
            y_val_proba: np.ndarray = np.asarray(
                self.model.predict(self.val_dmatrix), dtype=np.float32
            )
            y_val_pred: np.ndarray = (y_val_proba >= 0.5).astype(int)
            val_metrics = compute_binary_metrics(
                y_val_true, y_val_pred, y_val_proba, calibration=True
            )

            cm_train = confusion_matrix(y_train_true, y_train_pred)
            cm_val = confusion_matrix(y_val_true, y_val_pred)

            log_metrics_pretty(
                f"[RSU {self.rsu_id} | Vehicle {self.vehicle_id} | Round {round_num}] TRAIN",
                train_metrics,
                cm_train,
            )
            log_metrics_pretty(
                f"[RSU {self.rsu_id} | Vehicle {self.vehicle_id} | Round {round_num}] VAL",
                val_metrics,
                cm_val,
            )

            # Serialize only the delta booster as JSON bytes for FedXgbBagging
            model_bytes = booster_to_json_bytes(delta_booster)
            flwr_params = Parameters(
                tensor_type="xgboost_json",
                tensors=[model_bytes],
            )

            # Approximate communication from vehicle -> RSU in this round
            if isinstance(model_bytes, (bytes, bytearray)):
                upload_bytes = float(len(model_bytes))
            else:
                upload_bytes = 0.0

            total_bytes = float(upload_bytes) + float(download_bytes)

            metrics = {
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "train_auc": train_metrics["auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
                # Validation calibration quality (per vehicle)
                "val_brier": float(val_metrics.get("brier", 0.0)),
                "val_ece": float(val_metrics.get("ece", 0.0)),
                "train_time_sec": float(train_time),
                "num_train_examples": int(len(y_train_true)),
                "num_new_trees": int(num_new_trees),
                # Communication metrics (per vehicle / round)
                "upload_bytes": float(upload_bytes),
                "download_bytes": float(download_bytes),
                "total_bytes": float(total_bytes),
                # Early stopping info exported back to the strategy
                "early_stopping_enabled": bool(es_enabled),
                "early_stopping_rounds": int(self.xgb_cfg.early_stopping_rounds or 0),
                "best_iteration": int(best_iteration) if best_iteration is not None else -1,
                "best_score": float(best_score),
                "stopped_early": bool(stopped_early),
                "early_stopping_reason": reason,
            }

            return FitRes(
                status=Status(code=Code.OK, message="local training complete"),
                parameters=flwr_params,
                num_examples=len(y_train_true),
                metrics=metrics,
            )

        except Exception:
            self.logger.exception("fit failed")
            raise

    # ----- Local evaluation -----

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate during FL on VALIDATION only.

        The test split is deliberately reserved for final post-training RSU and
        GLOBAL evaluation, preventing test-set feedback into the FL loop.
        """
        try:
            self.set_parameters(ins.parameters)

            y_val_true = np.asarray(
                self.val_dmatrix.get_label(), dtype=np.int32
            )
            y_val_proba = np.asarray(
                self.model.predict(self.val_dmatrix), dtype=np.float64
            )
            y_val_proba = np.nan_to_num(
                y_val_proba, nan=0.0, posinf=1.0, neginf=0.0
            )
            y_val_proba = np.clip(y_val_proba, 1e-7, 1.0 - 1e-7)

            best_thr = find_optimal_threshold(
                y_val_true, y_val_proba, default_threshold=0.5
            )
            y_val_pred = (y_val_proba >= best_thr).astype(np.int32)

            val_metrics = compute_binary_metrics(
                y_val_true,
                y_val_pred,
                y_val_proba,
                calibration=True,
            )
            cm_val = confusion_matrix(y_val_true, y_val_pred)
            val_logloss = float(val_metrics["logloss"])

            log_metrics_pretty(
                f"[RSU {self.rsu_id} | Vehicle {self.vehicle_id}] "
                f"VAL (thr={best_thr:.4f}, logloss={val_logloss:.6f})",
                val_metrics,
                cm_val,
            )

            return EvaluateRes(
                status=Status(code=Code.OK, message="validation complete"),
                loss=val_logloss,
                num_examples=len(y_val_true),
                metrics={
                    "accuracy": val_metrics["accuracy"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                    "f1": val_metrics["f1"],
                    "auc": val_metrics["auc"],
                    "brier": val_metrics["brier"],
                    "ece": val_metrics["ece"],
                    "logloss": val_logloss,
                    "threshold": float(best_thr),
                    "confusion_matrix": json.dumps(cm_val.tolist()),
                },
            )
        except Exception:
            self.logger.exception("evaluate failed")
            raise


# ---------------------------------------------------------------------------
# RSU-level Strategy: FedXgbBagging (tree bagging across vehicles)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RSU-level Strategy helpers
# ---------------------------------------------------------------------------

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

class XgbBaggingStrategy(FedXgbBagging):
    """
    RSU-level strategy using Flower's FedXgbBagging:

    - Each vehicle trains an XGBoost model (we send boosters as JSON-encoded bytes).
    - FedXgbBagging concatenates the trees from all vehicles (bagging).
    - We keep track of the latest global parameters so we can save the RSU model.
    """
    def __init__(
        self,
        rsu_id: int,
        vehicles_per_rsu: int,
        num_rounds: int,
        num_local_rounds: int,
    ):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=vehicles_per_rsu,
            min_evaluate_clients=vehicles_per_rsu,
            min_available_clients=vehicles_per_rsu,
            # We still pass the round index so clients can log/use it
            on_fit_config_fn=lambda rnd: {"round": rnd},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
            # Aggregate validation metrics across vehicles by weighted average
            evaluate_metrics_aggregation_fn=weighted_average_eval_metrics,
        )
        self.rsu_id = rsu_id
        self.num_rounds = num_rounds
        self.num_local_rounds = num_local_rounds

        # Will hold the latest global model parameters
        self.latest_global_params: Parameters | None = None

        # Per-round summary for JSON dump (now based on bagging metrics)
        self.round_summaries: List[Dict[str, object]] = []

        # Track how many trees are present in the global booster
        self._global_num_trees: int = -1

        # Last good global model bytes (for saving RSU model at the end)
        self._last_global_model_bytes: bytes | None = None

        # Per-round, per-vehicle training/validation metrics for comparison script
        # Structure: {round: [ {client_id, num_examples, train_*, val_*, dp_*}, ... ]}
        self.vehicle_metrics_per_round: Dict[int, List[Dict[str, object]]] = {}

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Parameters | None, Dict[str, float]]:
        # Call FedXgbBagging's aggregation (tree-bagging logic)
        aggregated_params, agg_metrics = super().aggregate_fit(rnd, results, failures)

        if (
                aggregated_params is None
                or not aggregated_params.tensors
                or aggregated_params.tensors[0] is None
        ):
            logging.warning(
                "RSU %d - Round %d: no aggregated XGBoost parameters to inspect",
                self.rsu_id,
                rnd,
            )
        else:
            # Cache latest params and last good global model bytes
            self.latest_global_params = aggregated_params
            model_bytes = aggregated_params.tensors[0]
            self._last_global_model_bytes = model_bytes

            # Inspect the aggregated global booster to track tree growth
            try:
                booster = booster_from_json_bytes(model_bytes)
                num_trees = _count_trees_in_booster(booster)
                if num_trees >= 0:
                    if self._global_num_trees >= 0:
                        delta_trees = num_trees - self._global_num_trees
                    else:
                        delta_trees = num_trees
                    logging.info(
                        "RSU %d - Round %d: global booster trees=%d "
                        "(added this round=%d)",
                        self.rsu_id,
                        rnd,
                        num_trees,
                        delta_trees,
                    )
                    self._global_num_trees = num_trees
                else:
                    logging.info(
                        "RSU %d - Round %d: could not determine number of trees",
                        self.rsu_id,
                        rnd,
                    )
            except Exception as exc:
                logging.warning(
                    "RSU %d - Round %d: failed to inspect global booster: %s",
                    self.rsu_id,
                    rnd,
                    exc,
                )

        # Log high-level bagging metrics for this round (if any)
        if agg_metrics:
            logging.info(
                f"RSU {self.rsu_id} - Round {rnd} (bagging) metrics: {agg_metrics}"
            )

        # Capture per-vehicle metrics for this round
        vehicle_entries: List[Dict[str, object]] = []

        for client_proxy, fit_res in results:
            m = getattr(fit_res, "metrics", None) or {}

            entry: Dict[str, object] = {
                "client_id": str(getattr(client_proxy, "cid", "")),
                "num_examples": int(getattr(fit_res, "num_examples", 0)),
            }
            for key in [
                "train_accuracy",
                "train_f1",
                "train_auc",
                "val_accuracy",
                "val_f1",
                "val_auc",
                "val_brier",
                "val_ece",
                "train_time_sec",
                "num_train_examples",
                "num_new_trees",
                "upload_bytes",
                "download_bytes",
                "total_bytes",
            ]:
                if key in m:
                    val = m[key]
                    if isinstance(val, (int, float)):
                        entry[key] = float(val)
            vehicle_entries.append(entry)

        if vehicle_entries:
            self.vehicle_metrics_per_round[rnd] = vehicle_entries

        # --- Aggregate training-time and communication metrics across vehicles ---
        total_train_time = 0.0
        total_examples = 0
        num_clients = 0
        total_upload_bytes = 0.0
        total_download_bytes = 0.0

        for _, fit_res in results:
            num_clients += 1
            total_examples += getattr(fit_res, "num_examples", 0)
            m = getattr(fit_res, "metrics", None) or {}

            t_val = m.get("train_time_sec", None)
            if isinstance(t_val, (int, float)):
                total_train_time += float(t_val)

            up_val = m.get("upload_bytes", None)
            if isinstance(up_val, (int, float)):
                total_upload_bytes += float(up_val)

            dn_val = m.get("download_bytes", None)
            if isinstance(dn_val, (int, float)):
                total_download_bytes += float(dn_val)

        total_bytes = total_upload_bytes + total_download_bytes

        # Size of the (aggregated) global model this round
        global_model_size_bytes = 0.0
        if aggregated_params is not None and aggregated_params.tensors:
            first_tensor = aggregated_params.tensors[0]
            if isinstance(first_tensor, (bytes, bytearray)):
                global_model_size_bytes = float(len(first_tensor))

        fit_metrics: Dict[str, float] = {}
        fit_metrics["fit_total_train_time_sec"] = float(total_train_time)
        fit_metrics["fit_num_clients"] = float(num_clients)
        fit_metrics["fit_num_examples_total"] = float(total_examples)

        fit_metrics["fit_total_upload_bytes"] = float(total_upload_bytes)
        fit_metrics["fit_total_download_bytes"] = float(total_download_bytes)
        fit_metrics["fit_global_model_size_bytes"] = float(global_model_size_bytes)

        if num_clients > 0:
            fit_metrics["fit_avg_train_time_sec_per_client"] = (
                    total_train_time / float(num_clients)
            )
            fit_metrics["fit_avg_upload_bytes_per_client"] = (
                    total_upload_bytes / float(num_clients)
            )
            fit_metrics["fit_avg_download_bytes_per_client"] = (
                    total_download_bytes / float(num_clients)
            )
            fit_metrics["fit_avg_total_bytes_per_client"] = (
                    total_bytes / float(num_clients)
            )

        if total_examples > 0:
            fit_metrics["fit_avg_train_time_msec_per_example"] = (
                    (total_train_time * 1000.0) / float(total_examples)
            )
            fit_metrics["fit_avg_bytes_per_example"] = (
                    total_bytes / float(total_examples)
            )

        # Store JSON-friendly per-round summary
        round_summary: Dict[str, object] = {"round": rnd}
        for k, v in agg_metrics.items():
            try:
                round_summary[k] = float(v)
            except (TypeError, ValueError):
                continue

        # Include fit_* metrics so comparison code can pick them up
        for k, v in fit_metrics.items():
            round_summary[k] = v

        self.round_summaries.append(round_summary)

        return aggregated_params, agg_metrics

    def get_last_global_model_bytes(self) -> bytes | None:
        """Return the last successfully aggregated global model bytes, if any."""
        return self._last_global_model_bytes
# ---------------------------------------------------------------------------
# RSU orchestration
# ---------------------------------------------------------------------------
def run_rsu_federated_learning(
    rsu_cfg: RSUConfig,
    rsu_vehicles: List[Tuple[pd.DataFrame, pd.Series]],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    xgb_params: Dict,
) -> None:
    """
    Run a full FL process for one RSU over its vehicles.

    This version uses the Flower App-based runtime (ClientApp/ServerApp)
    and flwr.simulation.run_simulation instead of legacy start_simulation.
    """
    # NOTE: start_simulation is no longer used; we rely on run_simulation

    os.makedirs(rsu_cfg.output_dir, exist_ok=True)

    # Convert shared validation/test to NumPy arrays (picklable)
    X_val_np = X_val.values.astype(np.float32)
    y_val_np = y_val.values.astype(np.int32)
    X_test_np = X_test.values.astype(np.float32)
    y_test_np = y_test.values.astype(np.int32)

    feature_count = X_val_np.shape[1]

    # Prepare XGBoost config
    xgb_cfg = XGBoostConfig(
        params=xgb_params,
        num_local_rounds=rsu_cfg.num_local_rounds,
        early_stopping_rounds=10,  # for example; tune if you like
    )

    # Log num_parallel_tree so we know how many trees per boosting round
    num_parallel_tree = int(xgb_cfg.params.get("num_parallel_tree", 1))
    logging.info(
        "RSU %d - Using num_parallel_tree=%d",
        rsu_cfg.rsu_id,
        num_parallel_tree,
    )

    # Convert each vehicle's partition to NumPy arrays once
    vehicle_partitions: List[Tuple[np.ndarray, np.ndarray]] = []
    for (X_v, y_v) in rsu_vehicles:
        X_v_np = X_v.values.astype(np.float32)
        y_v_np = y_v.values.astype(np.int32)
        vehicle_partitions.append((X_v_np, y_v_np))

    # ---------------------------
    # Build ClientApp for this RSU
    # ---------------------------
    def client_fn(context) -> fl.client.Client:
        """
        Build a VehicleClient inside the simulation backend.

        Flower 1.23+:
          - Each Ray node has a `node_config` dict.
          - We pass a `"partition-id"` per node when starting the simulation.
          - That `"partition-id"` is what we should use to choose the vehicle partition.

        This function:
          1) Reads `partition-id` (and optional `num-partitions`) from `context.node_config`.
          2) Maps `partition-id` -> index in `vehicle_partitions`.
          3) Builds and returns a VehicleClient for that partition.
        """

        # Sanity check: we must have at least one vehicle partition
        num_parts = len(vehicle_partitions)
        if num_parts <= 0:
            raise RuntimeError("vehicle_partitions is empty, cannot build clients")

        # --- New: read partition-id from node_config instead of using context.node_id ---
        try:
            node_cfg = getattr(context, "node_config", None)
            if not isinstance(node_cfg, dict):
                raise KeyError(f"node_config is not a dict: {node_cfg!r}")

            partition_id = int(node_cfg["partition-id"])
            num_partitions = int(node_cfg.get("num-partitions", num_parts))
        except Exception as exc:
            raise RuntimeError(
                f"[RSU {rsu_cfg.rsu_id}] Failed to read partition-id from node_config: {exc}. "
                f"node_config={getattr(context, 'node_config', None)!r}"
            )

        # Bound check: partition_id must index into vehicle_partitions
        if not (0 <= partition_id < num_parts):
            raise RuntimeError(
                f"[RSU {rsu_cfg.rsu_id}] partition-id {partition_id} is outside "
                f"[0, {num_parts - 1}] for {num_parts} vehicle partitions "
                f"(num-partitions={num_partitions}, node_config={node_cfg!r})"
            )

        logging.info(
            "[RSU %d] Building client for partition-id=%d -> vehicle_partition_idx=%d/%d",
            rsu_cfg.rsu_id,
            partition_id,
            partition_id,
            num_parts,
        )

        # Select this vehicle's train split
        X_v_np, y_v_np = vehicle_partitions[partition_id]

        # Keep your global vehicle ID convention
        vehicle_id = rsu_cfg.rsu_id * 1000 + (partition_id + 1)

        # Build and return the VehicleClient
        return VehicleClient(
            vehicle_id=vehicle_id,
            rsu_id=rsu_cfg.rsu_id,
            X_train=X_v_np,
            y_train=y_v_np,
            X_val=X_val_np,
            y_val=y_val_np,
            X_test=X_test_np,
            y_test=y_test_np,
            xgb_cfg=xgb_cfg,
            feature_count=feature_count,
        )

    client_app = ClientApp(client_fn=client_fn)

    # Shared strategy instance so we can access latest_global_params after simulation
    strategy = XgbBaggingStrategy(
        rsu_id=rsu_cfg.rsu_id,
        vehicles_per_rsu=len(vehicle_partitions),
        num_rounds=rsu_cfg.num_rounds,
        num_local_rounds=rsu_cfg.num_local_rounds,
    )

    # ---------------------------
    # Build ServerApp for this RSU
    # ---------------------------
    # After adding: from flwr.server import ServerAppComponents, ServerConfig

    def server_fn(context) -> ServerAppComponents:
        # In your installed Flower version, ServerAppComponents expects `config=`
        config = ServerConfig(num_rounds=rsu_cfg.num_rounds)
        return ServerAppComponents(
            strategy=strategy,
            config=config,
        )

    server_app = ServerApp(server_fn=server_fn)

    rsu_start_time = time.time()
    logging.info(
        f"=== RSU {rsu_cfg.rsu_id}: starting FL with {len(vehicle_partitions)} vehicles, "
        f"{rsu_cfg.num_rounds} rounds, {rsu_cfg.num_local_rounds} local rounds "
        f"(App-based run_simulation) ==="
    )

    # Run the Flower App-based simulation (local backend by default)
    history = run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=len(vehicle_partitions),
    )

    rsu_runtime = time.time() - rsu_start_time
    logging.info(
        "=== RSU %d: FL run completed in %.3f sec (≈ %.2f min) ===",
        rsu_cfg.rsu_id,
        rsu_runtime,
        rsu_runtime / 60.0,
    )

    # Save RSU-global model if available, using last good global model bytes
    global_bytes = strategy.get_last_global_model_bytes()
    if not global_bytes:
        logging.error(
            "RSU %d - No global XGBoost model bytes available after FL run; "
            "cannot save RSU model",
            rsu_cfg.rsu_id,
        )
    else:
        try:
            booster = xgb.Booster()
            booster.load_model(bytearray(global_bytes))

            model_path = os.path.join(
                rsu_cfg.output_dir,
                f"iov_global_model_rsu_{rsu_cfg.rsu_id}.json",
            )
            booster.save_model(model_path)
            logging.info(
                "RSU %d - Saved RSU-global IoV model to %s",
                rsu_cfg.rsu_id,
                model_path,
            )
        except Exception as e:
            logging.error(
                "RSU %d - Failed to save global model: %s",
                rsu_cfg.rsu_id,
                e,
            )

    # ---------------------------
    # Inspect Flower History object (version-safe)
    # ---------------------------

    # Best-effort extraction that works across Flower versions and
    # with custom strategies like FedXgbBagging.
    losses_distributed, metrics_distributed_evaluate = (
        extract_history_distributed_summary(history)
    )

    logging.info(
        "RSU %d - FL losses_distributed (extracted): %s",
        rsu_cfg.rsu_id,
        losses_distributed,
    )
    logging.info(
        "RSU %d - FL metrics_distributed (evaluate, extracted): %s",
        rsu_cfg.rsu_id,
        metrics_distributed_evaluate,
    )

    # ---------------------------
    # Persist RSU-level summary as JSON
    # ---------------------------
    round_summaries = getattr(strategy, "round_summaries", [])
    final_round_summary = round_summaries[-1] if round_summaries else {}

    rsu_summary = {
        "rsu_id": rsu_cfg.rsu_id,
        "num_rsus": rsu_cfg.num_rsus,
        "vehicles_per_rsu": rsu_cfg.vehicles_per_rsu,
        "num_rounds": rsu_cfg.num_rounds,
        "num_local_rounds": rsu_cfg.num_local_rounds,
        "xgb_params": xgb_params,
        "early_stopping_rounds": xgb_cfg.early_stopping_rounds,
        # Wall-clock FL time for this RSU (used by comparison script)
        "simulation_wall_time_sec": float(rsu_runtime),
        "rsu_wall_time_sec": float(rsu_runtime),

        # These now match what Flower prints in the [SUMMARY] block
        "losses_distributed": losses_distributed,
        "metrics_distributed_evaluate": metrics_distributed_evaluate,

        # Bagging/train summaries from aggregate_fit (now include fit_* metrics)
        "round_summaries": round_summaries,
        "final_round_summary": final_round_summary,

        # Per-round, per-vehicle training/DP metrics
        "vehicle_metrics_per_round": getattr(
            strategy, "vehicle_metrics_per_round", {}
        ),

        # Optional: keep raw attributes + full string for debugging
        "flower_history_raw": {
            "losses_distributed_attr": getattr(history, "losses_distributed", []),
            "metrics_distributed_attr": getattr(history, "metrics_distributed", {}),
            "repr": str(history),
        },
    }

    summary_path = os.path.join(
        rsu_cfg.output_dir,
        f"iov_rsu_{rsu_cfg.rsu_id}_summary.json",
    )
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(rsu_summary, f, indent=2)
        logging.info(
            f"RSU {rsu_cfg.rsu_id} - Saved FL summary (including early stopping info) to {summary_path}"
        )
    except Exception as e:
        logging.error(f"RSU {rsu_cfg.rsu_id} - Failed to write summary JSON: {e}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate_saved_rsu_model_centralized(
    rsu_id: int,
    output_dir: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
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
    X_val_np = X_val.values.astype(np.float32)
    y_val_np = y_val.values.astype(np.int32)
    dval = xgb.DMatrix(X_val_np, label=y_val_np)

    y_val_proba = np.asarray(booster.predict(dval), dtype=np.float64)
    y_val_proba = np.nan_to_num(
        y_val_proba, nan=0.0, posinf=1.0, neginf=0.0
    )
    y_val_proba = np.clip(y_val_proba, 1e-7, 1.0 - 1e-7)
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
    X_test_np = X_test.values.astype(np.float32)
    y_test_np = y_test.values.astype(np.int32)
    dtest = xgb.DMatrix(X_test_np, label=y_test_np)

    y_proba = np.asarray(booster.predict(dtest), dtype=np.float64)
    y_proba = np.nan_to_num(
        y_proba, nan=0.0, posinf=1.0, neginf=0.0
    )
    y_proba = np.clip(y_proba, 1e-7, 1.0 - 1e-7)
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
    metrics["val_brier"] = float(val_metrics.get("brier", 0.0))
    metrics["val_ece"] = float(val_metrics.get("ece", 0.0))
    metrics["val_threshold"] = float(best_thr)

    return metrics

if __name__ == "__main__":
    setup_logging()

    # ---------------------------
    # IoV dataset selection
    # ---------------------------
    DATASET_NAME = "CSECICIDS2018"  # change to "CICIoV2024" when needed
    #DATASET_NAME = "CICIoV2024"

    DATASET_CONFIGS = {
        "CSECICIDS2018": {
            "preproc_dir": os.getenv("FLBCIDS_CSE_PREPROC_DIR", "data/preprocessed/CSE-CIC-IDS2018"),
            "train_csv": "CSECICIDS2018_train_preprocessed.csv",
            "val_csv": "CSECICIDS2018_val_preprocessed.csv",
            "test_csv": "CSECICIDS2018_test_preprocessed.csv",
            "label_col": "Label",
        },
        "CICIoV2024": {
            "preproc_dir": os.getenv("FLBCIDS_CICIOV_PREPROC_DIR", "data/preprocessed/CICIoV2024"),
            "train_csv": "CICIoV2024_train_preprocessed.csv",
            "val_csv": "CICIoV2024_val_preprocessed.csv",
            "test_csv": "CICIoV2024_test_preprocessed.csv",
            "label_col": "Label",
        },
    }

    if DATASET_NAME not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown DATASET_NAME='{DATASET_NAME}'. "
            f"Choose one of: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[DATASET_NAME]
    PREPROC_DIR = cfg["preproc_dir"]

    ds_cfg = IoVDatasetConfig(
        preproc_dir=PREPROC_DIR,
        train_csv=os.path.join(PREPROC_DIR, cfg["train_csv"]),
        val_csv=os.path.join(PREPROC_DIR, cfg["val_csv"]),
        test_csv=os.path.join(PREPROC_DIR, cfg["test_csv"]),
        label_col=cfg["label_col"],  # <-- IoV BENIGN vs ATTACK label
    )

    # ---------------------------
    # High-level RSU configuration
    # ---------------------------
    NUM_RSUS = 2  # number of RSUs
    VEHICLES_PER_RSU = 2  # vehicles attached to each RSU
    NUM_ROUNDS = 2  # FL rounds per RSU
    NUM_LOCAL_ROUNDS = 10  # manuscript-matched local boosting iterations per vehicle

    # Baseline (non-DP) run output directory
    OUTPUT_DIR = os.path.join(
        os.getenv("FLBCIDS_NONDP_OUTPUT_DIR", "artifacts/rsu_outputs_baseline"),
        DATASET_NAME,
    )

    # XGBoost hyperparameters for IoV IDS (baseline, non-DP)
    # Matched non-DP learner configuration used for the controlled ablation.
    # The DP-specific difference is the standard non-private hist tree method
    # and absence of DP accounting/noise parameters.
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.2,
        "tree_method": "hist",
        "min_child_weight": 500,
        "subsample": 0.2,
        "lambda": 1.0,
        "alpha": 1.0,
        "colsample_bytree": 0.9,
        "base_score": 0.5,  # updated to the training positive fraction below
        "max_delta_step": 1.0,
        "random_state": 42,
        "seed": 42,
        "num_parallel_tree": 1,
    }

    # ---------------------------
    # Persist high-level run configuration for comparison script
    # ---------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_config = {
        "dataset_name": DATASET_NAME,
        "dp_enabled": False,
        "dp_epsilon_round": 0.0,
        "dp_delta_round": 0.0,
        "dp_clip_l1": None,
        "num_rounds": NUM_ROUNDS,
        "num_local_rounds": NUM_LOCAL_ROUNDS,
        "num_rsus": NUM_RSUS,
        "vehicles_per_rsu": VEHICLES_PER_RSU,
        "tree_method": xgb_params.get("tree_method"),
        "dp_epsilon_total": 0.0,
        "experiment_seed": 42,
        "matched_ablation": True,
        "ablation_difference": (
            "DP tree mechanism/accounting disabled; learner hyperparameters "
            "otherwise matched to the principal FL-BC-IDS learner family"
        ),
        "publication_role": "matched_non_dp_hierarchical_reference",
    }

    run_cfg_path = os.path.join(OUTPUT_DIR, "iov_run_config.json")
    try:
        with open(run_cfg_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
        logging.info("Saved non-DP run configuration to %s", run_cfg_path)
    except Exception as e:
        logging.error("Failed to write non-DP run configuration JSON: %s", e)

    # ---------------------------
    # Load IoV dataset
    # ---------------------------
    df_train, df_val, df_test = load_iov_splits(ds_cfg)
    X_train, y_train, X_val, y_val, X_test, y_test = extract_numeric_features(
        df_train, df_val, df_test, label_col=ds_cfg.label_col
    )

    # ---------------------------
    # Imbalance handling (BENIGN vs ATTACK)
    # ---------------------------
    num_pos = int(y_train.sum())
    num_total = int(len(y_train))
    num_neg = num_total - num_pos

    if num_pos > 0 and num_neg > 0:
        spw = num_neg / num_pos
        pos_frac = num_pos / num_total
        logging.info(
            f"Setting XGBoost scale_pos_weight={spw:.2f} "
            f"(neg={num_neg}, pos={num_pos}), base_score≈{pos_frac:.4f}"
        )
        xgb_params["scale_pos_weight"] = spw
        xgb_params["base_score"] = float(pos_frac)
    else:
        raise ValueError(
            "Train split has only one class (pos or neg). "
        f"Check your preprocessed train file: {ds_cfg.train_csv}. "
        "scale_pos_weight cannot be computed."
        )

    # ---------------------------
    # Partition among RSUs/vehicles (balanced by label)
    # ---------------------------
    rsu_partitions = partition_train_among_rsus_and_vehicles(
        X_train,
        y_train,
        num_rsus=NUM_RSUS,
        vehicles_per_rsu=VEHICLES_PER_RSU,
        random_state=42,
        # k_min_pos / k_min_neg control how many examples of each class
        # we try to guarantee per vehicle (when global counts allow).
        k_min_pos=1,
        k_min_neg=1,
    )

    # ---------------------------
    # Partition diagnostics
    # ---------------------------
    log_partition_stats(rsu_partitions)

    # ---------------------------
    # Run FL per RSU
    # ---------------------------
    for rsu_id in range(1, NUM_RSUS + 1):
        rsu_cfg = RSUConfig(
            rsu_id=rsu_id,
            num_rsus=NUM_RSUS,
            vehicles_per_rsu=VEHICLES_PER_RSU,
            num_rounds=NUM_ROUNDS,
            num_local_rounds=NUM_LOCAL_ROUNDS,
            output_dir=OUTPUT_DIR,
        )
        rsu_vehicles = rsu_partitions[rsu_id - 1]
        run_rsu_federated_learning(
            rsu_cfg=rsu_cfg,
            rsu_vehicles=rsu_vehicles,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            xgb_params=xgb_params,
        )

    logging.info("All RSUs finished IoV federated training.")

    # ---------------------------
    # Centralized evaluation of saved RSU models
    # ---------------------------
    logging.info(
        "Starting centralized evaluation of saved RSU-global models on IoV TEST split..."
    )
    centralized_results: Dict[int, Dict[str, float]] = {}

    for rsu_id in range(1, NUM_RSUS + 1):
        metrics = evaluate_saved_rsu_model_centralized(
            rsu_id=rsu_id,
            output_dir=OUTPUT_DIR,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        if metrics:
            centralized_results[rsu_id] = metrics

    logging.info(f"[CENTRALIZED EVAL] Summary across RSUs: {centralized_results}")

    # Save centralized evaluation summary across all RSUs
    summary_all_path = os.path.join(OUTPUT_DIR, "iov_centralized_eval_summary.json")
    try:
        with open(summary_all_path, "w", encoding="utf-8") as f:
            json.dump(centralized_results, f, indent=2)
        logging.info(
            f"[CENTRALIZED EVAL] Saved summary across RSUs to {summary_all_path}"
        )
    except Exception as e:
        logging.error(f"[CENTRALIZED EVAL] Failed to write summary JSON: {e}")

    # ---------------------------
    # Global server ensemble aggregation across RSU models
    # ---------------------------
    logging.info(
        "[GLOBAL] Computing ensemble over all RSU-global models on IoV TEST split..."
    )

    # Reuse the same numeric feature subset/order as during training
    X_val_np = X_val.values.astype(np.float32)
    y_val_np = y_val.values.astype(np.int32)
    dval = xgb.DMatrix(X_val_np, label=y_val_np)

    X_test_np = X_test.values.astype(np.float32)
    y_test_np = y_test.values.astype(np.int32)
    dtest = xgb.DMatrix(X_test_np, label=y_test_np)

    sum_val_proba: np.ndarray | None = None
    sum_test_proba: np.ndarray | None = None
    used_rsus: List[int] = []

    # We already have centralized_results from the previous block
    per_rsu_test_metrics: Dict[int, Dict[str, float]] = centralized_results.copy()

    for rsu_id in range(1, NUM_RSUS + 1):
        model_path = os.path.join(OUTPUT_DIR, f"iov_global_model_rsu_{rsu_id}.json")
        if not os.path.exists(model_path):
            logging.warning(
                f"[GLOBAL] Skipping RSU {rsu_id}: model file not found at {model_path}"
            )
            continue

        booster = xgb.Booster()
        booster.load_model(model_path)

        # Predictions on VAL/TEST for ensemble averaging
        y_val_proba_rsu = np.asarray(
            booster.predict(dval), dtype=np.float64
        )
        y_test_proba_rsu = np.asarray(
            booster.predict(dtest), dtype=np.float64
        )
        y_val_proba_rsu = np.nan_to_num(
            y_val_proba_rsu, nan=0.0, posinf=1.0, neginf=0.0
        )
        y_test_proba_rsu = np.nan_to_num(
            y_test_proba_rsu, nan=0.0, posinf=1.0, neginf=0.0
        )
        y_val_proba_rsu = np.clip(
            y_val_proba_rsu, 1e-7, 1.0 - 1e-7
        )
        y_test_proba_rsu = np.clip(
            y_test_proba_rsu, 1e-7, 1.0 - 1e-7
        )

        if sum_val_proba is None:
            sum_val_proba = y_val_proba_rsu.astype(np.float64)
        else:
            sum_val_proba += y_val_proba_rsu.astype(np.float64)

        if sum_test_proba is None:
            sum_test_proba = y_test_proba_rsu.astype(np.float64)
        else:
            sum_test_proba += y_test_proba_rsu.astype(np.float64)

        used_rsus.append(rsu_id)

    if sum_test_proba is None or not used_rsus:
        logging.warning(
            "[GLOBAL] No RSU models available for ensemble aggregation; skipping."
        )
    else:
        # Mean probabilities across all RSUs
        mean_val_proba = sum_val_proba / float(len(used_rsus))
        mean_test_proba = sum_test_proba / float(len(used_rsus))

        # Calibrate an F1-optimal threshold on VAL for the ensemble
        ensemble_thr = find_optimal_threshold(
            y_val_np,
            mean_val_proba,
            default_threshold=0.5,
        )

        # VAL metrics at ensemble threshold
        ensemble_val_pred = (mean_val_proba >= ensemble_thr).astype(int)
        ensemble_val_metrics = compute_binary_metrics(
            y_val_np,
            ensemble_val_pred,
            mean_val_proba,
            calibration=True,
        )

        # TEST metrics at ensemble threshold
        y_pred_ensemble = (mean_test_proba >= ensemble_thr).astype(int)
        ensemble_metrics = compute_binary_metrics(
            y_test_np,
            y_pred_ensemble,
            mean_test_proba,
            calibration=True,
        )
        cm_ensemble = confusion_matrix(y_test_np, y_pred_ensemble)
        ensemble_metrics["threshold"] = float(ensemble_thr)
        ensemble_val_metrics["threshold"] = float(ensemble_thr)

        log_metrics_pretty(
            f"[GLOBAL ENSEMBLE] TEST (thr={ensemble_thr:.4f})",
            ensemble_metrics,
            cm_ensemble,
        )

        # Identify the best single RSU by AUC from centralized_results
        best_rsu_id = max(
            used_rsus,
            key=lambda rid: per_rsu_test_metrics[rid].get("auc", 0.0),
        )
        best_rsu_metrics = per_rsu_test_metrics[best_rsu_id]

        logging.info(
            f"[GLOBAL] Best single RSU on TEST is RSU {best_rsu_id} "
            f"with AUC={best_rsu_metrics['auc']:.6f}, "
            f"F1={best_rsu_metrics['f1']:.6f}"
        )

        # ---------------------------
        # Persist global server ensemble summary as JSON
        # ---------------------------
        global_summary = {
            "num_rsus_configured": NUM_RSUS,
            "num_rsus_used": len(used_rsus),
            "used_rsu_ids": used_rsus,
            # TEST-set ensemble metrics (Acc/Prec/Rec/F1/AUC/Brier/ECE + threshold)
            "ensemble_metrics": ensemble_metrics,
            "ensemble_confusion_matrix": cm_ensemble.tolist(),
            # VAL metrics used to choose threshold for the ensemble
            "ensemble_val_metrics": ensemble_val_metrics,
            "best_single_rsu_id": best_rsu_id,
            "best_single_rsu_metrics": best_rsu_metrics,
            "per_rsu_test_metrics": per_rsu_test_metrics,
        }

        global_summary_path = os.path.join(
            OUTPUT_DIR,
            "iov_global_server_ensemble_summary.json",
        )
        try:
            with open(global_summary_path, "w", encoding="utf-8") as f:
                json.dump(global_summary, f, indent=2)
            logging.info(
                f"[GLOBAL] Saved global server ensemble summary to {global_summary_path}"
            )
        except Exception as e:
            logging.error(
                f"[GLOBAL] Failed to write global ensemble summary JSON: {e}"
            )