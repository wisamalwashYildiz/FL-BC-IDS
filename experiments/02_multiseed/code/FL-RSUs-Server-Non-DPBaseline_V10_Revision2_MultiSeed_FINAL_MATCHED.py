#!/usr/bin/env python3
import json
import logging
import os
import time
import warnings
import tempfile
import ast  # <-- NEW: to safely parse str(history)
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb  # Standard (non-DP) XGBoost

import flwr as fl
import ray
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Scalar,
    Status,
)

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedXgbBagging
from flwr.server import ServerAppComponents, ServerConfig
# App-based runtime imports (Flower 1.23.0)
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation

# ---------------------------------------------------------------------------
# Ray/Flower runtime hardening for repeated RSU simulations on Windows
# ---------------------------------------------------------------------------
# Flower uses Ray as the Simulation Engine backend. Repeatedly terminating one
# local Ray runtime and immediately starting another on Windows can leave
# transient GCS/process/port state behind. Because this experiment executes
# RSU 1 and RSU 2 sequentially in one Python process, each RSU receives a
# fresh local Ray runtime with explicit shutdown/cleanup and startup-only
# retries.
#
# IMPORTANT:
# - This changes runtime lifecycle only.
# - It does NOT change the FL topology, client data, XGBoost parameters,
#   aggregation, number of rounds, local rounds, or evaluation semantics.
RAY_STARTUP_MAX_ATTEMPTS = 3
RAY_WINDOWS_CLEANUP_GRACE_SEC = 3.0
RAY_STARTUP_FAILURE_EXHAUSTED_MARKER = "RAY_STARTUP_FAILURE_EXHAUSTED"

RAY_STARTUP_RETRY_MARKERS = (
    "failed to start gcs",
    "failed to start the grpc server",
    "address already in use",
    "gcs_server.exe",
)


def _exception_chain_text(exc: BaseException) -> str:
    """Return the complete exception/cause/context chain as lowercase text."""
    parts: List[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc

    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(
            f"{type(cur).__name__}: {cur}"
        )

        if cur.__cause__ is not None:
            cur = cur.__cause__
        else:
            cur = cur.__context__

    return "\n".join(parts).lower()


def _is_transient_ray_startup_failure(
    exc: BaseException,
) -> bool:
    """
    Return True only for the Ray/GCS startup-failure family that is safe
    to retry before client training begins.
    """
    text = _exception_chain_text(exc)

    return any(
        marker in text
        for marker in RAY_STARTUP_RETRY_MARKERS
    )


def _ray_cleanup_barrier() -> None:
    """
    Best-effort cleanup of the current local Ray runtime.

    A short Windows barrier gives Ray child processes/GCS enough time to
    terminate before another local cluster is created.
    """
    try:
        ray.shutdown()
    except Exception as cleanup_exc:
        logging.warning(
            "Ray cleanup warning: %s",
            cleanup_exc,
        )

    if os.name == "nt":
        time.sleep(
            RAY_WINDOWS_CLEANUP_GRACE_SEC
        )


def _build_ray_backend_config(
    num_supernodes: int,
) -> Dict:
    """
    Build a fresh local Ray backend sized for exactly the required ClientApps.

    address='local' forces creation of a new local Ray instance rather than
    attaching to stale local-cluster state.

    Each vehicle ClientApp receives 2 CPUs, so for two vehicles the local
    runtime exposes 4 CPUs. This avoids Flower creating a much larger idle
    actor pool than this compact experiment requires.
    """
    num_supernodes = max(
        1,
        int(num_supernodes),
    )

    return {
        "init_args": {
            "address": "local",
            "include_dashboard": False,
            "num_cpus": int(
                2 * num_supernodes
            ),
            "num_gpus": 0,
        },
        "client_resources": {
            "num_cpus": 2,
            "num_gpus": 0,
        },
    }


def _run_simulation_hardened(
    *,
    server_app: ServerApp,
    client_app: ClientApp,
    num_supernodes: int,
    rsu_id: int,
):
    """
    Execute one RSU Flower simulation using a clean local Ray lifecycle.

    Only genuine Ray/GCS startup failures are retried here. Any failure after
    training begins is propagated immediately so the outer orchestrator can
    quarantine and restart the complete FL stage from a scientifically clean
    boundary.
    """
    backend_config = _build_ray_backend_config(
        num_supernodes
    )

    last_exc: BaseException | None = None

    for attempt in range(
        1,
        RAY_STARTUP_MAX_ATTEMPTS + 1,
    ):
        # Ensure no local Ray runtime from the previous RSU is still attached.
        _ray_cleanup_barrier()

        try:
            logging.info(
                "[RAY] RSU %d starting simulation attempt %d/%d "
                "(fresh local cluster, supernodes=%d, backend_cpus=%d)",
                int(rsu_id),
                int(attempt),
                int(RAY_STARTUP_MAX_ATTEMPTS),
                int(num_supernodes),
                int(
                    2 * max(
                        1,
                        int(num_supernodes),
                    )
                ),
            )

            history = run_simulation(
                server_app=server_app,
                client_app=client_app,
                num_supernodes=int(
                    num_supernodes
                ),
                backend_config=backend_config,
            )

            return history

        except Exception as exc:
            last_exc = exc

            transient_startup = (
                _is_transient_ray_startup_failure(
                    exc
                )
            )

            # Do NOT retry genuine FL/training/evaluation failures here.
            if not transient_startup:
                raise

            if (
                attempt
                >= RAY_STARTUP_MAX_ATTEMPTS
            ):
                logging.error(
                    "[%s] RSU %d exhausted %d Ray startup attempts. "
                    "Final cause chain: %s",
                    RAY_STARTUP_FAILURE_EXHAUSTED_MARKER,
                    int(rsu_id),
                    int(RAY_STARTUP_MAX_ATTEMPTS),
                    _exception_chain_text(exc),
                )

                raise

            logging.warning(
                "[RAY] RSU %d transient Ray startup failure on attempt %d/%d; "
                "cleaning the local runtime and retrying. Cause chain: %s",
                int(rsu_id),
                int(attempt),
                int(RAY_STARTUP_MAX_ATTEMPTS),
                _exception_chain_text(exc),
            )

            _ray_cleanup_barrier()

            # Progressive short backoff: 2 s after attempt 1,
            # 4 s after attempt 2.
            time.sleep(
                float(attempt) * 2.0
            )

        finally:
            # Flower normally terminates its Ray runtime itself, but this
            # explicitly clears any remaining local state before RSU 2.
            if ray.is_initialized():
                _ray_cleanup_barrier()

    if last_exc is not None:
        raise last_exc

    raise RuntimeError(
        "Ray simulation retry loop exited unexpectedly."
    )


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,  # <-- NEW: for threshold tuning
    brier_score_loss,        # <-- NEW: for calibration quality
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
    """Compute standard binary classification metrics for IoV IDS.

    If `calibration` is True, also compute Brier score and ECE.
    """
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
        try:
            brier = brier_score_loss(y_true, y_proba)
        except Exception:
            brier = 0.0
        ece = compute_ece(y_true, y_proba)
        metrics["brier"] = float(brier)
        metrics["ece"] = float(ece)

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

    # If only one class is present, threshold doesn't matter
    if len(np.unique(y_true)) < 2:
        return default_threshold

    # Regression-family predictions can slightly leave [0, 1].
    y_proba = np.clip(np.asarray(y_proba, dtype=np.float64), 0.0, 1.0)

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
    """
    Use the intersection of numeric columns across splits and separate features/labels.
    """
    num_cols_train = df_train.select_dtypes(include=["number"]).columns
    num_cols_val = df_val.select_dtypes(include=["number"]).columns
    num_cols_test = df_test.select_dtypes(include=["number"]).columns

    numeric_cols = list(set(num_cols_train) & set(num_cols_val) & set(num_cols_test))
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    if not numeric_cols:
        raise ValueError("No shared numeric feature columns found across splits")

    logging.info(f"Number of IoV numeric features (intersection): {len(numeric_cols)}")

    X_train = df_train[numeric_cols]
    y_train = df_train[label_col].astype(int)

    X_val = df_val[numeric_cols]
    y_val = df_val[label_col].astype(int)

    X_test = df_test[numeric_cols]
    y_test = df_test[label_col].astype(int)

    for name, X_, y_ in [("Train", X_train, y_train),
                         ("Val", X_val, y_val),
                         ("Test", X_test, y_test)]:
        if X_.empty or y_.empty:
            raise ValueError(f"{name} split is empty or invalid")

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

    # If one of the classes is missing globally, fall back to simple
    # equal-chunk partitioning.
    if n_pos_total == 0 or n_neg_total == 0:
        logging.warning(
            "Global train split has only one class (pos=%d, neg=%d); "
            "falling back to simple equal-chunk partitioning.",
            n_pos_total,
            n_neg_total,
        )

        idx_chunks = np.array_split(indices, total_vehicles)

        rsu_partitions_fallback: List[
            List[Tuple[pd.DataFrame, pd.Series]]
        ] = []

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

    def _allocate_counts(
        total: int,
        total_buckets: int,
        k_min: int,
    ) -> List[int]:
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
                f"_allocate_counts internal error: "
                f"sum(counts)={sum(counts)} != total={total}"
            )

        return counts

    # ------------------------------------------------------------------
    # Stage 1: split positives and negatives across RSUs
    # ------------------------------------------------------------------
    pos_chunks_rsu = np.array_split(pos_indices_all, num_rsus)
    neg_chunks_rsu = np.array_split(neg_indices_all, num_rsus)

    rsu_partitions: List[
        List[Tuple[pd.DataFrame, pd.Series]]
    ] = []

    for rsu_idx in range(num_rsus):
        pos_indices_rsu = pos_chunks_rsu[rsu_idx]
        neg_indices_rsu = neg_chunks_rsu[rsu_idx]

        n_pos_rsu = len(pos_indices_rsu)
        n_neg_rsu = len(neg_indices_rsu)
        n_rsu_total = n_pos_rsu + n_neg_rsu

        logging.info(
            "RSU %d: train subset total=%d, pos=%d, neg=%d "
            "(pos_frac=%.4f)",
            rsu_idx + 1,
            n_rsu_total,
            n_pos_rsu,
            n_neg_rsu,
            float(n_pos_rsu) / n_rsu_total
            if n_rsu_total > 0
            else 0.0,
        )

        # --------------------------------------------------------------
        # Stage 2: split each RSU's samples among its vehicles
        # --------------------------------------------------------------
        pos_counts = _allocate_counts(
            n_pos_rsu,
            vehicles_per_rsu,
            k_min_pos,
        )

        neg_counts = _allocate_counts(
            n_neg_rsu,
            vehicles_per_rsu,
            k_min_neg,
        )

        pos_cursor = 0
        neg_cursor = 0

        rsu_vehicles: List[
            Tuple[pd.DataFrame, pd.Series]
        ] = []

        for veh_idx in range(vehicles_per_rsu):
            n_pos_v = pos_counts[veh_idx]
            n_neg_v = neg_counts[veh_idx]

            idx_pos_v = pos_indices_rsu[
                pos_cursor : pos_cursor + n_pos_v
            ]

            idx_neg_v = neg_indices_rsu[
                neg_cursor : neg_cursor + n_neg_v
            ]

            pos_cursor += n_pos_v
            neg_cursor += n_neg_v

            combined_idx = np.concatenate(
                [idx_pos_v, idx_neg_v]
            )

            rng.shuffle(combined_idx)

            X_v = X_train.iloc[
                combined_idx
            ].reset_index(drop=True)

            y_v = y_train.iloc[
                combined_idx
            ].reset_index(drop=True)

            y_v_arr = np.asarray(
                y_v.to_numpy(copy=False),
                dtype=np.int32,
            )

            unique_labels = sorted(
                np.unique(y_v_arr).tolist()
            )

            n_pos_local = int(
                np.count_nonzero(y_v_arr == 1)
            )
            n_neg_local = int(
                np.count_nonzero(y_v_arr == 0)
            )

            if len(unique_labels) < 2:
                logging.info(
                    "RSU %d, vehicle %d: local shard has only labels %s "
                    "(pos=%d, neg=%d).",
                    rsu_idx + 1,
                    veh_idx + 1,
                    unique_labels,
                    n_pos_local,
                    n_neg_local,
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
            y_v_arr = np.asarray(
                y_v.to_numpy(copy=False),
                dtype=np.int32,
            )

            n_samples = int(y_v_arr.size)
            n_pos = int(
                np.count_nonzero(y_v_arr == 1)
            )
            n_neg = int(
                np.count_nonzero(y_v_arr == 0)
            )

            frac_pos = (
                float(n_pos) / float(n_samples)
                if n_samples > 0
                else 0.0
            )

            logging.info(
                "RSU %d | Vehicle %d: "
                "samples=%d, pos=%d, neg=%d, pos_frac=%.4f",
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

            if n_samples > 0 and (n_pos == 0 or n_neg == 0):
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
      - evaluates on the RSU-wide validation split during FL,
      - exchanges XGBoost models with the RSU (Flower server).

    The TEST split is reserved for final post-training evaluation.
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
        """Send the current XGBoost model encoded as JSON bytes."""
        try:
            # self.model is initialized as a valid Booster in __init__.
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

            best_iteration_raw = getattr(
                self.model,
                "best_iteration",
                None,
            )
            best_score_raw = getattr(
                self.model,
                "best_score",
                None,
            )

            # Normalize XGBoost's dynamically typed attributes to concrete
            # Python scalar types for safe arithmetic, formatting, and JSON.
            try:
                best_iteration: int = (
                    int(str(best_iteration_raw))
                    if best_iteration_raw is not None
                    else -1
                )
            except (TypeError, ValueError):
                best_iteration = -1

            try:
                best_score: float = (
                    float(str(best_score_raw))
                    if best_score_raw is not None
                    else -1.0
                )
            except (TypeError, ValueError):
                best_score = -1.0

            stopped_early = False
            reason = "Early stopping disabled"

            if es_enabled and best_iteration >= 0:
                # XGBoost uses a 0-based best_iteration.
                completed_iterations = best_iteration + 1

                stopped_early = (
                    completed_iterations
                    < self.xgb_cfg.num_local_rounds
                )

                if stopped_early:
                    reason = (
                        f"No improvement in eval_metric '{metric_name}' "
                        f"on validation set for "
                        f"{self.xgb_cfg.early_stopping_rounds} rounds"
                    )
                else:
                    reason = (
                        "Reached num_boost_round without triggering "
                        "early stopping "
                        f"(best_iteration={completed_iterations})"
                    )

            if es_enabled:
                if stopped_early:
                    completed_iterations = best_iteration + 1

                    self.logger.info(
                        f"[RSU {self.rsu_id} | "
                        f"Vehicle {self.vehicle_id} | "
                        f"Round {round_num}] "
                        f"Early stopping TRIGGERED at iteration "
                        f"{completed_iterations}/"
                        f"{self.xgb_cfg.num_local_rounds}, "
                        f"best {metric_name}={best_score:.6f}"
                    )
                else:
                    self.logger.info(
                        f"[RSU {self.rsu_id} | "
                        f"Vehicle {self.vehicle_id} | "
                        f"Round {round_num}] "
                        f"Early stopping ENABLED but ran full "
                        f"{self.xgb_cfg.num_local_rounds} iterations, "
                        f"best {metric_name}={best_score:.6f}"
                    )
            else:
                self.logger.info(
                    f"[RSU {self.rsu_id} | "
                    f"Vehicle {self.vehicle_id} | "
                    f"Round {round_num}] "
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
                "best_iteration": best_iteration,
                "best_score": best_score,
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
        """
        Evaluate the RSU-global model on the RSU-wide VALIDATION split.

        The TEST split is deliberately not used during federated training.
        TEST is reserved for the final post-training RSU and GLOBAL
        evaluations.
        """
        try:
            # Load the latest global model
            self.set_parameters(ins.parameters)

            # ----------------------------------------------------------
            # Predict on the RSU-wide validation split only
            # ----------------------------------------------------------
            y_val_true: np.ndarray = np.asarray(
                self.val_dmatrix.get_label(),
                dtype=np.int32,
            )

            y_val_proba: np.ndarray = np.asarray(
                self.model.predict(self.val_dmatrix),
                dtype=np.float64,
            )

            # Regression-family predictions can slightly leave [0, 1].
            y_val_proba = np.nan_to_num(
                y_val_proba,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )

            y_val_proba = np.clip(
                y_val_proba,
                1e-7,
                1.0 - 1e-7,
            )

            # ----------------------------------------------------------
            # Select threshold using VALIDATION only
            # ----------------------------------------------------------
            best_thr = find_optimal_threshold(
                y_val_true,
                y_val_proba,
                default_threshold=0.5,
            )

            y_pred_val: np.ndarray = (
                    y_val_proba >= best_thr
            ).astype(int)

            val_metrics = compute_binary_metrics(
                y_val_true,
                y_pred_val,
                y_val_proba,
                calibration=True,
            )

            # Proper probabilistic validation loss
            val_logloss = float(
                log_loss(
                    y_val_true,
                    y_val_proba,
                    labels=[0, 1],
                )
            )

            val_brier = float(
                val_metrics.get("brier", 0.0)
            )

            cm_val = confusion_matrix(
                y_val_true,
                y_pred_val,
            )

            log_metrics_pretty(
                f"[RSU {self.rsu_id} | "
                f"Vehicle {self.vehicle_id}] "
                f"VAL "
                f"(thr={best_thr:.4f}, "
                f"logloss={val_logloss:.6f})",
                val_metrics,
                cm_val,
            )

            return EvaluateRes(
                status=Status(
                    code=Code.OK,
                    message="evaluation complete",
                ),
                loss=val_logloss,
                num_examples=len(y_val_true),
                metrics={
                    "accuracy": val_metrics["accuracy"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                    "f1": val_metrics["f1"],
                    "auc": val_metrics["auc"],
                    "brier": val_brier,
                    "ece": float(
                        val_metrics.get("ece", 0.0)
                    ),
                    "logloss": val_logloss,
                    "threshold": float(best_thr),
                    "confusion_matrix": json.dumps(
                        cm_val.tolist()
                    ),
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
    results: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
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

        # Per-round, per-vehicle training/validation/DP metrics for comparison script
        # Structure: {round: [ {client_id, num_examples, train_*, val_*, dp_*}, ... ]}
        self.vehicle_metrics_per_round: Dict[int, List[Dict[str, object]]] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[
            Tuple[ClientProxy, FitRes] | BaseException
        ],
    ) -> Tuple[Parameters | None, Dict[str, Scalar]]:
        # Call FedXgbBagging's aggregation (tree-bagging logic).
        aggregated_params, agg_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        rnd = server_round

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

    # Convert shared validation/test data to concrete NumPy arrays.
    X_val_np: np.ndarray = np.asarray(
        X_val.to_numpy(copy=True),
        dtype=np.float32,
    )
    y_val_np: np.ndarray = np.asarray(
        y_val.to_numpy(copy=True),
        dtype=np.int32,
    )
    X_test_np: np.ndarray = np.asarray(
        X_test.to_numpy(copy=True),
        dtype=np.float32,
    )
    y_test_np: np.ndarray = np.asarray(
        y_test.to_numpy(copy=True),
        dtype=np.int32,
    )

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
    vehicle_partitions: List[
        Tuple[np.ndarray, np.ndarray]
    ] = []

    for X_v, y_v in rsu_vehicles:
        X_v_np: np.ndarray = np.asarray(
            X_v.to_numpy(copy=True),
            dtype=np.float32,
        )
        y_v_np: np.ndarray = np.asarray(
            y_v.to_numpy(copy=True),
            dtype=np.int32,
        )

        vehicle_partitions.append(
            (X_v_np, y_v_np)
        )

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

    # ------------------------------------------------------------------
    # Run Flower through the hardened Ray lifecycle wrapper.
    #
    # Each RSU receives a fresh local Ray runtime. Transient Ray/GCS
    # startup failures are retried locally; genuine FL/training failures
    # are propagated immediately.
    # ------------------------------------------------------------------
    history = _run_simulation_hardened(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=len(
            vehicle_partitions
        ),
        rsu_id=rsu_cfg.rsu_id,
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
        except Exception:
            logging.exception(
                "RSU %d - Failed to save global model",
                rsu_cfg.rsu_id,
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
    except Exception:
        logging.exception(
            "RSU %d - Failed to write summary JSON",
            rsu_cfg.rsu_id,
        )

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
    X_val_np: np.ndarray = np.asarray(
        X_val.to_numpy(copy=True),
        dtype=np.float32,
    )
    y_val_np: np.ndarray = np.asarray(
        y_val.to_numpy(copy=True),
        dtype=np.int32,
    )

    dval = xgb.DMatrix(
        X_val_np,
        label=y_val_np,
    )

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
    X_test_np: np.ndarray = np.asarray(
        X_test.to_numpy(copy=True),
        dtype=np.float32,
    )
    y_test_np: np.ndarray = np.asarray(
        y_test.to_numpy(copy=True),
        dtype=np.int32,
    )

    dtest = xgb.DMatrix(
        X_test_np,
        label=y_test_np,
    )

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
    metrics["val_brier"] = float(val_metrics.get("brier", 0.0))
    metrics["val_ece"] = float(val_metrics.get("ece", 0.0))
    metrics["val_threshold"] = float(best_thr)

    return metrics

if __name__ == "__main__":
    setup_logging()

    # -----------------------------------------------------------------------
    # Reviewer 1 / Revision 2 / Comment 1:
    # multi-seed, independently repartitioned evaluation of the non-DP
    # hierarchical FL baseline.
    #
    # The original baseline file remains untouched. Each seed reads the
    # corresponding seed-specific preprocessing artifacts and writes to an
    # isolated Revision-2 directory.
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Non-DP hierarchical FL baseline for Reviewer 1 Round-2 "
            "multi-seed statistical evaluation."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("CSECICIDS2018", "CICIoV2024"),
        default="CSECICIDS2018",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Experimental seed used by XGBoost and hierarchical "
            "RSU/vehicle partitioning. Preprocessing must already exist "
            "for the same seed."
        ),
    )
    parser.add_argument(
        "--revision-root",
        type=str,
        default=os.getenv("FLBCIDS_MULTI_SEED_RESULTS_DIR", "experiments/02_multiseed/results"),
        help="Root directory for Reviewer-1 Round-2 multi-seed result artifacts.",
    )
    parser.add_argument("--num-rsus", type=int, default=2)
    parser.add_argument("--vehicles-per-rsu", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument(
        "--num-local-rounds",
        type=int,
        default=10,
        help=(
            "Local boosting iterations per vehicle. Default=10 because the "
            "manuscript's matched non-DP hierarchical reference uses the same "
            "compact 10-local-round schedule as the principal FL-BC-IDS run."
        ),
    )
    args = parser.parse_args()

    # This retained baseline uses a set-intersection when reconstructing the
    # shared numeric feature list. The completed Round-2 study therefore fixed
    # Python's process-level hash seed to zero. Fail closed on direct execution
    # rather than silently changing feature order and producing non-comparable
    # results. The official orchestration runner sets this before child startup.
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "This matched non-DP multi-seed baseline requires PYTHONHASHSEED=0 "
            "at interpreter startup. Run it through "
            "Reviewer1_Comment1_Run_MultiSeed.py or set PYTHONHASHSEED=0 before "
            "launching Python."
        )

    DATASET_NAME = str(args.dataset)
    EXPERIMENT_SEED = int(args.seed)
    REVISION2_ROOT = os.path.abspath(str(args.revision_root))

    DATASET_CONFIGS = {
        "CSECICIDS2018": {
            "train_csv": "CSECICIDS2018_train_preprocessed.csv",
            "val_csv": "CSECICIDS2018_val_preprocessed.csv",
            "test_csv": "CSECICIDS2018_test_preprocessed.csv",
            "label_col": "Label",
        },
        "CICIoV2024": {
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

    PREPROC_DIR = os.path.join(
        REVISION2_ROOT,
        DATASET_NAME,
        f"seed_{EXPERIMENT_SEED}",
        "preprocessing",
    )

    ds_cfg = IoVDatasetConfig(
        preproc_dir=PREPROC_DIR,
        train_csv=os.path.join(PREPROC_DIR, cfg["train_csv"]),
        val_csv=os.path.join(PREPROC_DIR, cfg["val_csv"]),
        test_csv=os.path.join(PREPROC_DIR, cfg["test_csv"]),
        label_col=cfg["label_col"],
    )

    # ---------------------------
    # High-level RSU configuration
    # ---------------------------
    NUM_RSUS = int(args.num_rsus)
    VEHICLES_PER_RSU = int(args.vehicles_per_rsu)
    NUM_ROUNDS = int(args.num_rounds)
    NUM_LOCAL_ROUNDS = int(args.num_local_rounds)

    if NUM_RSUS <= 0 or VEHICLES_PER_RSU <= 0 or NUM_ROUNDS <= 0 or NUM_LOCAL_ROUNDS <= 0:
        raise ValueError(
            "num-rsus, vehicles-per-rsu, num-rounds, and num-local-rounds must all be > 0"
        )

    # Baseline (non-DP) run output directory.
    # The manuscript-matched configuration uses 10 local rounds. The local-round
    # count remains in the path so intentional auxiliary variants cannot collide.
    OUTPUT_DIR = os.path.join(
        REVISION2_ROOT,
        DATASET_NAME,
        f"seed_{EXPERIMENT_SEED}",
        f"nondp_local_{NUM_LOCAL_ROUNDS}",
    )

    logging.info("=" * 100)
    logging.info("REVIEWER 1 ROUND-2 MULTI-SEED NON-DP RUN")
    logging.info("Dataset              : %s", DATASET_NAME)
    logging.info("Experimental seed    : %d", EXPERIMENT_SEED)
    logging.info("Preprocessing input  : %s", PREPROC_DIR)
    logging.info("Baseline output dir  : %s", OUTPUT_DIR)
    logging.info(
        "Topology              : %d RSUs x %d vehicles, %d FL rounds, %d local rounds",
        NUM_RSUS,
        VEHICLES_PER_RSU,
        NUM_ROUNDS,
        NUM_LOCAL_ROUNDS,
    )
    logging.info("=" * 100)

    # Matched non-DP XGBoost hyperparameters.
    # These mirror the principal FL-BC-IDS learner configuration; the
    # privacy-specific difference is tree_method="hist" with no
    # dp_epsilon_per_tree parameter/accounting.
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
        "random_state": EXPERIMENT_SEED,
        "seed": EXPERIMENT_SEED,
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
        "experiment_seed": int(EXPERIMENT_SEED),
        "revision_context": {
            "round": 2,
            "reviewer": 1,
            "comment": 1,
            "purpose": "multi-seed and repartitioned statistical evaluation",
            "revision_root": str(REVISION2_ROOT),
            "preprocessing_dir": str(PREPROC_DIR),
            "output_dir": str(OUTPUT_DIR),
            "matched_ablation": True,
            "ablation_difference": "DP tree mechanism/accounting disabled; learner hyperparameters otherwise matched",
        },
    }

    run_cfg_path = os.path.join(OUTPUT_DIR, "iov_run_config.json")
    try:
        with open(run_cfg_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
        logging.info("Saved non-DP run configuration to %s", run_cfg_path)
    except Exception:
        logging.exception(
            "Failed to write non-DP run configuration JSON"
        )

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
        random_state=EXPERIMENT_SEED,
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
    except Exception:
        logging.exception(
            "[CENTRALIZED EVAL] Failed to write summary JSON"
        )

    # ---------------------------
    # Global server ensemble aggregation across RSU models
    # ---------------------------
    logging.info(
        "[GLOBAL] Computing ensemble over all RSU-global models on IoV TEST split..."
    )

    # Reuse the same numeric feature subset/order as during training.
    X_val_np: np.ndarray = np.asarray(
        X_val.to_numpy(copy=True),
        dtype=np.float32,
    )
    y_val_np: np.ndarray = np.asarray(
        y_val.to_numpy(copy=True),
        dtype=np.int32,
    )

    dval = xgb.DMatrix(
        X_val_np,
        label=y_val_np,
    )

    X_test_np: np.ndarray = np.asarray(
        X_test.to_numpy(copy=True),
        dtype=np.float32,
    )
    y_test_np: np.ndarray = np.asarray(
        y_test.to_numpy(copy=True),
        dtype=np.int32,
    )

    dtest = xgb.DMatrix(
        X_test_np,
        label=y_test_np,
    )

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
        y_val_proba_rsu = np.clip(
            np.asarray(booster.predict(dval), dtype=np.float64), 0.0, 1.0
        )
        y_test_proba_rsu = np.clip(
            np.asarray(booster.predict(dtest), dtype=np.float64), 0.0, 1.0
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
        except Exception:
            logging.exception(
                "[GLOBAL] Failed to write global ensemble summary JSON"
            )