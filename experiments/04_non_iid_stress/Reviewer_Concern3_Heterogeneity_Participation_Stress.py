#!/usr/bin/env python3
# Reviewer_Concern3_Heterogeneity_Participation_Stress.py
# Dedicated controlled robustness experiment for Reviewer 1, Comment 3.
# Derived from FL_DP_SSI_DualMerklePoseidon_RSU+Global_ZKVerifyABLATION_V10_2.py.
# The main FL-BC-IDS implementation is intentionally left unchanged.
import inspect
import json
import os
import shutil
import subprocess
import sys
os.environ["PYSNARK_BACKEND"] = "snarkjs"
import logging
from pathlib import Path
# AnchorZKP: locate the repository root, hardened core helpers, and Node modules.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = None
_CORE_DIR = None

_candidate_roots = []
_env_root = os.environ.get("FLBCIDS_REPO_ROOT", "").strip()
if _env_root:
    _candidate_roots.append(Path(_env_root).expanduser().resolve())
_candidate_roots.extend([_SCRIPT_DIR, *_SCRIPT_DIR.parents])

for _cand in _candidate_roots:
    _cand = _cand.resolve()
    if not (_cand / "node_modules").is_dir():
        continue
    for _helper_dir in (_cand, _cand / "code" / "core"):
        if (_helper_dir / "FL_IoV_DPxgb_UtilsV10.py").is_file():
            _PROJECT_ROOT = _cand
            _CORE_DIR = _helper_dir.resolve()
            break
    if _PROJECT_ROOT is not None:
        break

if _PROJECT_ROOT is None or _CORE_DIR is None:
    raise RuntimeError(
        "Could not locate the FL-BC-IDS repository root. Expected node_modules "
        "at the repository root and FL_IoV_DPxgb_UtilsV10.py either there or "
        "under code/core/. Set FLBCIDS_REPO_ROOT to the repository root if needed."
    )

if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

os.environ["ANCHOR_ZKP_PROJECT_ROOT"] = str(_PROJECT_ROOT)
_NODE_MODULES = (_PROJECT_ROOT / "node_modules").resolve()
_NODE_BIN = (_NODE_MODULES / ".bin").resolve()
os.environ["ANCHOR_ZKP_CIRCOM_LINK_LIBS"] = str(_NODE_MODULES)
os.environ["CIRCOM_LINK_LIBS"] = str(_NODE_MODULES)
if _NODE_BIN.is_dir():
    os.environ["PATH"] = str(_NODE_BIN) + os.pathsep + os.environ.get("PATH", "")

import time
from datetime import datetime, timezone
import glob
import math  # For DP epsilon accounting
import hashlib
from typing import Any, Callable, Dict, List, Tuple, cast
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import FL_IoV_AnchorZKP_UtilsV10 as azkp          # AnchorSum ZK lane
import FL_IoV_CanonicalSpecV10 as canon           # ✅ Phase-1 canonical bytes authority
import FL_IoV_MerkleSSI_UtilsV10 as mssi           # ✅ Phase-2 dual leaves + dual roots + audit
# MUST match your frozen spec (V8b)
POSEIDON_ARITY = 5
POSEIDON_DEPTH = 32
# ✅ Keep SHA depth explicit (can be same as Poseidon depth, but don’t overload names)
SHA_DEPTH = 32
# ✅ Make “be sure” the default
STRICT_MERKLE_AUDIT = True
# Flower partition-id is typically 0-based in Context.node_config
# (see Flower docs/examples using context.node_config["partition-id"]).
PARTITION_ID_BASE_DEFAULT: int = int(os.environ.get("PARTITION_ID_BASE_DEFAULT", "0"))
if PARTITION_ID_BASE_DEFAULT not in (0, 1):
    raise RuntimeError(f"Invalid PARTITION_ID_BASE_DEFAULT={PARTITION_ID_BASE_DEFAULT} (must be 0 or 1)")
import dp_xgboost as xgb  # DP-enabled XGBoost (Sarus)
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
# --- Monkey-patch FedXgbBagging for dp_xgboost compatibility --------------
import flwr.server.strategy.fedxgb_bagging as _fxgb
# --------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, log_loss
from FL_IoV_DPxgb_UtilsV10 import (
    IoVDatasetConfig,
    RSUConfig,
    XGBoostConfig,
    setup_logging,
    extract_history_distributed_summary,
    compute_binary_metrics,
    find_optimal_threshold,
    log_metrics_pretty,
    load_iov_splits,
    extract_numeric_features,
    partition_train_among_rsus_and_vehicles,
    log_partition_stats,
    initialize_booster_with_dummy_data,
    booster_to_json_bytes,
    booster_from_json_bytes,
    make_delta_booster,
    _count_trees_in_booster,
    robust_aggregate_probas,
    _as_int,
    _as_str,
    build_dp_round_record_and_sha256_v1,
    write_dp_record_bytes_v1,
    dp_record_sha256_from_bytes_v1,
    _safe_get_tree_nums,
    _sha256_to_nontrivial_field_v1,
    _b64e,
    _b64d,
    _did_from_pubkey_raw_ed25519,
    _ssi_vehicle_report_bytes_v1,
    _ssi_verify_vehicle_report_sig_v1,
    prove_verify_rsu_anchor_sum_from_meta_compat,
    weighted_average_eval_metrics,
    _looks_like_sha256_hex,
    _extract_public_inputs_v15,
    _write_public_sidecar_v15,
    _ensure_precompile_ok_v15,
    _assert_anchor_precompile_outputs_v15,
    _resolve_ssi_fp_v15,
    _get_round_keyed,
    _norm_sha256_hex_v15,
    _parse_intish_v15,
    _extract_ssi_def_fp_v15,
    _read_int_field_from,
    _resolve_policy_id_field_v1,
    _resolve_public_input_order_id_field_v1,
    _resolve_pins_hash_field_v1,
    _sha256_to_field,
    _canon_json_bytes,
    evaluate_saved_rsu_model_centralized,
    _make_dmatrix_version_safe,
)
# Apply monkey-patch
_fxgb._get_tree_nums = _safe_get_tree_nums

# ---------------------------------------------------------------------------
# SSI (Ed25519) helpers (Vehicle signs; RSU verifies)
# ---------------------------------------------------------------------------
STRICT_SSI_VERIFY = True  # refuse to proceed if any aggregated vehicle fails SSI verification

# ---------------------------------------------------------------------------
# Controlled ablation helpers
# ---------------------------------------------------------------------------
def _ablation_target_matches_key_v1(
    ablation_cfg: Dict[str, Any] | None,
    enable_key: str,
    target_key: str,
    *,
    rsu_id: int,
    vehicle_id: int,
    round_idx: int,
) -> bool:
    cfg = dict(ablation_cfg or {})
    if not bool(cfg.get(enable_key, False)):
        return False
    tgt = dict(cfg.get(target_key, {}) or {})
    return (
        int(tgt.get("rsu_id", -1)) == int(rsu_id)
        and int(tgt.get("vehicle_id", -1)) == int(vehicle_id)
        and int(tgt.get("round", -1)) == int(round_idx)
    )

def _ablation_target_matches_v1(
    ablation_cfg: Dict[str, Any] | None,
    *,
    rsu_id: int,
    vehicle_id: int,
    round_idx: int,
) -> bool:
    return _ablation_target_matches_key_v1(
        ablation_cfg,
        "enable_admission_failure_check",
        "admission_failure_target",
        rsu_id=rsu_id,
        vehicle_id=vehicle_id,
        round_idx=round_idx,
    )

def _corrupt_signature_b64_v1(sig_b64: str) -> str:
    try:
        raw = bytearray(_b64d(sig_b64))
        if not raw:
            return _b64e(b"\x00" * 64)
        raw[0] ^= 0x01
        return _b64e(bytes(raw))
    except Exception:
        return _b64e(b"\x00" * 64)

def _mutate_sha256_hex_v1(hex_str: str) -> str:
    s = str(hex_str or "").strip().lower()
    if len(s) != 64 or any(ch not in "0123456789abcdef" for ch in s):
        return ("0" * 63) + "1"
    last = s[-1]
    repl = "0" if last != "0" else "1"
    return s[:-1] + repl


# ---------------------------------------------------------------------------
# Reviewer Comment 3: controlled heterogeneity / participation stress helpers
# ---------------------------------------------------------------------------
REVIEWER3_EXPERIMENT_VERSION = "reviewer3-heterogeneity-participation-v1"


def _reviewer3_scenario_v1() -> str:
    """Resolve the controlled experiment condition from the environment."""
    scenario = str(os.environ.get("REVIEWER3_SCENARIO", "combined_stress") or "combined_stress").strip().lower()
    allowed = {"balanced_reference", "data_heterogeneity", "combined_stress"}
    if scenario not in allowed:
        raise ValueError(
            f"Invalid REVIEWER3_SCENARIO={scenario!r}; choose one of {sorted(allowed)}"
        )
    return scenario


def _reviewer3_context_target_v1(
    cfg: Dict[str, Any] | None,
    *,
    rsu_id: int,
    vehicle_id: int,
    round_idx: int,
) -> Dict[str, Any] | None:
    c: Dict[str, Any] = dict(cfg or {})
    if not bool(c.get("enable_context_binding_challenges", False)):
        return None

    targets = c.get("context_binding_targets", []) or []
    if not isinstance(targets, list):
        return None

    for tgt in targets:
        if not isinstance(tgt, dict):
            continue
        try:
            if (
                int(tgt.get("rsu_id", -1)) == int(rsu_id)
                and int(tgt.get("vehicle_id", -1)) == int(vehicle_id)
                and int(tgt.get("round", -1)) == int(round_idx)
            ):
                return {str(k): v for k, v in tgt.items()}
        except (TypeError, ValueError):
            continue
    return None


def _reviewer3_is_unavailable_v1(
    cfg: Dict[str, Any] | None,
    *,
    rsu_id: int,
    vehicle_id: int,
    round_idx: int,
) -> tuple[bool, str]:
    """
    Deterministic contribution-unavailability model.

    This is deliberately an FL-orchestration stress model, not a PHY/MAC simulator:
    - one client per RSU becomes persistently unavailable from a configured round;
    - selected local client indices experience multi-round burst outages;
    - selected local client indices can experience isolated outages.
    """
    c = dict(cfg or {})
    if not bool(c.get("enable_participation_stress", False)):
        return False, ""

    sched = dict(c.get("availability_schedule", {}) or {})
    local_index = int(vehicle_id) - int(rsu_id) * 1000

    permanent_idx = int(sched.get("permanent_dropout_local_index", -1))
    permanent_from = int(sched.get("permanent_dropout_from_round", 10**9))
    if local_index == permanent_idx and int(round_idx) >= permanent_from:
        return True, f"persistent_dropout_from_round_{permanent_from}"

    burst = dict(sched.get("burst_outage_rounds_by_local_index", {}) or {})
    rounds = burst.get(str(local_index), burst.get(local_index, [])) or []
    if int(round_idx) in {int(x) for x in rounds}:
        return True, "bursty_intermittent_link"

    isolated = dict(sched.get("isolated_outage_rounds_by_local_index", {}) or {})
    isolated_rounds = isolated.get(str(local_index), isolated.get(local_index, [])) or []
    if int(round_idx) in {int(x) for x in isolated_rounds}:
        return True, "isolated_dropout"

    return False, ""


def _reviewer3_allocate_total_v1(total: int, weights: np.ndarray) -> np.ndarray:
    """Largest-remainder integer allocation preserving an exact total."""
    total = int(total)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if total < 0 or w.size == 0:
        raise ValueError("invalid total/weights")
    w = np.where(np.isfinite(w) & (w > 0.0), w, 1e-12)
    raw = total * w / float(w.sum())
    out = np.floor(raw).astype(np.int64)
    rem = total - int(out.sum())
    if rem > 0:
        frac_order = np.argsort(-(raw - out), kind="mergesort")
        out[frac_order[:rem]] += 1
    if int(out.sum()) != total:
        raise RuntimeError("exact allocation failed")
    return out


def _reviewer3_allocate_bounded_v1(
    total: int,
    weights: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Allocate an exact integer total under per-client lower/upper bounds."""
    total = int(total)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower, dtype=np.int64).reshape(-1)
    hi = np.asarray(upper, dtype=np.int64).reshape(-1)
    if not (w.size == lo.size == hi.size):
        raise ValueError("bounded allocation shape mismatch")
    if np.any(lo < 0) or np.any(hi < lo):
        raise ValueError("invalid bounded allocation limits")
    if total < int(lo.sum()) or total > int(hi.sum()):
        raise ValueError(
            f"bounded allocation infeasible: total={total}, lower_sum={int(lo.sum())}, upper_sum={int(hi.sum())}"
        )

    out = lo.copy()
    remaining = total - int(out.sum())
    capacity = hi - out
    safe_w = np.where(np.isfinite(w) & (w > 0.0), w, 1e-12)

    while remaining > 0:
        active = capacity > 0
        if not np.any(active):
            raise RuntimeError("bounded allocation exhausted capacity")
        wa = np.where(active, safe_w, 0.0)
        raw = remaining * wa / float(wa.sum())
        add = np.minimum(np.floor(raw).astype(np.int64), capacity)
        progressed = int(add.sum())
        if progressed > 0:
            out += add
            capacity -= add
            remaining -= progressed
            continue

        # If every proportional increment is < 1, assign one-by-one by the
        # largest current fractional entitlement, with stable tie-breaking.
        candidates = np.nonzero(active)[0]
        score = raw[candidates]
        order = candidates[np.argsort(-score, kind="mergesort")]
        take = min(int(remaining), int(order.size))
        for idx in order[:take]:
            out[idx] += 1
            capacity[idx] -= 1
            remaining -= 1
            if remaining <= 0:
                break

    if int(out.sum()) != total or np.any(out < lo) or np.any(out > hi):
        raise RuntimeError("bounded allocation postcondition failed")
    return out


def _reviewer3_select_feature_for_skew_v1(
    X_train: pd.DataFrame,
    *,
    seed: int,
    sample_cap: int = 100_000,
) -> tuple[int, str]:
    """Choose a high-variance numeric feature using training data only."""
    n = len(X_train)
    if n <= 0 or X_train.shape[1] <= 0:
        raise ValueError("empty training matrix")
    rng = np.random.default_rng(int(seed) + 901)
    if n > int(sample_cap):
        sample_idx = rng.choice(n, size=int(sample_cap), replace=False)
        sample = X_train.iloc[sample_idx]
    else:
        sample = X_train
    arr = sample.to_numpy(dtype=np.float64, copy=False)
    variances = np.nanvar(arr, axis=0)
    variances = np.nan_to_num(variances, nan=-1.0, posinf=-1.0, neginf=-1.0)
    j = int(np.argmax(variances))
    return j, str(X_train.columns[j])


def _reviewer3_partition_heterogeneous_v1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    num_rsus: int,
    vehicles_per_rsu: int,
    seed: int = 42,
    label_alpha: float = 0.30,
    quantity_sigma: float = 0.90,
    max_size_ratio: float = 10.0,
) -> tuple[List[List[Tuple[pd.DataFrame, pd.Series]]], Dict[str, Any]]:
    """
    Create one deterministic, data-native heterogeneous FL partition.

    The construction jointly introduces:
      1) quantity skew via clipped log-normal client-size weights;
      2) label skew via Beta(alpha, alpha) client attack propensities while
         preserving the exact global class totals; and
      3) feature skew by allocating contiguous bands of a high-variance
         training feature within each class, without modifying feature values.

    Every row is assigned exactly once, validation/test are untouched, and (when
    feasible) each client retains at least one benign and one attack sample.
    """
    rng = np.random.default_rng(int(seed))
    V = int(num_rsus) * int(vehicles_per_rsu)
    n = int(len(X_train))
    if V <= 1 or n < 2 * V:
        raise ValueError(f"Need at least two records per vehicle: n={n}, vehicles={V}")

    y_arr = y_train.to_numpy(dtype=np.int32, copy=False)
    pos_idx = np.nonzero(y_arr == 1)[0]
    neg_idx = np.nonzero(y_arr == 0)[0]
    if len(pos_idx) < V or len(neg_idx) < V:
        raise ValueError(
            f"Heterogeneous experiment requires >=1 sample/class/client: pos={len(pos_idx)}, neg={len(neg_idx)}, V={V}"
        )

    # Quantity skew: log-normal weights, clipped to a controlled <= max_size_ratio envelope.
    q = rng.lognormal(mean=0.0, sigma=float(quantity_sigma), size=V)
    q_floor = max(float(np.quantile(q, 0.10)), 1e-12)
    q = np.clip(q, q_floor, q_floor * float(max_size_ratio))
    sizes = _reviewer3_allocate_total_v1(n - 2 * V, q) + 2  # reserve one sample/class/client

    # Label skew: binary Dirichlet analogue Beta(alpha, alpha).
    p_attack = rng.beta(float(label_alpha), float(label_alpha), size=V)
    p_attack = np.clip(p_attack, 0.01, 0.99)
    label_weights = sizes.astype(np.float64) * p_attack
    lower_pos = np.ones(V, dtype=np.int64)
    upper_pos = sizes.astype(np.int64) - 1
    pos_counts = _reviewer3_allocate_bounded_v1(
        len(pos_idx), label_weights, lower_pos, upper_pos
    )
    neg_counts = sizes.astype(np.int64) - pos_counts
    if int(pos_counts.sum()) != len(pos_idx) or int(neg_counts.sum()) != len(neg_idx):
        raise RuntimeError("class-total preservation failed")

    # Feature skew: use one high-variance training feature and allocate contiguous
    # score bands per class. RSUs are interleaved in the allocation order so each
    # RSU spans the feature range while individual clients remain heterogeneous.
    feat_j, feat_name = _reviewer3_select_feature_for_skew_v1(X_train, seed=int(seed))
    feat = X_train.iloc[:, feat_j].to_numpy(dtype=np.float64, copy=False)
    pos_sorted = pos_idx[np.argsort(feat[pos_idx], kind="mergesort")]
    neg_sorted = neg_idx[np.argsort(feat[neg_idx], kind="mergesort")]

    # Global client index = rsu_idx * vehicles_per_rsu + local_idx.
    client_order: List[int] = []
    for local_idx in range(int(vehicles_per_rsu)):
        for rsu_idx in range(int(num_rsus)):
            client_order.append(rsu_idx * int(vehicles_per_rsu) + local_idx)

    client_indices: List[List[int]] = [[] for _ in range(V)]
    p0 = 0
    n0 = 0
    for gidx in client_order:
        pc = int(pos_counts[gidx])
        nc = int(neg_counts[gidx])
        client_indices[gidx].extend(pos_sorted[p0:p0 + pc].tolist())
        client_indices[gidx].extend(neg_sorted[n0:n0 + nc].tolist())
        p0 += pc
        n0 += nc
    if p0 != len(pos_sorted) or n0 != len(neg_sorted):
        raise RuntimeError("feature-band allocation did not consume all rows")

    # Shuffle within each client only; no row is duplicated or dropped.
    rsu_partitions: List[List[Tuple[pd.DataFrame, pd.Series]]] = []
    assigned: List[int] = []
    for rsu_idx in range(int(num_rsus)):
        rsu_list: List[Tuple[pd.DataFrame, pd.Series]] = []
        for local_idx in range(int(vehicles_per_rsu)):
            gidx = rsu_idx * int(vehicles_per_rsu) + local_idx
            idx = np.asarray(client_indices[gidx], dtype=np.int64)
            rng_local = np.random.default_rng(int(seed) + 10_000 + gidx)
            rng_local.shuffle(idx)
            assigned.extend(idx.tolist())
            rsu_list.append(
                (
                    X_train.iloc[idx].reset_index(drop=True),
                    y_train.iloc[idx].reset_index(drop=True),
                )
            )
        rsu_partitions.append(rsu_list)

    assigned_arr = np.asarray(assigned, dtype=np.int64)
    if assigned_arr.size != n or np.unique(assigned_arr).size != n:
        raise RuntimeError("partition is not a one-to-one cover of the training rows")

    meta = {
        "experiment_version": REVIEWER3_EXPERIMENT_VERSION,
        "partition_mode": "joint_quantity_label_feature_heterogeneity",
        "seed": int(seed),
        "label_alpha": float(label_alpha),
        "quantity_sigma": float(quantity_sigma),
        "max_requested_size_ratio": float(max_size_ratio),
        "feature_skew_feature_index": int(feat_j),
        "feature_skew_feature_name": str(feat_name),
        "global_train_rows": int(n),
        "global_attack_rows": int(len(pos_idx)),
        "global_benign_rows": int(len(neg_idx)),
        "client_target_sizes": [int(x) for x in sizes.tolist()],
        "client_target_attack_counts": [int(x) for x in pos_counts.tolist()],
        "client_target_benign_counts": [int(x) for x in neg_counts.tolist()],
    }
    return rsu_partitions, meta


def _reviewer3_measure_and_write_partition_v1(
    rsu_partitions: List[List[Tuple[pd.DataFrame, pd.Series]]],
    *,
    output_dir: str,
    scenario: str,
    seed: int,
    heterogeneity_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    meta = dict(heterogeneity_meta or {})

    # Use the same selected feature when heterogeneous; otherwise choose the
    # highest-variance feature from the concatenated client samples (sampled).
    feat_name = str(meta.get("feature_skew_feature_name", "") or "")
    if not feat_name:
        first_X = rsu_partitions[0][0][0]
        feat_j = 0
        feat_name = str(first_X.columns[feat_j])
    else:
        first_X = rsu_partitions[0][0][0]
        feat_loc = first_X.columns.get_loc(feat_name)
        if not isinstance(feat_loc, (int, np.integer)):
            raise ValueError(
                f"Feature name {feat_name!r} does not resolve to one unique column"
            )
        feat_j = int(feat_loc)

    all_feature_parts = []
    rows: List[Dict[str, Any]] = []
    for rsu_idx, vehicles in enumerate(rsu_partitions, start=1):
        for local_idx, (X_v, y_v) in enumerate(vehicles, start=1):
            vals = np.asarray(
                X_v.iloc[:, feat_j].to_numpy(copy=False),
                dtype=np.float64,
            )
            valid_vals = vals[~np.isnan(vals)]
            all_feature_parts.append(vals)
            yarr = y_v.to_numpy(dtype=np.int32, copy=False)
            n = int(len(yarr))
            pos = int(np.sum(yarr == 1))
            neg = int(np.sum(yarr == 0))
            rows.append({
                "rsu_id": int(rsu_idx),
                "vehicle_id": int(rsu_idx * 1000 + local_idx),
                "local_vehicle_index": int(local_idx),
                "num_examples": n,
                "attack_count": pos,
                "benign_count": neg,
                "attack_fraction": float(pos / n if n else 0.0),
                "feature_name": str(feat_name),
                "feature_mean": float(valid_vals.mean()) if valid_vals.size else float("nan"),
                "feature_std": float(valid_vals.std()) if valid_vals.size else float("nan"),
                "feature_min": float(valid_vals.min()) if valid_vals.size else float("nan"),
                "feature_max": float(valid_vals.max()) if valid_vals.size else float("nan"),
            })

    all_vals = np.concatenate(all_feature_parts) if all_feature_parts else np.asarray([], dtype=np.float64)
    global_mean = float(np.nanmean(all_vals)) if all_vals.size else 0.0
    global_std = float(np.nanstd(all_vals)) if all_vals.size else 0.0
    for row in rows:
        row["feature_standardized_mean_shift"] = (
            float((row["feature_mean"] - global_mean) / global_std) if global_std > 0 else 0.0
        )

    csv_path = os.path.join(output_dir, "reviewer3_client_partition_stats.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    sizes = np.asarray([r["num_examples"] for r in rows], dtype=np.float64)
    attack_fracs = np.asarray([r["attack_fraction"] for r in rows], dtype=np.float64)
    shifts = np.asarray([r["feature_standardized_mean_shift"] for r in rows], dtype=np.float64)
    summary = {
        "experiment_version": REVIEWER3_EXPERIMENT_VERSION,
        "scenario": str(scenario),
        "seed": int(seed),
        "num_clients": int(len(rows)),
        "feature_diagnostic": str(feat_name),
        "global_feature_mean": float(global_mean),
        "global_feature_std": float(global_std),
        "client_size_min": int(np.min(sizes)) if sizes.size else 0,
        "client_size_median": float(np.median(sizes)) if sizes.size else 0.0,
        "client_size_max": int(np.max(sizes)) if sizes.size else 0,
        "client_size_max_min_ratio": float(np.max(sizes) / np.min(sizes)) if sizes.size and np.min(sizes) > 0 else 0.0,
        "client_attack_fraction_min": float(np.min(attack_fracs)) if attack_fracs.size else 0.0,
        "client_attack_fraction_median": float(np.median(attack_fracs)) if attack_fracs.size else 0.0,
        "client_attack_fraction_max": float(np.max(attack_fracs)) if attack_fracs.size else 0.0,
        "client_feature_mean_shift_min_sd": float(np.min(shifts)) if shifts.size else 0.0,
        "client_feature_mean_shift_max_sd": float(np.max(shifts)) if shifts.size else 0.0,
        "heterogeneity_meta": meta,
        "per_client_csv": os.path.basename(csv_path),
    }
    json_path = os.path.join(output_dir, "reviewer3_partition_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _reviewer3_write_experiment_summary_v1(
    *,
    output_dir: str,
    scenario: str,
    seed: int,
    num_rsus: int,
    vehicles_per_rsu: int,
    num_rounds: int,
    stress_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Create one compact reviewer-facing result bundle from native artifacts."""
    partition_summary = {}
    ppath = os.path.join(output_dir, "reviewer3_partition_summary.json")
    if os.path.exists(ppath):
        with open(ppath, "r", encoding="utf-8") as f:
            partition_summary = json.load(f)

    participation_rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    total_fit_success = 0
    total_fit_failures = 0
    for rsu_id in range(1, int(num_rsus) + 1):
        spath = os.path.join(output_dir, f"iov_rsu_{rsu_id}_summary.json")
        if not os.path.exists(spath):
            continue
        with open(spath, "r", encoding="utf-8") as f:
            sobj = json.load(f)
        for r in (sobj.get("round_summaries", []) or []):
            success = int(float(r.get("fit_results_received", r.get("fit_num_clients", 0)) or 0))
            failed = int(float(r.get("fit_failures_observed", 0) or 0))
            total_fit_success += success
            total_fit_failures += failed
            participation_rows.append({
                "rsu_id": int(rsu_id),
                "round": int(r.get("round", 0) or 0),
                "expected_clients": int(r.get("fit_expected_clients", vehicles_per_rsu) or vehicles_per_rsu),
                "successful_fit_results": success,
                "fit_failures": failed,
                "success_fraction": float(r.get("fit_success_fraction", 0.0) or 0.0),
                "admitted_clients": int(
                    r.get("rsu_anchor_N_used", 0) or 0
                ),
                "evidence_exclusions": int(
                    len(
                        r.get(
                            "preselect_exclusion_details",
                            [],
                        )
                        or []
                    )
                ),
                "full_evidence_materialized": int(
                    r.get(
                        "reviewer3_full_evidence_materialized",
                        0,
                    )
                    or 0
                ),
                "rsu_anchor_zkp_ok": int(
                    r.get("rsu_anchor_zkp_ok", 0) or 0
                ),
            })
            for e in (r.get("preselect_exclusion_details", []) or []):
                if not isinstance(e, dict):
                    continue

                ee: Dict[str, Any] = {str(k): v for k, v in e.items()}
                ee["rsu_id"] = int(rsu_id)
                ee["round"] = int(r.get("round", 0) or 0)
                exclusions.append(ee)

    participation_csv = os.path.join(output_dir, "reviewer3_round_participation.csv")
    pd.DataFrame(participation_rows).to_csv(participation_csv, index=False)
    exclusions_csv = os.path.join(output_dir, "reviewer3_context_exclusions.csv")
    pd.DataFrame(exclusions).to_csv(exclusions_csv, index=False)

    global_summary = {}
    gpath = os.path.join(output_dir, "global_ensemble_summary.json")
    if os.path.exists(gpath):
        with open(gpath, "r", encoding="utf-8") as f:
            global_summary = json.load(f)
    ensemble_metrics = dict(global_summary.get("ensemble_metrics", {}) or {})

    stale_rejections = sum(1 for e in exclusions if str(e.get("classification", "")) == "stale_round_context")
    cross_rsu_rejections = sum(1 for e in exclusions if str(e.get("classification", "")) == "cross_rsu_context")
    challenge_targets = list(stress_cfg.get("context_binding_targets", []) or [])
    requested_stale = sum(1 for t in challenge_targets if str(t.get("mode", "")).lower() == "stale_round")
    requested_cross = sum(1 for t in challenge_targets if str(t.get("mode", "")).lower() == "cross_rsu_scope")

    summary = {
        "experiment_version": REVIEWER3_EXPERIMENT_VERSION,
        "scenario": str(scenario),
        "seed": int(seed),
        "purpose": (
            "Controlled federated data-heterogeneity, quantity-heterogeneity, "
            "unstable-participation, intermittent-availability, stale-round "
            "delayed-arrival, and cross-RSU context stress evaluation; "
            "not a packet-level V2X mobility or seamless-handoff simulation."
        ),
        "topology": {
            "num_rsus": int(num_rsus),
            "vehicles_per_rsu": int(vehicles_per_rsu),
            "total_vehicles": int(num_rsus * vehicles_per_rsu),
            "rounds": int(num_rounds),
        },
        "partition_summary": partition_summary,
        "participation": {
            "successful_fit_results_total": int(total_fit_success),
            "fit_failures_total": int(total_fit_failures),
            "successful_fit_fraction": float(total_fit_success / (total_fit_success + total_fit_failures)) if (total_fit_success + total_fit_failures) else 0.0,
            "round_csv": os.path.basename(participation_csv),
        },
        "context_binding_challenges": {
            "targets": challenge_targets,
            "requested_stale_round_challenges": int(requested_stale),
            "observed_stale_round_rejections": int(stale_rejections),
            "requested_cross_rsu_challenges": int(requested_cross),
            "observed_cross_rsu_rejections": int(cross_rsu_rejections),
            "exclusions_csv": os.path.basename(exclusions_csv),
        },
        "global_ensemble_metrics": ensemble_metrics,
        "global_confusion_matrix": global_summary.get("ensemble_confusion_matrix", []),
        "zkp_global_anchor_ok": bool(global_summary.get("zkp_global_anchor_ok", False)),
        "used_rsu_ids": global_summary.get("used_rsu_ids", []),
        "stress_configuration": dict(stress_cfg or {}),
    }
    out_path = os.path.join(output_dir, "REVIEWER3_EXPERIMENT_SUMMARY.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("[REVIEWER3] Saved experiment summary to %s", out_path)
    return summary

def _reviewer3_validate_completed_run_v1(
    *,
    output_dir: str,
    scenario: str,
    summary: Dict[str, Any],
    num_rsus: int,
    vehicles_per_rsu: int,
    num_rounds: int,
    commitment_report: Dict[str, Any],
    public_input_report: Dict[str, Any],
    layered_overhead_report: Dict[str, Any],
) -> None:
    errors: List[str] = []

    # ------------------------------------------------------------
    # Final predictive outputs
    # ------------------------------------------------------------
    metrics = dict(
        summary.get("global_ensemble_metrics", {}) or {}
    )

    for name in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    ):
        if _reviewer3_optional_float_v1(
            metrics.get(name)
        ) is None:
            errors.append(
                f"missing/non-numeric global metric: {name}"
            )

    cm = summary.get("global_confusion_matrix", [])

    if not (
        isinstance(cm, list)
        and len(cm) == 2
        and all(
            isinstance(row, list) and len(row) == 2
            for row in cm
        )
    ):
        errors.append(
            "global confusion matrix is not 2x2"
        )

    # ------------------------------------------------------------
    # Final GLOBAL proof
    # ------------------------------------------------------------
    if not bool(
        summary.get("zkp_global_anchor_ok", False)
    ):
        errors.append(
            "GLOBAL AnchorSum proof did not verify"
        )

    expected_rsus = list(
        range(1, int(num_rsus) + 1)
    )

    used_rsus = [
        int(x)
        for x in (
            summary.get("used_rsu_ids", []) or []
        )
    ]

    if used_rsus != expected_rsus:
        errors.append(
            f"global ensemble used_rsu_ids={used_rsus}, "
            f"expected={expected_rsus}"
        )

    global_summary_path = os.path.join(
        output_dir,
        "global_ensemble_summary.json",
    )

    if os.path.isfile(global_summary_path):
        gobj = _read_json_v1(
            global_summary_path
        )

        zkp_used_rsus = [
            int(x)
            for x in (
                gobj.get("zkp_used_rsu_ids", []) or []
            )
        ]

        if zkp_used_rsus != expected_rsus:
            errors.append(
                f"GLOBAL proof used_rsu_ids={zkp_used_rsus}, "
                f"expected={expected_rsus}"
            )
    else:
        errors.append(
            "missing global_ensemble_summary.json"
        )

    # ------------------------------------------------------------
    # Independent artifact checks
    # ------------------------------------------------------------
    if str(
        commitment_report.get("result", "")
    ) != "pass":
        errors.append(
            "commitment recomputation report != pass"
        )

    if str(
        public_input_report.get("result", "")
    ) != "pass":
        errors.append(
            "public-input agreement report != pass"
        )

    if str(
        layered_overhead_report.get("result", "")
    ) != "pass":
        errors.append(
            "layered-overhead report != pass"
        )

    # ------------------------------------------------------------
    # Scenario-specific expected participation
    # ------------------------------------------------------------
    if scenario == "combined_stress":
        expected_success = {
            1: 10,
            2: 9,
            3: 9,
            4: 9,
            5: 9,
            6: 8,
            7: 8,
            8: 9,
        }

        expected_admitted = {
            1: 10,
            2: 9,
            3: 9,
            4: 9,
            5: 9,
            6: 7,
            7: 8,
            8: 9,
        }

        expected_full_evidence = {
            6,
            int(num_rounds),
        }

    else:
        expected_success = {
            r: int(vehicles_per_rsu)
            for r in range(
                1,
                int(num_rounds) + 1,
            )
        }

        expected_admitted = dict(
            expected_success
        )

        expected_full_evidence = {
            int(num_rounds)
        }

    # ------------------------------------------------------------
    # Check every RSU and every round
    # ------------------------------------------------------------
    for rsu_id in range(
        1,
        int(num_rsus) + 1,
    ):
        spath = os.path.join(
            output_dir,
            f"iov_rsu_{int(rsu_id)}_summary.json",
        )

        if not os.path.isfile(spath):
            errors.append(
                f"missing RSU summary: rsu_id={rsu_id}"
            )
            continue

        sobj = _read_json_v1(spath)

        by_round = {
            int(r.get("round", 0) or 0): r
            for r in (
                sobj.get("round_summaries", []) or []
            )
            if isinstance(r, dict)
        }

        for rnd in range(
            1,
            int(num_rounds) + 1,
        ):
            ent = by_round.get(rnd)

            if ent is None:
                errors.append(
                    f"RSU {rsu_id}: "
                    f"missing round {rnd} summary"
                )
                continue

            success = int(
                ent.get(
                    "fit_results_received",
                    0,
                )
                or 0
            )

            failed = int(
                ent.get(
                    "fit_failures_observed",
                    0,
                )
                or 0
            )

            admitted = int(
                ent.get(
                    "rsu_anchor_N_used",
                    0,
                )
                or 0
            )

            full = int(
                ent.get(
                    "reviewer3_full_evidence_materialized",
                    0,
                )
                or 0
            )

            zkp_ok = int(
                ent.get(
                    "rsu_anchor_zkp_ok",
                    0,
                )
                or 0
            )

            expected_success_r = int(
                expected_success[rnd]
            )

            expected_failed_r = (
                int(vehicles_per_rsu)
                - expected_success_r
            )

            expected_admitted_r = int(
                expected_admitted[rnd]
            )

            expected_full_r = (
                1
                if rnd in expected_full_evidence
                else 0
            )

            if success != expected_success_r:
                errors.append(
                    f"RSU {rsu_id} round {rnd}: "
                    f"fit_results_received={success}, "
                    f"expected={expected_success_r}"
                )

            if failed != expected_failed_r:
                errors.append(
                    f"RSU {rsu_id} round {rnd}: "
                    f"fit_failures_observed={failed}, "
                    f"expected={expected_failed_r}"
                )

            if admitted != expected_admitted_r:
                errors.append(
                    f"RSU {rsu_id} round {rnd}: "
                    f"admitted={admitted}, "
                    f"expected={expected_admitted_r}"
                )

            if full != expected_full_r:
                errors.append(
                    f"RSU {rsu_id} round {rnd}: "
                    f"full_evidence_materialized={full}, "
                    f"expected={expected_full_r}"
                )

            if (
                expected_full_r == 1
                and zkp_ok != 1
            ):
                errors.append(
                    f"RSU {rsu_id} round {rnd}: "
                    "materialized AnchorSum proof "
                    "did not verify"
                )

    # ------------------------------------------------------------
    # Run-level participation / context totals
    # ------------------------------------------------------------
    participation = dict(
        summary.get("participation", {}) or {}
    )

    context = dict(
        summary.get(
            "context_binding_challenges",
            {},
        )
        or {}
    )

    if scenario == "combined_stress":
        if int(
            participation.get(
                "successful_fit_results_total",
                -1,
            )
        ) != 142:
            errors.append(
                "combined_stress "
                "successful_fit_results_total != 142"
            )

        if int(
            participation.get(
                "fit_failures_total",
                -1,
            )
        ) != 18:
            errors.append(
                "combined_stress "
                "fit_failures_total != 18"
            )

        success_fraction = (
            _reviewer3_optional_float_v1(
                participation.get(
                    "successful_fit_fraction"
                )
            )
        )

        if (
            success_fraction is None
            or not math.isclose(
                success_fraction,
                0.8875,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            errors.append(
                "combined_stress "
                "successful_fit_fraction != 0.8875"
            )

        if int(
            context.get(
                "observed_stale_round_rejections",
                -1,
            )
        ) != 1:
            errors.append(
                "combined_stress "
                "observed_stale_round_rejections != 1"
            )

        if int(
            context.get(
                "observed_cross_rsu_rejections",
                -1,
            )
        ) != 1:
            errors.append(
                "combined_stress "
                "observed_cross_rsu_rejections != 1"
            )

    else:
        expected_total = (
            int(num_rsus)
            * int(vehicles_per_rsu)
            * int(num_rounds)
        )

        if int(
            participation.get(
                "successful_fit_results_total",
                -1,
            )
        ) != expected_total:
            errors.append(
                f"{scenario} "
                "successful_fit_results_total "
                f"!= {expected_total}"
            )

        if int(
            participation.get(
                "fit_failures_total",
                -1,
            )
        ) != 0:
            errors.append(
                f"{scenario} "
                "fit_failures_total != 0"
            )

        if int(
            context.get(
                "observed_stale_round_rejections",
                0,
            )
        ) != 0:
            errors.append(
                f"{scenario} unexpectedly observed "
                "stale-round rejection"
            )

        if int(
            context.get(
                "observed_cross_rsu_rejections",
                0,
            )
        ) != 0:
            errors.append(
                f"{scenario} unexpectedly observed "
                "cross-RSU rejection"
            )

    # ------------------------------------------------------------
    # Fail hard and remove completion marker
    # ------------------------------------------------------------
    if errors:
        completion_marker = os.path.join(
            output_dir,
            "REVIEWER3_EXPERIMENT_SUMMARY.json",
        )

        try:
            if os.path.exists(completion_marker):
                os.remove(completion_marker)
        except OSError:
            pass

        raise RuntimeError(
            "[REVIEWER3][VALIDATION] FAILED:\n - "
            + "\n - ".join(errors)
        )

    logging.info(
        "[REVIEWER3][VALIDATION] PASS | scenario=%s",
        scenario,
    )

def _reviewer3_optional_float_v1(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None

def _reviewer3_write_cross_scenario_comparison_v1(project_root: str, *, seed: int) -> None:
    """If scenario outputs exist, write a compact latest-per-scenario comparison."""
    results_root = os.path.join(str(project_root), "reviewer_concern3_results")
    if not os.path.isdir(results_root):
        return
    scenarios = ["balanced_reference", "data_heterogeneity", "combined_stress"]
    latest: Dict[str, tuple[str, Dict[str, Any]]] = {}
    for name in os.listdir(results_root):
        run_dir = os.path.join(results_root, name)
        if not os.path.isdir(run_dir):
            continue
        spath = os.path.join(run_dir, "REVIEWER3_EXPERIMENT_SUMMARY.json")
        if not os.path.isfile(spath):
            continue
        try:
            with open(spath, "r", encoding="utf-8") as f:
                obj = json.load(f)
            scenario = str(obj.get("scenario", "") or "")
            if scenario not in scenarios or int(obj.get("seed", -1)) != int(seed):
                continue
            if scenario not in latest or name > latest[scenario][0]:
                latest[scenario] = (name, obj)
        except Exception:
            continue

    rows: List[Dict[str, Any]] = []
    baseline_f1 = None
    baseline_recall = None
    if "balanced_reference" in latest:
        bm = dict(latest["balanced_reference"][1].get("global_ensemble_metrics", {}) or {})
        baseline_f1 = _reviewer3_optional_float_v1(bm.get("f1"))
        baseline_recall = _reviewer3_optional_float_v1(bm.get("recall"))

    for scenario in scenarios:
        if scenario not in latest:
            continue
        run_name, obj = latest[scenario]
        pm = dict(obj.get("partition_summary", {}) or {})
        part = dict(obj.get("participation", {}) or {})
        cm = dict(obj.get("context_binding_challenges", {}) or {})
        met = dict(obj.get("global_ensemble_metrics", {}) or {})
        f1 = _reviewer3_optional_float_v1(met.get("f1"))
        recall = _reviewer3_optional_float_v1(met.get("recall"))
        rows.append({
            "scenario": scenario,
            "run_folder": run_name,
            "accuracy": met.get("accuracy", ""),
            "precision": met.get("precision", ""),
            "recall": recall if recall is not None else "",
            "f1": f1 if f1 is not None else "",
            "auc": met.get("auc", ""),
            "log_loss": met.get(
                "logloss",
                met.get("log_loss", ""),
            ),
            "delta_f1_vs_balanced":
                (f1 - baseline_f1)
                if f1 is not None and baseline_f1 is not None else "",
            "delta_recall_vs_balanced":
                (recall - baseline_recall)
                if recall is not None and baseline_recall is not None else "",
            "client_size_ratio": pm.get("client_size_max_min_ratio", ""),
            "attack_fraction_min": pm.get("client_attack_fraction_min", ""),
            "attack_fraction_max": pm.get("client_attack_fraction_max", ""),
            "feature_shift_min_sd": pm.get("client_feature_mean_shift_min_sd", ""),
            "feature_shift_max_sd": pm.get("client_feature_mean_shift_max_sd", ""),
            "fit_success_fraction": part.get("successful_fit_fraction", ""),
            "fit_failures_total": part.get("fit_failures_total", ""),
            "stale_round_rejections": cm.get("observed_stale_round_rejections", ""),
            "cross_rsu_rejections": cm.get("observed_cross_rsu_rejections", ""),
            "global_zkp_ok": obj.get("zkp_global_anchor_ok", False),
        })

    if not rows:
        return
    csv_path = os.path.join(results_root, "REVIEWER3_CROSS_SCENARIO_COMPARISON.csv")
    json_path = os.path.join(results_root, "REVIEWER3_CROSS_SCENARIO_COMPARISON.json")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"seed": int(seed), "latest_per_scenario": rows}, f, indent=2)
    logging.info("[REVIEWER3] Cross-scenario comparison written to %s", csv_path)


def _sha256_file_hex_top_v1(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _write_json_pretty_v1(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _resolve_output_path_v1(output_dir: str, path_value: str) -> str:
    p = str(path_value or "").strip()
    if not p:
        return ""
    return p if os.path.isabs(p) else os.path.join(output_dir, p)

def _read_json_v1(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _find_round_summary_v1(rsu_summary: Dict[str, Any], round_idx: int) -> Dict[str, Any]:
    for ent in (rsu_summary.get("round_summaries", []) or []):
        try:
            if int(ent.get("round", -1)) == int(round_idx):
                return ent
        except Exception:
            continue
    return {}

def _extract_public_inputs_from_json_v1(obj: Any) -> List[str]:
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        vals = _extract_public_inputs_v15(obj)
        if isinstance(vals, list):
            return [str(x) for x in vals]
    return []

def _spec_public_input_names_v1(spec_obj: Any) -> List[str]:
    for attr in ("public_inputs", "public_input_names", "public_input_order"):
        vals = getattr(spec_obj, attr, None)
        if isinstance(vals, (list, tuple)):
            return [str(x) for x in vals]
    return []

def _build_expected_public_inputs_v1(
    public_input_names: List[str],
    context_vals: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    expected: List[str] = []
    unresolved: List[str] = []

    for name in public_input_names:
        low = str(name).strip().lower()
        val: Any = None

        if low == "anchor_root_poseidon_field":
            val = context_vals.get("anchor_root_poseidon_field")
        elif low.endswith("root_poseidon_field") or low == "root_poseidon_field":
            if "anchor" in low:
                val = context_vals.get("anchor_root_poseidon_field")
            else:
                val = context_vals.get("root_poseidon_field")
        elif low in ("agg_commit_field", "rsu_commit_field"):
            val = context_vals.get("agg_commit_field")
        elif low == "global_commit_field":
            val = context_vals.get("global_commit_field")
        elif low == "commit_field":
            val = context_vals.get("commit_field")
        elif low == "pins_hash_field":
            val = context_vals.get("pins_hash_field")
        elif low == "policy_id_field":
            val = context_vals.get("policy_id_field")
        elif low == "public_input_order_id_field":
            val = context_vals.get("public_input_order_id_field")

        if val is None or str(val).strip() == "":
            expected.append("")
            unresolved.append(str(name))
        else:
            expected.append(str(val))

    return expected, unresolved

def _reviewer3_optional_int_v1(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None
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
            feature_min: np.ndarray,
            feature_max: np.ndarray,
            dp_output_dir: str,
            anchor_ctx: Dict[str, Any],
            ablation_cfg: Dict[str, Any] | None = None,
    ):
        self.vehicle_id = vehicle_id
        self.rsu_id = rsu_id
        self.xgb_cfg = xgb_cfg
        self.feature_count = feature_count
        self.dp_output_dir = dp_output_dir  # <-- NEW
        self.ablation_cfg = dict(ablation_cfg or {})
        # Global feature bounds (same ordering as columns in X_train)
        self.feature_min = feature_min
        self.feature_max = feature_max
        # FedXgbBagging uses num_parallel_tree to interpret tree groups
        self.num_parallel_tree = int(self.xgb_cfg.params.get("num_parallel_tree", 1))
        if self.num_parallel_tree < 1:
            self.num_parallel_tree = 1
        # Reviewer-3 stress run:
        # clients require only their local training shard.
        # Full validation/test evaluation is performed once at the final
        # RSU/GLOBAL evaluation stage, not redundantly inside every client.
        self.X_train = X_train
        self.y_train = y_train

        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.train_dmatrix = xgb.DMatrix(
            self.X_train,
            label=self.y_train,
            feature_min=self.feature_min,
            feature_max=self.feature_max,
        )

        self.val_dmatrix = None
        self.test_dmatrix = None
        # ---------------------------
        # AnchorZKP context (audit lane)
        # ---------------------------
        self.anchor_ctx = anchor_ctx
        self.anchor_enabled = bool(anchor_ctx)
        self.anchor_id_field = str(anchor_ctx.get("anchor_id_field", ""))
        self.anchor_version = str(anchor_ctx.get("anchor_version", ""))
        self.anchor_M = _as_int(anchor_ctx.get("M"), 0)
        self.anchor_SCALE = _as_int(anchor_ctx.get("SCALE"), 0)
        self.anchor_X_path = str(anchor_ctx.get("anchor_X_path", ""))
        self.logger = logging.getLogger(f"RSU{self.rsu_id}-Vehicle{self.vehicle_id}")
        # ---------------------------
        # SSI identity (Ed25519) — persisted so evidence remains stable across rounds/restarts
        # ---------------------------
        self.ssi_did: str = ""
        self.ssi_pubkey_b64: str = ""
        self._ssi_priv: Ed25519PrivateKey | None = None
        self._load_or_create_ssi_identity_v1()
        self.anchor_dmatrix = None
        try:
            if self.anchor_enabled and self.anchor_X_path and os.path.exists(self.anchor_X_path):
                X_anchor = np.load(self.anchor_X_path).astype(np.float32)
                # Safety: ensure shape matches M if provided
                if self.anchor_M > 0 and X_anchor.shape[0] != self.anchor_M:
                    self.logger.warning(
                        "[RSU %s | Vehicle %s] anchor_X rows=%d != M=%d (continuing)",
                        self.rsu_id,
                        self.vehicle_id,
                        X_anchor.shape[0],
                        self.anchor_M,
                    )
                self.anchor_dmatrix = xgb.DMatrix(
                    X_anchor,
                    feature_min=self.feature_min,
                    feature_max=self.feature_max,
                )
            else:
                self.anchor_enabled = False
        except Exception as exc:
            self.anchor_enabled = False
            self.anchor_dmatrix = None
            self.logger.warning(
                "[RSU %s | Vehicle %s] Failed to init anchor_dmatrix: %s",
                self.rsu_id,
                self.vehicle_id,
                exc,
            )
        # Initialize a minimal booster (safe, created in worker) using the same bounds
        self.model = initialize_booster_with_dummy_data(
            self.xgb_cfg.params,
            feature_count=self.feature_count,
            feature_min=self.feature_min,
            feature_max=self.feature_max,
        )
        # Track cumulative DP epsilon spent across all local rounds for this vehicle
        self.dp_epsilon_cumulative: float = 0.0
        # In-memory DP ledger entries, flushed to JSON every fit call
        self.dp_ledger: List[Dict[str, Any]] = []
        # Try to restore previous DP state from disk so that ε_cumulative
        # and the ledger are truly cumulative across FL rounds.
        self._load_dp_state_from_disk()
        # Ensure there is at least an empty DP ledger file for this vehicle
        try:
            ledger_path = self._get_ledger_path()
            if not os.path.exists(ledger_path):
                with open(ledger_path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)
                self.logger.info(
                    "Created new DP ledger file for vehicle %s at %s",
                    self.vehicle_id,
                    ledger_path,
                )
            else:
                self.logger.info(
                    "Using existing DP ledger file for vehicle %s at %s",
                    self.vehicle_id,
                    ledger_path,
                )
        except Exception as exc:
            self.logger.warning(
                "[RSU %s | Vehicle %s] Could not create empty DP ledger file: %s",
                self.rsu_id,
                self.vehicle_id,
                exc,
            )
        self.logger.info(
            f"Initialized Vehicle {self.vehicle_id} on RSU {self.rsu_id} "
            f"with {self.train_dmatrix.num_row()} local samples "
            f"(starting ε_cumulative≈{self.dp_epsilon_cumulative:.4f}, "
            f"ledger_entries={len(self.dp_ledger)})"
        )
    def _get_ledger_path(self) -> str:
        """
        Return the filesystem path for this vehicle's DP ledger file,
        creating the dp_ledgers directory if needed.
        """
        ledger_dir = os.path.join(self.dp_output_dir, "dp_ledgers")
        os.makedirs(ledger_dir, exist_ok=True)
        return os.path.join(ledger_dir, f"dp_ledger_rsu_{self.rsu_id}_vehicle_{self.vehicle_id}.json")
    def _get_dp_record_path(self, round_num: int) -> str:
        """
        Return the filesystem path for this vehicle's canonical DP record bytes file for this round.
        The bytes written here must be exactly the bytes used to compute dp_record_sha256.
        """
        rec_dir = os.path.join(
            self.dp_output_dir,
            "dp_records",
            f"rsu_{int(self.rsu_id)}",
            f"vehicle_{int(self.vehicle_id)}",
        )
        os.makedirs(rec_dir, exist_ok=True)
        return os.path.join(rec_dir, f"round_{int(round_num)}.json")
    def _get_ssi_key_path(self) -> str:
        ssi_dir = os.path.join(
            self.dp_output_dir,
            "ssi_keys",
            f"rsu_{int(self.rsu_id)}",
        )
        os.makedirs(ssi_dir, exist_ok=True)
        return os.path.join(ssi_dir, f"vehicle_{int(self.vehicle_id)}.json")
    def _load_or_create_ssi_identity_v1(self) -> None:
        """
        Create or load a persistent Ed25519 identity for this vehicle.
        Evidence is recorded in metrics each round and verified at RSU.
        """
        key_path = self._get_ssi_key_path()
        try:
            if os.path.exists(key_path):
                with open(key_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                priv_b64 = str(obj.get("ed25519_priv_b64", "") or "")
                pub_b64 = str(obj.get("ed25519_pub_b64", "") or "")
                priv_raw = _b64d(priv_b64)
                pub_raw = _b64d(pub_b64)
                self._ssi_priv = Ed25519PrivateKey.from_private_bytes(priv_raw)
                # sanity: public key must match
                pub2 = self._ssi_priv.public_key().public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
                if pub2 != pub_raw:
                    raise RuntimeError("SSI key mismatch: stored pubkey != derived pubkey")
                self.ssi_pubkey_b64 = pub_b64
                self.ssi_did = str(obj.get("did", "") or _did_from_pubkey_raw_ed25519(pub_raw))
                self.logger.info(
                    "[SSI][Vehicle %s] Loaded persistent SSI key | did=%s | key_path=%s",
                    self.vehicle_id,
                    self.ssi_did,
                    key_path,
                )
                return
            # create new persistent key
            priv = Ed25519PrivateKey.generate()
            priv_raw = priv.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            pub_raw = priv.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            did = _did_from_pubkey_raw_ed25519(pub_raw)
            obj = {
                "did": did,
                "ed25519_priv_b64": _b64e(priv_raw),
                "ed25519_pub_b64": _b64e(pub_raw),
            }
            with open(key_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
            self._ssi_priv = priv
            self.ssi_pubkey_b64 = str(obj["ed25519_pub_b64"])
            self.ssi_did = str(obj["did"])
            self.logger.info(
                "[SSI][Vehicle %s] Created persistent SSI key | did=%s | key_path=%s",
                self.vehicle_id,
                self.ssi_did,
                key_path,
            )
        except Exception as exc:
            # If identity cannot be created/loaded, fail hard (otherwise you cannot claim SSI verified)
            self.logger.error(
                "[SSI][Vehicle %s] Failed to load/create SSI identity: %s",
                self.vehicle_id,
                exc,
            )
            raise
    def _ssi_sign_report_v1(self, report_bytes: bytes) -> str:
        if self._ssi_priv is None:
            raise RuntimeError("SSI private key missing")
        sig = self._ssi_priv.sign(report_bytes)
        return _b64e(sig)
    def _load_dp_state_from_disk(self) -> None:
        """
        Load previous DP ledger and cumulative epsilon for this vehicle
        from <dp_output_dir>/dp_ledgers/dp_ledger_vehicle_<vehicle_id>.json, if it exists.
        This makes dp_epsilon_cumulative truly cumulative across FL rounds
        even if the Flower App runtime recreates VehicleClient instances.
        """
        try:
            ledger_path = self._get_ledger_path()
            if not os.path.exists(ledger_path):
                # No previous DP state for this vehicle
                return
            with open(ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.dp_ledger = data
                if self.dp_ledger:
                    last_entry = self.dp_ledger[-1]
                    if isinstance(last_entry, dict):
                        # Prefer scaled-int cumulative epsilon first (new schema)
                        if "epsilon_cumulative_u" in last_entry:
                            self.dp_epsilon_cumulative = float(int(last_entry.get("epsilon_cumulative_u", 0))) / 1_000_000.0
                        # Backward compatibility: older schema stored float
                        elif "epsilon_cumulative" in last_entry:
                            self.dp_epsilon_cumulative = float(last_entry.get("epsilon_cumulative", 0.0))
                        else:
                            # Fallback: sum dp_epsilon_used across entries (supports both schemas)
                            total_u = 0
                            total_f = 0.0
                            for e in self.dp_ledger:
                                if not isinstance(e, dict):
                                    continue
                                if "dp_epsilon_used_u" in e:
                                    try:
                                        total_u += int(e.get("dp_epsilon_used_u", 0))
                                    except Exception:
                                        pass
                                elif "dp_epsilon_used" in e:
                                    try:
                                        total_f += float(e.get("dp_epsilon_used", 0.0))
                                    except Exception:
                                        pass
                            if total_u > 0:
                                self.dp_epsilon_cumulative = float(total_u) / 1_000_000.0
                            else:
                                self.dp_epsilon_cumulative = float(total_f)
            self.logger.info(
                "[RSU %s | Vehicle %s] Loaded existing DP ledger from %s "
                "(entries=%d, epsilon_cumulative≈%.4f)",
                self.rsu_id,
                self.vehicle_id,
                ledger_path,
                len(self.dp_ledger),
                self.dp_epsilon_cumulative,
            )
        except Exception as exc:
            # If anything goes wrong, start with a clean slate but do not crash.
            self.logger.warning(
                "[RSU %s | Vehicle %s] Failed to load existing DP ledger: %s "
                "-- starting with fresh DP state.",
                self.rsu_id,
                self.vehicle_id,
                exc,
            )
            self.dp_ledger = []
            self.dp_epsilon_cumulative = 0.0
    # ----- Parameter exchange -----
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Send current XGBoost model encoded as JSON bytes.
        If self.model is None (e.g., if we later start from scratch for dp_xgboost),
        we send empty parameters and let the strategy handle it.
        """
        try:
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
            reviewer3_unavailable, reviewer3_unavailability_reason = _reviewer3_is_unavailable_v1(
                self.ablation_cfg,
                rsu_id=int(self.rsu_id),
                vehicle_id=int(self.vehicle_id),
                round_idx=int(round_num),
            )
            if reviewer3_unavailable:
                self.logger.warning(
                    "[REVIEWER3][UNAVAILABLE] RSU=%d vehicle=%d round=%d reason=%s",
                    int(self.rsu_id), int(self.vehicle_id), int(round_num),
                    str(reviewer3_unavailability_reason),
                )
                raise RuntimeError(
                    f"REVIEWER3_SIMULATED_UNAVAILABILITY:rsu={int(self.rsu_id)}:"
                    f"vehicle={int(self.vehicle_id)}:round={int(round_num)}:"
                    f"reason={reviewer3_unavailability_reason}"
                )
            self.logger.info(
                f"Vehicle {self.vehicle_id} on RSU {self.rsu_id}: "
                f"starting local training for round {round_num}"
            )
            fit_wall_t0 = time.perf_counter()
            set_parameters_time_sec = 0.0
            local_train_time_sec = 0.0
            dp_record_time_sec = 0.0
            delta_build_time_sec = 0.0
            model_serialize_time_sec = 0.0
            anchor_payload_time_sec = 0.0
            ssi_report_time_sec = 0.0

            # --- NEW: measure download size of incoming global model ---
            download_bytes = 0
            try:
                if ins.parameters is not None and ins.parameters.tensors:
                    for t in ins.parameters.tensors:
                        if isinstance(t, (bytes, bytearray)):
                            download_bytes += len(t)
            except Exception as exc:
                self.logger.warning(
                    "[RSU %s | Vehicle %s | Round %s] "
                    "Failed to measure download_bytes: %s",
                    self.rsu_id,
                    self.vehicle_id,
                    round_num,
                    exc,
                )

            # Load current global model if present
            _t_set0 = time.perf_counter()
            self.set_parameters(ins.parameters)
            set_parameters_time_sec = time.perf_counter() - _t_set0

            # Remember how many boosting rounds the incoming global model has.
            # We will use this to slice out only the newly grown trees.
            try:
                prev_num_rounds = (
                    self.model.num_boosted_rounds() if self.model is not None else 0
                )
            except Exception:
                prev_num_rounds = 0

            _t_train0 = time.perf_counter()

            # Reviewer-3 stress run:
            # training behavior is preserved at the configured 10 local
            # boosting rounds, but the shared 450k-row validation split is not
            # redundantly evaluated inside every client and every boosting step.
            evals = [(self.train_dmatrix, "train")]

            self.model = xgb.train(
                self.xgb_cfg.params,
                self.train_dmatrix,
                num_boost_round=self.xgb_cfg.num_local_rounds,
                xgb_model=self.model,
                evals=evals,
            )
            local_train_time_sec = time.perf_counter() - _t_train0
            train_time = local_train_time_sec
            self.logger.info(
                "[RSU %s | Vehicle %s | Round %s] Local DP-XGBoost training time = %.3f sec",
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
            # Approximate local DP budget spent in this fit call, using
            # the DP-XGBoost formula: n * log(1 + γ (e^{ε_tree} - 1)).
            dp_eps_tree = float(self.xgb_cfg.params.get("dp_epsilon_per_tree", 0.0))
            subsample = float(self.xgb_cfg.params.get("subsample", 1.0))
            if dp_eps_tree > 0.0 and subsample > 0.0 and num_new_trees > 0:
                eps_factor = math.log(1.0 + subsample * (math.exp(dp_eps_tree) - 1.0))
                dp_epsilon_used = num_new_trees * eps_factor
            else:
                dp_epsilon_used = 0.0
            if dp_epsilon_used > 0.0:
                # Update cumulative DP budget for this vehicle
                self.dp_epsilon_cumulative += dp_epsilon_used
                self.logger.info(
                    "[RSU %s | Vehicle %s | Round %s] "
                    "Approx. local DP ε spent this round ≈ %.4f "
                    "(trees=%d, ε_tree=%.4f, subsample=%.3f) | cumulative ε≈%.4f",
                    self.rsu_id,
                    self.vehicle_id,
                    round_num,
                    dp_epsilon_used,
                    num_new_trees,
                    dp_eps_tree,
                    subsample,
                    self.dp_epsilon_cumulative,
                )
            else:
                self.logger.info(
                    "[RSU %s | Vehicle %s | Round %s] "
                    "DP-XGBoost parameters indicate no DP budget consumed "
                    "(dp_epsilon_per_tree<=0 or subsample<=0 or no new trees). "
                    "Cumulative ε≈%.4f",
                    self.rsu_id,
                    self.vehicle_id,
                    round_num,
                    self.dp_epsilon_cumulative,
                )
            # ---- DP boundary (MANDATORY): build canonical DP record bytes + sha, persist exact bytes, then log ledger ----
            dp_round_json_sha256 = ""
            dp_record_path = ""
            _t_dp0 = time.perf_counter()
            try:
                # Extra payload bound into the DP record as a deterministic string (no floats)
                extra_payload = {
                    "rsu_id": int(self.rsu_id),
                    "vehicle_id": int(self.vehicle_id),
                    "num_new_trees": int(num_new_trees),
                    "dp_epsilon_per_tree_u": int(round(float(dp_eps_tree) * 1_000_000)),
                    "subsample_u": int(round(float(subsample) * 1_000_000)),
                    "epsilon_cumulative_u": int(round(float(self.dp_epsilon_cumulative) * 1_000_000)),
                }
                extra_str = canon.canon_json_bytes_v1(extra_payload).decode("utf-8")
                # V15: build_dp_round_record_and_sha256_v1 expects int-like fields (no floats)
                clip_l1_int = int(1_000_000_000)  # >0; effectively “no per-example L1 clip”
                epsilon_u_int = int(round(float(dp_epsilon_used) * 1_000_000))  # micro-epsilon
                delta_u_int = int(0)
                dp_record_bytes, dp_round_json_sha256 = build_dp_round_record_and_sha256_v1(
                    round_idx=int(round_num),
                    mechanism="dp_xgboost_tree_budget_v1",
                    clip_l1=int(clip_l1_int),
                    epsilon=int(epsilon_u_int),
                    delta=int(delta_u_int),
                    accountant="tree_budget_v1",
                    extra=extra_str,
                )
                # Fail fast if anything produces an invalid DP hash (prevents “empty merkle” cascades)
                if (not isinstance(dp_round_json_sha256, str)) or (len(dp_round_json_sha256) != 64):
                    raise RuntimeError(f"[DP] Invalid dp_round_json_sha256='{dp_round_json_sha256}'")
                dp_record_path = self._get_dp_record_path(int(round_num))
                write_dp_record_bytes_v1(dp_record_path, dp_record_bytes=dp_record_bytes)
                # Self-check: sha must match recomputation from bytes
                if dp_record_sha256_from_bytes_v1(dp_record_bytes) != dp_round_json_sha256:
                    raise RuntimeError("dp_record_sha256_from_bytes_v1 mismatch")
                dp_record_time_sec = time.perf_counter() - _t_dp0
            except Exception as exc_dp:
                self.logger.error(
                    "[RSU %s | Vehicle %s | Round %s] DP record boundary FAILED (cannot proceed): %s",
                    self.rsu_id,
                    self.vehicle_id,
                    round_num,
                    exc_dp,
                )
                # On-chain lane requires a valid dp_round_json_sha256, so this must stop the round.
                raise RuntimeError(
                    f"[DP] Mandatory DP record boundary failed for RSU={int(self.rsu_id)} "
                    f"Vehicle={int(self.vehicle_id)} Round={int(round_num)}"
                ) from exc_dp
            # Persist a DP ledger entry (audit-only; not used as canonical bytes)
            try:
                self.dp_ledger.append(
                    {
                        "round": int(round_num),
                        "dp_record_sha256": str(dp_round_json_sha256),
                        "dp_record_path": str(dp_record_path),
                        "num_new_trees": int(num_new_trees),
                        "dp_epsilon_used_u": int(round(float(dp_epsilon_used) * 1_000_000)),
                        "epsilon_cumulative_u": int(round(float(self.dp_epsilon_cumulative) * 1_000_000)),
                        "dp_epsilon_per_tree_u": int(round(float(dp_eps_tree) * 1_000_000)),
                        "subsample_u": int(round(float(subsample) * 1_000_000)),
                    }
                )
                ledger_path = self._get_ledger_path()
                with open(ledger_path, "w", encoding="utf-8") as f:
                    json.dump(self.dp_ledger, f, indent=2)
                self.logger.info(
                    "[RSU %s | Vehicle %s | Round %s] Updated DP ledger at %s (entries=%d) | dp_record_sha256=%s",
                    self.rsu_id,
                    self.vehicle_id,
                    round_num,
                    ledger_path,
                    len(self.dp_ledger),
                    str(dp_round_json_sha256),
                )
            except Exception as exc_ledger:
                self.logger.warning(
                    "[RSU %s | Vehicle %s | Round %s] Failed to update DP ledger: %s",
                    self.rsu_id,
                    self.vehicle_id,
                    round_num,
                    exc_ledger,
                )
            # Build a delta booster containing only the newly added trees
            _t_delta0 = time.perf_counter()
            delta_booster = make_delta_booster(
                booster=self.model,
                prev_num_rounds=prev_num_rounds,
            )
            delta_build_time_sec = time.perf_counter() - _t_delta0
            # Early stopping diagnostics
            # ---------------------------
            es_enabled = (
                    self.xgb_cfg.early_stopping_rounds is not None
                    and self.xgb_cfg.early_stopping_rounds > 0
            )
            metric_name = self.xgb_cfg.params.get("eval_metric", "auc")
            best_iteration_raw = getattr(self.model, "best_iteration", None)
            best_iteration: int | None = _reviewer3_optional_int_v1(
                best_iteration_raw
            )
            best_score_raw = getattr(self.model, "best_score", None)
            best_score_opt = _reviewer3_optional_float_v1(best_score_raw)
            best_score: float = (
                best_score_opt if best_score_opt is not None else -1.0
            )
            stopped_early = False
            reason = "Early stopping disabled"
            if es_enabled and best_iteration is not None:
                # XGBoost uses 0-based iteration; +1 = number of trees
                stopped_early = (best_iteration + 1) < self.xgb_cfg.num_local_rounds
                if stopped_early:
                    reason = (
                        f"No improvement in eval_metric '{metric_name}' on validation "
                        f"set for {self.xgb_cfg.early_stopping_rounds} rounds"
                    )
                else:
                    reason = (
                        "Reached num_boost_round without triggering early stopping "
                        f"(best_iteration={best_iteration + 1})"
                    )
            if es_enabled:
                if stopped_early:
                    self.logger.info(
                        f"[RSU {self.rsu_id} | Vehicle {self.vehicle_id} | Round {round_num}] "
                        f"Early stopping TRIGGERED at iteration {best_iteration + 1}/"
                        f"{self.xgb_cfg.num_local_rounds}, best {metric_name}={best_score:.6f}"
                    )
                else:
                    self.logger.info(
                        f"[RSU {self.rsu_id} | Vehicle {self.vehicle_id} | Round {round_num}] "
                        f"Early stopping ENABLED but ran full {self.xgb_cfg.num_local_rounds} "
                        f"iterations, best {metric_name}={best_score:.6f}"
                    )
            else:
                self.logger.info(
                    f"[RSU {self.rsu_id} | Vehicle {self.vehicle_id} | Round {round_num}] "
                    "Early stopping DISABLED"
                )
            # Reviewer-3 stress run:
            # per-client full-dataset metric computation is intentionally
            # omitted. The reviewer-facing endpoint is the final held-out
            # RSU/GLOBAL evaluation, which remains unchanged.
            y_train_true: np.ndarray = np.asarray(
                self.train_dmatrix.get_label(), dtype=np.int32
            )

            self.logger.info(
                "[REVIEWER3][FIT-COMPLETE] RSU=%d vehicle=%d round=%d "
                "local_examples=%d train_time_sec=%.3f",
                int(self.rsu_id),
                int(self.vehicle_id),
                int(round_num),
                int(len(y_train_true)),
                float(train_time),
            )
            # Serialize only the delta booster as JSON bytes for FedXgbBagging
            _t_ser0 = time.perf_counter()
            model_bytes = booster_to_json_bytes(delta_booster)
            model_serialize_time_sec = time.perf_counter() - _t_ser0
            # --- NEW: measure upload size of outgoing delta model ---
            upload_bytes = 0
            try:
                if isinstance(model_bytes, (bytes, bytearray)):
                    upload_bytes = len(model_bytes)
            except Exception as exc:
                self.logger.warning(
                    "[RSU %s | Vehicle %s | Round %s] "
                    "Failed to measure upload_bytes: %s",
                    self.rsu_id,
                    self.vehicle_id,
                    round_num,
                    exc,
                )
            flwr_params = Parameters(
                tensor_type="xgboost_json",
                tensors=[model_bytes],
            )
            # --- NEW: include communication metrics in FitRes.metrics ---
            total_bytes = upload_bytes + download_bytes
            # ---------------------------
            # AnchorZKP payload (q_anchor_*) for RSU aggregation verification
            # ---------------------------
            q_anchor_b64 = ""
            q_anchor_sha256 = ""
            _t_anchor0 = time.perf_counter()
            try:
                if self.anchor_enabled:
                    # Anchor is enabled => anchor_dmatrix must exist
                    if self.anchor_dmatrix is None:
                        raise RuntimeError("anchor_enabled=True but anchor_dmatrix is None")
                    # Predict on anchor set (treat output as probability-like; clip to [0,1])
                    p_anchor = np.asarray(self.model.predict(self.anchor_dmatrix), dtype=np.float64)
                    p_anchor = np.nan_to_num(p_anchor, nan=0.0, posinf=1.0, neginf=0.0)
                    p_anchor = np.clip(p_anchor, 0.0, 1.0)
                    q_anchor = azkp.quantize_proba_to_int(p_anchor, SCALE=int(self.anchor_SCALE))
                    q_anchor_b64, q_anchor_sha256 = azkp.encode_int_vector_b64(q_anchor)
                    # If anchor is enabled, q_anchor_sha256 MUST be present and SHA256-like
                    q_anchor_sha256 = str(q_anchor_sha256 or "").strip()
                    if len(q_anchor_sha256) != 64:
                        raise RuntimeError(f"Invalid q_anchor_sha256='{q_anchor_sha256}'")
                else:
                    # Anchor disabled => keep empty values (this is allowed)
                    q_anchor_b64 = ""
                    q_anchor_sha256 = ""
            except Exception as exc:
                self.logger.error(
                    "[RSU %s | Vehicle %s | Round %s] Anchor payload FAILED (cannot proceed): %s",
                    self.rsu_id,
                    self.vehicle_id,
                    round_num,
                    exc,
                )
                # On-chain lane requires q_anchor_sha256 when anchor is enabled, so stop.
                raise RuntimeError(
                    f"[ANCHOR] Mandatory anchor payload failed for RSU={int(self.rsu_id)} "
                    f"Vehicle={int(self.vehicle_id)} Round={int(round_num)}"
                ) from exc
            anchor_payload_time_sec = time.perf_counter() - _t_anchor0

            # Hash of the exact delta bytes we send (binds SSI report + RSU Merkle record)
            model_delta_sha256 = hashlib.sha256(model_bytes).hexdigest() if model_bytes else ""
            # SSI report (canonical bytes) + signature (Vehicle → RSU verification evidence)
            _t_ssi0 = time.perf_counter()
            reviewer3_context_target = _reviewer3_context_target_v1(
                self.ablation_cfg,
                rsu_id=int(self.rsu_id),
                vehicle_id=int(self.vehicle_id),
                round_idx=int(round_num),
            )
            reviewer3_context_challenge_target = 0
            reviewer3_context_challenge_mode = ""
            signed_report_rsu_id = int(self.rsu_id)
            signed_report_round = int(round_num)
            if reviewer3_context_target is not None:
                reviewer3_context_challenge_target = 1
                reviewer3_context_challenge_mode = str(
                    reviewer3_context_target.get("mode", "") or ""
                ).strip().lower()
                if reviewer3_context_challenge_mode == "stale_round":
                    signed_report_round = max(0, int(round_num) - 1)
                elif reviewer3_context_challenge_mode == "cross_rsu_scope":
                    signed_report_rsu_id = int(
                        reviewer3_context_target.get(
                            "signed_rsu_id", 2 if int(self.rsu_id) == 1 else 1
                        )
                    )
                    if signed_report_rsu_id == int(self.rsu_id):
                        raise RuntimeError("cross_rsu_scope target must sign for a different RSU")
                else:
                    raise RuntimeError(
                        f"Unsupported reviewer3 context challenge mode={reviewer3_context_challenge_mode!r}"
                    )
                self.logger.warning(
                    "[REVIEWER3][CONTEXT] RSU=%d vehicle=%d active_round=%d mode=%s signed_rsu=%d signed_round=%d",
                    int(self.rsu_id), int(self.vehicle_id), int(round_num),
                    reviewer3_context_challenge_mode, int(signed_report_rsu_id), int(signed_report_round),
                )

            ssi_report_bytes = _ssi_vehicle_report_bytes_v1(
                rsu_id=int(signed_report_rsu_id),
                vehicle_id=int(self.vehicle_id),
                round_idx=int(signed_report_round),
                model_delta_sha256=str(model_delta_sha256),
                dp_round_json_sha256=str(dp_round_json_sha256),
                q_anchor_sha256=str(q_anchor_sha256 or ""),
            )
            ssi_report_sha256_hex = hashlib.sha256(ssi_report_bytes).hexdigest()
            ssi_sig_b64 = self._ssi_sign_report_v1(ssi_report_bytes)

            ablation_admission_target = 0
            ablation_admission_mode = ""
            ablation_digest_target = 0
            ablation_digest_mode = ""

            if _ablation_target_matches_v1(
                self.ablation_cfg,
                rsu_id=int(self.rsu_id),
                vehicle_id=int(self.vehicle_id),
                round_idx=int(round_num),
            ):
                ablation_admission_target = 1
                ablation_admission_mode = str(
                    self.ablation_cfg.get("admission_failure_mode", "bad_signature") or "bad_signature"
                ).strip().lower()

                if ablation_admission_mode == "bad_signature":
                    ssi_sig_b64 = _corrupt_signature_b64_v1(ssi_sig_b64)
                else:
                    raise RuntimeError(
                        f"Unsupported admission_failure_mode={ablation_admission_mode!r}"
                    )

            if _ablation_target_matches_key_v1(
                self.ablation_cfg,
                "enable_digest_binding_failure_check",
                "digest_binding_failure_target",
                rsu_id=int(self.rsu_id),
                vehicle_id=int(self.vehicle_id),
                round_idx=int(round_num),
            ):
                ablation_digest_target = 1
                ablation_digest_mode = str(
                    self.ablation_cfg.get("digest_binding_failure_mode", "bad_dp_record_binding")
                    or "bad_dp_record_binding"
                ).strip().lower()

                if ablation_digest_mode == "bad_dp_record_binding":
                    dp_round_json_sha256 = _mutate_sha256_hex_v1(dp_round_json_sha256)
                elif ablation_digest_mode == "bad_q_anchor_binding":
                    q_anchor_sha256 = _mutate_sha256_hex_v1(q_anchor_sha256)
                elif ablation_digest_mode == "bad_model_delta_binding":
                    model_delta_sha256 = _mutate_sha256_hex_v1(model_delta_sha256)
                else:
                    raise RuntimeError(
                        f"Unsupported digest_binding_failure_mode={ablation_digest_mode!r}"
                    )

            ssi_report_time_sec = time.perf_counter() - _t_ssi0
            fit_wall_time_sec = time.perf_counter() - fit_wall_t0
            metrics = {
                # ---- identity (so RSU records are stable and not based on Flower partition ids) ----
                "rsu_id": int(self.rsu_id),
                "vehicle_id": int(self.vehicle_id),
                # ---- Reviewer 3 controlled signed-context challenge metadata ----
                "reviewer3_context_challenge_target": int(reviewer3_context_challenge_target),
                "reviewer3_context_challenge_mode": str(reviewer3_context_challenge_mode),
                "reviewer3_signed_report_rsu_id": int(signed_report_rsu_id),
                "reviewer3_signed_report_round": int(signed_report_round),
                # ---- SSI evidence (Vehicle signs; RSU verifies) ----
                "ssi_did": str(self.ssi_did),
                "ssi_pubkey_b64": str(self.ssi_pubkey_b64),
                "ssi_sig_b64": str(ssi_sig_b64),
                "ssi_report_sha256_hex": str(ssi_report_sha256_hex),
                "model_delta_sha256": str(model_delta_sha256),
                "ablation_admission_target": int(ablation_admission_target),
                "ablation_admission_mode": str(ablation_admission_mode),
                "ablation_digest_target": int(ablation_digest_target),
                "ablation_digest_mode": str(ablation_digest_mode),
                "reviewer3_fit_completed": 1,
                "train_time_sec": float(train_time),
                "fit_wall_time_sec": float(fit_wall_time_sec),
                "set_parameters_time_sec": float(set_parameters_time_sec),
                "local_train_time_sec": float(local_train_time_sec),
                "dp_record_time_sec": float(dp_record_time_sec),
                "delta_build_time_sec": float(delta_build_time_sec),
                "model_serialize_time_sec": float(model_serialize_time_sec),
                "anchor_payload_time_sec": float(anchor_payload_time_sec),
                "ssi_report_time_sec": float(ssi_report_time_sec),
                "num_train_examples": int(len(y_train_true)),
                "dp_epsilon_used": float(dp_epsilon_used),
                "dp_epsilon_cumulative": float(self.dp_epsilon_cumulative),
                "dp_epsilon_per_tree": float(dp_eps_tree),
                "dp_round_json_sha256": str(dp_round_json_sha256),
                "dp_record_ok": 1 if str(dp_round_json_sha256).strip() else 0,
                "dp_record_err": "" if str(dp_round_json_sha256).strip() else "dp_record_boundary_failed",
                "num_new_trees": int(num_new_trees),
                "upload_bytes": int(upload_bytes),
                "download_bytes": int(download_bytes),
                "total_bytes": int(total_bytes),
                "early_stopping_enabled": bool(es_enabled),
                "early_stopping_rounds": int(self.xgb_cfg.early_stopping_rounds or 0),
                "best_iteration":
                best_iteration if best_iteration is not None else -1,
                "best_score": float(best_score),
                "stopped_early": bool(stopped_early),
                "early_stopping_reason": str(reason),
                # ---- AnchorZKP ----
                "anchor_version": str(self.anchor_version),
                "anchor_id_field": str(self.anchor_id_field),
                "anchor_M": int(self.anchor_M),
                "anchor_SCALE": int(self.anchor_SCALE),
                "q_anchor_b64": q_anchor_b64 if isinstance(q_anchor_b64, str) else "",
                "q_anchor_sha256": q_anchor_sha256 if isinstance(q_anchor_sha256, str) else "",
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
        We deliberately avoid using the TEST split here to prevent test leakage
        during federated training. TEST is only used later in centralized and
        global-ensemble evaluation.
        """
        try:
            round_num = int(ins.config.get("round", 0))
            reviewer3_unavailable, reviewer3_unavailability_reason = _reviewer3_is_unavailable_v1(
                self.ablation_cfg,
                rsu_id=int(self.rsu_id),
                vehicle_id=int(self.vehicle_id),
                round_idx=int(round_num),
            )
            if reviewer3_unavailable:
                self.logger.warning(
                    "[REVIEWER3][EVAL-UNAVAILABLE] RSU=%d vehicle=%d round=%d reason=%s",
                    int(self.rsu_id), int(self.vehicle_id), int(round_num),
                    str(reviewer3_unavailability_reason),
                )
                raise RuntimeError(
                    f"REVIEWER3_SIMULATED_EVAL_UNAVAILABILITY:rsu={int(self.rsu_id)}:"
                    f"vehicle={int(self.vehicle_id)}:round={int(round_num)}:"
                    f"reason={reviewer3_unavailability_reason}"
                )
            # Load the latest global model
            self.set_parameters(ins.parameters)

            # ---- 1) Predict on the RSU-wide validation split ----
            y_val_true: np.ndarray = np.asarray(
                self.val_dmatrix.get_label(), dtype=np.int32
            )
            y_val_proba: np.ndarray = np.asarray(
                self.model.predict(self.val_dmatrix), dtype=np.float64
            )

            # Clamp to a valid probability range for stable log-loss computation
            y_val_proba = np.nan_to_num(y_val_proba, nan=0.0, posinf=1.0, neginf=0.0)
            y_val_proba = np.clip(y_val_proba, 1e-7, 1.0 - 1e-7)

            # ---- 2) Threshold-dependent metrics ----
            best_thr = find_optimal_threshold(
                y_val_true,
                y_val_proba,
                default_threshold=0.5,
            )
            y_pred_val: np.ndarray = (y_val_proba >= best_thr).astype(int)

            val_metrics = compute_binary_metrics(
                y_val_true, y_pred_val, y_val_proba, calibration=True
            )

            # ---- 3) Proper probabilistic losses ----
            val_logloss = float(log_loss(y_val_true, y_val_proba, labels=[0, 1]))
            val_brier = float(val_metrics.get("brier", 0.0))

            cm_val = confusion_matrix(y_val_true, y_pred_val)
            log_metrics_pretty(
                f"[RSU {self.rsu_id} | Vehicle {self.vehicle_id}] "
                f"VAL (thr={best_thr:.4f}, logloss={val_logloss:.6f})",
                val_metrics,
                cm_val,
            )

            return EvaluateRes(
                status=Status(code=Code.OK, message="evaluation complete"),
                loss=val_logloss,
                num_examples=len(y_val_true),
                metrics={
                    "accuracy": val_metrics["accuracy"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                    "f1": val_metrics["f1"],
                    "auc": val_metrics["auc"],
                    "brier": val_brier,
                    "ece": float(val_metrics.get("ece", 0.0)),
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
            output_dir: str,  # <-- NEW
            anchor_ctx: Dict[str, Any],
    ):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=vehicles_per_rsu,
            min_evaluate_clients=0,
            min_available_clients=vehicles_per_rsu,
            accept_failures=True,
            # We still pass the round index so clients can log/use it
            on_fit_config_fn=lambda rnd: {"round": rnd},
            on_evaluate_config_fn=lambda rnd: {"round": rnd},
            # Aggregate test metrics across vehicles by weighted average
            evaluate_metrics_aggregation_fn=weighted_average_eval_metrics,
        )
        self.rsu_id = rsu_id
        self.num_rounds = num_rounds
        self.num_local_rounds = num_local_rounds
        # Will hold the latest global model parameters
        self.latest_global_params: Parameters | None = None
        # Per-round summary for JSON dump (now based on bagging metrics)
        self.round_summaries: List[Dict[str, Any]] = []

        # Manual distributed evaluation history (do not rely only on Flower History)
        self.losses_distributed_manual: List[Tuple[int, float]] = []
        self.metrics_distributed_evaluate_manual: Dict[str, List[Tuple[int, float]]] = {}
        # Track how many trees are present in the global booster
        self._global_num_trees: int = -1
        # Last good global model bytes (for saving RSU model at the end)
        self._last_global_model_bytes: bytes | None = None
        # Per-round, per-vehicle training/validation/DP metrics for comparison script
        # Structure: {round: [ {client_id, num_examples, train_*, val_*, dp_*}, ... ]}
        self.vehicle_metrics_per_round: Dict[int, List[Dict[str, Any]]] = {}
        # ---------------------------
        # AnchorZKP lane state (per RSU)
        # ---------------------------
        self.output_dir = output_dir
        self.anchor_ctx = anchor_ctx
        self.anchor_id_field = str(anchor_ctx.get("anchor_id_field", ""))
        self.anchor_version = str(anchor_ctx.get("anchor_version", ""))
        self.anchor_M = _as_int(anchor_ctx.get("M"), 0)
        self.anchor_SCALE = _as_int(anchor_ctx.get("SCALE"), 0)
        # RSU circuit cap: vehicles_per_rsu for this RSU run
        self.anchor_NMAX = int(vehicles_per_rsu)
        # Per-round validity + artifact paths
        self.rsu_round_valid: Dict[int, int] = {}  # round -> 0/1
        self.rsu_anchor_artifact: Dict[int, str] = {}  # round -> summary path
        # ---- NEW: Poseidon root per RSU round (Phase 2 bridge) ----
        self.rsu_root_poseidon_by_round: Dict[int, int] = {}  # round -> root_poseidon_field
        self.rsu_round_manifest_path: Dict[int, str] = {}  # round -> manifest JSON path

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Parameters | None, Dict[str, Scalar]]:
        rnd = int(server_round)
        rsu_round_t0 = time.perf_counter()
        rsu_preselect_latency_sec = 0.0
        rsu_model_aggregation_latency_sec = 0.0
        rsu_ssi_verification_latency_sec = 0.0
        rsu_merkle_build_latency_sec = 0.0
        rsu_verification_pipeline_latency_sec = 0.0
        rsu_zkp_latency_sec = 0.0

        # ---- Reviewer 3: participation/failure accounting ----
        reviewer3_fit_results_received = int(len(results or []))
        reviewer3_fit_failures_observed = int(len(failures or []))
        reviewer3_failure_proxy_cids: List[str] = []
        for _f in (failures or []):
            try:
                if isinstance(_f, tuple) and len(_f) >= 1:
                    reviewer3_failure_proxy_cids.append(str(getattr(_f[0], "cid", "")))
            except Exception:
                pass

        # ---- Anchor preselect (so: aggregation == manifest roots == ZK proof inputs) ----
        results_all = results
        results_used = results
        tmp_used: List[Tuple[Tuple[int, str, str], str, np.ndarray]] = []
        q_list: List[np.ndarray] = []
        used_vehicle_cids: List[str] = []
        excluded_vehicle_cids: List[str] = []
        preselect_exclusion_details: List[Dict[str, Any]] = []
        rsu_anchor_summary_path = ""
        # NEW: total number of clients whose anchor payload decoded successfully (before NMAX truncation)
        n_total_anchor_decoded: int = 0
        _t_preselect0 = time.perf_counter()
        try:
            M = int(self.anchor_M)
            if M > 0 and self.anchor_id_field:
                NMAX = int(self.anchor_NMAX)
                tmp: List[Tuple[Tuple[int, str, str], str, np.ndarray]] = []
                for client_proxy, fit_res in results_all:
                    m = getattr(fit_res, "metrics", None) or {}
                    cid = str(getattr(client_proxy, "cid", ""))
                    # decode anchor payload
                    try:
                        q_vec = azkp.decode_vehicle_anchor_metrics(
                            m,
                            expected_anchor_id_field=int(str(self.anchor_id_field)),
                            expected_M=int(self.anchor_M),
                            expected_SCALE=int(self.anchor_SCALE),
                        )
                    except Exception:
                        continue
                    # require DP + q_anchor sha
                    dp_sha = str(m.get("dp_round_json_sha256", "") or "").strip()
                    q_anchor_sha = str(m.get("q_anchor_sha256", "") or "").strip()
                    if not _looks_like_sha256_hex(dp_sha):
                        continue
                    if not _looks_like_sha256_hex(q_anchor_sha):
                        continue
                    # require delta bytes
                    delta_bytes = b""
                    try:
                        if fit_res is not None and fit_res.parameters is not None and fit_res.parameters.tensors:
                            delta_bytes = fit_res.parameters.tensors[0] or b""
                    except Exception:
                        delta_bytes = b""
                    if not delta_bytes:
                        continue
                    # use vehicle_id from metrics (stable identity)
                    veh_id = 0
                    try:
                        v_from_metrics = m.get("vehicle_id", None)
                        if str(v_from_metrics).strip() != "":
                            veh_id = _reviewer3_optional_int_v1(v_from_metrics) or 0
                    except Exception:
                        veh_id = 0

                    # require SSI pubkey+sig and cryptographically verify the signed report
                    ssi_pub_b64 = str(m.get("ssi_pubkey_b64", "") or "").strip()
                    ssi_sig_b64 = str(m.get("ssi_sig_b64", "") or "").strip()
                    ssi_report_sha_client = str(m.get("ssi_report_sha256_hex", "") or "").strip()

                    if not ssi_pub_b64 or not ssi_sig_b64:
                        preselect_exclusion_details.append(
                            {
                                "cid": str(cid),
                                "vehicle_id": int(veh_id),
                                "reason": "missing_ssi_pubkey_or_signature",
                                "classification": "missing_ssi_material",
                                "ablation_target": int(m.get("ablation_admission_target", 0) or 0),
                                "ablation_mode": str(m.get("ablation_admission_mode", "") or ""),
                                "ablation_digest_target": int(m.get("ablation_digest_target", 0) or 0),
                                "ablation_digest_mode": str(m.get("ablation_digest_mode", "") or ""),
                                "reviewer3_context_challenge_target": int(m.get("reviewer3_context_challenge_target", 0) or 0),
                                "reviewer3_context_challenge_mode": str(m.get("reviewer3_context_challenge_mode", "") or ""),
                                "reviewer3_signed_report_rsu_id": int(m.get("reviewer3_signed_report_rsu_id", self.rsu_id) or self.rsu_id),
                                "reviewer3_signed_report_round": int(m.get("reviewer3_signed_report_round", rnd) or rnd),
                            }
                        )
                        continue

                    try:
                        _pub_raw = _b64d(ssi_pub_b64)
                        _sig_raw = _b64d(ssi_sig_b64)
                        if not _pub_raw or not _sig_raw:
                            raise ValueError("empty decoded pubkey/signature")

                        did_from_pub = _did_from_pubkey_raw_ed25519(_pub_raw)
                        if not str(did_from_pub).strip():
                            raise ValueError("empty DID derived from pubkey")

                        delta_sha = hashlib.sha256(delta_bytes).hexdigest()
                        expected_report_bytes = _ssi_vehicle_report_bytes_v1(
                            rsu_id=int(self.rsu_id),
                            vehicle_id=int(veh_id),
                            round_idx=int(rnd),
                            model_delta_sha256=str(delta_sha),
                            dp_round_json_sha256=str(dp_sha),
                            q_anchor_sha256=str(q_anchor_sha),
                        )
                        expected_report_sha = hashlib.sha256(expected_report_bytes).hexdigest()

                        _ssi_verify_vehicle_report_sig_v1(
                            report_bytes=expected_report_bytes,
                            pubkey_b64=ssi_pub_b64,
                            sig_b64=ssi_sig_b64,
                        )

                        if ssi_report_sha_client and ssi_report_sha_client != expected_report_sha:
                            raise ValueError("ssi_report_sha256_hex mismatch vs RSU recompute")

                    except Exception as exc_ssi_verify:
                        exclusion_classification = "bad_signature_or_report_binding"
                        reviewer3_mode = str(m.get("reviewer3_context_challenge_mode", "") or "").strip().lower()
                        if int(m.get("reviewer3_context_challenge_target", 0) or 0) == 1:
                            if reviewer3_mode == "stale_round":
                                exclusion_classification = "stale_round_context"
                            elif reviewer3_mode == "cross_rsu_scope":
                                exclusion_classification = "cross_rsu_context"
                        elif int(m.get("ablation_digest_target", 0) or 0) == 1:
                            exclusion_classification = "bad_digest_binding"
                        elif int(m.get("ablation_admission_target", 0) or 0) == 1:
                            exclusion_classification = "bad_signature"

                        preselect_exclusion_details.append(
                            {
                                "cid": str(cid),
                                "vehicle_id": int(veh_id),
                                "reason": "bad_signature_or_report_binding",
                                "classification": str(exclusion_classification),
                                "detail": str(exc_ssi_verify),
                                "ablation_target": int(m.get("ablation_admission_target", 0) or 0),
                                "ablation_mode": str(m.get("ablation_admission_mode", "") or ""),
                                "ablation_digest_target": int(m.get("ablation_digest_target", 0) or 0),
                                "ablation_digest_mode": str(m.get("ablation_digest_mode", "") or ""),
                                "reviewer3_context_challenge_target": int(m.get("reviewer3_context_challenge_target", 0) or 0),
                                "reviewer3_context_challenge_mode": str(m.get("reviewer3_context_challenge_mode", "") or ""),
                                "reviewer3_signed_report_rsu_id": int(m.get("reviewer3_signed_report_rsu_id", self.rsu_id) or self.rsu_id),
                                "reviewer3_signed_report_round": int(m.get("reviewer3_signed_report_round", rnd) or rnd),
                            }
                        )
                        continue

                    sort_key = (int(veh_id), str(did_from_pub), str(cid))
                    tmp.append((sort_key, cid, q_vec))
                tmp.sort(key=lambda t: t[0])
                n_total_anchor_decoded = int(len(tmp))
                tmp_used = tmp[:NMAX]
                used_vehicle_cids = [cid for (_k, cid, _q) in tmp_used]
                q_list = [q for (_k, _cid, q) in tmp_used]
                if used_vehicle_cids:
                    used_set = set(used_vehicle_cids)
                    results_used = [
                        (cp, fr) for (cp, fr) in results_all
                        if str(getattr(cp, "cid", "")) in used_set
                    ]
                    excluded_vehicle_cids = [
                        str(getattr(cp, "cid", "")) for (cp, _) in results_all
                        if str(getattr(cp, "cid", "")) not in used_set
                    ]
                else:
                    excluded_vehicle_cids = [
                        str(getattr(cp, "cid", ""))
                        for (cp, _) in results_all
                    ]

                    raise RuntimeError(
                        f"[ADMISSION][RSU {int(self.rsu_id)}] "
                        f"Round {int(rnd)}: no eligible contribution "
                        "remained after anchor/DP/SSI admission; "
                        "refusing to aggregate unverified updates."
                    )
        except Exception as exc_preselect:
            logging.exception(
                "[ADMISSION][RSU %d] Round %d: "
                "preselection/admission failed; "
                "refusing ungated aggregation",
                int(self.rsu_id),
                int(rnd),
            )

            raise RuntimeError(
                f"RSU {int(self.rsu_id)} round {int(rnd)} "
                "admission/preselection failed"
            ) from exc_preselect
        # ✅ Gated aggregation: only clients committed into roots/proof affect the RSU model
        # Deterministic order: align aggregation order with manifest_cids sorting
        rsu_preselect_latency_sec = time.perf_counter() - _t_preselect0

        logging.info(
            "[ZKP][RSU %d] Round %d: preselect decoded=%d used=%d excluded=%d used_cids=%s excluded_cids=%s",
            int(self.rsu_id),
            int(rnd),
            int(n_total_anchor_decoded),
            int(len(used_vehicle_cids)),
            int(len(excluded_vehicle_cids)),
            str(used_vehicle_cids),
            str(excluded_vehicle_cids),
        )
        results_used = sorted(results_used, key=lambda t: str(getattr(t[0], "cid", "")))
        _t_agg0 = time.perf_counter()
        aggregated_params, agg_metrics = super().aggregate_fit(rnd, results_used, failures)
        rsu_model_aggregation_latency_sec = time.perf_counter() - _t_agg0
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
        # ---- New: summarize DP epsilon across vehicles (if present)
        #          and capture per-vehicle metrics for this round ----
        dp_eps_used: List[float] = []
        dp_eps_cum: List[float] = []
        vehicle_entries: List[Dict[str, Any]] = []
        for client_proxy, fit_res in results_used:
            m = getattr(fit_res, "metrics", None) or {}
            # DP epsilon stats
            val_used = m.get("dp_epsilon_used", None)
            if isinstance(val_used, (int, float)):
                dp_eps_used.append(float(val_used))
            val_cum = m.get("dp_epsilon_cumulative", None)
            if isinstance(val_cum, (int, float)):
                dp_eps_cum.append(float(val_cum))
            # Per-vehicle metrics entry
            entry: Dict[str, Any] = {
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
                "val_logloss",
                "val_brier",
                "val_ece",
                "train_time_sec",
                "num_train_examples",
                "dp_epsilon_used",
                "dp_epsilon_cumulative",
                # NEW: per-vehicle communication metrics
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
        dp_stats: Dict[str, float] = {}
        if dp_eps_used:
            dp_stats["dp_epsilon_used_min"] = float(np.min(dp_eps_used))
            dp_stats["dp_epsilon_used_mean"] = float(np.mean(dp_eps_used))
            dp_stats["dp_epsilon_used_max"] = float(np.max(dp_eps_used))
            logging.info(
                "RSU %d - Round %d: DP ε_used across vehicles "
                "(min=%.4f, mean=%.4f, max=%.4f)",
                self.rsu_id,
                rnd,
                dp_stats["dp_epsilon_used_min"],
                dp_stats["dp_epsilon_used_mean"],
                dp_stats["dp_epsilon_used_max"],
            )
        if dp_eps_cum:
            dp_stats["dp_epsilon_cumulative_min"] = float(np.min(dp_eps_cum))
            dp_stats["dp_epsilon_cumulative_mean"] = float(np.mean(dp_eps_cum))
            dp_stats["dp_epsilon_cumulative_max"] = float(np.max(dp_eps_cum))
            logging.info(
                "RSU %d - Round %d: DP ε_cumulative across vehicles "
                "(min=%.4f, mean=%.4f, max=%.4f)",
                self.rsu_id,
                rnd,
                dp_stats["dp_epsilon_cumulative_min"],
                dp_stats["dp_epsilon_cumulative_mean"],
                dp_stats["dp_epsilon_cumulative_max"],
            )
        # --- NEW: aggregate training-time, communication, and client-side latency metrics ---
        total_train_time = 0.0
        total_examples = 0
        num_clients = 0
        total_upload_bytes = 0
        total_download_bytes = 0

        total_client_wall_time = 0.0
        total_set_parameters_time = 0.0
        total_local_train_time = 0.0
        total_dp_record_time = 0.0
        total_delta_build_time = 0.0
        total_model_serialize_time = 0.0
        total_anchor_payload_time = 0.0
        total_ssi_report_time = 0.0

        for _, fit_res in results_used:
            num_clients += 1
            total_examples += getattr(fit_res, "num_examples", 0)
            m = getattr(fit_res, "metrics", None) or {}

            t_val = m.get("train_time_sec", None)
            if isinstance(t_val, (int, float)):
                total_train_time += float(t_val)

            up_val = m.get("upload_bytes", None)
            if isinstance(up_val, (int, float)):
                total_upload_bytes += int(up_val)

            down_val = m.get("download_bytes", None)
            if isinstance(down_val, (int, float)):
                total_download_bytes += int(down_val)

            cur = m.get("fit_wall_time_sec", None)
            if isinstance(cur, (int, float)):
                total_client_wall_time += float(cur)

            cur = m.get("set_parameters_time_sec", None)
            if isinstance(cur, (int, float)):
                total_set_parameters_time += float(cur)

            cur = m.get("local_train_time_sec", None)
            if isinstance(cur, (int, float)):
                total_local_train_time += float(cur)

            cur = m.get("dp_record_time_sec", None)
            if isinstance(cur, (int, float)):
                total_dp_record_time += float(cur)

            cur = m.get("delta_build_time_sec", None)
            if isinstance(cur, (int, float)):
                total_delta_build_time += float(cur)

            cur = m.get("model_serialize_time_sec", None)
            if isinstance(cur, (int, float)):
                total_model_serialize_time += float(cur)

            cur = m.get("anchor_payload_time_sec", None)
            if isinstance(cur, (int, float)):
                total_anchor_payload_time += float(cur)

            cur = m.get("ssi_report_time_sec", None)
            if isinstance(cur, (int, float)):
                total_ssi_report_time += float(cur)

        # Size of the aggregated global model (same for all clients this round)
        global_model_size_bytes = 0
        if aggregated_params is not None and aggregated_params.tensors:
            try:
                global_model_size_bytes = len(aggregated_params.tensors[0])
            except Exception:
                global_model_size_bytes = 0

        fit_metrics: Dict[str, float] = {}
        fit_metrics["fit_total_train_time_sec"] = float(total_train_time)
        fit_metrics["fit_total_client_wall_time_sec"] = float(total_client_wall_time)
        fit_metrics["fit_total_set_parameters_time_sec"] = float(total_set_parameters_time)
        fit_metrics["fit_total_local_train_time_sec"] = float(total_local_train_time)
        fit_metrics["fit_total_dp_record_time_sec"] = float(total_dp_record_time)
        fit_metrics["fit_total_delta_build_time_sec"] = float(total_delta_build_time)
        fit_metrics["fit_total_model_serialize_time_sec"] = float(total_model_serialize_time)
        fit_metrics["fit_total_anchor_payload_time_sec"] = float(total_anchor_payload_time)
        fit_metrics["fit_total_ssi_report_time_sec"] = float(total_ssi_report_time)
        fit_metrics["fit_num_clients"] = float(num_clients)
        fit_metrics["fit_num_examples_total"] = float(total_examples)

        if num_clients > 0:
            fit_metrics["fit_avg_train_time_sec_per_client"] = (
                total_train_time / float(num_clients)
            )
            fit_metrics["fit_avg_client_wall_time_sec_per_client"] = (
                total_client_wall_time / float(num_clients)
            )
            fit_metrics["fit_avg_set_parameters_time_sec_per_client"] = (
                total_set_parameters_time / float(num_clients)
            )
            fit_metrics["fit_avg_local_train_time_sec_per_client"] = (
                total_local_train_time / float(num_clients)
            )
            fit_metrics["fit_avg_dp_record_time_sec_per_client"] = (
                total_dp_record_time / float(num_clients)
            )
            fit_metrics["fit_avg_delta_build_time_sec_per_client"] = (
                total_delta_build_time / float(num_clients)
            )
            fit_metrics["fit_avg_model_serialize_time_sec_per_client"] = (
                total_model_serialize_time / float(num_clients)
            )
            fit_metrics["fit_avg_anchor_payload_time_sec_per_client"] = (
                total_anchor_payload_time / float(num_clients)
            )
            fit_metrics["fit_avg_ssi_report_time_sec_per_client"] = (
                total_ssi_report_time / float(num_clients)
            )

        if total_examples > 0:
            fit_metrics["fit_avg_train_time_msec_per_example"] = (
                (total_train_time * 1000.0) / float(total_examples)
            )

        # --- NEW: communication overhead & model size (RSU-level) ---
        total_bytes = total_upload_bytes + total_download_bytes
        fit_metrics["fit_total_upload_bytes"] = float(total_upload_bytes)
        fit_metrics["fit_total_download_bytes"] = float(total_download_bytes)
        fit_metrics["fit_total_bytes"] = float(total_bytes)

        if num_clients > 0:
            fit_metrics["fit_avg_upload_bytes_per_client"] = (
                float(total_upload_bytes) / float(num_clients)
            )
            fit_metrics["fit_avg_download_bytes_per_client"] = (
                float(total_download_bytes) / float(num_clients)
            )
            fit_metrics["fit_avg_bytes_per_client"] = (
                float(total_bytes) / float(num_clients)
            )

        if total_examples > 0:
            fit_metrics["fit_avg_upload_bytes_per_example"] = (
                float(total_upload_bytes) / float(total_examples)
            )
            fit_metrics["fit_avg_download_bytes_per_example"] = (
                float(total_download_bytes) / float(total_examples)
            )
            fit_metrics["fit_avg_bytes_per_example"] = (
                float(total_bytes) / float(total_examples)
            )

        if global_model_size_bytes > 0:
            fit_metrics["fit_global_model_size_bytes"] = float(global_model_size_bytes)
        # Store JSON-friendly per-round summary
        round_summary: Dict[str, Any] = {"round": rnd}
        for k, v in agg_metrics.items():
            try:
                round_summary[k] = float(v)
            except (TypeError, ValueError):
                continue
        # Include fit_* metrics so comparison code can pick them up
        for k, v in fit_metrics.items():
            round_summary[k] = v
        # Also include DP epsilon stats (if any)
        for k, v in dp_stats.items():
            round_summary[k] = v
        # ------------------------------------------------------------------
        # AnchorZKP: RSU-level aggregation verification (P0 audit lane)
        # ------------------------------------------------------------------
        rsu_anchor_ok = 0
        rsu_anchor_N_used = 0
        Q_rsu_b64 = ""
        global_commit_field = ""
        # ✅ ALSO define RSU-root defaults here (so round_summary is always safe)
        rsu_root_poseidon_field = 0
        rsu_root_sha256_hex = ""
        # ✅ V15 public-input gating fields (persist to round_summary so GLOBAL can reuse them)
        pins_hash_field: int = 0
        policy_id_field: int = 0
        public_input_order_id_field: int = 0
        # ✅ NEW: SSI preimage-definition fingerprint (MUST be stable across RSUs + GLOBAL)
        ssi_preimage_def_sha256_v1_hex: str = ""
        ssi_preimage_def_field_bn254_v1: int = 0
        try:
            if hasattr(canon, "ssi_preimage_def_sha256_v1") and callable(getattr(canon, "ssi_preimage_def_sha256_v1")):
                ssi_preimage_def_sha256_v1_hex = str(canon.ssi_preimage_def_sha256_v1())
            elif hasattr(mssi, "ssi_preimage_def_sha256_v1") and callable(getattr(mssi, "ssi_preimage_def_sha256_v1")):
                ssi_preimage_def_sha256_v1_hex = str(mssi.ssi_preimage_def_sha256_v1())
        except Exception:
            ssi_preimage_def_sha256_v1_hex = ""
        try:
            if hasattr(canon, "ssi_preimage_def_field_bn254_v1") and callable(
                    getattr(canon, "ssi_preimage_def_field_bn254_v1")):
                ssi_preimage_def_field_bn254_v1 = int(canon.ssi_preimage_def_field_bn254_v1())
            elif hasattr(mssi, "ssi_preimage_def_field_bn254_v1") and callable(
                    getattr(mssi, "ssi_preimage_def_field_bn254_v1")):
                ssi_preimage_def_field_bn254_v1 = int(mssi.ssi_preimage_def_field_bn254_v1())
        except Exception:
            ssi_preimage_def_field_bn254_v1 = 0
        # ✅ Optional: log if canon vs mssi disagree (this is the root-cause signal)
        try:
            _sha_canon = ""
            _sha_mssi = ""
            if hasattr(canon, "ssi_preimage_def_sha256_v1") and callable(getattr(canon, "ssi_preimage_def_sha256_v1")):
                _sha_canon = str(canon.ssi_preimage_def_sha256_v1())
            if hasattr(mssi, "ssi_preimage_def_sha256_v1") and callable(getattr(mssi, "ssi_preimage_def_sha256_v1")):
                _sha_mssi = str(mssi.ssi_preimage_def_sha256_v1())
            if _sha_canon and _sha_mssi and _sha_canon != _sha_mssi:
                logging.warning(
                    "[SSI][RSU %d] Round %d: ssi_preimage_def_sha256_v1 mismatch between canon and mssi | canon=%s mssi=%s",
                    int(self.rsu_id), int(rnd), str(_sha_canon), str(_sha_mssi)
                )
        except Exception:
            pass
        # Reviewer-3 evidence materialization policy:
        # - always materialize complete evidence on the final round;
        # - additionally materialize Round 6 for combined_stress because that
        #   round carries the stale-round and cross-RSU signed-context challenges;
        # - all rounds still execute contribution admission, SSI verification,
        #   dropout handling, aggregation, DP accounting, and exclusion recording.
        reviewer3_scenario_now = _reviewer3_scenario_v1()
        reviewer3_full_evidence_round = (
            int(rnd) == int(self.num_rounds)
            or (
                reviewer3_scenario_now == "combined_stress"
                and int(rnd) == 6
            )
        )

        _t_verify0 = time.perf_counter()
        try:
            M = (
                int(self.anchor_M)
                if reviewer3_full_evidence_round
                else 0
            )

            if not reviewer3_full_evidence_round:
                # Admission has already been completed above. Preserve the
                # admitted-client count for Reviewer-3 participation reporting,
                # while skipping only the expensive commitment/proof materialization.
                rsu_anchor_N_used = int(len(results_used))
                self.rsu_round_valid[rnd] = 0
                self.rsu_anchor_artifact[rnd] = ""

                logging.info(
                    "[REVIEWER3][FAST-EVIDENCE] RSU=%d round=%d "
                    "admitted=%d | full Merkle/Groth16 materialization skipped",
                    int(self.rsu_id),
                    int(rnd),
                    int(rsu_anchor_N_used),
                )

            if M > 0 and self.anchor_id_field:
                NMAX = int(self.anchor_NMAX)
                # Use the preselected set so: aggregation == roots == proof inputs
                rsu_anchor_N_used = int(len(q_list))
                # ---- Build canonical per-vehicle record bytes, dual leaves, and RSU roots ----
                # Always try to write a stable manifest so GLOBAL has something to read.
                rsu_manifest_path = os.path.join(
                    str(self.output_dir),
                    "round_manifests",
                    "rsu",
                    f"rsu_{self.rsu_id}",
                    f"round_{rnd}.json",
                )
                os.makedirs(os.path.dirname(rsu_manifest_path), exist_ok=True)
                try:
                    # Map FitRes by cid for deterministic access
                    fit_by_cid: Dict[str, FitRes] = {
                        str(getattr(cp, "cid", "")): fr for (cp, fr) in results_used
                    }
                    _t_rsu_ssi0 = time.perf_counter()

                    # ✅ Step 1: canonical bytes list (in the exact order used)
                    record_bytes_list: List[bytes] = []
                    # ✅ Optional: compact audit payload (store record + leaves)
                    records_compact: List[Dict[str, Any]] = []
                    # Aggregation set (always): what super().aggregate_fit actually used
                    manifest_cids: List[str] = sorted(
                        [str(getattr(cp, "cid", "")) for (cp, _fr) in results_used]
                    )
                    clients_with_valid_anchor_payload: List[str] = list(used_vehicle_cids)  # already sorted
                    # ✅ If no anchors decoded, do NOT attempt SSI/Merkle proofs for everyone.
                    # This preserves your stated behavior: keep training, skip audit/proof.
                    if len(clients_with_valid_anchor_payload) == 0:
                        rsu_root_poseidon_field = 0
                        rsu_root_sha256_hex = ""
                        self.rsu_root_poseidon_by_round[rnd] = 0
                        self.rsu_round_manifest_path[rnd] = str(rsu_manifest_path)
                        # still write a minimal manifest so GLOBAL has an auditable record of the skip
                        mssi.write_round_roots_manifest_v1(
                            rsu_manifest_path,
                            round_idx=int(rnd),
                            rsu_id=int(self.rsu_id),
                            root_sha256_hex="",
                            root_poseidon_field="0",
                            identity_state_root_field="0",
                            poseidon_pad_leaf_field="0",
                            extra={
                                "scope": "rsu_round",
                                "poseidon_arity": int(POSEIDON_ARITY),
                                "poseidon_depth": int(POSEIDON_DEPTH),
                                "sha_depth": int(SHA_DEPTH),
                                "client_ids_sorted": manifest_cids,
                                "clients_with_valid_anchor_payload": [],
                                "excluded_client_ids": excluded_vehicle_cids,
                                "N_total_decoded": int(n_total_anchor_decoded),
                                "N_used_for_anchorsum": 0,
                                "ssi_verify_total": 0,
                                "ssi_verify_ok": 0,
                                "ssi_verify_fail": 0,
                                "records": [],
                                "auditor_recompute_ok": False,
                                "record_count": 0,
                                "merkle_ssi_skipped": True,
                                "skip_reason": "no_anchor_payloads_decoded",
                            },
                        )
                        logging.warning(
                            "[MERKLE][RSU %d] Round %d: no anchor payloads decoded; "
                            "continuing training but skipping Merkle/SSI audit + AnchorSum proof.",
                            int(self.rsu_id),
                            int(rnd),
                        )
                    else:
                        for cid in manifest_cids:
                            fr = fit_by_cid.get(str(cid))
                            m_for_cid = (getattr(fr, "metrics", None) or {}) if fr is not None else {}
                            delta_bytes = b""
                            try:
                                if fr is not None and fr.parameters is not None and fr.parameters.tensors:
                                    delta_bytes = fr.parameters.tensors[0] or b""
                            except Exception:
                                delta_bytes = b""
                            # Prefer real vehicle_id from client metrics; fall back to cid if needed
                            veh_id = 0
                            try:
                                v_from_metrics = m_for_cid.get("vehicle_id", None)
                                if isinstance(v_from_metrics, (int, float, str)) and str(v_from_metrics).strip() != "":
                                    veh_id = _reviewer3_optional_int_v1(v_from_metrics) or 0
                                else:
                                    veh_id = int(str(cid))  # only if cid is numeric
                            except Exception:
                                veh_id = 0
                            dp_sha = str(m_for_cid.get("dp_round_json_sha256", "") or "").strip()
                            q_anchor_sha = str(m_for_cid.get("q_anchor_sha256", "") or "").strip()
                            delta_sha = hashlib.sha256(delta_bytes).hexdigest() if delta_bytes else ""
                            # ---------------------------
                            # SSI inputs (MUST be extracted before checks)
                            # ---------------------------
                            ssi_pub_b64 = str(m_for_cid.get("ssi_pubkey_b64", "") or "").strip()
                            ssi_sig_b64 = str(m_for_cid.get("ssi_sig_b64", "") or "").strip()
                            ssi_did_claim = str(m_for_cid.get("ssi_did", "") or "").strip()
                            ssi_report_sha_client = str(m_for_cid.get("ssi_report_sha256_hex", "") or "").strip()
                            if not dp_sha:
                                raise RuntimeError(
                                    f"[DP][RSU {int(self.rsu_id)}] Round {int(rnd)} cid={cid}: missing dp_round_json_sha256"
                                )
                            if not q_anchor_sha:
                                raise RuntimeError(
                                    f"[ANCHOR][RSU {int(self.rsu_id)}] Round {int(rnd)} cid={cid}: missing q_anchor_sha256"
                                )
                            if not delta_sha:
                                raise RuntimeError(
                                    f"[MODEL][RSU {int(self.rsu_id)}] Round {int(rnd)} cid={cid}: missing model delta bytes (delta_sha empty)"
                                )
                            if not ssi_pub_b64 or not ssi_sig_b64:
                                raise RuntimeError(
                                    f"[SSI][RSU {int(self.rsu_id)}] Round {int(rnd)} cid={cid}: missing ssi_pubkey_b64/ssi_sig_b64"
                                )
                            # Derive DID from pubkey (binding identity to the signing key)
                            ssi_did_from_pub = ""
                            try:
                                _pub_raw = _b64d(ssi_pub_b64)
                                if not _pub_raw:
                                    raise ValueError("empty decoded pubkey")
                                ssi_did_from_pub = _did_from_pubkey_raw_ed25519(_pub_raw)
                            except Exception:
                                ssi_did_from_pub = ""
                            if not ssi_did_from_pub:
                                raise RuntimeError(
                                    f"[SSI][RSU {int(self.rsu_id)}] Round {int(rnd)} cid={cid}: could not derive DID from pubkey"
                                )
                            # ---------------------------
                            # SSI verification at RSU (cryptographic evidence)
                            # ---------------------------
                            expected_report_bytes = _ssi_vehicle_report_bytes_v1(
                                rsu_id=int(self.rsu_id),
                                vehicle_id=int(veh_id),
                                round_idx=int(rnd),
                                model_delta_sha256=str(delta_sha),
                                dp_round_json_sha256=str(dp_sha),
                                q_anchor_sha256=str(q_anchor_sha),
                            )
                            expected_report_sha = hashlib.sha256(expected_report_bytes).hexdigest()
                            ssi_ok = 0
                            ssi_err = ""
                            try:
                                if ssi_did_claim and ssi_did_claim != ssi_did_from_pub:
                                    raise ValueError("ssi_did mismatch vs derived pubkey DID")
                                _ssi_verify_vehicle_report_sig_v1(
                                    report_bytes=expected_report_bytes,
                                    pubkey_b64=ssi_pub_b64,
                                    sig_b64=ssi_sig_b64,
                                )
                                if ssi_report_sha_client and ssi_report_sha_client != expected_report_sha:
                                    raise ValueError("ssi_report_sha256_hex mismatch vs RSU recompute")
                                ssi_ok = 1
                            except Exception as exc_ssi:
                                ssi_ok = 0
                                ssi_err = str(exc_ssi)
                            # Use ONLY the verified DID going forward
                            ssi_did = str(ssi_did_from_pub)
                            logging.info(
                                "[SSI][RSU %d] Round %d verify vehicle_id=%d cid=%s ok=%d did=%s report_sha=%s",
                                int(self.rsu_id),
                                int(rnd),
                                int(veh_id),
                                str(cid),
                                int(ssi_ok),
                                str(ssi_did),
                                str(expected_report_sha),
                            )
                            if STRICT_SSI_VERIFY and int(ssi_ok) != 1:
                                raise RuntimeError(
                                    f"[SSI][RSU {int(self.rsu_id)}] Round {int(rnd)} vehicle_id={int(veh_id)} cid={str(cid)} "
                                    f"verification failed: {ssi_err}"
                                )
                            # ---------------------------
                            # MerkleSSI input MUST be ClientUpdateEnvelopeV1 (matches parse_client_update_envelope_v1)
                            # ---------------------------
                            # sig_sha256 = SHA256(raw signature bytes)
                            _sig_raw = b""
                            try:
                                _sig_raw = _b64d(ssi_sig_b64) if ssi_sig_b64 else b""
                            except Exception:
                                _sig_raw = b""
                            sig_sha256 = hashlib.sha256(_sig_raw).hexdigest() if _sig_raw else ""
                            # auth_evidence_sha256 = SHA256(canonical AuthEvidenceV1 bytes)
                            ssi_evidence_obj = {
                                "schema": "VehicleSSIVerificationEvidenceV1",
                                "v": 1,
                                "ssi_verify_ok": int(ssi_ok),
                                "did": str(ssi_did),
                                "dp_round_json_sha256": str(dp_sha),
                                "ssi_report_sha256_hex": str(expected_report_sha),
                            }
                            ssi_evidence_bytes = canon.canon_json_bytes_v1(ssi_evidence_obj)
                            auth_evidence_sha256 = hashlib.sha256(ssi_evidence_bytes).hexdigest()
                            # optional data fingerprint (allowed to be empty by the parser)
                            data_fingerprint_sha256 = str(m_for_cid.get("data_fingerprint_sha256", "") or "")
                            # nonce/timestamp: prefer client-provided, otherwise RSU fills
                            nonce_val = 0
                            try:
                                nonce_val = int(m_for_cid.get("nonce", m_for_cid.get("ssi_nonce", 0)) or 0)
                            except Exception:
                                nonce_val = 0
                            timestamp_val = 0
                            try:
                                timestamp_val = int(m_for_cid.get("timestamp", m_for_cid.get("ssi_timestamp", 0)) or 0)
                            except Exception:
                                timestamp_val = 0
                            if timestamp_val <= 0:
                                timestamp_val = int(time.time())
                            # Build envelope (ONLY the fields your parser requires + optional data_fingerprint_sha256)
                            env = {
                                "schema": "ClientUpdateEnvelopeV1",
                                "did": str(ssi_did),
                                "signed_payload_sha256": str(expected_report_sha),
                                "update_commit_sha256": str(delta_sha),
                                "dp_record_sha256": str(dp_sha),
                                "sig_sha256": str(sig_sha256),
                                "auth_evidence_sha256": str(auth_evidence_sha256),
                                "data_fingerprint_sha256": str(data_fingerprint_sha256),
                                "rsu_id": int(self.rsu_id),
                                "round_idx": int(rnd),
                                "nonce": int(nonce_val),
                                "timestamp": int(timestamp_val),
                            }
                            env_bytes = canon.canon_json_bytes_v1(env)
                            # Fail fast with the EXACT same validator MerkleSSI expects
                            try:
                                canon.parse_client_update_envelope_v1(env_bytes)
                            except Exception as _e:
                                raise RuntimeError(f"Bad canonical envelope for MerkleSSI (cid={cid}): {_e}") from _e
                            # ✅ Merkle roots/leaves MUST be computed over envelope bytes
                            record_bytes_list.append(env_bytes)
                            leaf_sha_hex, leaf_pose_f = mssi.record_bytes_to_dual_leaves_v1(env_bytes)
                            # ---------------------------
                            # Keep a detailed record for manifest/debug (OK to keep ClientUpdateRecordV1 here)
                            # ---------------------------
                            rec_dbg = {
                                "schema": "ClientUpdateRecordV1",
                                "v": 1,
                                "rsu_id": int(self.rsu_id),
                                "round": int(rnd),
                                "vehicle_id": int(veh_id),
                                "client_id": str(cid),
                                "anchor_version": str(self.anchor_version),
                                "anchor_id_field": str(self.anchor_id_field),
                                "M": int(self.anchor_M),
                                "SCALE": int(self.anchor_SCALE),
                                "q_anchor_sha256": str(q_anchor_sha),
                                "dp_round_json_sha256": str(dp_sha),
                                "model_delta_sha256": str(delta_sha),
                                # SSI evidence (verified DID)
                                "ssi_did": str(ssi_did),
                                "ssi_pubkey_b64": str(ssi_pub_b64),
                                "ssi_sig_b64": str(ssi_sig_b64),
                                "ssi_report_sha256_hex": str(expected_report_sha),
                                "ssi_verify_ok": int(ssi_ok),
                            }
                            records_compact.append(
                                {
                                    "client_id": str(cid),
                                    "record": rec_dbg,
                                    # MerkleSSI is over envelope bytes:
                                    "envelope": env,
                                    "envelope_sha256_hex": hashlib.sha256(env_bytes).hexdigest(),
                                    "sha256_leaf_hex": str(leaf_sha_hex),
                                    "poseidon_leaf_field": int(leaf_pose_f),
                                    "ssi": {
                                        "did": str(ssi_did),
                                        "report_sha256_hex": str(expected_report_sha),
                                        "verify_ok": int(ssi_ok),
                                    },
                                }
                            )
                        # ✅ FINAL sanity checks MUST be after the loop (never inside it)
                        if STRICT_MERKLE_AUDIT:
                            if len(record_bytes_list) == 0:
                                raise RuntimeError(
                                    f"[MERKLE][RSU {int(self.rsu_id)}] Round {int(rnd)}: "
                                    f"record_bytes_list is empty; refusing to publish constant roots."
                                )
                            if len(record_bytes_list) != len(manifest_cids):
                                raise RuntimeError(
                                    f"[MERKLE][RSU {int(self.rsu_id)}] Round {int(rnd)}: "
                                    f"record_bytes_list ({len(record_bytes_list)}) != manifest_cids ({len(manifest_cids)})."
                                )
                    rsu_ssi_verification_latency_sec = time.perf_counter() - _t_rsu_ssi0

                    # ✅ Step 2: compute dual roots from record bytes via mssi specs
                    _t_rsu_merkle0 = time.perf_counter()
                    sha_spec = mssi.Sha256MerkleSpecV1(depth=int(SHA_DEPTH))
                    poseidon_spec = mssi.PoseidonMerkleSpecV1(
                        arity=int(POSEIDON_ARITY), depth=int(POSEIDON_DEPTH)
                    )
                    computed = mssi.build_dual_roots_from_record_bytes_v1(
                        record_bytes_list,
                        sha_spec=sha_spec,
                        poseidon_spec=poseidon_spec,
                        pad_leaf_field=0,
                    )
                    rsu_root_sha256_hex = str(computed["root_sha256_hex"])
                    rsu_root_poseidon_field = int(computed["root_poseidon_field"])
                    poseidon_pad_leaf_field = str(computed.get("poseidon_pad_leaf_field", "0"))
                    # ✅ Step 2: write manifest via authoritative helper
                    # SSI totals for this RSU round (evidence persisted)
                    ssi_total = int(len(records_compact))
                    ssi_ok_count = 0
                    for rr in records_compact:
                        try:
                            ssi_ok_count += 1 if int(rr.get("record", {}).get("ssi_verify_ok", 0)) == 1 else 0
                        except Exception:
                            pass
                    mssi.write_round_roots_manifest_v1(
                        rsu_manifest_path,
                        round_idx=int(rnd),
                        rsu_id=int(self.rsu_id),
                        root_sha256_hex=rsu_root_sha256_hex,
                        root_poseidon_field=str(rsu_root_poseidon_field),
                        identity_state_root_field="0",
                        poseidon_pad_leaf_field=str(poseidon_pad_leaf_field),
                        extra={
                            "scope": "rsu_round",
                            "poseidon_arity": int(POSEIDON_ARITY),
                            "poseidon_depth": int(POSEIDON_DEPTH),
                            "sha_depth": int(SHA_DEPTH),
                            "client_ids_sorted": manifest_cids,
                            "clients_with_valid_anchor_payload": clients_with_valid_anchor_payload,
                            "excluded_client_ids": excluded_vehicle_cids,
                            "N_total_decoded": int(n_total_anchor_decoded),
                            "N_used_for_anchorsum": int(rsu_anchor_N_used),
                            # SSI evidence summary
                            "ssi_verify_total": int(ssi_total),
                            "ssi_verify_ok": int(ssi_ok_count),
                            "ssi_verify_fail": int(ssi_total - ssi_ok_count),
                            "records": records_compact,
                        },
                    )
                    # ✅ Step 3: auditor recomputation check (FAILS if drift)
                    canon_scheme = getattr(canon, "CANON_SCHEME", "JCS/RFC8785")
                    record_count = int(len(record_bytes_list))
                    audit_ok = True
                    try:
                        mssi.audit_round_roots_manifest_v1(
                            rsu_manifest_path,
                            record_bytes_list=record_bytes_list,
                            sha_spec=sha_spec,
                            poseidon_spec=poseidon_spec,
                        )
                    except Exception:
                        audit_ok = False
                        raise
                    # ✅ After audit succeeds, rewrite manifest via authoritative helper (no manual json editing)
                    try:
                        record_sha256_hex_list = [hashlib.sha256(rb).hexdigest() for rb in record_bytes_list]
                        # Start from the same extra you already wrote earlier, then add audit fields
                        extra_final = {
                            "scope": "rsu_round",
                            "poseidon_arity": int(POSEIDON_ARITY),
                            "poseidon_depth": int(POSEIDON_DEPTH),
                            "sha_depth": int(SHA_DEPTH),
                            "client_ids_sorted": manifest_cids,
                            "clients_with_valid_anchor_payload": clients_with_valid_anchor_payload,
                            "excluded_client_ids": excluded_vehicle_cids,
                            "N_total_decoded": int(n_total_anchor_decoded),
                            "N_used_for_anchorsum": int(rsu_anchor_N_used),
                            "ssi_verify_total": int(ssi_total),
                            "ssi_verify_ok": int(ssi_ok_count),
                            "ssi_verify_fail": int(ssi_total - ssi_ok_count),
                            "records": records_compact,
                            # ✅ Canonical audit evidence (stable fields for on-chain export)
                            "canon_scheme": str(canon_scheme),
                            "record_count": int(record_count),
                            "auditor_recompute_ok": bool(audit_ok),
                            "record_sha256_hex_list": record_sha256_hex_list,
                            # ✅ SSI preimage-definition fingerprint (stable cross-layer)
                            "ssi_preimage_def_sha256_v1": str(ssi_preimage_def_sha256_v1_hex),
                            "ssi_preimage_def_sha256": str(ssi_preimage_def_sha256_v1_hex),
                            "ssi_preimage_def_field_bn254_v1": int(ssi_preimage_def_field_bn254_v1),
                            "ssi_preimage_def_field_bn254": int(ssi_preimage_def_field_bn254_v1),
                        }
                        mssi.write_round_roots_manifest_v1(
                            rsu_manifest_path,
                            round_idx=int(rnd),
                            rsu_id=int(self.rsu_id),
                            root_sha256_hex=str(rsu_root_sha256_hex),
                            root_poseidon_field=str(rsu_root_poseidon_field),
                            identity_state_root_field="0",
                            poseidon_pad_leaf_field=str(poseidon_pad_leaf_field),
                            extra=extra_final,
                        )
                    except Exception as exc_mani:
                        logging.warning(
                            "[MERKLE][RSU %d] Round %d: could not rewrite manifest via helper: %s",
                            int(self.rsu_id), int(rnd), exc_mani
                        )
                    # 3) Log (now self-evident)
                    rsu_merkle_build_latency_sec = time.perf_counter() - _t_rsu_merkle0

                    logging.info(
                        "[MERKLE][RSU %d] Round %d: canon=%s records=%d audit=%s | root_sha256=%s | root_poseidon=%d | manifest=%s",
                        int(self.rsu_id), int(rnd),
                        str(canon_scheme), int(record_count),
                        "OK" if audit_ok else "FAIL",
                        str(rsu_root_sha256_hex), int(rsu_root_poseidon_field),
                        str(rsu_manifest_path),
                    )
                    # 4) Save for GLOBAL stage
                    self.rsu_root_poseidon_by_round[rnd] = int(rsu_root_poseidon_field)
                    self.rsu_round_manifest_path[rnd] = str(rsu_manifest_path)
                except Exception:
                    logging.exception(
                        "[MERKLE][RSU %d] Round %d: manifest/audit FAILED",
                        int(self.rsu_id),
                        int(rnd),
                    )
                    # ✅ “Be sure” means: do not silently continue with fake roots
                    if STRICT_MERKLE_AUDIT:
                        raise
                    rsu_root_poseidon_field = 0
                    rsu_root_sha256_hex = ""
                # ------------------------------------------------------------------
                # AnchorZKP: RSU-level aggregation verification (P0 audit lane)
                # ------------------------------------------------------------------
                # ✅ Critical fix: skip BEFORE proving if N_used==0
                if rsu_anchor_N_used <= 0:
                    rsu_anchor_ok = 0
                    Q_rsu_b64 = ""
                    rsu_commit_field = ""
                    global_commit_field = ""
                    self.rsu_round_valid[rnd] = 0
                    self.rsu_anchor_artifact[rnd] = ""
                    logging.warning(
                        "[ZKP][RSU %d] Round %d: N_used=0, skipping anchor-sum proof.",
                        self.rsu_id, rnd
                    )
                else:
                    # Load meta (binds anchor_version / M / SCALE)
                    meta_path = str(self.anchor_ctx.get("anchor_meta_path", ""))
                    if not meta_path or not os.path.exists(meta_path):
                        raise FileNotFoundError(f"anchor_meta_path not found: {meta_path}")
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = azkp.AnchorMeta.from_dict(json.load(f))
                    # ---- Consistency checks (RSU) ----
                    if str(getattr(meta, "anchor_version", "")) != str(self.anchor_version):
                        raise ValueError("anchor_version mismatch")
                    if int(getattr(meta, "M", -1)) != int(self.anchor_M):
                        raise ValueError("M mismatch")
                    if int(getattr(meta, "SCALE", -1)) != int(self.anchor_SCALE):
                        raise ValueError("SCALE mismatch")
                    base_out = Path(self.output_dir).resolve()
                    cfg = azkp.AnchorZKPConfig(
                        M=_as_int(self.anchor_M, 0),
                        SCALE=_as_int(self.anchor_SCALE, 0),
                        enable_range_checks=False,
                        generated_circuits_dir=(base_out / "circuits_generated" / "anchorsum"),
                        artifacts_dir=(base_out / "zkp_artifacts" / "anchorsum"),
                        run_root_dir=(base_out / "zkp_runs" / "anchorsum"),
                    )
                    # Anchor meta root (this is what the AnchorZKP circuit should bind to)
                    anchor_root_poseidon_field = _as_int(getattr(meta, "root_poseidon_field", 0), 0)
                    if anchor_root_poseidon_field <= 0:
                        anchor_root_poseidon_field = _as_int(
                            self.anchor_ctx.get("anchor_root_poseidon_field",
                                                self.anchor_ctx.get("root_poseidon_field", 0)),
                            0
                        )
                    # ✅ V15 public-input gating fields (define upfront so never “maybe undefined”)
                    pins_hash_field: int = 0
                    policy_id_field: int = 0
                    public_input_order_id_field: int = 0
                    # ✅ Deterministic actor-local computation (works even when env vars are missing)
                    prime: int = int(getattr(cfg, "prime"))
                    m_val: int = int(getattr(cfg, "M"))
                    scale_val: int = int(getattr(cfg, "SCALE"))
                    nmax_val: int = int(NMAX)
                    selection_mode: str = "pubmask"
                    range_checks_bit: int = 1 if bool(getattr(cfg, "enable_range_checks", False)) else 0
                    pins_hash_field = _sha256_to_nontrivial_field_v1(
                        "pins_v1|rsu_anchorsum|"
                        f"selection={selection_mode}|"
                        f"M={m_val}|SCALE={scale_val}|NMAX={nmax_val}|"
                        f"enable_range_checks={range_checks_bit}",
                        prime,
                    )
                    policy_id_field = _sha256_to_nontrivial_field_v1(
                        "policy_v1|rsu_anchorsum|"
                        f"selection={selection_mode}|"
                        f"M={m_val}|SCALE={scale_val}|"
                        f"enable_range_checks={range_checks_bit}",
                        prime,
                    )
                    public_input_order_id_field = _sha256_to_nontrivial_field_v1(
                        "pio_v1|rsu_anchorsum|"
                        "root_poseidon_field,anchor_root_poseidon_field,pins_hash_field,policy_id_field,public_input_order_id_field",
                        prime,
                    )
                    # ✅ Optional env override (only if non-trivial); otherwise keep computed values above
                    try:
                        _env_pins = int(str(os.environ.get("ZK_PINS_HASH_FIELD", "")).strip() or "0")
                        if _env_pins not in (0, 1):
                            pins_hash_field = int(_env_pins) % prime
                        _env_policy = int(str(os.environ.get("ZK_POLICY_ID_FIELD", "")).strip() or "0")
                        if _env_policy not in (0, 1):
                            policy_id_field = int(_env_policy) % prime
                        _env_pio = int(str(os.environ.get("ZK_PUBLIC_INPUT_ORDER_ID_FIELD", "")).strip() or "0")
                        if _env_pio not in (0, 1):
                            public_input_order_id_field = int(_env_pio) % prime
                    except Exception:
                        pass
                    _t_zkp0 = time.perf_counter()
                    art = prove_verify_rsu_anchor_sum_from_meta_compat(
                        cfg=cfg,
                        meta=meta,
                        anchor_id_field=int(str(self.anchor_id_field)),
                        round_idx=int(rnd),
                        rsu_id=int(self.rsu_id),
                        q_list=q_list,
                        mask_list=None,
                        NMAX=int(NMAX),
                        out_dir=cfg.artifacts_dir,
                        include_Q_b64=True,
                        root_poseidon_field=int(rsu_root_poseidon_field),
                        anchor_root_poseidon_field=int(anchor_root_poseidon_field),
                        pins_hash_field=int(pins_hash_field),
                        policy_id_field=int(policy_id_field),
                        public_input_order_id_field=int(public_input_order_id_field),
                    )
                    rsu_zkp_latency_sec = time.perf_counter() - _t_zkp0
                    Q_rsu_b64 = str(art.payload.get("Q_rsu_b64", "") or "")
                    # RSU commit key in utils is agg_commit_field (be tolerant to key drift)
                    rsu_commit_field = (
                            str(art.public.get("agg_commit_field", "") or "")
                            or str(art.public.get("rsu_commit_field", "") or "")
                            or str(art.public.get("commit_field", "") or "")
                    )
                    global_commit_field = str(rsu_commit_field)  # carry forward for GLOBAL stage
                    rsu_anchor_ok = 1 if (bool(art.ok) and bool(Q_rsu_b64)) else 0
                    # 1) Prefer the path returned by the utils (authoritative)
                    rsu_anchor_proof_path = ""
                    try:
                        if isinstance(getattr(art, "payload", None), dict):
                            rsu_anchor_proof_path = str(art.payload.get("artifact_path", "") or "")
                    except Exception:
                        rsu_anchor_proof_path = ""
                    # 2) Fallback to your convention (only if utils didn’t return one)
                    if not rsu_anchor_proof_path:
                        rsu_anchor_proof_path = os.path.join(
                            str(cfg.artifacts_dir),
                            f"rsu_{self.rsu_id}",
                            f"round_{rnd}.json",
                        )
                    # 3) Fail fast if missing (prevents “ok but no proof visible”)
                    if STRICT_MERKLE_AUDIT and not os.path.exists(rsu_anchor_proof_path):
                        raise FileNotFoundError(
                            f"[ZKP][RSU {self.rsu_id}] Expected proof artifact missing: {rsu_anchor_proof_path}"
                        )
                    # ✅ Copy proof to a stable, shared location for packaging/audit (+ public-input sidecar)
                    stable_proof_copy = os.path.join(
                        str(self.output_dir),
                        "zkp_anchor_proofs",
                        "anchorsum",
                        f"rsu_{self.rsu_id}",
                        f"round_{rnd}.json",
                    )
                    os.makedirs(os.path.dirname(stable_proof_copy), exist_ok=True)
                    # Always-defined locals (safe for later artifact_norm / index)
                    public_sidecar_src = ""
                    public_sidecar_sha256 = ""
                    public_sidecar_copy = ""
                    stable_proof_copy_sha256 = ""
                    proof_copy_relpath = ""
                    def _sha256_file_v1(p: str) -> str:
                        h = hashlib.sha256()
                        with open(p, "rb") as f_in:
                            for chunk in iter(lambda: f_in.read(1024 * 1024), b""):
                                h.update(chunk)
                        return h.hexdigest()
                    def _relpath_posix_v1(p: str, base: str) -> str:
                        return os.path.relpath(p, base).replace("\\", "/")
                    try:
                        base_abs = str(Path(self.output_dir).resolve())
                        # ---- 1) Copy the RSU proof JSON ----
                        with open(rsu_anchor_proof_path, "r", encoding="utf-8") as pf:
                            proof_obj = json.load(pf)
                        with open(stable_proof_copy, "w", encoding="utf-8") as outf:
                            json.dump(proof_obj, outf, indent=2)
                        stable_proof_copy_sha256 = _sha256_file_v1(stable_proof_copy)
                        proof_copy_relpath = _relpath_posix_v1(stable_proof_copy, base_abs)
                        # ---- 2) Discover + copy the stable public-input sidecar if present ----
                        payload = proof_obj.get("payload", {}) if isinstance(proof_obj, dict) else {}
                        public_sidecar_src = str(
                            payload.get("public_sidecar_path", "")
                            or payload.get("public_inputs_sidecar_path", "")
                            or payload.get("public_inputs_v1_path", "")
                            or proof_obj.get("public_sidecar_path", "")
                            or ""
                        ).strip()
                        public_sidecar_sha256 = str(
                            payload.get("public_sidecar_sha256", "")
                            or payload.get("public_inputs_sidecar_sha256", "")
                            or proof_obj.get("public_sidecar_sha256", "")
                            or ""
                        ).strip()
                        # Normalize to absolute path if it exists
                        if public_sidecar_src:
                            try:
                                public_sidecar_src = str(Path(public_sidecar_src).resolve())
                            except Exception:
                                pass
                        if public_sidecar_src and os.path.exists(public_sidecar_src):
                            public_sidecar_copy = os.path.join(
                                str(self.output_dir),
                                "zkp_public_inputs",
                                "anchorsum",
                                f"rsu_{self.rsu_id}",
                                f"round_{rnd}_public_inputs_v1.json",
                            )
                            os.makedirs(os.path.dirname(public_sidecar_copy), exist_ok=True)
                            shutil.copyfile(public_sidecar_src, public_sidecar_copy)
                            # If sha wasn’t provided or invalid, compute it from the copied bytes
                            if not _looks_like_sha256_hex(public_sidecar_sha256):
                                public_sidecar_sha256 = _sha256_file_v1(public_sidecar_copy)
                    except Exception as exc_copy:
                        if STRICT_MERKLE_AUDIT:
                            raise
                        logging.warning(
                            "[ZKP][RSU %d] Could not copy proof/public sidecar: %s",
                            self.rsu_id,
                            exc_copy,
                        )
                    # ✅ (RE-ADD) Your normalized summary path (safe, separate)
                    rsu_anchor_summary_path = os.path.join(
                        str(self.output_dir),
                        "zkp_anchor_summaries",
                        "anchorsum",
                        f"rsu_{self.rsu_id}",
                        f"round_{rnd}.json",
                    )
                    artifact_norm = {
                        "ok": bool(rsu_anchor_ok),
                        "round": int(rnd),
                        "rsu_id": int(self.rsu_id),
                        "root_sha256_hex": str(rsu_root_sha256_hex),
                        "root_poseidon_field": int(rsu_root_poseidon_field),
                        "rsu_round_manifest_path": str(self.rsu_round_manifest_path.get(rnd, "")),
                        "anchor_version": str(self.anchor_version),
                        "anchor_id_field": str(self.anchor_id_field),
                        "M": int(self.anchor_M),
                        "SCALE": int(self.anchor_SCALE),
                        "N_used": int(rsu_anchor_N_used),
                        "Q_rsu_b64": str(Q_rsu_b64),
                        "rsu_commit_field": str(rsu_commit_field),
                        "global_commit_field": str(rsu_commit_field),
                        "anchor_root_poseidon_field": int(anchor_root_poseidon_field),
                        "proof_artifact_path": str(rsu_anchor_proof_path),
                        "proof_artifact_copy_path": str(stable_proof_copy),
                        "proof_artifact_copy_relpath": str(proof_copy_relpath),
                        "proof_artifact_copy_sha256": str(stable_proof_copy_sha256),
                        "public_inputs_sidecar_path": str(public_sidecar_src),
                        "public_inputs_sidecar_copy_path": str(public_sidecar_copy),
                        "public_inputs_sidecar_copy_relpath": (
                            _relpath_posix_v1(str(public_sidecar_copy), str(self.output_dir))
                            if public_sidecar_copy else ""
                        ),
                        "public_inputs_sidecar_sha256": str(public_sidecar_sha256),
                        "pins_hash_field": int(pins_hash_field),
                        "policy_id_field": int(policy_id_field),
                        "public_input_order_id_field": int(public_input_order_id_field),
                        "selection_mode": "pubmask",
                        # ✅ NEW: GLOBAL exclusion fix — persist SSI preimage-definition fingerprint in the summary itself
                        "ssi_preimage_def_sha256_v1": str(ssi_preimage_def_sha256_v1_hex),
                        "ssi_preimage_def_sha256": str(ssi_preimage_def_sha256_v1_hex),
                        "ssi_preimage_def_field_bn254_v1": int(ssi_preimage_def_field_bn254_v1),
                        "ssi_preimage_def_field_bn254": int(ssi_preimage_def_field_bn254_v1),
                    }
                    os.makedirs(os.path.dirname(rsu_anchor_summary_path), exist_ok=True)
                    with open(rsu_anchor_summary_path, "w", encoding="utf-8") as f:
                        json.dump(artifact_norm, f, indent=2)
                    # Track validity + summary path (what GLOBAL should read)
                    self.rsu_round_valid[rnd] = int(rsu_anchor_ok)
                    self.rsu_anchor_artifact[rnd] = str(rsu_anchor_summary_path)
                    if int(rsu_anchor_ok) == 1:
                        logging.info(
                            "[ZKP][RSU %d] ✅ AnchorSum verification SUCCEEDED | round=%d N_used=%d root_poseidon_field=%d summary=%s proof=%s",
                            self.rsu_id, rnd, rsu_anchor_N_used, int(rsu_root_poseidon_field),
                            rsu_anchor_summary_path, rsu_anchor_proof_path
                        )
                    else:
                        logging.error(
                            "[ZKP][RSU %d] ❌ AnchorSum verification FAILED | round=%d N_used=%d summary=%s proof=%s",
                            self.rsu_id, rnd, rsu_anchor_N_used,
                            rsu_anchor_summary_path, rsu_anchor_proof_path
                        )
                logging.info(
                    "[ZKP][RSU %d] Round %d anchor-sum proof ok=%d (N_used=%d) artifact=%s",
                    int(self.rsu_id),
                    int(rnd),
                    int(rsu_anchor_ok),
                    int(rsu_anchor_N_used),
                    str(self.rsu_anchor_artifact.get(rnd, "")),
                )
        except Exception:
            self.rsu_round_valid[rnd] = 0
            self.rsu_anchor_artifact[rnd] = ""
            logging.exception(
                "[ZKP][RSU %d] Round %d: RSU round verification pipeline FAILED",
                self.rsu_id,
                rnd,
            )
            if STRICT_MERKLE_AUDIT:
                raise
        rsu_verification_pipeline_latency_sec = time.perf_counter() - _t_verify0
        rsu_round_total_latency_sec = time.perf_counter() - rsu_round_t0

        # Add to round_summary (JSON-friendly)
        round_summary["rsu_preselect_latency_sec"] = float(rsu_preselect_latency_sec)
        round_summary["rsu_model_aggregation_latency_sec"] = float(rsu_model_aggregation_latency_sec)
        round_summary["rsu_ssi_verification_latency_sec"] = float(rsu_ssi_verification_latency_sec)
        round_summary["rsu_merkle_build_latency_sec"] = float(rsu_merkle_build_latency_sec)
        round_summary["rsu_verification_pipeline_latency_sec"] = float(rsu_verification_pipeline_latency_sec)
        round_summary["rsu_zkp_latency_sec"] = float(rsu_zkp_latency_sec)
        round_summary["rsu_round_total_latency_sec"] = float(rsu_round_total_latency_sec)

        round_summary["rsu_anchor_zkp_ok"] = int(rsu_anchor_ok)
        round_summary["reviewer3_full_evidence_materialized"] = int(
            reviewer3_full_evidence_round
        )
        round_summary["rsu_anchor_N_used"] = int(rsu_anchor_N_used)
        round_summary["rsu_anchor_artifact"] = str(self.rsu_anchor_artifact.get(rnd, ""))
        # ✅ Persist public-input gating fields for GLOBAL stage reuse (prevents spec-derivation fallback)
        round_summary["pins_hash_field"] = int(pins_hash_field)
        round_summary["policy_id_field"] = int(policy_id_field)
        round_summary["public_input_order_id_field"] = int(public_input_order_id_field)
        # Reviewer 3 participation/dropout accounting
        round_summary["fit_expected_clients"] = int(self.anchor_NMAX)
        round_summary["fit_results_received"] = int(reviewer3_fit_results_received)
        round_summary["fit_failures_observed"] = int(reviewer3_fit_failures_observed)
        round_summary["fit_success_fraction"] = (
            float(reviewer3_fit_results_received) / float(self.anchor_NMAX)
            if int(self.anchor_NMAX) > 0 else 0.0
        )
        round_summary["fit_failure_proxy_cids"] = list(reviewer3_failure_proxy_cids)
        # NEW: audit who was excluded by the RSU gating
        round_summary["excluded_client_ids"] = excluded_vehicle_cids
        round_summary["preselect_exclusion_details"] = preselect_exclusion_details
        # ---- NEW: Phase-2 bridge outputs for GLOBAL stage ----
        round_summary["rsu_root_sha256_hex"] = str(rsu_root_sha256_hex)
        round_summary["rsu_root_poseidon_field"] = int(rsu_root_poseidon_field)
        round_summary["rsu_round_manifest_path"] = str(self.rsu_round_manifest_path.get(rnd, ""))
        round_summary["Q_rsu_b64"] = str(Q_rsu_b64)
        round_summary["global_commit_field"] = str(global_commit_field)

        manifest_path_cur = str(self.rsu_round_manifest_path.get(rnd, ""))
        round_summary["rsu_manifest_size_bytes"] = (
            int(os.path.getsize(manifest_path_cur))
            if manifest_path_cur and os.path.exists(manifest_path_cur)
            else 0
        )

        proof_summary_path_cur = str(self.rsu_anchor_artifact.get(rnd, ""))
        round_summary["rsu_anchor_summary_size_bytes"] = (
            int(os.path.getsize(proof_summary_path_cur))
            if proof_summary_path_cur and os.path.exists(proof_summary_path_cur)
            else 0
        )

        proof_copy_path_cur = ""
        public_sidecar_copy_path_cur = ""
        try:
            if proof_summary_path_cur and os.path.exists(proof_summary_path_cur):
                with open(proof_summary_path_cur, "r", encoding="utf-8") as f:
                    anchor_summary_obj = json.load(f)
                proof_copy_path_cur = str(anchor_summary_obj.get("proof_artifact_copy_path", "") or "")
                public_sidecar_copy_path_cur = str(anchor_summary_obj.get("public_inputs_sidecar_copy_path", "") or "")
        except Exception:
            proof_copy_path_cur = ""
            public_sidecar_copy_path_cur = ""

        round_summary["rsu_proof_artifact_size_bytes"] = (
            int(os.path.getsize(proof_copy_path_cur))
            if proof_copy_path_cur and os.path.exists(proof_copy_path_cur)
            else 0
        )
        round_summary["rsu_public_inputs_sidecar_size_bytes"] = (
            int(os.path.getsize(public_sidecar_copy_path_cur))
            if public_sidecar_copy_path_cur and os.path.exists(public_sidecar_copy_path_cur)
            else 0
        )
        round_summary["selection_mode"] = "pubmask"
        round_summary["ssi_preimage_def_sha256_v1"] = str(ssi_preimage_def_sha256_v1_hex)
        round_summary["ssi_preimage_def_sha256"] = str(ssi_preimage_def_sha256_v1_hex)
        round_summary["ssi_preimage_def_field_bn254_v1"] = int(ssi_preimage_def_field_bn254_v1)
        round_summary["ssi_preimage_def_field_bn254"] = int(ssi_preimage_def_field_bn254_v1)
        # SSI evidence summary (RSU verified)
        try:
            with open(str(self.rsu_round_manifest_path.get(rnd, "")), "r", encoding="utf-8") as f:
                mani_tmp = json.load(f)
            extra_tmp = mani_tmp.get("extra", {}) if isinstance(mani_tmp, dict) else {}
            round_summary["ssi_verify_total"] = int(extra_tmp.get("ssi_verify_total", 0))
            round_summary["ssi_verify_ok"] = int(extra_tmp.get("ssi_verify_ok", 0))
            round_summary["ssi_verify_fail"] = int(extra_tmp.get("ssi_verify_fail", 0))
        except Exception:
            round_summary["ssi_verify_total"] = 0
            round_summary["ssi_verify_ok"] = 0
            round_summary["ssi_verify_fail"] = 0
        self.round_summaries.append(round_summary)
        return aggregated_params, agg_metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[
                Tuple[ClientProxy, EvaluateRes] | BaseException
                ],
    ) -> Tuple[float | None, Dict[str, Scalar]]:
        rnd = int(server_round)
        """Manually aggregate distributed evaluation loss/metrics.

        This avoids relying on Flower's History(loss, distributed), which can be
        unreliable for the XGBoost bagging path in this runtime.
        """
        if not results:
            return None, {}

        # Weighted average of client losses
        total_examples = 0
        weighted_loss_sum = 0.0
        metrics_records: List[Tuple[int, Dict[str, float]]] = []

        for _, eval_res in results:
            n = int(eval_res.num_examples)
            total_examples += n
            weighted_loss_sum += n * float(eval_res.loss)

            numeric_metrics: Dict[str, float] = {}
            for k, v in (eval_res.metrics or {}).items():
                if isinstance(v, (int, float)):
                    numeric_metrics[k] = float(v)
            metrics_records.append((n, numeric_metrics))

        aggregated_loss = (
            weighted_loss_sum / float(total_examples)
            if total_examples > 0
            else None
        )

        aggregated_metrics: Dict[str, Scalar] = {}
        if metrics_records:
            try:
                aggregated_metrics = weighted_average_eval_metrics(metrics_records)
            except Exception as exc:
                logging.warning(
                    "RSU %d - Round %d: failed to aggregate evaluation metrics: %s",
                    self.rsu_id,
                    rnd,
                    exc,
                )
                aggregated_metrics = {}

        # Persist manual history
        if aggregated_loss is not None:
            self.losses_distributed_manual.append((int(rnd), float(aggregated_loss)))

        for k, v in aggregated_metrics.items():
            if isinstance(v, (int, float)):
                self.metrics_distributed_evaluate_manual.setdefault(k, []).append(
                    (int(rnd), float(v))
                )

        logging.info(
            "RSU %d - Round %d: manual distributed eval loss=%.8f metrics=%s",
            self.rsu_id,
            rnd,
            float(aggregated_loss) if aggregated_loss is not None else float("nan"),
            aggregated_metrics,
        )

        return aggregated_loss, aggregated_metrics
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
    feature_min: np.ndarray,
    feature_max: np.ndarray,
    anchor_ctx: Dict[str, Any],  # <-- NEW
    ablation_cfg: Dict[str, Any] | None = None,
) -> None:
    """
    Run a full FL process for one RSU over its vehicles.
    This version uses the Flower App-based runtime (ClientApp/ServerApp)
    and flwr.simulation.run_simulation instead of legacy start_simulation.
    """
    # NOTE: start_simulation is no longer used; we rely on run_simulation
    os.makedirs(rsu_cfg.output_dir, exist_ok=True)
    # ✅ isolate all RSU artifacts to a per-RSU folder
    rsu_run_dir = os.path.join(rsu_cfg.output_dir, f"rsu_{rsu_cfg.rsu_id}")
    os.makedirs(rsu_run_dir, exist_ok=True)
    # Reviewer-3 stress run:
    # do not replicate the full shared validation/test matrices into every
    # Ray client. Full VAL/TEST evaluation remains enabled after RSU training.
    feature_count = int(len(feature_min))

    X_val_np = np.empty((0, feature_count), dtype=np.float32)
    y_val_np = np.empty((0,), dtype=np.int32)

    X_test_np = np.empty((0, feature_count), dtype=np.float32)
    y_test_np = np.empty((0,), dtype=np.int32)

    logging.info(
        "[REVIEWER3][FAST] Per-client VAL/TEST matrices disabled; "
        "full held-out evaluation remains at the final RSU/GLOBAL stage."
    )
    # Prepare XGBoost config
    xgb_cfg = XGBoostConfig(
        params=dict(xgb_params),  # ✅ isolate per RSU
        num_local_rounds=rsu_cfg.num_local_rounds,
        early_stopping_rounds=None,
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
        X_v_np = np.asarray(
            np.clip(
                X_v.to_numpy(dtype=np.float32, copy=False),
                feature_min,
                feature_max,
            ),
            dtype=np.float32,
        )
        y_v_np = np.asarray(
            y_v.to_numpy(copy=False),
            dtype=np.int32,
        )

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
        # V15 safety: enforce required pins in the Ray actor process explicitly
        for _k, _v in env_v15.items():
            os.environ[str(_k)] = str(_v)
        # Sanity check: we must have at least one vehicle partition
        num_parts = len(vehicle_partitions)
        if num_parts <= 0:
            raise RuntimeError("vehicle_partitions is empty, cannot build clients")
        # -------------------------------------------------------------------
        # V15: enforce PUBMASK-only inside Ray actor (evidence in actor logs)
        # -------------------------------------------------------------------
        require_pubmask_raw = getattr(
            azkp,
            "_require_pubmask_only_v15",
            None,
        )

        if not callable(require_pubmask_raw):
            raise RuntimeError(
                "Missing azkp._require_pubmask_only_v15; "
                "update AnchorZKP utils to V15 PUBMASK-only."
            )

        require_pubmask = cast(
            Callable[..., Any],
            require_pubmask_raw,
        )

        require_pubmask(
            anchor_ctx,
            where=f"RSU {rsu_cfg.rsu_id} client_fn (Ray actor)",
        )
        # --- V15: deterministic, base-aware partition mapping (no Ray node_id fallback) ---
        def _resolve_partition_idx0_v15(ctx, *, num_parts_local: int, partition_id_base_default: int) -> tuple[
            int, int, dict]:
            node_cfg_raw = getattr(ctx, "node_config", None)
            if not isinstance(node_cfg_raw, dict):
                raise RuntimeError(f"node_config is not a dict: {node_cfg_raw!r}")
            node_cfg = dict(node_cfg_raw)  # copy so we can safely inject fields for logging
            if "partition-id" not in node_cfg:
                raise RuntimeError(f"node_config missing 'partition-id': {node_cfg!r}")
            pid = int(node_cfg["partition-id"])
            num_partitions_local = int(node_cfg.get("num-partitions", num_parts_local))
            # Flower examples use 0-based partition-id (0..num-partitions-1).
            base_raw = node_cfg.get("partition-id-base", None)
            if base_raw is None:
                # Prefer 0-based when pid is in-range, otherwise fall back to 1-based if that fits.
                # Flower docs/examples commonly use 0-based partition-id.
                if 0 <= pid < num_parts_local:
                    base = 0
                elif 1 <= pid <= num_parts_local:
                    base = 1
                else:
                    base = int(partition_id_base_default)
                node_cfg["partition-id-base"] = base
            else:
                base_opt = _reviewer3_optional_int_v1(base_raw)
                if base_opt is None:
                    raise RuntimeError(
                        f"invalid partition-id-base={base_raw!r}"
                    )
                base = base_opt
            if base not in (0, 1):
                raise RuntimeError(f"invalid partition-id-base={base!r} (must be 0 or 1)")
            idx0 = pid - base
            return idx0, num_partitions_local, node_cfg
        try:
            partition_id_base_default_env = int(os.environ.get("PARTITION_ID_BASE_DEFAULT", "0"))
            partition_idx0, num_partitions, node_cfg = _resolve_partition_idx0_v15(
                context,
                num_parts_local=num_parts,
                partition_id_base_default=partition_id_base_default_env,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[RSU {rsu_cfg.rsu_id}] Failed to resolve partition index from node_config: {exc}. "
                f"node_config={getattr(context, 'node_config', None)!r}"
            ) from exc
        if not (0 <= partition_idx0 < num_parts):
            raise RuntimeError(
                f"[RSU {rsu_cfg.rsu_id}] partition_idx0 {partition_idx0} is outside "
                f"[0, {num_parts - 1}] for {num_parts} vehicle partitions "
                f"(num-partitions={num_partitions}, node_config={node_cfg!r})"
            )
        logging.info(
            "[RSU %d] Building client | partition_idx0=%d | num_parts=%d | node_config=%s",
            rsu_cfg.rsu_id,
            partition_idx0,
            num_parts,
            node_cfg,
        )
        # Select this vehicle's train split
        X_v_np, y_v_np = vehicle_partitions[partition_idx0]
        # Stable vehicle identity: rsu_id*1000 + 1..N
        vehicle_id = rsu_cfg.rsu_id * 1000 + (partition_idx0 + 1)
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
            feature_min=feature_min,
            feature_max=feature_max,
            dp_output_dir=rsu_cfg.output_dir,
            anchor_ctx=anchor_ctx,  # <-- NEW
            ablation_cfg=ablation_cfg,
        )
    client_app = ClientApp(client_fn=client_fn)
    # Shared strategy instance so we can access latest_global_params after simulation
    strategy = XgbBaggingStrategy(
        rsu_id=rsu_cfg.rsu_id,
        vehicles_per_rsu=len(vehicle_partitions),
        num_rounds=rsu_cfg.num_rounds,
        num_local_rounds=rsu_cfg.num_local_rounds,
        output_dir=rsu_run_dir,  # ✅ prevents RSU overwrite
        anchor_ctx=anchor_ctx,  # <-- NEW
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
    # -------------------------------------------------------------------
    # V15: FORCE PUBMASK-only pins into the process env (do NOT use setdefault)
    # AND push them into Ray runtime_env when supported.
    # This prevents “driver has pins, Ray actors don’t” and prevents stale env drift.
    # -------------------------------------------------------------------
    env_v15 = {
        "ZK_SELECTION_MODE": "pubmask",
        "ZK_CIRCUIT_NAME": "rootaware_pubmask",
        "STRICT_CIRCUITS_PRECOMPILE": "1",
        "PARTITION_ID_BASE_DEFAULT": "0",
    }
    for k, v in env_v15.items():
        os.environ[k] = v
    run_kwargs: dict[str, Any] = {
        "server_app": server_app,
        "client_app": client_app,
        "num_supernodes": len(vehicle_partitions),
    }
    # IMPORTANT (V15): Do NOT inject runtime_env via init_args or ray_init_args.
    # Flower’s Ray backend manages ray.init(...) and may already pass runtime_env.
    # We only set os.environ (see below) and also enforce pins inside client_fn.
    # (This block is intentionally removed to prevent duplicate runtime_env.)
    history = run_simulation(**run_kwargs)
    rsu_runtime = time.time() - rsu_start_time
    logging.info(
        "=== RSU %d: FL run completed in %.3f sec (≈ %.2f min) ===",
        rsu_cfg.rsu_id,
        rsu_runtime,
        rsu_runtime / 60.0,
    )
    # -------------------------------------------------------------------
    # V15: Emit stable RSU public sidecars (required for GLOBAL audit)
    # We write: <anchorsum_artifacts_dir>/rsu_<rsu_id>/round_<r>_public.json
    # from the actual proof artifacts produced by the RSU strategy.
    # -------------------------------------------------------------------
    try:
        artifacts_dir_str = str(anchor_ctx.get("anchorsum_artifacts_dir", "") or "")
        rsu_art_map = getattr(strategy, "rsu_anchor_artifact", {}) or {}
        if artifacts_dir_str and isinstance(rsu_art_map, dict):
            for rk, av in rsu_art_map.items():
                try:
                    r_int = int(rk)
                except Exception:
                    continue
                proof_path = ""
                if isinstance(av, str):
                    proof_path = av
                elif isinstance(av, dict):
                    proof_path = str(
                        av.get("proof_artifact_path")
                        or av.get("artifact_path")
                        or av.get("path")
                        or ""
                    )
                def _abs_existing(*bases: str, p: str) -> str:
                    if not p:
                        return ""
                    if os.path.isabs(p) and os.path.exists(p):
                        return p
                    for b in bases:
                        if not b:
                            continue
                        cand = os.path.join(str(b), str(p))
                        if os.path.exists(cand):
                            return cand
                    # last-resort: return absolute candidate under OUTPUT_DIR (even if missing)
                    return os.path.join(str(OUTPUT_DIR), str(p))
                if proof_path and not os.path.isabs(proof_path):
                    proof_path = _abs_existing(rsu_run_dir, rsu_cfg.output_dir, str(OUTPUT_DIR), p=proof_path)
                if not proof_path or not os.path.exists(proof_path):
                    continue
                with open(proof_path, "r", encoding="utf-8") as f:
                    art = json.load(f)
                # Follow wrapper pointer if needed
                if isinstance(art, dict) and (
                        "public_inputs" not in art and "publicSignals" not in art and "public" not in art
                ):
                    hinted = str(
                        art.get("proof_artifact_path", "") or art.get("artifact_path", "") or art.get("path", "") or "")
                    if hinted:
                        hinted_abs = hinted
                        if not os.path.isabs(hinted_abs):
                            hinted_abs = _abs_existing(rsu_run_dir, rsu_cfg.output_dir, str(OUTPUT_DIR), p=hinted_abs)
                        if os.path.exists(hinted_abs):
                            with open(hinted_abs, "r", encoding="utf-8") as ff:
                                art = json.load(ff)
                            proof_path = hinted_abs  # keep directory consistent for public.json fallback
                pub_inputs = _extract_public_inputs_v15(art)
                # V15: if artifact doesn't embed publics, derive them from snarkjs public output next to proof.
                if not pub_inputs:
                    pub_path = ""
                    if isinstance(art, dict):
                        pub_path = str(
                            art.get("public_json_path")
                            or art.get("public_path")
                            or art.get("publicSignalsPath")
                            or ""
                        )
                    if pub_path and not os.path.isabs(pub_path):
                        pub_path = _abs_existing(rsu_run_dir, rsu_cfg.output_dir, str(OUTPUT_DIR), p=pub_path)
                    if (not pub_path) or (not os.path.exists(pub_path)):
                        base_dir = os.path.dirname(proof_path)
                        for name in ("public.json", "pub.json", "publicSignals.json"):
                            cand = os.path.join(base_dir, name)
                            if os.path.exists(cand):
                                pub_path = cand
                                break
                    if pub_path and os.path.exists(pub_path):
                        with open(pub_path, "r", encoding="utf-8") as ff:
                            pub_obj = json.load(ff)
                        if isinstance(pub_obj, list):
                            pub_inputs = [str(x) for x in pub_obj]
                        elif isinstance(pub_obj, dict):
                            pub_inputs = _extract_public_inputs_v15(pub_obj)
                if not pub_inputs:
                    continue
                sidecar_path = os.path.join(
                    artifacts_dir_str,
                    f"rsu_{int(rsu_cfg.rsu_id)}",
                    f"round_{int(r_int)}_public.json",
                )
                _write_public_sidecar_v15(
                    sidecar_path,
                    pub_inputs,
                    who=f"rsu_{int(rsu_cfg.rsu_id)}",
                    round_idx=int(r_int),
                )
        else:
            logging.warning(
                "[RSU %d] Sidecar emit skipped: missing anchorsum_artifacts_dir or rsu_anchor_artifact map",
                rsu_cfg.rsu_id,
            )
    except Exception as exc:
        logging.warning("[RSU %d] Failed to emit RSU public sidecars: %s", rsu_cfg.rsu_id, exc)
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
            # ✅ keep saved model at the top level (so your centralized eval paths remain unchanged)
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
    losses_distributed_extracted, metrics_distributed_evaluate_extracted = (
        extract_history_distributed_summary(history)
    )

    losses_distributed_manual = getattr(strategy, "losses_distributed_manual", [])
    metrics_distributed_evaluate_manual = getattr(
        strategy, "metrics_distributed_evaluate_manual", {}
    )

    # Prefer manual aggregation because Flower's History(loss, distributed)
    # is unreliable in this XGBoost bagging path.
    losses_distributed = (
        losses_distributed_manual
        if losses_distributed_manual
        else losses_distributed_extracted
    )
    metrics_distributed_evaluate = (
        metrics_distributed_evaluate_manual
        if metrics_distributed_evaluate_manual
        else metrics_distributed_evaluate_extracted
    )

    logging.info(
        "RSU %d - FL losses_distributed (manual preferred): %s",
        rsu_cfg.rsu_id,
        losses_distributed,
    )
    logging.info(
        "RSU %d - FL metrics_distributed (evaluate, manual preferred): %s",
        rsu_cfg.rsu_id,
        metrics_distributed_evaluate,
    )
    # ---------------------------
    # Persist RSU-level summary as JSON
    # ---------------------------
    round_summaries = getattr(strategy, "round_summaries", [])
    final_round_summary = round_summaries[-1] if round_summaries else {}
    # -------------------------------------------------------------------
    # V15: persist SSI preimage fingerprint (audit lane)
    # -------------------------------------------------------------------
    fp_fn_raw = getattr(
        azkp,
        "ssi_preimage_fingerprint_v1",
        None,
    )

    if not callable(fp_fn_raw):
        raise RuntimeError(
            "Missing azkp.ssi_preimage_fingerprint_v1 "
            "(required for audit evidence)."
        )

    fp_fn = cast(Callable[[], Any], fp_fn_raw)
    fp_obj = fp_fn()

    if not isinstance(fp_obj, dict):
        raise RuntimeError(
            "ssi_preimage_fingerprint_v1 must return a dict"
        )

    ssi_fp: Dict[str, Any] = {
        str(k): v for k, v in fp_obj.items()
    }
    # -------------------------------------------------------------------
    # V15: expected stable public sidecars for RSU proofs
    # -------------------------------------------------------------------
    artifacts_dir_str = str(anchor_ctx.get("anchorsum_artifacts_dir", "") or "")
    if not artifacts_dir_str:
        raise RuntimeError("anchor_ctx missing anchorsum_artifacts_dir (required for sidecar evidence).")
    materialized_rounds = sorted(
        {
            int(r.get("round", 0) or 0)
            for r in round_summaries
            if isinstance(r, dict)
            and int(
                r.get(
                    "reviewer3_full_evidence_materialized",
                    0,
                )
                or 0
            ) == 1
        }
    )

    rsu_public_sidecar_by_round = {
        str(r): os.path.join(
            artifacts_dir_str,
            f"rsu_{int(rsu_cfg.rsu_id)}",
            f"round_{int(r)}_public.json",
        )
        for r in materialized_rounds
    }
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
        "overhead_summary": {
            "simulation_wall_time_sec": float(rsu_runtime),
            "sum_fit_total_upload_bytes": float(sum(float((r or {}).get("fit_total_upload_bytes", 0.0)) for r in round_summaries)),
            "sum_fit_total_download_bytes": float(sum(float((r or {}).get("fit_total_download_bytes", 0.0)) for r in round_summaries)),
            "sum_fit_total_bytes": float(sum(float((r or {}).get("fit_total_bytes", 0.0)) for r in round_summaries)),
            "sum_fit_total_train_time_sec": float(sum(float((r or {}).get("fit_total_train_time_sec", 0.0)) for r in round_summaries)),
            "sum_rsu_preselect_latency_sec": float(sum(float((r or {}).get("rsu_preselect_latency_sec", 0.0)) for r in round_summaries)),
            "sum_rsu_model_aggregation_latency_sec": float(sum(float((r or {}).get("rsu_model_aggregation_latency_sec", 0.0)) for r in round_summaries)),
            "sum_rsu_ssi_verification_latency_sec": float(sum(float((r or {}).get("rsu_ssi_verification_latency_sec", 0.0)) for r in round_summaries)),
            "sum_rsu_merkle_build_latency_sec": float(sum(float((r or {}).get("rsu_merkle_build_latency_sec", 0.0)) for r in round_summaries)),
            "sum_rsu_verification_pipeline_latency_sec": float(sum(float((r or {}).get("rsu_verification_pipeline_latency_sec", 0.0)) for r in round_summaries)),
            "sum_rsu_zkp_latency_sec": float(sum(float((r or {}).get("rsu_zkp_latency_sec", 0.0)) for r in round_summaries)),
            "sum_rsu_round_total_latency_sec": float(sum(float((r or {}).get("rsu_round_total_latency_sec", 0.0)) for r in round_summaries)),
        },
        "rsu_anchor_round_valid": getattr(strategy, "rsu_round_valid", {}),
        "rsu_anchor_artifacts": getattr(strategy, "rsu_anchor_artifact", {}),
        # NEW: persist anchor_ctx (at least the canonical fields auditors need)
        "anchor_ctx": {
            "anchor_version": str(anchor_ctx.get("anchor_version", "")),
            "anchor_id_field": str(anchor_ctx.get("anchor_id_field", "")),
            "M": int(_as_int(anchor_ctx.get("M"), 0)),
            "SCALE": int(_as_int(anchor_ctx.get("SCALE"), 0)),
            # ✅ explicit anchor-set root
            "anchor_root_poseidon_field": int(_as_int(anchor_ctx.get("anchor_root_poseidon_field"), 0)),
            # ⚠️ legacy alias (keeps old readers alive)
            "root_poseidon_field": int(_as_int(anchor_ctx.get("root_poseidon_field"), 0)),
        },
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
        "rsu_run_dir": str(rsu_run_dir),
        # -------------------------------------------------------------------
        # V15 audit evidence: SSI fingerprint + PUBMASK pins + public sidecar paths
        # -------------------------------------------------------------------
        "zk_selection_mode": "pubmask",
        "zk_circuit_name": "rootaware_pubmask",
        "ssi_preimage_fingerprint_v1": ssi_fp,
        "rsu_public_sidecar_by_round": rsu_public_sidecar_by_round,
        "ablation_cfg": dict(ablation_cfg or {}),
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
def write_digest_binding_failure_ablation_report_v1(
    output_dir: str,
    ablation_cfg: Dict[str, Any] | None,
    num_rsus: int,
) -> Dict[str, Any]:
    cfg = dict(ablation_cfg or {})
    report_dir = os.path.join(output_dir, "ablation_reports")
    target = dict(cfg.get("digest_binding_failure_target", {}) or {})

    report: Dict[str, Any] = {
        "enabled": bool(cfg.get("enable_digest_binding_failure_check", False)),
        "mode": str(cfg.get("digest_binding_failure_mode", "") or ""),
        "target": target,
        "detected": False,
        "matched_detail": {},
        "result": "not_run",
    }

    if not report["enabled"]:
        out_path = os.path.join(report_dir, "digest_binding_failure_report.json")
        _write_json_pretty_v1(out_path, report)
        return report

    target_round = int(target.get("round", -1))
    target_vehicle_id = int(target.get("vehicle_id", -1))
    target_rsu_id = int(target.get("rsu_id", -1))

    for rsu_id in range(1, int(num_rsus) + 1):
        if int(rsu_id) != int(target_rsu_id):
            continue

        summary_path = os.path.join(output_dir, f"iov_rsu_{int(rsu_id)}_summary.json")
        if not os.path.exists(summary_path):
            continue

        rsu_summary = _read_json_v1(summary_path)

        for round_summary in (rsu_summary.get("round_summaries", []) or []):
            if int(round_summary.get("round", -999)) != int(target_round):
                continue

            for detail in (round_summary.get("preselect_exclusion_details", []) or []):
                if (
                    int(detail.get("vehicle_id", -1)) == int(target_vehicle_id)
                    and int(detail.get("ablation_digest_target", 0) or 0) == 1
                ):
                    report["detected"] = True
                    report["matched_detail"] = detail
                    break

            if report["detected"]:
                break

        if report["detected"]:
            break

    report["result"] = "pass" if report["detected"] else "fail"

    out_path = os.path.join(report_dir, "digest_binding_failure_report.json")
    _write_json_pretty_v1(out_path, report)
    logging.info("[ABLATION] Saved digest-binding failure report to %s", out_path)
    return report

def run_commitment_recompute_check_v1(
    output_dir: str,
    anchor_ctx: Dict[str, Any],
    ablation_cfg: Dict[str, Any] | None,
) -> Dict[str, Any]:
    cfg = dict(ablation_cfg or {})
    report_dir = os.path.join(output_dir, "ablation_reports")
    target = dict(cfg.get("commitment_recompute_target", {}) or {})

    report: Dict[str, Any] = {
        "enabled": bool(cfg.get("enable_commitment_recompute_check", False)),
        "target": target,
        "rsu_check": {},
        "global_check": {},
        "result": "not_run",
    }

    if not report["enabled"]:
        out_path = os.path.join(report_dir, "commitment_recompute_report.json")
        _write_json_pretty_v1(out_path, report)
        return report

    rsu_id = int(target.get("rsu_id", 2))
    round_idx = int(target.get("round", 1))
    global_round = int(target.get("global_round", 2))

    sha_spec = mssi.Sha256MerkleSpecV1(depth=int(SHA_DEPTH))
    poseidon_spec = mssi.PoseidonMerkleSpecV1(
        arity=int(POSEIDON_ARITY),
        depth=int(POSEIDON_DEPTH),
    )

    rsu_report: Dict[str, Any] = {
        "target": {"rsu_id": int(rsu_id), "round": int(round_idx)},
        "manifest_path": "",
        "record_count": 0,
        "record_envelope_sha256_match": False,
        "manifest_root_sha256_hex": "",
        "recomputed_root_sha256_hex": "",
        "manifest_root_poseidon_field": 0,
        "recomputed_root_poseidon_field": 0,
        "result": "fail",
    }

    try:
        rsu_summary_path = os.path.join(output_dir, f"iov_rsu_{int(rsu_id)}_summary.json")
        rsu_summary = _read_json_v1(rsu_summary_path)
        round_summary = _find_round_summary_v1(rsu_summary, int(round_idx))
        manifest_path = _resolve_output_path_v1(
            output_dir,
            str(round_summary.get("rsu_round_manifest_path", "") or ""),
        )
        rsu_report["manifest_path"] = str(manifest_path)

        manifest_obj = _read_json_v1(manifest_path)
        extra = manifest_obj.get("extra", {}) if isinstance(manifest_obj, dict) else {}
        records = extra.get("records", []) if isinstance(extra, dict) else []

        record_bytes_list: List[bytes] = []
        envelope_sha_match = True

        for rec in records:
            if not isinstance(rec, dict):
                raise RuntimeError("manifest record entry is not a dict")
            env = rec.get("envelope", None)
            if not isinstance(env, dict):
                raise RuntimeError("manifest record missing canonical envelope")
            env_bytes = canon.canon_json_bytes_v1(env)
            record_bytes_list.append(env_bytes)

            stored_env_sha = str(rec.get("envelope_sha256_hex", "") or "").strip().lower()
            actual_env_sha = hashlib.sha256(env_bytes).hexdigest()
            if stored_env_sha and stored_env_sha != actual_env_sha:
                envelope_sha_match = False

        computed = mssi.build_dual_roots_from_record_bytes_v1(
            record_bytes_list,
            sha_spec=sha_spec,
            poseidon_spec=poseidon_spec,
            pad_leaf_field=0,
        )

        rsu_report["record_count"] = int(len(record_bytes_list))
        rsu_report["record_envelope_sha256_match"] = bool(envelope_sha_match)
        rsu_report["manifest_root_sha256_hex"] = str(manifest_obj.get("root_sha256_hex", "") or "")
        rsu_report["recomputed_root_sha256_hex"] = str(computed.get("root_sha256_hex", "") or "")
        rsu_report["manifest_root_poseidon_field"] = int(_as_int(manifest_obj.get("root_poseidon_field", 0), 0))
        rsu_report["recomputed_root_poseidon_field"] = int(_as_int(computed.get("root_poseidon_field", 0), 0))

        rsu_pass = bool(
            len(record_bytes_list) > 0
            and envelope_sha_match
            and rsu_report["manifest_root_sha256_hex"] == rsu_report["recomputed_root_sha256_hex"]
            and int(rsu_report["manifest_root_poseidon_field"]) == int(rsu_report["recomputed_root_poseidon_field"])
        )
        rsu_report["result"] = "pass" if rsu_pass else "fail"

    except Exception as exc:
        rsu_report["error"] = str(exc)
        rsu_report["result"] = "fail"

    global_report: Dict[str, Any] = {
        "target": {"round": int(global_round)},
        "gate_manifest_path": "",
        "global_round_manifest_path": "",
        "global_manifest_canon_sha256_hex": "",
        "gate_root_poseidon_field": 0,
        "recomputed_root_poseidon_field": 0,
        "summary_root_poseidon_field": 0,
        "rsu_ids_sorted_match": False,
        "mask_list_match": False,
        "rsu_round_roots_match": False,
        "pins_hash_field_match": False,
        "policy_id_field_match": False,
        "public_input_order_id_field_match": False,
        "result": "fail",
    }

    try:
        gate_path = os.path.join(output_dir, "zkp_anchor_gating_manifest.json")
        gate_obj = _read_json_v1(gate_path)
        global_manifest_path = _resolve_output_path_v1(
            output_dir,
            str(gate_obj.get("global_round_manifest_path", "") or ""),
        )

        global_report["gate_manifest_path"] = str(gate_path)
        global_report["global_round_manifest_path"] = str(global_manifest_path)

        global_manifest_obj = _read_json_v1(global_manifest_path)
        canon_bytes = _canon_json_bytes(global_manifest_obj)
        canon_sha = hashlib.sha256(canon_bytes).hexdigest()
        recomputed_root = int(_sha256_to_field(canon_bytes))

        global_report["global_manifest_canon_sha256_hex"] = str(canon_sha)
        global_report["gate_root_poseidon_field"] = int(_as_int(gate_obj.get("root_poseidon_field", 0), 0))
        global_report["recomputed_root_poseidon_field"] = int(recomputed_root)

        summary_root = 0
        global_summary_path = os.path.join(output_dir, "global_ensemble_summary.json")
        if os.path.exists(global_summary_path):
            global_summary_obj = _read_json_v1(global_summary_path)
            anchor_ctx_summary = global_summary_obj.get("anchor_ctx", {}) if isinstance(global_summary_obj, dict) else {}
            summary_root = int(_as_int(anchor_ctx_summary.get("root_poseidon_field", 0), 0))
        global_report["summary_root_poseidon_field"] = int(summary_root)

        gate_rsu_ids = [int(x) for x in (gate_obj.get("rsu_ids_sorted", []) or [])]
        mani_rsu_ids = [int(x) for x in (global_manifest_obj.get("rsu_ids_sorted", []) or [])]
        gate_mask = [int(x) for x in (gate_obj.get("mask_list", []) or [])]
        mani_mask = [int(x) for x in (global_manifest_obj.get("mask_list", []) or [])]
        gate_roots = [int(x) for x in (gate_obj.get("rsu_round_roots", []) or [])]
        mani_roots = [int(x) for x in (global_manifest_obj.get("rsu_round_roots", []) or [])]

        global_report["rsu_ids_sorted_match"] = bool(gate_rsu_ids == mani_rsu_ids)
        global_report["mask_list_match"] = bool(gate_mask == mani_mask)
        global_report["rsu_round_roots_match"] = bool(gate_roots == mani_roots)
        global_report["pins_hash_field_match"] = (
            int(_as_int(gate_obj.get("pins_hash_field", 0), 0))
            == int(_as_int(global_manifest_obj.get("pins_hash_field", 0), 0))
        )
        global_report["policy_id_field_match"] = (
            int(_as_int(gate_obj.get("policy_id_field", 0), 0))
            == int(_as_int(global_manifest_obj.get("policy_id_field", 0), 0))
        )
        global_report["public_input_order_id_field_match"] = (
            int(_as_int(gate_obj.get("public_input_order_id_field", 0), 0))
            == int(_as_int(global_manifest_obj.get("public_input_order_id_field", 0), 0))
        )

        global_pass = bool(
            int(_as_int(global_manifest_obj.get("round", -1), -1)) == int(global_round)
            and int(global_report["gate_root_poseidon_field"]) == int(global_report["recomputed_root_poseidon_field"])
            and (int(summary_root) in (0, int(global_report["recomputed_root_poseidon_field"])))
            and global_report["rsu_ids_sorted_match"]
            and global_report["mask_list_match"]
            and global_report["rsu_round_roots_match"]
            and global_report["pins_hash_field_match"]
            and global_report["policy_id_field_match"]
            and global_report["public_input_order_id_field_match"]
        )
        global_report["result"] = "pass" if global_pass else "fail"

    except Exception as exc:
        global_report["error"] = str(exc)
        global_report["result"] = "fail"

    report["rsu_check"] = rsu_report
    report["global_check"] = global_report
    report["result"] = (
        "pass"
        if rsu_report.get("result") == "pass" and global_report.get("result") == "pass"
        else "fail"
    )

    out_path = os.path.join(report_dir, "commitment_recompute_report.json")
    _write_json_pretty_v1(out_path, report)
    logging.info("[ABLATION] Saved commitment recomputation report to %s", out_path)
    return report

def run_public_input_agreement_check_v1(
    output_dir: str,
    anchor_ctx: Dict[str, Any],
    spec_rsu: Any,
    spec_global: Any,
    ablation_cfg: Dict[str, Any] | None,
) -> Dict[str, Any]:
    del anchor_ctx, spec_rsu, spec_global

    cfg = dict(ablation_cfg or {})
    report_dir = os.path.join(output_dir, "ablation_reports")
    target = dict(cfg.get("public_input_agreement_target", {}) or {})

    enabled = bool(cfg.get("enable_public_input_agreement_check", False))

    if not enabled:
        report: Dict[str, Any] = {
            "enabled": False,
            "target": target,
            "authoritative_verifier": "verify_public_inputs_from_artifacts.py",
            "rsu_check": {},
            "global_check": {},
            "result": "not_run",
        }

        out_path = os.path.join(
            report_dir,
            "public_input_agreement_report.json",
        )
        _write_json_pretty_v1(out_path, report)
        return report

    rsu_id = int(target.get("rsu_id", 2))
    rsu_round = int(target.get("round", 1))
    global_round = int(target.get("global_round", 1))

    try:
        import verify_public_inputs_from_artifacts as pia
    except Exception as exc:
        raise RuntimeError(
            "Reviewer-3 public-input verification requires "
            "verify_public_inputs_from_artifacts.py in the project root."
        ) from exc

    authoritative = pia.build_report(
        root=Path(output_dir).resolve(),
        rsu_id=int(rsu_id),
        rsu_round=int(rsu_round),
        global_round=int(global_round),
    )

    report = {
        "enabled": True,
        "target": target,
        "authoritative_verifier": "verify_public_inputs_from_artifacts.py",
        "rsu_check": dict(
            authoritative.get("rsu_scope", {}) or {}
        ),
        "global_check": dict(
            authoritative.get("global_scope", {}) or {}
        ),
        "result": str(
            authoritative.get("result", "fail") or "fail"
        ),
    }

    out_path = os.path.join(
        report_dir,
        "public_input_agreement_report.json",
    )
    _write_json_pretty_v1(out_path, report)

    logging.info(
        "[ABLATION] Saved authoritative public-input "
        "agreement report to %s",
        out_path,
    )

    return report

def write_layered_overhead_breakdown_report_v1(
    output_dir: str,
    ablation_cfg: Dict[str, Any] | None,
    num_rsus: int,
    export_index_write_latency_sec: float,
) -> Dict[str, Any]:
    cfg = dict(ablation_cfg or {})
    report_dir = os.path.join(output_dir, "ablation_reports")

    report: Dict[str, Any] = {
        "enabled": bool(cfg.get("enable_layered_overhead_breakdown", False)),
        "supports_actual_onchain_publication_measurement": False,
        "note": (
            "This V8c run measures off-chain learning, DP, SSI, Merkle, proof, and export-index packaging layers. "
            "Actual blockchain submission latency/gas must still be taken from the on-chain publication run/report."
        ),
        "measured_layers_sec": {},
        "fine_grained_components_sec": {},
        "per_rsu_overhead_summary": {},
        "global_overhead_summary": {},
        "result": "not_run",
    }

    if not report["enabled"]:
        out_path = os.path.join(report_dir, "layered_overhead_report.json")
        _write_json_pretty_v1(out_path, report)
        return report

    total_local_train = 0.0
    total_rsu_model_aggregation = 0.0
    total_dp_record = 0.0
    total_ssi_report = 0.0
    total_rsu_preselect = 0.0
    total_rsu_ssi_verify = 0.0
    total_rsu_merkle = 0.0
    total_rsu_zkp = 0.0

    for rsu_id in range(1, int(num_rsus) + 1):
        summary_path = os.path.join(output_dir, f"iov_rsu_{int(rsu_id)}_summary.json")
        if not os.path.exists(summary_path):
            continue

        rsu_summary = _read_json_v1(summary_path)
        report["per_rsu_overhead_summary"][str(rsu_id)] = dict(rsu_summary.get("overhead_summary", {}) or {})

        for round_summary in (rsu_summary.get("round_summaries", []) or []):
            total_local_train += float(round_summary.get("fit_total_local_train_time_sec", 0.0) or 0.0)
            total_rsu_model_aggregation += float(round_summary.get("rsu_model_aggregation_latency_sec", 0.0) or 0.0)
            total_dp_record += float(round_summary.get("fit_total_dp_record_time_sec", 0.0) or 0.0)
            total_ssi_report += float(round_summary.get("fit_total_ssi_report_time_sec", 0.0) or 0.0)
            total_rsu_preselect += float(round_summary.get("rsu_preselect_latency_sec", 0.0) or 0.0)
            total_rsu_ssi_verify += float(round_summary.get("rsu_ssi_verification_latency_sec", 0.0) or 0.0)
            total_rsu_merkle += float(round_summary.get("rsu_merkle_build_latency_sec", 0.0) or 0.0)
            total_rsu_zkp += float(round_summary.get("rsu_zkp_latency_sec", 0.0) or 0.0)

    global_summary_path = os.path.join(output_dir, "global_ensemble_summary.json")
    global_overhead = {}
    if os.path.exists(global_summary_path):
        global_summary = _read_json_v1(global_summary_path)
        global_overhead = dict(global_summary.get("overhead", {}) or {})
    report["global_overhead_summary"] = dict(global_overhead)

    global_aggregation = float(global_overhead.get("global_aggregation_latency_sec", 0.0) or 0.0)
    global_zkp = float(global_overhead.get("global_zkp_latency_sec", 0.0) or 0.0)
    global_artifact_export = float(global_overhead.get("global_artifact_export_latency_sec", 0.0) or 0.0)

    report["fine_grained_components_sec"] = {
        "local_train_sec": float(total_local_train),
        "rsu_model_aggregation_sec": float(total_rsu_model_aggregation),
        "global_aggregation_sec": float(global_aggregation),
        "dp_record_sec": float(total_dp_record),
        "ssi_vehicle_report_sec": float(total_ssi_report),
        "rsu_preselect_sec": float(total_rsu_preselect),
        "rsu_ssi_verification_sec": float(total_rsu_ssi_verify),
        "rsu_merkle_build_sec": float(total_rsu_merkle),
        "rsu_zkp_sec": float(total_rsu_zkp),
        "global_zkp_sec": float(global_zkp),
        "global_artifact_export_sec": float(global_artifact_export),
        "export_index_write_sec": float(export_index_write_latency_sec),
    }

    report["measured_layers_sec"] = {
        "learning_core_sec": float(total_local_train + total_rsu_model_aggregation + global_aggregation),
        "dp_evidence_sec": float(total_dp_record),
        "ssi_admission_sec": float(total_ssi_report + total_rsu_preselect + total_rsu_ssi_verify),
        "commitment_sec": float(total_rsu_merkle),
        "proof_sec": float(total_rsu_zkp + global_zkp),
        "artifact_export_sec": float(global_artifact_export + float(export_index_write_latency_sec)),
    }

    report["result"] = "pass"

    out_path = os.path.join(report_dir, "layered_overhead_report.json")
    _write_json_pretty_v1(out_path, report)
    logging.info("[ABLATION] Saved layered overhead report to %s", out_path)
    return report

def write_admission_failure_ablation_report_v1(
    output_dir: str,
    ablation_cfg: Dict[str, Any] | None,
    num_rsus: int,
) -> Dict[str, Any]:
    cfg = dict(ablation_cfg or {})
    report_dir = os.path.join(output_dir, "ablation_reports")
    target = dict(cfg.get("admission_failure_target", {}) or {})

    report: Dict[str, Any] = {
        "enabled": bool(cfg.get("enable_admission_failure_check", False)),
        "mode": str(cfg.get("admission_failure_mode", "") or ""),
        "target": target,
        "detected": False,
        "matched_detail": {},
        "result": "not_run",
    }

    if not report["enabled"]:
        out_path = os.path.join(report_dir, "admission_failure_report.json")
        _write_json_pretty_v1(out_path, report)
        return report

    target_round = int(target.get("round", -1))
    target_vehicle_id = int(target.get("vehicle_id", -1))
    target_rsu_id = int(target.get("rsu_id", -1))

    for rsu_id in range(1, int(num_rsus) + 1):
        if int(rsu_id) != int(target_rsu_id):
            continue

        summary_path = os.path.join(output_dir, f"iov_rsu_{int(rsu_id)}_summary.json")
        if not os.path.exists(summary_path):
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            rsu_summary = json.load(f)

        for round_summary in (rsu_summary.get("round_summaries", []) or []):
            if int(round_summary.get("round", -999)) != int(target_round):
                continue

            for detail in (round_summary.get("preselect_exclusion_details", []) or []):
                if int(detail.get("vehicle_id", -1)) == int(target_vehicle_id):
                    report["detected"] = True
                    report["matched_detail"] = detail
                    break

            if report["detected"]:
                break

        if report["detected"]:
            break

    report["result"] = "pass" if report["detected"] else "fail"

    out_path = os.path.join(report_dir, "admission_failure_report.json")
    _write_json_pretty_v1(out_path, report)
    logging.info("[ABLATION] Saved admission-failure report to %s", out_path)
    return report

def run_artifact_tamper_check_v1(
    output_dir: str,
    ablation_cfg: Dict[str, Any] | None,
) -> Dict[str, Any]:
    cfg = dict(ablation_cfg or {})
    report_dir = os.path.join(output_dir, "ablation_reports", "tamper_probe")
    target = dict(cfg.get("tamper_target", {}) or {})

    report: Dict[str, Any] = {
        "enabled": bool(cfg.get("enable_artifact_tamper_check", False)),
        "target": target,
        "consistency_scope": "dp_record_digest_binding",
        "original_dp_record_path": "",
        "tampered_copy_path": "",
        "expected_sha256_from_ledger": "",
        "original_sha256": "",
        "tampered_sha256": "",
        "detected": False,
        "result": "not_run",
    }

    if not report["enabled"]:
        out_path = os.path.join(report_dir, "tamper_report.json")
        _write_json_pretty_v1(out_path, report)
        return report

    rsu_id = int(target.get("rsu_id", -1))
    vehicle_id = int(target.get("vehicle_id", -1))
    round_idx = int(target.get("round", -1))

    dp_record_path = os.path.join(
        output_dir,
        "dp_records",
        f"rsu_{int(rsu_id)}",
        f"vehicle_{int(vehicle_id)}",
        f"round_{int(round_idx)}.json",
    )
    ledger_path = os.path.join(
        output_dir,
        "dp_ledgers",
        f"dp_ledger_rsu_{int(rsu_id)}_vehicle_{int(vehicle_id)}.json",
    )

    report["original_dp_record_path"] = str(dp_record_path)

    if (not os.path.exists(dp_record_path)) or (not os.path.exists(ledger_path)):
        report["result"] = "fail"
        out_path = os.path.join(report_dir, "tamper_report.json")
        _write_json_pretty_v1(out_path, report)
        return report

    expected_sha256 = ""
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger_entries = json.load(f)

    for entry in (ledger_entries or []):
        try:
            if int(entry.get("round", -1)) == int(round_idx):
                expected_sha256 = str(entry.get("dp_record_sha256", "") or "").strip()
                break
        except Exception:
            continue

    report["expected_sha256_from_ledger"] = str(expected_sha256)
    report["original_sha256"] = _sha256_file_hex_top_v1(dp_record_path)

    tampered_copy_path = os.path.join(
        report_dir,
        f"rsu_{int(rsu_id)}_vehicle_{int(vehicle_id)}_round_{int(round_idx)}_tampered.json",
    )

    with open(dp_record_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            obj["_tampered_probe"] = True
            tampered_text = json.dumps(obj, indent=2, sort_keys=True)
        else:
            tampered_text = raw_text + "\n "
    except Exception:
        tampered_text = raw_text + "\n "

    os.makedirs(os.path.dirname(tampered_copy_path), exist_ok=True)
    with open(tampered_copy_path, "w", encoding="utf-8") as f:
        f.write(tampered_text)

    report["tampered_copy_path"] = str(tampered_copy_path)
    report["tampered_sha256"] = _sha256_file_hex_top_v1(tampered_copy_path)

    report["detected"] = bool(
        expected_sha256
        and report["original_sha256"] == expected_sha256
        and report["tampered_sha256"] != expected_sha256
    )
    report["result"] = "pass" if report["detected"] else "fail"

    out_path = os.path.join(report_dir, "tamper_report.json")
    _write_json_pretty_v1(out_path, report)
    logging.info("[ABLATION] Saved artifact-tamper report to %s", out_path)
    return report

def write_ablation_summary_v1(
    output_dir: str,
    admission_report: Dict[str, Any],
    digest_report: Dict[str, Any],
    tamper_report: Dict[str, Any],
    commitment_report: Dict[str, Any],
    public_input_report: Dict[str, Any],
    layered_overhead_report: Dict[str, Any],
) -> None:
    out_path = os.path.join(output_dir, "ablation_reports", "ablation_summary.json")
    _write_json_pretty_v1(
        out_path,
        {
            "admission_failure_check": dict(admission_report or {}),
            "digest_binding_failure_check": dict(digest_report or {}),
            "artifact_tamper_check": dict(tamper_report or {}),
            "commitment_recompute_check": dict(commitment_report or {}),
            "public_input_agreement_check": dict(public_input_report or {}),
            "layered_overhead_breakdown": dict(layered_overhead_report or {}),
        },
    )
    logging.info("[ABLATION] Saved combined ablation summary to %s", out_path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # Reviewer 1 Comment 3: one-click batch execution
    #
    # The parent process automatically launches all three experimental
    # conditions as independent Python processes. Independent processes are
    # intentional because Flower/Ray and the ZKP runtime should start from a
    # clean process state for each condition.
    # -----------------------------------------------------------------------
    _reviewer3_batch_child = (
        str(os.environ.get("REVIEWER3_BATCH_CHILD", "0")).strip() == "1"
    )

    if not _reviewer3_batch_child:
        _reviewer3_all_scenarios = [
            "balanced_reference",
            "data_heterogeneity",
            "combined_stress",
        ]

        _reviewer3_script_path = str(Path(__file__).resolve())

        print("=" * 79, flush=True)
        print(
            "REVIEWER 1 COMMENT 3 — AUTOMATIC THREE-SCENARIO EXECUTION",
            flush=True,
        )
        print("=" * 79, flush=True)

        for _scenario in _reviewer3_all_scenarios:
            print(flush=True)
            print("=" * 79, flush=True)
            print(
                f"STARTING SCENARIO: {_scenario}",
                flush=True,
            )
            print("=" * 79, flush=True)

            _child_env = os.environ.copy()
            _child_env["REVIEWER3_BATCH_CHILD"] = "1"
            _child_env["REVIEWER3_SCENARIO"] = str(_scenario)

            subprocess.run(
                [sys.executable, _reviewer3_script_path],
                env=_child_env,
                check=True,
            )

            print(flush=True)
            print("=" * 79, flush=True)
            print(
                f"COMPLETED SCENARIO: {_scenario}",
                flush=True,
            )
            print("=" * 79, flush=True)

        print(flush=True)
        print("=" * 79, flush=True)
        print(
            "ALL REVIEWER 1 COMMENT 3 SCENARIOS COMPLETED SUCCESSFULLY",
            flush=True,
        )
        print("=" * 79, flush=True)

        sys.exit(0)

    setup_logging()

    # ---------------------------
    # IoV dataset selection
    # ---------------------------
    DATASET_NAME = "CSECICIDS2018"   # change to "CICIoV2024" when needed
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
    if not os.path.isabs(PREPROC_DIR):
        PREPROC_DIR = os.path.join(str(_PROJECT_ROOT), PREPROC_DIR)

    ds_cfg = IoVDatasetConfig(
        preproc_dir=PREPROC_DIR,
        train_csv=os.path.join(PREPROC_DIR, cfg["train_csv"]),
        val_csv=os.path.join(PREPROC_DIR, cfg["val_csv"]),
        test_csv=os.path.join(PREPROC_DIR, cfg["test_csv"]),
        label_col=cfg["label_col"],  # IoV BENIGN vs ATTACK label
    )
    # ---------------------------
    # Reviewer 3 controlled stress condition
    # ---------------------------
    REVIEWER3_SCENARIO = _reviewer3_scenario_v1()
    REVIEWER3_SEED = 42

    # ---------------------------
    # High-level RSU configuration
    # ---------------------------
    # Larger than the compact 2-RSU/4-vehicle experiment, but intentionally
    # still a controlled FL stress test rather than a mobility/network simulator.
    NUM_RSUS = 2
    VEHICLES_PER_RSU = 10
    NUM_ROUNDS = 8
    NUM_LOCAL_ROUNDS = 10
    # ✅ Define once and reuse everywhere (training + global stage)
    rsu_ids_sorted: List[int] = list(range(1, int(NUM_RSUS) + 1))
    # ---------------------------
    # AnchorZKP configuration (audit lane)
    # ---------------------------
    ANCHOR_M = 64
    ANCHOR_SCALE = 100_000
    ANCHOR_SEED = 42
    ANCHOR_VERSION = "v1"
    ANCHOR_OVERWRITE = True  # <-- add this
    # Results are isolated from every prior experiment. Each execution gets a
    # timestamped scenario-specific directory under reviewer_concern3_results/.
    _reviewer3_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _project_root_runtime = str(_PROJECT_ROOT)
    OUTPUT_DIR = os.path.join(
        _project_root_runtime,
        "reviewer_concern3_results",
        f"{_reviewer3_ts}_{REVIEWER3_SCENARIO}_seed{REVIEWER3_SEED}",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _enable_participation = REVIEWER3_SCENARIO == "combined_stress"
    _enable_context = REVIEWER3_SCENARIO == "combined_stress"

    ABLATION_CONFIG = {
        # Disable the older security ablations so this experiment isolates Comment 3.
        "enable_admission_failure_check": False,
        "enable_digest_binding_failure_check": False,
        "enable_artifact_tamper_check": False,

        # Controlled unstable participation / intermittent contribution delivery.
        "enable_participation_stress": bool(_enable_participation),
        "availability_schedule": {
            # Vehicle local-index 10 at each RSU is unavailable from round 5 onward.
            "permanent_dropout_local_index": 10,
            "permanent_dropout_from_round": 5,
            # Multi-round outages emulate intermittent link availability at the FL
            # contribution level (not packet-level wireless behavior).
            "burst_outage_rounds_by_local_index": {
                "3": [2, 3],
                "7": [6, 7],
            },
            # One isolated missed contribution per RSU.
            "isolated_outage_rounds_by_local_index": {
                "5": [4],
            },
        },

        # Valid signatures bound to a deliberately wrong run context.
        # These are not forged signatures: the RSU must reject them because its
        # expected (RSU, round) context differs from the signed context.
        "enable_context_binding_challenges": bool(_enable_context),
        "context_binding_targets": [
            {
                "rsu_id": 1,
                "vehicle_id": 1002,
                "round": 6,
                "mode": "stale_round",
            },
            {
                "rsu_id": 2,
                "vehicle_id": 2002,
                "round": 6,
                "mode": "cross_rsu_scope",
                "signed_rsu_id": 1,
            },
        ] if _enable_context else [],

        # Preserve independent evidence reconstruction on the final stressed round.
        "enable_commitment_recompute_check": True,
        "commitment_recompute_target": {
            "rsu_id": 2,
            "round": int(NUM_ROUNDS),
            "global_round": int(NUM_ROUNDS),
        },
        "enable_public_input_agreement_check": True,
        "public_input_agreement_target": {
            "rsu_id": 2,
            "round": int(NUM_ROUNDS),
            "global_round": int(NUM_ROUNDS),
        },
        "enable_layered_overhead_breakdown": True,
    }

    with open(os.path.join(OUTPUT_DIR, "reviewer3_experiment_condition.json"), "w", encoding="utf-8") as _f:
        json.dump(
            {
                "experiment_version": REVIEWER3_EXPERIMENT_VERSION,
                "scenario": REVIEWER3_SCENARIO,
                "seed": REVIEWER3_SEED,
                "num_rsus": NUM_RSUS,
                "vehicles_per_rsu": VEHICLES_PER_RSU,
                "num_rounds": NUM_ROUNDS,
                "num_local_rounds": NUM_LOCAL_ROUNDS,
                "stress_configuration": ABLATION_CONFIG,
                "interpretation_boundary": (
                    "Controlled FL-level data/participation/context stress evaluation; "
                    "not physical mobility, seamless handoff, or packet-level V2X deployment validation."
                ),
            },
            _f,
            indent=2,
        )
    # -------------------------------------------------------------------
    # ✅ On-chain export package (constant location + per-run snapshot)
    # -------------------------------------------------------------------
    ONCHAIN_EXPORT_DIR = os.path.join(OUTPUT_DIR, "onchain_export")
    os.makedirs(ONCHAIN_EXPORT_DIR, exist_ok=True)
    _run_ts_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _run_hash8 = hashlib.sha256(f"{_run_ts_utc}|pid={os.getpid()}".encode("utf-8")).hexdigest()[:8]
    RUN_ID = f"{_run_ts_utc}_{_run_hash8}"
    def _sha256_file_hex_v1(path: str) -> str:
        try:
            if not path or (not os.path.exists(path)) or (not os.path.isfile(path)):
                return ""
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""
    def _safe_relpath_posix_v1(path: str, root_dir: str) -> str:
        try:
            if not path:
                return ""
            abs_root = os.path.abspath(root_dir)
            abs_p = os.path.abspath(path)
            rel = os.path.relpath(abs_p, abs_root)
            return rel.replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")
    def _atomic_write_json_v1(path: str, obj: dict) -> None:
        tmp = f"{path}.tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(s)
            f.write("\n")
        os.replace(tmp, path)
    def _extract_proof_path_from_summary_map_v1(val: Any) -> str:
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            return str(
                val.get("proof_artifact_path")
                or val.get("artifact_path")
                or val.get("path")
                or ""
            )
        return ""
    def _resolve_real_proof_artifact_path_v1(proof_path_abs: str, out_root_abs: str) -> str:
        """
        If proof_path_abs points to a wrapper/summary JSON, follow its pointer to the real proof artifact JSON.
        We treat a file as a "real proof artifact" ONLY if it has:
          - a proof object AND public inputs
          - AND at least one of the RSU-specific payload bindings (root/Q/pins fields)
        """
        try:
            if not proof_path_abs or (not os.path.exists(proof_path_abs)):
                return proof_path_abs
            with open(proof_path_abs, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                return proof_path_abs
            has_proof = ("proof" in obj or "pi_a" in obj or "pi_b" in obj or "pi_c" in obj)
            has_publics = ("public_inputs" in obj or "publicSignals" in obj or "public" in obj)
            # RSU/global artifacts usually carry at least one of these binding fields
            has_binding = any(
                k in obj for k in (
                    "root_poseidon_field",
                    "Q_rsu_b64",
                    "Q_global_b64",
                    "pins_hash_field",
                    "policy_id_field",
                    "public_input_order_id_field",
                    "rsu_round_manifest_path",
                    "round_manifest_path",
                    "manifest_path",
                )
            )
            if has_proof and has_publics and has_binding:
                return proof_path_abs
            # Follow pointer keys used in your pipeline
            hinted = str(
                obj.get("proof_artifact_path")
                or obj.get("proof_artifact_relpath")
                or obj.get("artifact_path")
                or obj.get("proof_path")
                or obj.get("proof_json_path")
                or obj.get("path")
                or ""
            ).strip()
            if not hinted:
                return proof_path_abs
            hinted_abs = hinted
            if not os.path.isabs(hinted_abs):
                hinted_abs = os.path.join(out_root_abs, hinted_abs)
            if os.path.exists(hinted_abs):
                return hinted_abs
            return proof_path_abs
        except Exception:
            return proof_path_abs
    def _build_onchain_export_index_v1(
            *,
            output_dir: str,
            run_id: str,
            rsu_ids_sorted: List[int],
            num_rounds: int,
            vehicles_per_rsu: int,
            anchor_ctx: Dict[str, Any],
            run_cfg_path: str,
            gate_path: str,
            global_ensemble_summary_path: str,
    ) -> dict:
        out_root = os.path.abspath(output_dir)
        def _p(p: str) -> str:
            return _safe_relpath_posix_v1(p, out_root)
        export_index: dict = {
            "schema": "OnChainExportIndexV1",
            "run_id": str(run_id),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "output_root": _p(out_root),
            "selection_mode": "pubmask",
            "zk_circuit_name": "rootaware_pubmask",
            "topology": {
                "num_rounds": int(num_rounds),
                "num_rsus": int(len(rsu_ids_sorted)),
                "vehicles_per_rsu": int(vehicles_per_rsu),
                "rsu_ids_sorted": [int(x) for x in rsu_ids_sorted],
                "rsu_vehicle_ids": {
                    str(rsu_id): [int(rsu_id * 1000 + i) for i in range(1, int(vehicles_per_rsu) + 1)]
                    for rsu_id in rsu_ids_sorted
                },
            },
            "anchors": {
                "anchor_version": str(anchor_ctx.get("anchor_version", "")),
                "anchor_id_field": str(anchor_ctx.get("anchor_id_field", "")),
                "M": int(_as_int(anchor_ctx.get("M"), 0)),
                "SCALE": int(_as_int(anchor_ctx.get("SCALE"), 0)),
                "anchor_root_poseidon_field": int(_as_int(anchor_ctx.get("anchor_root_poseidon_field"), 0)),
                "anchor_X_path": _p(str(anchor_ctx.get("anchor_X_path", "") or "")),
                "anchor_meta_path": _p(str(anchor_ctx.get("anchor_meta_path", "") or "")),
            },
            "run_files": {
                "iov_run_config_json": {
                    "path": _p(run_cfg_path),
                    "sha256": _sha256_file_hex_v1(run_cfg_path),
                },
                "zkp_anchor_gating_manifest_json": {
                    "path": _p(gate_path),
                    "sha256": _sha256_file_hex_v1(gate_path),
                },
                "global_ensemble_summary_json": {
                    "path": _p(global_ensemble_summary_path),
                    "sha256": _sha256_file_hex_v1(global_ensemble_summary_path),
                },
            },
            "circuits": [],
            "rsu_rounds": {},
            "global_round": {},
        }
        # -------------------------------------------------------------------
        # Circuits discovery (RSU + GLOBAL) from generated_circuits_dir
        # -------------------------------------------------------------------
        gen_dir = str(anchor_ctx.get("anchorsum_generated_circuits_dir", "") or "")
        art_dir = str(anchor_ctx.get("anchorsum_artifacts_dir", "") or "")
        rsu_circom_files = sorted(glob.glob(os.path.join(gen_dir, "AggRSU_*.circom")))
        global_circom_files = sorted(glob.glob(os.path.join(gen_dir, "AggGlobal_*.circom")))
        for circom_path in (rsu_circom_files + global_circom_files):
            stem = os.path.splitext(os.path.basename(circom_path))[0]
            circuit_dir = os.path.join(art_dir, stem)
            vkey_candidates = [
                os.path.join(circuit_dir, "verification_key.json"),
                os.path.join(circuit_dir, "vkey.json"),
            ]
            vkey_path = ""
            for c in vkey_candidates:
                if os.path.exists(c):
                    vkey_path = c
                    break
            verifier_sol_exact = os.path.join(circuit_dir, f"Verifier_{stem}.sol")
            if os.path.exists(verifier_sol_exact):
                verifier_sol_path = verifier_sol_exact
            else:
                verifier_sol_candidates = sorted(glob.glob(os.path.join(circuit_dir, "Verifier_*.sol")))
                verifier_sol_path = verifier_sol_candidates[0] if verifier_sol_candidates else ""
            export_index["circuits"].append(
                {
                    "circuit_name": str(stem),
                    "circom_path": _p(circom_path),
                    "circom_sha256": _sha256_file_hex_v1(circom_path),
                    "artifacts_dir": _p(circuit_dir),
                    "verification_key_path": _p(vkey_path),
                    "vkey_sha256": _sha256_file_hex_v1(vkey_path) if vkey_path else "",
                    "verifier_sol_path": _p(verifier_sol_path),
                    "verifier_sol_sha256": _sha256_file_hex_v1(verifier_sol_path) if verifier_sol_path else "",
                }
            )
        # -------------------------------------------------------------------
        # RSU per-round objects
        # -------------------------------------------------------------------
        for rsu_id in rsu_ids_sorted:
            rsu_summary_path = os.path.join(out_root, f"iov_rsu_{int(rsu_id)}_summary.json")

            rsu_rounds_entry: Dict[str, Any] = {}

            rsu_entry: Dict[str, Any] = {
                "summary_json": {"path": _p(rsu_summary_path),
                    "sha256": _sha256_file_hex_v1(rsu_summary_path),},
                "rounds": rsu_rounds_entry,
                "model_json": {"path": _p(os.path.join(out_root, f"iov_global_model_rsu_{int(rsu_id)}.json")),
                    "sha256": _sha256_file_hex_v1(os.path.join(out_root, f"iov_global_model_rsu_{int(rsu_id)}.json")),},
            }
            summary_obj: dict = {}
            if os.path.exists(rsu_summary_path):
                try:
                    with open(rsu_summary_path, "r", encoding="utf-8") as f:
                        summary_obj = json.load(f)
                except Exception:
                    summary_obj = {}
            art_map = summary_obj.get("rsu_anchor_artifacts", summary_obj.get("rsu_anchor_artifact", {})) or {}
            valid_map = summary_obj.get("rsu_anchor_round_valid", summary_obj.get("rsu_round_valid", {})) or {}
            artifacts_dir_str = str(anchor_ctx.get("anchorsum_artifacts_dir", "") or "")
            for r in range(1, int(num_rounds) + 1):
                art_val = _get_round_keyed(art_map, r, "")
                ok_val = _get_round_keyed(valid_map, r, 0)
                proof_path = _extract_proof_path_from_summary_map_v1(art_val)
                if proof_path and (not os.path.isabs(proof_path)):
                    proof_path = os.path.join(out_root, proof_path)
                # ✅ IMPORTANT: resolve wrapper -> real proof artifact
                proof_path = _resolve_real_proof_artifact_path_v1(proof_path, out_root)
                sidecar_path = os.path.join(
                    artifacts_dir_str,
                    f"rsu_{int(rsu_id)}",
                    f"round_{int(r)}_public.json",
                )
                rsu_manifest_path = ""
                root_poseidon_field = 0
                if proof_path and os.path.exists(proof_path):
                    try:
                        with open(proof_path, "r", encoding="utf-8") as f:
                            proof_obj = json.load(f)
                        # ✅ If the resolved proof artifact points to a manifest, pin it
                        rsu_manifest_path = str(proof_obj.get("rsu_round_manifest_path", "") or "")
                        if rsu_manifest_path and (not os.path.isabs(rsu_manifest_path)):
                            rsu_manifest_path = os.path.join(out_root, rsu_manifest_path)
                        # ✅ Pin the real RSU round root bound into the proof artifact
                        root_poseidon_field = int(_as_int(proof_obj.get("root_poseidon_field", 0), 0))
                    except Exception:
                        rsu_manifest_path = ""
                        root_poseidon_field = 0
                # ----------------------------
                # Resolve RSU round manifest path robustly
                # ----------------------------
                def _abs_if_exists_v1(p: str) -> str:
                    if not p:
                        return ""
                    try:
                        p = str(p).strip()
                        if not p:
                            return ""
                        abs_p = p if os.path.isabs(p) else os.path.join(out_root, p)
                        return abs_p if os.path.exists(abs_p) else ""
                    except Exception:
                        return ""
                def _find_manifest_path_from_proof_v1(proof_obj: dict) -> str:
                    if not isinstance(proof_obj, dict):
                        return ""
                    meta0 = proof_obj.get("meta", {})
                    meta0 = meta0 if isinstance(meta0, dict) else {}
                    for k in (
                            "rsu_round_manifest_path",
                            "rsu_round_manifest_relpath",
                            "round_manifest_path",
                            "round_manifest_relpath",
                            "roots_manifest_path",
                            "roots_manifest_relpath",
                            "manifest_path",
                            "manifest_relpath",
                    ):
                        v = str(proof_obj.get(k, "") or meta0.get(k, "") or "").strip()
                        if v:
                            v_abs = _abs_if_exists_v1(v)
                            if v_abs:
                                return v_abs
                    return ""
                def _find_manifest_path_from_summary_v1(summary_obj: dict, round_idx: int) -> str:
                    if not isinstance(summary_obj, dict):
                        return ""
                    rs = summary_obj.get("round_summaries", []) or []
                    if not isinstance(rs, list):
                        return ""
                    for ent in rs:
                        if not isinstance(ent, dict):
                            continue
                        rr = _reviewer3_optional_int_v1(ent.get("round"))

                        if rr is None or rr != int(round_idx):
                            continue
                        for k in (
                                "rsu_round_manifest_path",
                                "rsu_round_manifest_relpath",
                                "round_manifest_path",
                                "round_manifest_relpath",
                                "roots_manifest_path",
                                "roots_manifest_relpath",
                                "manifest_path",
                                "manifest_relpath",
                        ):
                            v = str(ent.get(k, "") or "").strip()
                            if v:
                                v_abs = _abs_if_exists_v1(v)
                                if v_abs:
                                    return v_abs
                    return ""
                def _guess_manifest_path_by_convention_v1(rsu_id_int: int, round_idx: int) -> str:
                    # Add the most common locations you’ve used in this repo
                    candidates = [
                        os.path.join(out_root, "round_manifests", f"rsu_{rsu_id_int}", f"round_{round_idx}.json"),
                        os.path.join(out_root, "round_manifests", f"rsu_{rsu_id_int}",
                                     f"round_{round_idx}_manifest.json"),
                        os.path.join(out_root, "zkp_round_manifests", f"rsu_{rsu_id_int}", f"round_{round_idx}.json"),
                        os.path.join(out_root, "zkp_round_manifests", f"rsu_{rsu_id_int}",
                                     f"round_{round_idx}_manifest.json"),
                        os.path.join(out_root, f"rsu_{rsu_id_int}_round_{round_idx}_manifest.json"),
                    ]
                    for c in candidates:
                        if os.path.exists(c):
                            return c
                    return ""
                # Ensure we have the real proof object (not a wrapper)
                proof_obj: dict = {}
                if proof_path and os.path.exists(proof_path):
                    try:
                        with open(proof_path, "r", encoding="utf-8") as f:
                            _tmp = json.load(f)
                        proof_obj = _tmp if isinstance(_tmp, dict) else {}
                    except Exception:
                        proof_obj = {}
                # Extract manifest path using multiple sources
                rsu_manifest_path = (
                        _find_manifest_path_from_proof_v1(proof_obj)
                        or _find_manifest_path_from_summary_v1(summary_obj, r)
                        or _guess_manifest_path_by_convention_v1(int(rsu_id), int(r))
                )
                # -------------------------------------------------------------------
                # Prefer stable packaged proof/sidecar copies (what on-chain code should anchor)
                # -------------------------------------------------------------------
                proof_copy_path = str(proof_obj.get("proof_artifact_copy_path", "") or "").strip()
                public_copy_path = str(proof_obj.get("public_inputs_sidecar_copy_path", "") or "").strip()
                # Resolve abs paths if needed
                if proof_copy_path and (not os.path.isabs(proof_copy_path)):
                    proof_copy_path = os.path.join(out_root, proof_copy_path)
                if public_copy_path and (not os.path.isabs(public_copy_path)):
                    public_copy_path = os.path.join(out_root, public_copy_path)
                # Use copy paths if they exist; otherwise fall back to original paths
                proof_path_onchain = proof_copy_path if (
                            proof_copy_path and os.path.exists(proof_copy_path)) else proof_path
                public_path_onchain = public_copy_path if (
                            public_copy_path and os.path.exists(public_copy_path)) else sidecar_path
                # Prefer provided sha256 fields (exact packaged bytes), else compute
                proof_sha256 = str(proof_obj.get("proof_artifact_copy_sha256", "") or "").strip()
                if (not proof_sha256) or (len(proof_sha256) != 64):
                    proof_sha256 = _sha256_file_hex_v1(proof_path_onchain)
                public_sha256 = str(proof_obj.get("public_inputs_sidecar_sha256", "") or "").strip()
                if (not public_sha256) or (len(public_sha256) != 64):
                    public_sha256 = _sha256_file_hex_v1(public_path_onchain)
                manifest_sha256 = _sha256_file_hex_v1(rsu_manifest_path)
                # Also export pins fields explicitly (on-chain submitter should not need to open the proof JSON)
                meta0 = proof_obj.get("meta", {})
                meta0 = meta0 if isinstance(meta0, dict) else {}
                pins_hash_field_r = int(_as_int(proof_obj.get("pins_hash_field", meta0.get("pins_hash_field", 0)), 0))
                policy_id_field_r = int(_as_int(proof_obj.get("policy_id_field", meta0.get("policy_id_field", 0)), 0))
                pio_id_field_r = int(
                    _as_int(proof_obj.get("public_input_order_id_field", meta0.get("public_input_order_id_field", 0)),
                            0))
                rsu_rounds_entry[str(r)] = {
                    "ok": bool(int(ok_val)) if isinstance(ok_val, (int, float, str)) else bool(ok_val),
                    # stable on-chain objects (prefer packaged copies)
                    "proof_json": {
                        "path": _p(proof_path_onchain),
                        "sha256": str(proof_sha256),
                    },
                    "public_inputs_sidecar": {
                        "path": _p(public_path_onchain),
                        "sha256": str(public_sha256),
                    },
                    "round_manifest_json": {
                        "path": _p(rsu_manifest_path),
                        "sha256": str(manifest_sha256),
                    },
                    "root_poseidon_field": int(root_poseidon_field),
                    # ✅ pins fields exported for submitter
                    "pins_hash_field": int(pins_hash_field_r),
                    "policy_id_field": int(policy_id_field_r),
                    "public_input_order_id_field": int(pio_id_field_r),
                    # flattened hashes (what your on-chain code reads easily)
                    "proof_sha256": str(proof_sha256),
                    "public_sha256": str(public_sha256),
                    "manifest_sha256": str(manifest_sha256),
                    "pins_sha256": {
                        "proof_sha256": str(proof_sha256),
                        "public_sha256": str(public_sha256),
                        "manifest_sha256": str(manifest_sha256),
                    },
                }
            export_index["rsu_rounds"][str(int(rsu_id))] = rsu_entry
        # -------------------------------------------------------------------
        # GLOBAL anchor proof objects (read from gating manifest if present)
        # -------------------------------------------------------------------
        gate_obj: dict = {}
        if gate_path and os.path.exists(gate_path):
            try:
                with open(gate_path, "r", encoding="utf-8") as f:
                    gate_obj = json.load(f)
            except Exception:
                gate_obj = {}
        global_anchor_summary_path = str(gate_obj.get("global_artifact_path", "") or "")
        if global_anchor_summary_path and (not os.path.isabs(global_anchor_summary_path)):
            global_anchor_summary_path = os.path.join(out_root, global_anchor_summary_path)
        global_proof_path = ""
        global_sidecar_path = str(gate_obj.get("global_public_inputs_sidecar_path", "") or "")
        if global_sidecar_path and (not os.path.isabs(global_sidecar_path)):
            global_sidecar_path = os.path.join(out_root, global_sidecar_path)
        global_round_root_poseidon_field = int(_as_int(gate_obj.get("root_poseidon_field", 0), 0))
        if global_anchor_summary_path and os.path.exists(global_anchor_summary_path):
            try:
                with open(global_anchor_summary_path, "r", encoding="utf-8") as f:
                    gsum = json.load(f)
                global_proof_path = str(gsum.get("proof_artifact_path", "") or "")
                if global_proof_path and (not os.path.isabs(global_proof_path)):
                    global_proof_path = os.path.join(out_root, global_proof_path)
            except Exception:
                global_proof_path = ""
        global_round_manifest_path = str(gate_obj.get("global_round_manifest_path", "") or "")
        if global_round_manifest_path and (not os.path.isabs(global_round_manifest_path)):
            global_round_manifest_path = os.path.join(out_root, global_round_manifest_path)
        global_anchor_ok_flag = bool(gate_obj.get("global_anchor_ok", False))
        # ✅ Pull preimage fields directly from gate manifest (so on-chain submitter does NOT need extra reads)
        gate_rsu_ids_sorted = [int(x) for x in (gate_obj.get("rsu_ids_sorted", []) or [])]
        gate_mask_list = [int(x) for x in (gate_obj.get("mask_list", []) or [])]
        gate_rsu_round_roots = [int(x) for x in (gate_obj.get("rsu_round_roots", []) or [])]
        gate_pins_hash_field = int(_as_int(gate_obj.get("pins_hash_field", 0), 0))
        gate_policy_id_field = int(_as_int(gate_obj.get("policy_id_field", 0), 0))
        gate_pio_id_field = int(_as_int(gate_obj.get("public_input_order_id_field", 0), 0))
        # ✅ Resolve GLOBAL proof wrapper → real proof artifact if needed
        global_proof_path = _resolve_real_proof_artifact_path_v1(global_proof_path, out_root)
        # ✅ Enforce on-chain required files when global_anchor_ok is True
        if global_anchor_ok_flag:
            if (not global_round_manifest_path) or (not os.path.exists(global_round_manifest_path)):
                raise RuntimeError(f"[ONCHAIN] GLOBAL manifest missing: {global_round_manifest_path!r}")
            if (not global_proof_path) or (not os.path.exists(global_proof_path)):
                raise RuntimeError(f"[ONCHAIN] GLOBAL proof artifact missing: {global_proof_path!r}")
            if (not global_sidecar_path) or (not os.path.exists(global_sidecar_path)):
                raise RuntimeError(f"[ONCHAIN] GLOBAL public sidecar missing: {global_sidecar_path!r}")
            if len(gate_rsu_ids_sorted) != len(gate_mask_list) or len(gate_rsu_ids_sorted) != len(gate_rsu_round_roots):
                raise RuntimeError(
                    f"[ONCHAIN] GLOBAL preimage shape mismatch: "
                    f"len(rsu_ids_sorted)={len(gate_rsu_ids_sorted)} "
                    f"len(mask_list)={len(gate_mask_list)} "
                    f"len(rsu_round_roots)={len(gate_rsu_round_roots)}"
                )
            if gate_pins_hash_field <= 0 or gate_policy_id_field <= 0 or gate_pio_id_field <= 0:
                raise RuntimeError("[ONCHAIN] GLOBAL pins/policy/pio fields missing or invalid in gate manifest")
        export_index["global_round"] = {
            "anchor_ok": bool(global_anchor_ok_flag),
            "root_poseidon_field": int(global_round_root_poseidon_field),
            "used_rsu_ids": [int(x) for x in (gate_obj.get("used_rsu_ids", []) or [])],
            # ✅ REQUIRED: Full GLOBAL manifest preimage fields for on-chain reproducibility
            "rsu_ids_sorted": gate_rsu_ids_sorted,
            "mask_list": gate_mask_list,
            "rsu_round_roots": gate_rsu_round_roots,
            "pins_hash_field": int(gate_pins_hash_field),
            "policy_id_field": int(gate_policy_id_field),
            "public_input_order_id_field": int(gate_pio_id_field),
            "anchor_summary_json": {
                "path": _p(global_anchor_summary_path),
                "sha256": _sha256_file_hex_v1(global_anchor_summary_path),
            },
            "proof_json": {
                "path": _p(global_proof_path),
                "sha256": _sha256_file_hex_v1(global_proof_path),
            },
            "public_inputs_sidecar": {
                "path": _p(global_sidecar_path),
                "sha256": _sha256_file_hex_v1(global_sidecar_path),
            },
            "round_manifest_json": {
                "path": _p(global_round_manifest_path),
                "sha256": _sha256_file_hex_v1(global_round_manifest_path),
            },
        }
        return export_index
    # Optional: clear existing DP ledgers for a clean run
    dp_ledgers_dir = os.path.join(OUTPUT_DIR, "dp_ledgers")
    os.makedirs(dp_ledgers_dir, exist_ok=True)
    for fname in os.listdir(dp_ledgers_dir):
        if fname.endswith(".json"):
            try:
                os.remove(os.path.join(dp_ledgers_dir, fname))
            except OSError as e:
                logging.warning("Could not remove old DP ledger %s: %s", fname, e)
    # XGBoost hyperparameters for IoV IDS
    # Note: DP-XGBoost is enabled via tree_method="approxDP" and dp_epsilon_per_tree.
    # We follow the dp_xgboost examples: train as regression on {0,1} labels and
    # do classification via thresholding on the predicted mean.
    xgb_params = {
        "objective": "reg:squarederror",
        # Internal optimization/early-stopping metric; AUC/F1 are computed externally
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.2,
        "tree_method": "approxDP",  # <-- Sarus DP tree updater
        "dp_epsilon_per_tree": 0.25,  # <-- per-tree privacy budget (DP-friendly)
        "min_child_weight": 500,  # <-- large -> lower sensitivity/noise
        "subsample": 0.2,  # <-- matches dp_xgboost examples
        "lambda": 1.0,
        "alpha": 1.0,
        "colsample_bytree": 0.9,
        "base_score": 0.5,  # will be updated to global pos rate below
        "max_delta_step": 1.0,
        "random_state": 42,
        "seed": 42,
        # Control how many parallel trees are added per boosting round.
        # With FedXgbBagging this also controls how many trees are appended per update.
        "num_parallel_tree": 1,
    }
    # --- Learner-stage per-record-instance DP composition estimate (ignoring early stopping) ---
    dp_eps_tree = float(xgb_params.get("dp_epsilon_per_tree", 0.0))
    subsample = float(xgb_params.get("subsample", 1.0))
    num_parallel_tree = int(xgb_params.get("num_parallel_tree", 1))
    eps_per_round_upper: float | None = None
    eps_total_upper: float | None = None
    if dp_eps_tree > 0.0 and subsample > 0.0 and num_parallel_tree > 0:
        eps_factor = math.log(1.0 + subsample * (math.exp(dp_eps_tree) - 1.0))
        eps_per_round_upper = NUM_LOCAL_ROUNDS * num_parallel_tree * eps_factor
        eps_total_upper = NUM_ROUNDS * eps_per_round_upper
        logging.info(
            "Learner-stage per-record-instance DP composition estimate "
            "conditional on the fixed preprocessing artifact (ignoring early stopping): "
            "eps_per_round<=%.4f, eps_total<=%.4f "
            "(local_rounds=%d, global_rounds=%d, num_parallel_tree=%d)",
            eps_per_round_upper,
            eps_total_upper,
            NUM_LOCAL_ROUNDS,
            NUM_ROUNDS,
            num_parallel_tree,
        )
    else:
        logging.info(
            "Could not compute a meaningful DP epsilon bound "
            "(dp_epsilon_per_tree<=0, subsample<=0, or num_parallel_tree<=0)."
        )
    # ---------------------------
    # Persist high-level run configuration for comparison script
    # ---------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dp_enabled_flag = dp_eps_tree > 0.0
    run_config = {
        "dp_enabled": dp_enabled_flag,
        # Use the analytical upper bound if available, else 0.0
        "dp_epsilon_round": float(eps_per_round_upper) if eps_per_round_upper is not None else 0.0,
        "dp_delta_round": 0.0,
        # No explicit L1 clipping for dp_xgboost; keep None so comparison code prints b≈None
        "dp_clip_l1": None,
        "num_rounds": NUM_ROUNDS,
        "num_local_rounds": NUM_LOCAL_ROUNDS,
        "reviewer3_experiment_version": REVIEWER3_EXPERIMENT_VERSION,
        "reviewer3_scenario": REVIEWER3_SCENARIO,
        "reviewer3_seed": REVIEWER3_SEED,
        "reviewer3_stress_configuration": ABLATION_CONFIG,
        "num_rsus": NUM_RSUS,
        "vehicles_per_rsu": VEHICLES_PER_RSU,
        "tree_method": xgb_params.get("tree_method"),
        "dp_epsilon_per_tree": dp_eps_tree,
        "subsample": subsample,
        "num_parallel_tree": num_parallel_tree,
        "dp_epsilon_total": float(eps_total_upper) if eps_total_upper is not None else 0.0,
    }
    run_cfg_path = os.path.join(OUTPUT_DIR, "iov_run_config.json")
    try:
        with open(run_cfg_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
        logging.info("Saved DP run configuration to %s", run_cfg_path)
    except Exception:
        logging.exception(
            "Failed to write DP run configuration JSON"
        )
    # ---------------------------
    # Load IoV dataset
    # ---------------------------
    df_train, df_val, df_test = load_iov_splits(ds_cfg)
    X_train, y_train, X_val, y_val, X_test, y_test = extract_numeric_features(
        df_train, df_val, df_test, label_col=ds_cfg.label_col
    )
    # ---------------------------
    # Build/load anchor set once (shared across all RSUs)
    # ---------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    anchor_X_path = os.path.join(OUTPUT_DIR, "anchor_X.npy")
    anchor_meta_path = os.path.join(OUTPUT_DIR, "anchor_meta.json")
    # Load if present (and not overwriting), else build
    if (not ANCHOR_OVERWRITE) and os.path.exists(anchor_X_path) and os.path.exists(anchor_meta_path):
        X_anchor, meta_raw, anchor_id_field_raw = azkp.load_anchor_set(OUTPUT_DIR)
    else:
        X_anchor, meta_raw, anchor_id_field_raw = azkp.build_anchor_set(
            X_source=X_val,
            M=int(ANCHOR_M),
            seed=int(ANCHOR_SEED),
            out_dir=OUTPUT_DIR,
            SCALE=int(ANCHOR_SCALE),
            anchor_version=str(ANCHOR_VERSION),
            overwrite=bool(ANCHOR_OVERWRITE),
        )
    # --- FIX 1: ensure anchor_id_field exists and is int-like for later int(...) use ---
    anchor_id_field = _as_int(anchor_id_field_raw, 0)
    if anchor_id_field <= 0:
        raise RuntimeError(f"Invalid anchor_id_field from utils: {anchor_id_field_raw!r} -> {anchor_id_field}")
    # Normalize meta_raw -> AnchorMeta (type-checker safe)
    meta: azkp.AnchorMeta | None = None
    if isinstance(meta_raw, azkp.AnchorMeta):
        meta = meta_raw
    elif isinstance(meta_raw, dict):
        meta = azkp.AnchorMeta.from_dict(meta_raw)
    else:
        # Fallback: load from disk
        try:
            with open(anchor_meta_path, "r", encoding="utf-8") as f:
                meta = azkp.AnchorMeta.from_dict(json.load(f))
        except Exception as exc:
            raise RuntimeError(f"Failed to load anchor meta from {anchor_meta_path}: {exc}") from exc
    if meta is None:
        raise RuntimeError("AnchorMeta is None after normalization (unexpected)")
    # --- FIX 2: optionally enforce meta matches your run config (prevents silent drift) ---
    if str(getattr(meta, "anchor_version", "")) != str(ANCHOR_VERSION):
        raise RuntimeError(
            f"anchor_version mismatch: meta={getattr(meta, 'anchor_version', None)!r} vs cfg={ANCHOR_VERSION!r}"
        )
    if int(getattr(meta, "M", -1)) != int(ANCHOR_M):
        raise RuntimeError(f"M mismatch: meta={getattr(meta, 'M', None)} vs cfg={ANCHOR_M}")
    if int(getattr(meta, "SCALE", -1)) != int(ANCHOR_SCALE):
        raise RuntimeError(f"SCALE mismatch: meta={getattr(meta, 'SCALE', None)} vs cfg={ANCHOR_SCALE}")
    # Canonical context shipped to vehicles + RSUs
    root_poseidon_field = _as_int(getattr(meta, "root_poseidon_field", 0), 0)
    if root_poseidon_field <= 0:
        raise RuntimeError(
            "AnchorMeta is missing root_poseidon_field (or it's 0). "
            "Rebuild anchors with the updated AnchorZKP utils that writes this field."
        )
    anchor_root_poseidon_field = int(root_poseidon_field)
    anchor_ctx: Dict[str, Any] = {
        "anchor_version": _as_str(getattr(meta, "anchor_version", ANCHOR_VERSION), ANCHOR_VERSION),
        "anchor_id_field": str(anchor_id_field),
        "M": _as_int(getattr(meta, "M", ANCHOR_M), ANCHOR_M),
        "SCALE": _as_int(getattr(meta, "SCALE", ANCHOR_SCALE), ANCHOR_SCALE),
        # ✅ explicit anchor-set root
        "anchor_root_poseidon_field": int(anchor_root_poseidon_field),
        # ⚠️ temporary alias for older code paths
        "root_poseidon_field": int(anchor_root_poseidon_field),
        "anchor_X_path": str(anchor_X_path),
        "anchor_meta_path": str(anchor_meta_path),
    }
    logging.info("[ANCHOR] anchor_ctx=%s", anchor_ctx)
    # -------------------------------------------------------------------
    # V15 (AnchorZKP Utils V8b): precompile Groth16 artifacts BEFORE any Ray/Flower actor runs
    # V8b layout (matches your precompile_anchorsum_groth16 implementation + your actual tree):
    #   - Circuit source (*.circom) is written under: cfg.generated_circuits_dir
    #   - Circom outputs (*.r1cs/*.wasm/*.sym) are emitted under: cfg.artifacts_dir/<circuit_stem>/
    #   - Groth16 artifacts (*.zkey + verification_key.json/vkey.json) are also under that same <circuit_stem>/
    # Strict precompile: verify the full set; if partial state exists, self-heal by forcing per-circuit rebuild.
    # -------------------------------------------------------------------
    base_out = Path(OUTPUT_DIR).resolve()
    anchorsum_cfg = azkp.AnchorZKPConfig(
        M=int(_as_int(anchor_ctx.get("M"), 0)),
        SCALE=int(_as_int(anchor_ctx.get("SCALE"), 0)),
        enable_range_checks=False,
        generated_circuits_dir=(base_out / "circuits_generated" / "anchorsum"),
        artifacts_dir=(base_out / "zkp_artifacts" / "anchorsum"),
        run_root_dir=(base_out / "zkp_runs" / "anchorsum"),
    )
    anchorsum_spec_rsu = anchorsum_cfg.spec_rsu(NMAX=int(VEHICLES_PER_RSU))
    anchorsum_spec_global = anchorsum_cfg.spec_global(RMAX=int(NUM_RSUS), NMAX=int(VEHICLES_PER_RSU))
    logging.info("[ZKP] anchorsum_spec_rsu public_inputs=%s", getattr(anchorsum_spec_rsu, "public_inputs", None))
    logging.info("[ZKP] anchorsum_spec_global public_inputs=%s", getattr(anchorsum_spec_global, "public_inputs", None))

    # Precompile both RSU and GLOBAL specs (strict, with self-heal)
    _ensure_precompile_ok_v15(anchorsum_cfg, anchorsum_spec_rsu, label="anchorsum RSU")
    _ensure_precompile_ok_v15(anchorsum_cfg, anchorsum_spec_global, label="anchorsum GLOBAL")
    # Final invariant checks (both circuits)
    _assert_anchor_precompile_outputs_v15(anchorsum_cfg, anchorsum_spec_rsu, label="anchorsum RSU (final)")
    _assert_anchor_precompile_outputs_v15(anchorsum_cfg, anchorsum_spec_global, label="anchorsum GLOBAL (final)")
    # Ship deterministic artifact locations to RSU/Vehicle code paths
    anchor_ctx["anchorsum_generated_circuits_dir"] = str(anchorsum_cfg.generated_circuits_dir)
    anchor_ctx["anchorsum_artifacts_dir"] = str(anchorsum_cfg.artifacts_dir)
    anchor_ctx["anchorsum_run_root_dir"] = str(anchorsum_cfg.run_root_dir)
    # ---------------------------
    # Global feature bounds for DP-XGBoost
    # ---------------------------
    feature_min = X_train.min(axis=0).to_numpy(dtype=np.float32)
    feature_max = X_train.max(axis=0).to_numpy(dtype=np.float32)
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
        # For reg:squarederror with {0,1} labels, this sets the prior mean of y.
        xgb_params["base_score"] = float(pos_frac)
    else:
        raise ValueError(
            "Train split has only one class (pos or neg). "
            "Check your preprocessed CSECICIDS2018_train_preprocessed.csv: "
            "scale_pos_weight cannot be computed."
        )
    # ---------------------------
    # Reviewer 3 controlled partition condition
    # ---------------------------
    reviewer3_partition_meta: Dict[str, Any] = {}
    if REVIEWER3_SCENARIO == "balanced_reference":
        rsu_partitions = partition_train_among_rsus_and_vehicles(
            X_train,
            y_train,
            num_rsus=NUM_RSUS,
            vehicles_per_rsu=VEHICLES_PER_RSU,
            random_state=REVIEWER3_SEED,
            k_min_pos=1,
            k_min_neg=1,
        )

        # Use exactly the same deterministic feature diagnostic as the
        # heterogeneous scenarios so feature-shift values are comparable.
        ref_feat_j, ref_feat_name = _reviewer3_select_feature_for_skew_v1(
            X_train,
            seed=REVIEWER3_SEED,
        )

        reviewer3_partition_meta = {
            "partition_mode": "existing_balanced_stratified_reference",
            "seed": int(REVIEWER3_SEED),
            "feature_skew_feature_index": int(ref_feat_j),
            "feature_skew_feature_name": str(ref_feat_name),
        }
    else:
        rsu_partitions, reviewer3_partition_meta = _reviewer3_partition_heterogeneous_v1(
            X_train,
            y_train,
            num_rsus=NUM_RSUS,
            vehicles_per_rsu=VEHICLES_PER_RSU,
            seed=REVIEWER3_SEED,
            label_alpha=0.30,
            quantity_sigma=0.90,
            max_size_ratio=10.0,
        )

    # Existing diagnostics plus reviewer-specific machine-readable statistics.
    log_partition_stats(rsu_partitions)
    _reviewer3_measure_and_write_partition_v1(
        rsu_partitions,
        output_dir=OUTPUT_DIR,
        scenario=REVIEWER3_SCENARIO,
        seed=REVIEWER3_SEED,
        heterogeneity_meta=reviewer3_partition_meta,
    )
    # ---------------------------
    # Run FL per RSU
    # ---------------------------
    ssi_fp = _resolve_ssi_fp_v15(azkp)
    anchor_ctx["ssi_preimage_fingerprint_v1"] = dict(ssi_fp)
    for rsu_id in rsu_ids_sorted:
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
            feature_min=feature_min,
            feature_max=feature_max,
            anchor_ctx=anchor_ctx,
            ablation_cfg=ABLATION_CONFIG,
        )
    logging.info("All RSUs finished IoV federated training.")
    # -------------------------------------------------------------------
    # GLOBAL AnchorZKP stage (RSU -> Global anchor-sum verification)
    # -------------------------------------------------------------------
    TARGET_ROUND = int(NUM_ROUNDS)  # Flower rounds are 1..NUM_ROUNDS
    M_global = _as_int(anchor_ctx.get("M"), 0)
    # Will be set when we build the GLOBAL manifest (0 means “not computed”)
    global_round_root_poseidon_field: int = 0
    rsu_ok_map: Dict[int, bool] = {}
    rsu_artifact_path_map: Dict[int, str] = {}
    rsu_used_ids: List[int] = []
    pins_hash_field: int = 0
    policy_id_field: int = 0
    public_input_order_id_field: int = 0
    # 1) Load per-RSU summary -> find valid flag + artifact path for TARGET_ROUND
    for rsu_id in rsu_ids_sorted:
        summary_path = os.path.join(OUTPUT_DIR, f"iov_rsu_{rsu_id}_summary.json")
        ok = False
        art_path = ""
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                # Support both possible summary key spellings (keeps you compatible)
                valid_map = s.get("rsu_anchor_round_valid", s.get("rsu_round_valid", {}))
                artifacts_map = s.get("rsu_anchor_artifacts", s.get("rsu_anchor_artifact", {}))
                ok_val = _get_round_keyed(valid_map, TARGET_ROUND, 0)
                ok = bool(int(ok_val)) if isinstance(ok_val, (int, float, str)) else bool(ok_val)
                art_val = _get_round_keyed(artifacts_map, TARGET_ROUND, "")
                if isinstance(art_val, str):
                    art_path = art_val
                elif isinstance(art_val, dict):
                    art_path = str(
                        art_val.get("proof_artifact_path")
                        or art_val.get("artifact_path")
                        or art_val.get("path")
                        or ""
                    )
                else:
                    art_path = ""
                rsu_run_dir_from_summary = str(s.get("rsu_run_dir", "") or "")
                # If strategy stored a relative path, prefer rsu_run_dir, then OUTPUT_DIR
                if art_path and not os.path.isabs(art_path):
                    cand = ""
                    if rsu_run_dir_from_summary:
                        cand = os.path.join(rsu_run_dir_from_summary, art_path)
                        if os.path.exists(cand):
                            art_path = cand
                        else:
                            art_path = os.path.join(str(OUTPUT_DIR), art_path)
                    else:
                        art_path = os.path.join(str(OUTPUT_DIR), art_path)
            except Exception as exc:
                logging.warning("[GLOBAL] Failed to read RSU summary rsu=%d: %s", rsu_id, exc)
                ok = False
                art_path = ""
        else:
            logging.warning("[GLOBAL] Missing RSU summary for rsu=%d at %s", rsu_id, summary_path)
        rsu_ok_map[rsu_id] = bool(ok)
        rsu_artifact_path_map[rsu_id] = str(art_path)
    # 2) Build fixed-order Q_rsu_list + mask_list across RSU IDs (bind mask positions)
    global_total_t0 = time.perf_counter()
    global_anchor_ok = False
    global_artifact_path = ""
    global_proof_path = ""
    global_sidecar_path = ""
    Q_global_b64 = ""
    global_commit_field = ""
    global_round_root_poseidon_field = 0  # ✅ keep manifest honest on failure
    global_aggregation_latency_sec = 0.0
    global_zkp_latency_sec = 0.0
    global_artifact_export_latency_sec = 0.0
    global_total_latency_sec = 0.0
    mask_all: List[int] = []
    try:
        if M_global > 0:
            # load meta once
            with open(
                    str(anchor_ctx["anchor_meta_path"]),
                    "r",
                    encoding="utf-8",
            ) as f:
                global_meta: azkp.AnchorMeta = (
                    azkp.AnchorMeta.from_dict(json.load(f))
                )
            # Reuse the driver-precompiled config/spec (V8b invariant)
            cfg = anchorsum_cfg
            global_spec = anchorsum_spec_global
            rsu_Q_all: List[np.ndarray] = []
            mask_all = []
            rsu_used_ids = []
            rsu_round_root_map: Dict[int, int] = {}
            for rsu_id in rsu_ids_sorted:
                ok = bool(rsu_ok_map.get(rsu_id, False))
                art_path = str(rsu_artifact_path_map.get(rsu_id, "") or "")
                if ok and art_path and os.path.exists(art_path):
                    try:
                        with open(art_path, "r", encoding="utf-8") as f:
                            art = json.load(f)
                        # ✅ If this is a wrapper/summary, follow the pointer to the real proof artifact
                        if isinstance(art, dict) and ("Q_rsu_b64" not in art):
                            meta0 = art.get("meta", {})
                            meta0 = meta0 if isinstance(meta0, dict) else {}
                            hinted = str(
                                art.get("proof_artifact_path")
                                or art.get("proof_artifact_relpath")
                                or art.get("artifact_path")
                                or art.get("proof_path")
                                or art.get("proof_json_path")
                                or art.get("path")
                                or meta0.get("proof_artifact_path")
                                or meta0.get("proof_artifact_relpath")
                                or meta0.get("artifact_path")
                                or meta0.get("proof_path")
                                or meta0.get("proof_json_path")
                                or meta0.get("path")
                                or ""
                            ).strip()
                            if hinted:
                                hinted_abs = hinted
                                if not os.path.isabs(hinted_abs):
                                    hinted_abs = os.path.join(str(OUTPUT_DIR), hinted_abs)
                                if os.path.exists(hinted_abs):
                                    with open(hinted_abs, "r", encoding="utf-8") as ff:
                                        art = json.load(ff)
                                else:
                                    raise ValueError(f"proof artifact pointer points to missing file: {hinted_abs!r}")
                        # -------------------------------------------------------------------
                        # Bind anchor identity + enforce V15 audit invariants (PUBMASK + SSI fp + pins + sidecar)
                        # -------------------------------------------------------------------
                        art_meta = art.get("meta", {}) if isinstance(art.get("meta", {}), dict) else {}
                        # ---- Consistency checks (GLOBAL reading RSU summary) ----
                        if str(art.get("anchor_version", "")) != str(anchor_ctx.get("anchor_version", "")):
                            raise ValueError("anchor_version mismatch")
                        if int(str(art.get("anchor_id_field", -1))) != int(str(anchor_ctx.get("anchor_id_field", -2))):
                            raise ValueError("anchor_id_field mismatch")
                        if int(str(art.get("M", -1))) != int(_as_int(anchor_ctx.get("M"), 0)):
                            raise ValueError("M mismatch")
                        if int(str(art.get("SCALE", -1))) != int(_as_int(anchor_ctx.get("SCALE"), 0)):
                            raise ValueError("SCALE mismatch")
                        # ---- Bind to the same anchor-set Poseidon root ----
                        art_anchor_root = art.get("anchor_root_poseidon_field", None)
                        if art_anchor_root is None or str(art_anchor_root).strip() == "":
                            raise ValueError(
                                "missing anchor_root_poseidon_field (RSU artifact must bind to anchor root)")
                        if int(str(art_anchor_root)) != int(_as_int(anchor_ctx.get("anchor_root_poseidon_field"), 0)):
                            raise ValueError("anchor_root_poseidon_field mismatch")
                        # ---- V15: PUBMASK-only enforcement ----
                        art_selection = str(
                            art.get("selection_mode")
                            or art_meta.get("selection_mode")
                            or art.get("zk_selection_mode")
                            or art_meta.get("zk_selection_mode")
                            or ""
                        ).strip().lower()
                        if art_selection != "pubmask":
                            raise ValueError(f"selection_mode mismatch (expected 'pubmask', got {art_selection!r})")
                        # ---- V15: SSI fingerprint binding (must match runtime fingerprint) ----
                        ssi_fp_local = anchor_ctx.get("ssi_preimage_fingerprint_v1", None)
                        ssi_fp_local = ssi_fp_local if isinstance(ssi_fp_local, dict) else ssi_fp
                        expected_def_sha = _norm_sha256_hex_v15(ssi_fp_local.get("ssi_preimage_def_sha256_v1", ""))
                        expected_def_field_int = _parse_intish_v15(
                            ssi_fp_local.get("ssi_preimage_def_field_bn254_v1", 0)
                        )
                        if not expected_def_sha or expected_def_field_int <= 0:
                            raise ValueError("runtime SSI fingerprint missing required fields")
                        got_def_sha, got_def_field_int = _extract_ssi_def_fp_v15(art, art_meta)
                        if got_def_sha != expected_def_sha:
                            raise ValueError("ssi_preimage_def_sha256_v1 mismatch")
                        if got_def_field_int != expected_def_field_int:
                            raise ValueError("ssi_preimage_def_field_bn254_v1 mismatch")
                        # ---- V15: Required pins fields must exist (audit lane) ----
                        pins_hash_field_rsu = _read_int_field_from(art, "pins_hash_field") or _read_int_field_from(
                            art_meta, "pins_hash_field")
                        policy_id_field_rsu = _read_int_field_from(art, "policy_id_field") or _read_int_field_from(
                            art_meta, "policy_id_field")
                        pio_id_field_rsu = _read_int_field_from(art,
                                                                "public_input_order_id_field") or _read_int_field_from(
                            art_meta, "public_input_order_id_field")
                        if pins_hash_field_rsu <= 0:
                            raise ValueError("missing/invalid pins_hash_field in RSU artifact")
                        if policy_id_field_rsu <= 0:
                            raise ValueError("missing/invalid policy_id_field in RSU artifact")
                        if pio_id_field_rsu <= 0:
                            raise ValueError("missing/invalid public_input_order_id_field in RSU artifact")
                        # ---- V15: Public sidecar must exist (stable evidence of public inputs) ----
                        rsu_sidecar = os.path.join(
                            str(cfg.artifacts_dir),
                            f"rsu_{int(rsu_id)}",
                            f"round_{int(TARGET_ROUND)}_public.json",
                        )
                        if not os.path.exists(rsu_sidecar):
                            raise ValueError(f"missing RSU public sidecar: {rsu_sidecar}")
                        # ---- Require RSU proof to be bound to a published RSU-round root ----
                        rsu_round_root = art.get("root_poseidon_field", None)
                        if rsu_round_root is None or str(rsu_round_root).strip() == "" or int(str(rsu_round_root)) <= 0:
                            raise ValueError(
                                "missing/invalid root_poseidon_field in RSU artifact (RSU proof not bound to round root)")
                        # ---- Require and read RSU round manifest ----
                        rsu_round_manifest_path = str(art.get("rsu_round_manifest_path", "") or "")
                        if rsu_round_manifest_path and (not os.path.isabs(rsu_round_manifest_path)):
                            rsu_round_manifest_path = os.path.join(str(OUTPUT_DIR), rsu_round_manifest_path)
                        if (not rsu_round_manifest_path) or (not os.path.exists(rsu_round_manifest_path)):
                            raise ValueError(
                                f"missing/invalid rsu_round_manifest_path in RSU artifact: {rsu_round_manifest_path!r}")
                        # ---- Hard bind: artifact root must match manifest root (prevents drift X vs Y) ----
                        try:
                            with open(rsu_round_manifest_path, "r", encoding="utf-8") as f:
                                rsu_mani = json.load(f)
                        except Exception as exc:
                            raise ValueError(f"failed to read rsu_round_manifest_path JSON: {exc}") from exc
                        mani_root = rsu_mani.get("root_poseidon_field", None)
                        if mani_root is None or str(mani_root).strip() == "" or int(str(mani_root)) <= 0:
                            raise ValueError("RSU round manifest missing/invalid root_poseidon_field")
                        if int(str(mani_root)) != int(str(rsu_round_root)):
                            raise ValueError(
                                f"RSU artifact root_poseidon_field != manifest root_poseidon_field "
                                f"({int(str(rsu_round_root))} != {int(str(mani_root))})"
                            )
                        # ---- Optional but recommended: ensure round + rsu_id match (if present in manifest) ----
                        mani_round = rsu_mani.get("round", None)
                        if mani_round is not None and int(str(mani_round)) != int(TARGET_ROUND):
                            raise ValueError(f"RSU round manifest round mismatch ({mani_round} != {TARGET_ROUND})")
                        mani_rsu_id = rsu_mani.get("rsu_id", None)
                        if mani_rsu_id is not None and int(str(mani_rsu_id)) != int(rsu_id):
                            raise ValueError(f"RSU round manifest rsu_id mismatch ({mani_rsu_id} != {rsu_id})")
                        # ---- Load payload fields needed for GLOBAL proof ----
                        Q_b64 = art.get("Q_rsu_b64", "")
                        gcf = str(
                            art.get("global_commit_field")
                            or art.get("agg_commit_field")
                            or art.get("rsu_commit_field")
                            or ""
                        )
                        if not isinstance(Q_b64, str) or not Q_b64.strip():
                            raise ValueError("missing Q_rsu_b64")
                        if not gcf.strip():
                            raise ValueError("missing rsu commit field (global_commit_field/agg_commit_field)")
                        q = azkp.decode_int_vector_b64(Q_b64, expected_len=int(M_global))
                        # ---- Accept RSU into GLOBAL proof inputs ----
                        rsu_round_root_map[rsu_id] = int(str(rsu_round_root))
                        rsu_Q_all.append(q)
                        mask_all.append(1)
                        rsu_used_ids.append(rsu_id)
                        continue
                    except Exception as exc:
                        logging.warning(
                            "[ZKP][GLOBAL] RSU %d excluded: artifact invalid/unusable (%s) | path=%s",
                            rsu_id,
                            exc,
                            art_path,
                        )
                rsu_round_root_map[rsu_id] = 0  # ✅ explicit “not included / invalid”
                rsu_Q_all.append(np.zeros((int(M_global),), dtype=np.int64))
                mask_all.append(0)
                rsu_ok_map[rsu_id] = False
            # 3) Prove/verify GLOBAL anchor-sum (only if >=1 RSU is valid)
            if any(mask_all):
                policy_id_field = int(_resolve_policy_id_field_v1())
                public_input_order_id_field = int(_resolve_public_input_order_id_field_v1(global_spec, cfg_obj=cfg))
                pins_hash_field = int(
                    _resolve_pins_hash_field_v1(
                        spec_obj=global_spec,
                        cfg_obj=cfg,
                        anchor_ctx_obj=anchor_ctx,
                        RMAX=int(NUM_RSUS),
                        NMAX=int(VEHICLES_PER_RSU),
                    )
                )
                global_round_manifest = {
                    "schema": "GlobalRoundManifestV1",
                    "round": int(TARGET_ROUND),
                    "anchor_root_poseidon_field": int(_as_int(anchor_ctx.get("anchor_root_poseidon_field"), 0)),
                    "rsu_ids_sorted": [int(x) for x in rsu_ids_sorted],
                    "mask_list": [int(m) for m in mask_all],
                    "rsu_round_roots": [int(rsu_round_root_map.get(int(rid), 0)) for rid in rsu_ids_sorted],
                    # ✅ bind required public-input pins into the round root
                    "pins_hash_field": int(pins_hash_field),
                    "policy_id_field": int(policy_id_field),
                    "public_input_order_id_field": int(public_input_order_id_field),
                }
                global_round_manifest_path = os.path.join(
                    str(OUTPUT_DIR),
                    "zkp_round_manifests",
                    "global",
                    f"round_{int(TARGET_ROUND)}.json",
                )
                os.makedirs(os.path.dirname(global_round_manifest_path), exist_ok=True)
                with open(global_round_manifest_path, "w", encoding="utf-8") as f:
                    json.dump(global_round_manifest, f, indent=2)
                global_round_root_poseidon_field = _sha256_to_field(_canon_json_bytes(global_round_manifest))
                print("azkp module file:", azkp.__file__)
                print("prove_verify_global_anchor_sum_from_meta file:",
                      inspect.getsourcefile(azkp.prove_verify_global_anchor_sum_from_meta))
                print("prove_verify_global_anchor_sum_from_meta signature:",
                      inspect.signature(azkp.prove_verify_global_anchor_sum_from_meta))
                print("prove_verify_global_anchor_sum file:",
                      inspect.getsourcefile(azkp.prove_verify_global_anchor_sum))
                print("prove_verify_global_anchor_sum signature:",
                      inspect.signature(azkp.prove_verify_global_anchor_sum))
                _t_global_zkp0 = time.perf_counter()
                global_art = azkp.prove_verify_global_anchor_sum_from_meta(
                    cfg=cfg,
                    spec=global_spec,
                    meta=global_meta,
                    anchor_id_field=_as_int(anchor_ctx.get("anchor_id_field"), 0),
                    round_idx=int(TARGET_ROUND),
                    global_id=0,
                    root_poseidon_field=int(global_round_root_poseidon_field),
                    Q_rsu_list=rsu_Q_all,
                    mask_list=mask_all,
                    RMAX=int(NUM_RSUS),
                    NMAX=int(VEHICLES_PER_RSU),
                    out_dir=cfg.artifacts_dir,
                    # ✅ REQUIRED by V15 spec/circuit
                    pins_hash_field=int(pins_hash_field),
                    policy_id_field=int(policy_id_field),
                    public_input_order_id_field=int(public_input_order_id_field),
                    include_Q_b64=True,
                )
                global_zkp_latency_sec = time.perf_counter() - _t_global_zkp0
                Q_global_b64 = str(global_art.payload.get("Q_global_b64", "") or "")
                global_commit_field = str(
                    global_art.public.get("global_commit_field")
                    or global_art.public.get("agg_commit_field")
                    or global_art.public.get("commit_field")
                    or ""
                )
                global_anchor_ok = bool(global_art.ok) and bool(Q_global_b64) and bool(global_commit_field)
                # REAL proof artifact path (utils output)
                # utils writes: <artifacts_dir>/global/round_<r>.json
                # Prefer path returned by utils (if present), else fall back to your convention
                global_proof_path = ""
                if isinstance(getattr(global_art, "payload", None), dict):
                    global_proof_path = str(global_art.payload.get("artifact_path", "") or "")
                if not global_proof_path:
                    global_proof_path = os.path.join(str(cfg.artifacts_dir), "global", f"round_{TARGET_ROUND}.json")

                if global_proof_path and not os.path.exists(global_proof_path):
                    logging.warning(
                        "[ZKP][GLOBAL] Proof path does not exist (check utils output): %s",
                        global_proof_path,
                    )
                # keep your normalized summaries aligned (avoid "global_0" confusion)
                global_summary_path = os.path.join(
                    str(OUTPUT_DIR),
                    "zkp_anchor_summaries",
                    "anchorsum",
                    "global",
                    f"round_{TARGET_ROUND}.json",
                )
                # -------------------------------------------------------------------
                # Extract public inputs defensively (snarkjs writes them to public.json)
                # -------------------------------------------------------------------
                global_public_inputs = []
                if isinstance(getattr(global_art, "public", None), dict):
                    global_public_inputs = (
                            global_art.public.get("public_inputs")
                            or global_art.public.get("publicSignals")
                            or global_art.public.get("public_signals")
                            or global_art.public.get("public")
                            or []
                    )
                elif isinstance(getattr(global_art, "payload", None), dict):
                    global_public_inputs = (
                            global_art.payload.get("public_inputs")
                            or global_art.payload.get("publicSignals")
                            or global_art.payload.get("public_signals")
                            or []
                    )
                if not isinstance(global_public_inputs, list):
                    global_public_inputs = [global_public_inputs]
                global_public_inputs = [str(x) for x in global_public_inputs]
                _t_global_export0 = time.perf_counter()
                global_sidecar_path = os.path.join(
                    str(cfg.artifacts_dir),
                    "global",
                    f"round_{int(TARGET_ROUND)}_public.json",
                )
                # V15: ensure GLOBAL sidecar exists (prefer in-memory publics; fallback to disk proof/public.json)
                if not os.path.exists(global_sidecar_path):
                    if not global_public_inputs:
                        # Fallback 1: read publics from the proof artifact JSON on disk
                        if global_proof_path and os.path.exists(global_proof_path):
                            try:
                                with open(global_proof_path, "r", encoding="utf-8") as f:
                                    g_art_disk = json.load(f)
                                global_public_inputs = _extract_public_inputs_v15(g_art_disk)
                            except Exception:
                                global_public_inputs = []
                        # Fallback 2: read sibling snarkjs public output next to the proof
                        if not global_public_inputs and global_proof_path and os.path.exists(global_proof_path):
                            base_dir = os.path.dirname(global_proof_path)
                            pub_path = ""
                            for name in ("public.json", "pub.json", "publicSignals.json"):
                                cand = os.path.join(base_dir, name)
                                if os.path.exists(cand):
                                    pub_path = cand
                                    break
                            if pub_path:
                                try:
                                    with open(pub_path, "r", encoding="utf-8") as f:
                                        pub_obj = json.load(f)
                                    if isinstance(pub_obj, list):
                                        global_public_inputs = [str(x) for x in pub_obj]
                                    elif isinstance(pub_obj, dict):
                                        global_public_inputs = _extract_public_inputs_v15(pub_obj)
                                except Exception:
                                    global_public_inputs = []
                    if not global_public_inputs:
                        raise RuntimeError(
                            f"Missing GLOBAL public sidecar and could not derive public inputs from disk: {global_sidecar_path}"
                        )
                    _write_public_sidecar_v15(
                        global_sidecar_path,
                        [str(x) for x in global_public_inputs],
                        who="global",
                        round_idx=int(TARGET_ROUND),
                    )
                    logging.info("[ZKP][GLOBAL] Wrote GLOBAL public sidecar to %s", global_sidecar_path)
                artifact_norm = {
                    "ok": bool(global_anchor_ok),
                    "round": int(TARGET_ROUND),
                    "global_id": 0,
                    "anchor_version": str(anchor_ctx.get("anchor_version", "")),
                    "anchor_id_field": str(anchor_ctx.get("anchor_id_field", "")),
                    "M": _as_int(anchor_ctx.get("M"), 0),
                    "SCALE": _as_int(anchor_ctx.get("SCALE"), 0),
                    "used_rsu_ids": [int(x) for x in rsu_used_ids],
                    "Q_global_b64": str(Q_global_b64),
                    "global_commit_field": str(global_commit_field),
                    # ✅ include these so later readers never crash
                    "public_inputs": global_public_inputs,
                    "proof_artifact_path": str(global_proof_path),
                    "root_poseidon_field": int(global_round_root_poseidon_field),
                    "anchor_root_poseidon_field": int(_as_int(anchor_ctx.get("anchor_root_poseidon_field"), 0)),
                    "pins_hash_field": int(pins_hash_field),
                    "policy_id_field": int(policy_id_field),
                    "public_input_order_id_field": int(public_input_order_id_field),
                    "public_inputs_sidecar_path": str(global_sidecar_path),
                    "selection_mode": "pubmask",
                }
                os.makedirs(os.path.dirname(global_summary_path), exist_ok=True)
                with open(global_summary_path, "w", encoding="utf-8") as f:
                    json.dump(artifact_norm, f, indent=2)
                global_artifact_path = str(global_summary_path)
                global_artifact_export_latency_sec = time.perf_counter() - _t_global_export0
                global_total_latency_sec = (
                        float(global_aggregation_latency_sec)
                        + float(global_zkp_latency_sec)
                        + float(global_artifact_export_latency_sec)
                )
                if global_anchor_ok:
                    logging.info(
                        "[ZKP][GLOBAL] ✅ AnchorSum verification SUCCEEDED | round=%d used_rsus=%s root_poseidon_field=%s artifact=%s",
                        TARGET_ROUND,
                        rsu_used_ids,
                        str(global_round_root_poseidon_field),
                        global_artifact_path,
                    )
                else:
                    logging.error(
                        "[ZKP][GLOBAL] ❌ AnchorSum verification FAILED | used_rsus=%s artifact=%s",
                        rsu_used_ids,
                        global_artifact_path,
                    )
            else:
                logging.warning("[ZKP][GLOBAL] No RSUs passed RSU-level gating; skipping global proof.")
    except Exception as exc:
        logging.warning("[ZKP][GLOBAL] Global anchor-sum proof FAILED: %s", exc)
        global_anchor_ok = False
        global_artifact_path = ""
        Q_global_b64 = ""
        global_commit_field = ""
        # ✅ Keep manifest shape stable even on failure
        if not mask_all or len(mask_all) != len(rsu_ids_sorted):
            mask_all = [0] * len(rsu_ids_sorted)
        if not rsu_used_ids:
            rsu_used_ids = []
    # 4) Persist a small gating manifest for the ensemble stage
    # -------------------------------------------------------------------
    # ✅ Gate manifest (GLOBAL stage) — now includes rsu_round_roots + global_round_manifest_path
    # -------------------------------------------------------------------
    # Safe defaults so gate_manifest never crashes on failure paths
    _rsu_round_root_map_safe: Dict[int, int]
    _rsu_round_root_map_obj = locals().get(
        "rsu_round_root_map"
    )

    if isinstance(_rsu_round_root_map_obj, dict):
        _rsu_round_root_map_safe = {
            int(k): int(_as_int(v, 0))
            for k, v in _rsu_round_root_map_obj.items()
        }
    else:
        _rsu_round_root_map_safe = {
            int(rid): 0 for rid in rsu_ids_sorted
        }
    _global_round_manifest_path_safe = ""
    if "global_round_manifest_path" in locals():
        _global_round_manifest_path_safe = str(locals().get("global_round_manifest_path") or "")
    if not _global_round_manifest_path_safe:
        _global_round_manifest_path_safe = os.path.join(
            str(OUTPUT_DIR),
            "zkp_round_manifests",
            "global",
            f"round_{int(TARGET_ROUND)}.json",
        )
    gate_manifest = {
        "round": int(TARGET_ROUND),
        # Anchor identity + public input
        "root_poseidon_field": int(global_round_root_poseidon_field),  # ✅ GLOBAL round-binding root
        "anchor_root_poseidon_field": int(_as_int(anchor_ctx.get("anchor_root_poseidon_field"), 0)),
        "anchor_id_field": str(anchor_ctx.get("anchor_id_field", "")),
        "anchor_version": str(anchor_ctx.get("anchor_version", "")),
        "M": int(_as_int(anchor_ctx.get("M"), 0)),
        "SCALE": int(_as_int(anchor_ctx.get("SCALE"), 0)),
        # ✅ EXACT RSU order used for (Q_rsu_list, mask_list)
        "rsu_ids_sorted": [int(x) for x in rsu_ids_sorted],
        # ✅ Mask positions correspond to rsu_ids_sorted
        "mask_list": [int(m) for m in mask_all],
        # ✅ REQUIRED for on-chain reproducibility of GlobalRoundManifestV1 preimage
        "rsu_round_roots": [int(_rsu_round_root_map_safe.get(int(rid), 0)) for rid in rsu_ids_sorted],
        "global_round_manifest_path": str(_global_round_manifest_path_safe),
        # ✅ RSUs that had valid Q payload and were masked-in (eligible for proof)
        "rsus_with_valid_anchor_payload": [int(x) for x in rsu_used_ids],
        # NEW: persist file pointers for audit reproducibility
        "anchor_X_path": str(anchor_ctx.get("anchor_X_path", "")),
        "anchor_meta_path": str(anchor_ctx.get("anchor_meta_path", "")),
        # ZKP outputs
        "global_commit_field": str(global_commit_field),
        "rsu_ok_map": {str(k): bool(v) for k, v in rsu_ok_map.items()},
        "used_rsu_ids": [int(x) for x in rsu_used_ids],
        "global_anchor_ok": bool(global_anchor_ok),
        "global_artifact_path": str(global_artifact_path),
        "Q_global_b64": str(Q_global_b64),
        # Diagnostics
        "num_rsus_configured": int(NUM_RSUS),
        "num_rsus_masked_in": int(sum(1 for m in mask_all if int(m) == 1)),
        # Required pins for deterministic verification
        "pins_hash_field": int(pins_hash_field),
        "policy_id_field": int(policy_id_field),
        "public_input_order_id_field": int(public_input_order_id_field),
        # SSI fingerprint evidence
        "ssi_preimage_fingerprint_v1": ssi_fp,
        # Stable GLOBAL public inputs sidecar path
        "global_public_inputs_sidecar_path": os.path.join(
            str(anchorsum_cfg.artifacts_dir),
            "global",
            f"round_{int(TARGET_ROUND)}_public.json",
        ),
    }
    gate_path = os.path.join(OUTPUT_DIR, "zkp_anchor_gating_manifest.json")
    try:
        with open(gate_path, "w", encoding="utf-8") as f:
            json.dump(gate_manifest, f, indent=2)
        logging.info("[ZKP][GLOBAL] Saved anchor gating manifest to %s", gate_path)
    except Exception as exc:
        logging.warning("[ZKP][GLOBAL] Failed to write gating manifest: %s", exc)
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
            feature_min=feature_min,
            feature_max=feature_max,
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
    # Reuse the same numeric feature subset/order as during training
    X_val_np = np.asarray(
        X_val.to_numpy(dtype=np.float32, copy=False),
        dtype=np.float32,
    )
    y_val_np = np.asarray(
        y_val.to_numpy(copy=False),
        dtype=np.int32,
    )

    X_val_np = np.asarray(
        np.clip(X_val_np, feature_min, feature_max),
        dtype=np.float32,
    )

    dval = _make_dmatrix_version_safe(
        X_val_np,
        y_val_np,
        feature_min,
        feature_max,
    )

    X_test_np = np.asarray(
        X_test.to_numpy(dtype=np.float32, copy=False),
        dtype=np.float32,
    )
    y_test_np = np.asarray(
        y_test.to_numpy(copy=False),
        dtype=np.int32,
    )

    X_test_np = np.asarray(
        np.clip(X_test_np, feature_min, feature_max),
        dtype=np.float32,
    )
    dtest = _make_dmatrix_version_safe(
        X_test_np,
        y_test_np,
        feature_min,
        feature_max,
    )
    # Store per-RSU probability vectors instead of manual sums, so we can
    # apply robust aggregation (clipped / trimmed mean) over RSUs.
    val_probas_per_rsu: List[np.ndarray] = []
    test_probas_per_rsu: List[np.ndarray] = []
    used_rsus: List[int] = []
    # We already have centralized_results from the previous block
    per_rsu_test_metrics: Dict[int, Dict[str, float]] = centralized_results.copy()
    # AnchorZKP gating for ensemble:
    # - If GLOBAL proof ok: restrict strictly to the RSUs used in that proof
    # - Else: restrict to RSUs that passed RSU-level proof (best-effort)
    if global_anchor_ok:
        allowed_rsus = {int(x) for x in rsu_used_ids}
    else:
        allowed_rsus = {int(k) for k, v in rsu_ok_map.items() if bool(v)}
    if not allowed_rsus:
        logging.warning("[GLOBAL] No RSUs passed AnchorZKP gating; skipping ensemble.")
    else:
        for rsu_id in range(1, NUM_RSUS + 1):
            if rsu_id not in allowed_rsus:
                logging.warning(
                    "[GLOBAL] Skipping RSU %d in ensemble due to AnchorZKP gating (round=%d)",
                    rsu_id,
                    TARGET_ROUND,
                )
                continue
            model_path = os.path.join(OUTPUT_DIR, f"iov_global_model_rsu_{rsu_id}.json")
            if not os.path.exists(model_path):
                logging.warning(
                    f"[GLOBAL] Skipping RSU {rsu_id}: model file not found at {model_path}"
                )
                continue
            booster = xgb.Booster()
            booster.load_model(model_path)
            # Predictions on VAL/TEST for ensemble aggregation
            y_val_proba_rsu = np.clip(booster.predict(dval), 0.0, 1.0).astype(np.float64)
            y_test_proba_rsu = np.clip(booster.predict(dtest), 0.0, 1.0).astype(np.float64)
            val_probas_per_rsu.append(y_val_proba_rsu)
            test_probas_per_rsu.append(y_test_proba_rsu)
            used_rsus.append(rsu_id)
    if not used_rsus or not val_probas_per_rsu or not test_probas_per_rsu:
        logging.warning(
            "[GLOBAL] No RSU models available for ensemble aggregation; skipping."
        )
    else:
        # Robust aggregation across RSUs:
        # coordinate-wise clipped mean on probability vectors.
        _t_global_agg0 = time.perf_counter()
        mean_val_proba = robust_aggregate_probas(
            val_probas_per_rsu,
            clip_min=0.0,
            clip_max=1.0,
        )
        mean_test_proba = robust_aggregate_probas(
            test_probas_per_rsu,
            clip_min=0.0,
            clip_max=1.0,
        )
        global_aggregation_latency_sec = time.perf_counter() - _t_global_agg0
        # ---------- NEW: audit / sanity logging for robust aggregation ----------
        try:
            val_min = float(np.min(mean_val_proba))
            val_max = float(np.max(mean_val_proba))
            test_min = float(np.min(mean_test_proba))
            test_max = float(np.max(mean_test_proba))
        except Exception:
            val_min = val_max = test_min = test_max = float("nan")
        logging.info(
            "[GLOBAL] Robust aggregation completed over %d RSUs "
            "(val shape=%s, test shape=%s, "
            "val range=[%.4f, %.4f], test range=[%.4f, %.4f])",
            len(used_rsus),
            mean_val_proba.shape,
            mean_test_proba.shape,
            val_min,
            val_max,
            test_min,
            test_max,
        )
        logging.info(
            "[GLOBAL] RSUs used in robust ensemble: %s",
            ", ".join(str(rid) for rid in used_rsus),
        )
        # -----------------------------------------------------------------------
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
            key=lambda rid: per_rsu_test_metrics.get(rid, {}).get("auc", 0.0),
        )
        best_rsu_metrics = per_rsu_test_metrics.get(best_rsu_id, {})
        if not best_rsu_metrics:
            logging.warning(
                "[GLOBAL] Best RSU %d had no centralized metrics (missing eval?)",
                best_rsu_id,
            )
        else:
            logging.info(
                "[GLOBAL] Best single RSU on TEST is RSU %d with AUC=%.6f, F1=%.6f",
                best_rsu_id,
                float(best_rsu_metrics.get("auc", 0.0)),
                float(best_rsu_metrics.get("f1", 0.0)),
            )
        # ---------------------------
        # Persist global server ensemble summary as JSON
        # ---------------------------
        global_summary = {
            "num_rsus_configured": NUM_RSUS,
            "num_rsus_used": len(used_rsus),
            "used_rsu_ids": used_rsus,
            # TEST-set ensemble metrics (Acc/Prec/Rec/F1/AUC/LogLoss/Brier/ECE + threshold)
            "ensemble_metrics": ensemble_metrics,
            "ensemble_confusion_matrix": cm_ensemble.tolist(),
            # VAL metrics used to choose threshold for the ensemble (including LogLoss/Brier/ECE)
            "ensemble_val_metrics": ensemble_val_metrics,
            "best_single_rsu_id": best_rsu_id,
            "best_single_rsu_metrics": best_rsu_metrics,
            "anchor_ctx": {
                "anchor_version": str(anchor_ctx.get("anchor_version", "")),
                "anchor_id_field": str(anchor_ctx.get("anchor_id_field", "")),
                "M": int(_as_int(anchor_ctx.get("M"), 0)),
                "SCALE": int(_as_int(anchor_ctx.get("SCALE"), 0)),
                # ✅ GLOBAL round-binding root (what the GLOBAL ZKP binds to)
                "root_poseidon_field": int(global_round_root_poseidon_field),
                # ✅ anchor-set identity root
                "anchor_root_poseidon_field": int(_as_int(anchor_ctx.get("anchor_root_poseidon_field"), 0)),
                # NEW: persist file pointers for audit reproducibility
                "anchor_X_path": str(anchor_ctx.get("anchor_X_path", "")),
                "anchor_meta_path": str(anchor_ctx.get("anchor_meta_path", "")),
                "pins_hash_field": int(pins_hash_field) if "pins_hash_field" in locals() else 0,
                "policy_id_field": int(policy_id_field) if "policy_id_field" in locals() else 0,
                "public_input_order_id_field": int(
                    public_input_order_id_field) if "public_input_order_id_field" in locals() else 0,
            },
            "zkp_anchor_target_round": int(TARGET_ROUND),
            "zkp_rsu_ok_map": {str(k): bool(v) for k, v in rsu_ok_map.items()},
            "zkp_used_rsu_ids": [int(x) for x in (rsu_used_ids if "rsu_used_ids" in locals() else [])],
            "zkp_global_anchor_ok": bool(global_anchor_ok),
            "zkp_global_anchor_artifact": str(global_artifact_path),
            "overhead": {
                "global_aggregation_latency_sec": float(global_aggregation_latency_sec),
                "global_zkp_latency_sec": float(global_zkp_latency_sec),
                "global_artifact_export_latency_sec": float(global_artifact_export_latency_sec),
                "global_total_latency_sec": float(global_total_latency_sec),
                "global_summary_artifact_size_bytes": (
                    int(os.path.getsize(global_artifact_path))
                    if global_artifact_path and os.path.exists(global_artifact_path)
                    else 0
                ),
                "global_proof_artifact_size_bytes": (
                    int(os.path.getsize(global_proof_path))
                    if global_proof_path and os.path.exists(global_proof_path)
                    else 0
                ),
                "global_public_inputs_sidecar_size_bytes": (
                    int(os.path.getsize(global_sidecar_path))
                    if global_sidecar_path and os.path.exists(global_sidecar_path)
                    else 0
                ),
            },
            # Canonical GLOBAL audit outputs
            "zkp_global_commit_field": str(global_commit_field) if "global_commit_field" in locals() else "",
            "zkp_Q_global_b64": str(Q_global_b64) if "Q_global_b64" in locals() else "",
        }
        global_summary_path = os.path.join(OUTPUT_DIR, "global_ensemble_summary.json")
        try:
            with open(global_summary_path, "w", encoding="utf-8") as f:
                json.dump(global_summary, f, indent=2)
            logging.info(
                f"[GLOBAL] Saved global ensemble summary to {global_summary_path}"
            )
        except Exception:
            logging.exception(
                "[GLOBAL] Failed to write global ensemble summary JSON"
            )
    # -------------------------------------------------------------------
    # ✅ On-chain export package (latest pointer + immutable run snapshot)
    # -------------------------------------------------------------------
    export_index_write_latency_sec = 0.0
    _t_export_index0 = time.perf_counter()
    try:
        # Capture run_id in anchor_ctx (useful for auditors)
        anchor_ctx["run_id"] = str(RUN_ID)
        run_cfg_path_abs = os.path.join(OUTPUT_DIR, "iov_run_config.json")
        gate_path_abs = os.path.join(OUTPUT_DIR, "zkp_anchor_gating_manifest.json")
        global_ensemble_summary_path_abs = os.path.join(OUTPUT_DIR, "global_ensemble_summary.json")
        export_index = _build_onchain_export_index_v1(
            output_dir=str(OUTPUT_DIR),
            run_id=str(RUN_ID),
            rsu_ids_sorted=[int(x) for x in rsu_ids_sorted],
            num_rounds=int(NUM_ROUNDS),
            vehicles_per_rsu=int(VEHICLES_PER_RSU),
            anchor_ctx=dict(anchor_ctx),
            run_cfg_path=str(run_cfg_path_abs),
            gate_path=str(gate_path_abs),
            global_ensemble_summary_path=str(global_ensemble_summary_path_abs),
        )
        latest_path = os.path.join(ONCHAIN_EXPORT_DIR, "latest_run_index_v1.json")
        archived_path = os.path.join(ONCHAIN_EXPORT_DIR, f"run_{RUN_ID}_index_v1.json")
        _atomic_write_json_v1(archived_path, export_index)
        _atomic_write_json_v1(latest_path, export_index)
        logging.info(
            "[ONCHAIN] Export index written | run_id=%s | archived=%s | latest=%s",
            str(RUN_ID),
            archived_path,
            latest_path,
        )
    except Exception as exc:
        logging.warning("[ONCHAIN] Failed to write on-chain export index: %s", exc)
    finally:
        export_index_write_latency_sec = time.perf_counter() - _t_export_index0

    admission_report = write_admission_failure_ablation_report_v1(
        output_dir=OUTPUT_DIR,
        ablation_cfg=ABLATION_CONFIG,
        num_rsus=NUM_RSUS,
    )

    digest_report = write_digest_binding_failure_ablation_report_v1(
        output_dir=OUTPUT_DIR,
        ablation_cfg=ABLATION_CONFIG,
        num_rsus=NUM_RSUS,
    )

    tamper_report = run_artifact_tamper_check_v1(
        output_dir=OUTPUT_DIR,
        ablation_cfg=ABLATION_CONFIG,
    )

    commitment_report = run_commitment_recompute_check_v1(
        output_dir=OUTPUT_DIR,
        anchor_ctx=anchor_ctx,
        ablation_cfg=ABLATION_CONFIG,
    )

    public_input_report = run_public_input_agreement_check_v1(
        output_dir=OUTPUT_DIR,
        anchor_ctx=anchor_ctx,
        spec_rsu=anchorsum_spec_rsu,
        spec_global=anchorsum_spec_global,
        ablation_cfg=ABLATION_CONFIG,
    )

    layered_overhead_report = write_layered_overhead_breakdown_report_v1(
        output_dir=OUTPUT_DIR,
        ablation_cfg=ABLATION_CONFIG,
        num_rsus=NUM_RSUS,
        export_index_write_latency_sec=float(export_index_write_latency_sec),
    )

    write_ablation_summary_v1(
        output_dir=OUTPUT_DIR,
        admission_report=admission_report,
        digest_report=digest_report,
        tamper_report=tamper_report,
        commitment_report=commitment_report,
        public_input_report=public_input_report,
        layered_overhead_report=layered_overhead_report,
    )

    reviewer3_summary = _reviewer3_write_experiment_summary_v1(
        output_dir=OUTPUT_DIR,
        scenario=REVIEWER3_SCENARIO,
        seed=REVIEWER3_SEED,
        num_rsus=NUM_RSUS,
        vehicles_per_rsu=VEHICLES_PER_RSU,
        num_rounds=NUM_ROUNDS,
        stress_cfg=ABLATION_CONFIG,
    )

    _reviewer3_validate_completed_run_v1(
        output_dir=OUTPUT_DIR,
        scenario=REVIEWER3_SCENARIO,
        summary=reviewer3_summary,
        num_rsus=NUM_RSUS,
        vehicles_per_rsu=VEHICLES_PER_RSU,
        num_rounds=NUM_ROUNDS,
        commitment_report=commitment_report,
        public_input_report=public_input_report,
        layered_overhead_report=layered_overhead_report,
    )

    _reviewer3_write_cross_scenario_comparison_v1(
        str(_PROJECT_ROOT),
        seed=REVIEWER3_SEED,
    )

    logging.info(
        "[REVIEWER3] COMPLETE | scenario=%s | results=%s",
        REVIEWER3_SCENARIO,
        OUTPUT_DIR,
    )
