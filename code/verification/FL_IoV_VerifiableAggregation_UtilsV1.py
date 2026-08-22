#!/usr/bin/env python3
"""
FL_IoV_VerifiableAggregation_UtilsV1.py

Reviewer-1 Comment-4 support utilities.

Purpose
-------
1. Bind an anchor deterministically to the exact serialized XGBoost delta bytes
   whose SHA-256 digest is carried by the signed vehicle report.
2. Provide deterministic replay of the intended all-tree RSU aggregation.
3. Reproduce Flower 1.23.0 FedXgbBagging aggregation semantics for a controlled
   contract diagnostic.
4. Fail closed on malformed or model-contract-incompatible XGBoost artifacts.

These utilities establish artifact/model consistency. They do NOT prove honest
local training, raw-data provenance, benign updates, or correct runtime DP.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


ALL_TREE_AGGREGATION_SPEC_V1 = "FLBCIDS-AllTreeAggregationReplay-V1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def _normalize_sha256_hex(value: str, *, field_name: str) -> str:
    s = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(s):
        raise ValueError(f"{field_name} must be exactly 64 lowercase/uppercase hexadecimal characters")
    return s


def _loads_model(model_bytes: bytes) -> dict[str, Any]:
    if not isinstance(model_bytes, (bytes, bytearray)) or not model_bytes:
        raise ValueError("model bytes must be non-empty")
    try:
        obj = json.loads(bytearray(model_bytes))
    except Exception as exc:
        raise ValueError("model bytes are not valid XGBoost JSON") from exc
    if not isinstance(obj, dict):
        raise ValueError("XGBoost JSON root must be an object")
    return obj


def _dumps_model(obj: dict[str, Any]) -> bytes:
    # Frozen serialization for this replay protocol. We preserve insertion order
    # from XGBoost JSON but remove insignificant whitespace.
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _model_tree_block(obj: dict[str, Any]) -> dict[str, Any]:
    try:
        block = obj["learner"]["gradient_booster"]["model"]
    except Exception as exc:
        raise ValueError("unsupported XGBoost JSON layout: missing tree model block") from exc
    if not isinstance(block, dict):
        raise ValueError("invalid XGBoost tree model block")
    return block


def _declared_tree_nums(model_bytes: bytes) -> tuple[int, int]:
    """Return (num_trees, num_parallel_tree) from the XGBoost JSON fields."""
    obj = _loads_model(model_bytes)
    block = _model_tree_block(obj)
    params = block.get("gbtree_model_param")
    if not isinstance(params, dict):
        raise ValueError("missing gbtree_model_param")
    try:
        n = int(params["num_trees"])
        p = int(params["num_parallel_tree"])
    except Exception as exc:
        raise ValueError("invalid num_trees/num_parallel_tree") from exc
    if n < 0:
        raise ValueError("num_trees must be non-negative")
    if p <= 0:
        raise ValueError("num_parallel_tree must be positive")
    return n, p


def _iteration_indptr(block: dict[str, Any], *, num_parallel: int) -> list[int]:
    trees = block.get("trees")
    if not isinstance(trees, list):
        raise ValueError("XGBoost trees field is not a list")

    raw = block.get("iteration_indptr")
    if raw is None:
        if len(trees) % int(num_parallel) != 0:
            raise ValueError(
                f"cannot infer iteration_indptr: trees={len(trees)} "
                f"num_parallel_tree={num_parallel}"
            )
        return list(range(0, len(trees) + 1, int(num_parallel)))

    if not isinstance(raw, list) or len(raw) < 1:
        raise ValueError("iteration_indptr must be a non-empty list")

    try:
        out = [int(v) for v in raw]
    except Exception as exc:
        raise ValueError("iteration_indptr contains non-integer values") from exc

    if out[0] != 0 or out[-1] != len(trees):
        raise ValueError(
            f"invalid iteration_indptr endpoints: first={out[0]} "
            f"last={out[-1]} trees={len(trees)}"
        )

    if any(b <= a for a, b in zip(out, out[1:])):
        if len(trees) == 0 and out == [0]:
            return out
        raise ValueError("iteration_indptr must be strictly increasing")

    # The evaluated FL-BC-IDS binary configuration uses one output group, so
    # every boosting-iteration group must contain num_parallel_tree trees.
    widths = [b - a for a, b in zip(out, out[1:])]
    if any(w != int(num_parallel) for w in widths):
        raise ValueError(
            f"iteration_indptr group width mismatch: widths={widths[:8]} "
            f"num_parallel_tree={num_parallel}"
        )
    return out


def validate_xgboost_model_structure_v1(model_bytes: bytes) -> None:
    """Fail closed if the retained JSON is structurally inconsistent."""
    obj = _loads_model(model_bytes)
    block = _model_tree_block(obj)
    declared_n, p = _declared_tree_nums(model_bytes)

    trees = block.get("trees")
    tree_info = block.get("tree_info")
    if not isinstance(trees, list):
        raise ValueError("trees must be a list")
    if declared_n != len(trees):
        raise ValueError(
            f"declared num_trees mismatch: declared={declared_n} actual={len(trees)}"
        )
    if not isinstance(tree_info, list) or len(tree_info) != len(trees):
        raise ValueError(
            f"tree_info length mismatch: tree_info={len(tree_info) if isinstance(tree_info, list) else 'invalid'} "
            f"trees={len(trees)}"
        )

    expected_ids = list(range(len(trees)))
    actual_ids: list[int] = []
    for tree in trees:
        if not isinstance(tree, dict):
            raise ValueError("tree entry is not an object")
        try:
            actual_ids.append(int(tree["id"]))
        except Exception as exc:
            raise ValueError("tree is missing a valid integer id") from exc
    if actual_ids != expected_ids:
        raise ValueError(
            f"tree ids are not canonical sequential ids: "
            f"first_actual={actual_ids[:8]} expected={expected_ids[:8]}"
        )

    _iteration_indptr(block, num_parallel=p)

    learner = obj.get("learner")
    if not isinstance(learner, dict):
        raise ValueError("missing learner object")
    learner_params = learner.get("learner_model_param")
    if not isinstance(learner_params, dict):
        raise ValueError("missing learner_model_param")
    try:
        num_feature = int(learner_params["num_feature"])
        num_target = int(learner_params.get("num_target", 1))
        num_class = int(learner_params.get("num_class", 0))
    except Exception as exc:
        raise ValueError("invalid learner model dimensions") from exc
    if num_feature <= 0:
        raise ValueError("num_feature must be positive")
    if num_target != 1:
        raise ValueError(f"unsupported num_target={num_target}; expected 1")
    if num_class != 0:
        raise ValueError(
            f"this replay utility is frozen for the binary FL-BC-IDS contract; "
            f"got num_class={num_class}"
        )


def tree_count(model_bytes: bytes) -> int:
    validate_xgboost_model_structure_v1(model_bytes)
    obj = _loads_model(model_bytes)
    return len(_model_tree_block(obj)["trees"])


def num_parallel_tree(model_bytes: bytes) -> int:
    validate_xgboost_model_structure_v1(model_bytes)
    _, p = _declared_tree_nums(model_bytes)
    return p


def _model_contract_obj_v1(model_bytes: bytes) -> dict[str, Any]:
    """Extract fields that must agree before tree concatenation is allowed."""
    validate_xgboost_model_structure_v1(model_bytes)
    obj = _loads_model(model_bytes)
    learner = obj["learner"]
    gb = learner["gradient_booster"]
    block = gb["model"]
    params = block["gbtree_model_param"]

    return {
        "schema": "XGBoostModelContractV1",
        "xgboost_json_version": obj.get("version"),
        "gradient_booster_name": gb.get("name"),
        "learner_model_param": learner.get("learner_model_param"),
        "objective": learner.get("objective"),
        "feature_names": learner.get("feature_names"),
        "feature_types": learner.get("feature_types"),
        "num_parallel_tree": int(params["num_parallel_tree"]),
    }


def model_contract_sha256_v1(model_bytes: bytes) -> str:
    payload = json.dumps(
        _model_contract_obj_v1(model_bytes),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_hex(payload)


def assert_compatible_model_contract_v1(
    previous_model_bytes: bytes,
    current_delta_bytes: bytes,
) -> None:
    prev = _model_contract_obj_v1(previous_model_bytes)
    curr = _model_contract_obj_v1(current_delta_bytes)
    if prev != curr:
        raise ValueError(
            "XGBoost model-contract mismatch: "
            f"previous={model_contract_sha256_v1(previous_model_bytes)} "
            f"delta={model_contract_sha256_v1(current_delta_bytes)}"
        )


def flower_1_23_aggregate_exact(
    previous_model_bytes: bytes | None,
    current_delta_bytes: bytes,
) -> bytes:
    """Reproduce Flower 1.23.0 ``FedXgbBagging.aggregate`` semantics exactly.

    Flower 1.23.0 appends ``num_parallel_tree`` trees from each later update.
    That matches Flower's expected one-boosting-iteration update contract, but
    truncates a multi-iteration sliced delta when ``num_parallel_tree == 1``.
    """
    if not previous_model_bytes:
        return bytes(current_delta_bytes)

    tree_num_prev, _ = _declared_tree_nums(previous_model_bytes)
    _, p_curr = _declared_tree_nums(current_delta_bytes)

    prev = _loads_model(previous_model_bytes)
    curr = _loads_model(current_delta_bytes)
    prev_block = _model_tree_block(prev)
    curr_block = _model_tree_block(curr)

    prev_block["gbtree_model_param"]["num_trees"] = str(tree_num_prev + p_curr)
    iteration_indptr = prev_block["iteration_indptr"]
    prev_block["iteration_indptr"].append(iteration_indptr[-1] + p_curr)

    curr_trees = curr_block["trees"]
    if len(curr_trees) < p_curr:
        raise ValueError(
            f"delta contains fewer trees ({len(curr_trees)}) than "
            f"num_parallel_tree ({p_curr})"
        )

    for tree_count_local in range(p_curr):
        curr_trees[tree_count_local]["id"] = tree_num_prev + tree_count_local
        prev_block["trees"].append(curr_trees[tree_count_local])
        # Flower 1.23.0 appends 0 here.
        prev_block["tree_info"].append(0)

    return bytes(json.dumps(prev), "utf-8")


def aggregate_all_delta_trees_v1(
    previous_model_bytes: bytes | None,
    current_delta_bytes: bytes,
) -> bytes:
    """Append every boosting-iteration group contained in ``current_delta_bytes``.

    This is the frozen FL-BC-IDS replay contract for multi-round sliced deltas.
    """
    if not current_delta_bytes:
        raise ValueError("current_delta_bytes is empty")

    validate_xgboost_model_structure_v1(current_delta_bytes)

    if not previous_model_bytes:
        return _dumps_model(_loads_model(current_delta_bytes))

    validate_xgboost_model_structure_v1(previous_model_bytes)
    assert_compatible_model_contract_v1(previous_model_bytes, current_delta_bytes)

    prev = _loads_model(previous_model_bytes)
    curr = _loads_model(current_delta_bytes)
    prev_block = _model_tree_block(prev)
    curr_block = _model_tree_block(curr)

    prev_trees = prev_block["trees"]
    curr_trees = curr_block["trees"]
    prev_tree_info = prev_block["tree_info"]
    curr_tree_info = curr_block["tree_info"]

    prev_n = len(prev_trees)
    curr_n = len(curr_trees)
    if curr_n == 0:
        return _dumps_model(prev)

    _, p_prev = _declared_tree_nums(previous_model_bytes)
    _, p_curr = _declared_tree_nums(current_delta_bytes)
    if p_prev != p_curr:
        raise ValueError(
            f"num_parallel_tree mismatch: previous={p_prev}, delta={p_curr}"
        )

    prev_iptr = _iteration_indptr(prev_block, num_parallel=p_prev)
    curr_iptr = _iteration_indptr(curr_block, num_parallel=p_curr)

    for local_idx, raw_tree in enumerate(curr_trees):
        tree = json.loads(json.dumps(raw_tree))
        tree["id"] = prev_n + local_idx
        prev_trees.append(tree)
        prev_tree_info.append(curr_tree_info[local_idx])

    prev_block["gbtree_model_param"]["num_trees"] = str(prev_n + curr_n)
    prev_block["iteration_indptr"] = list(prev_iptr) + [
        prev_n + int(v) for v in curr_iptr[1:]
    ]

    out = _dumps_model(prev)
    validate_xgboost_model_structure_v1(out)
    return out


def replay_all_tree_aggregation_v1(
    ordered_delta_bytes: Sequence[bytes],
    *,
    previous_model_bytes: bytes | None = None,
) -> bytes:
    if not ordered_delta_bytes:
        raise ValueError("no deltas supplied")

    out = previous_model_bytes
    for delta in ordered_delta_bytes:
        out = aggregate_all_delta_trees_v1(out, delta)
    if out is None:
        raise RuntimeError("aggregation replay produced no model")
    return out


def quantize_probabilities_v1(p: np.ndarray, scale: int) -> np.ndarray:
    if int(scale) <= 0:
        raise ValueError("scale must be positive")
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)
    # Matches FL_IoV_AnchorZKP_UtilsV10.quantize_proba_to_int.
    return np.rint(a * int(scale)).astype(np.dtype("<i8"))


def pack_int64_le_v1(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.dtype("<i8")).reshape(-1).tobytes(order="C")


def q_sha256_v1(vec: np.ndarray) -> str:
    # Matches encode_int_vector_b64: the SHA-256 is over raw little-endian int64
    # bytes, not over the compressed/base64 transport representation.
    return sha256_hex(pack_int64_le_v1(vec))


def derive_anchor_from_delta_bytes_v1(
    delta_bytes: bytes,
    anchor_X: np.ndarray,
    *,
    scale: int,
    xgb_module: Any,
) -> tuple[np.ndarray, str]:
    """Derive the contribution anchor from the exact retained delta bytes."""
    validate_xgboost_model_structure_v1(delta_bytes)

    anchor = np.asarray(anchor_X, dtype=np.float32)
    if anchor.ndim != 2 or anchor.shape[0] <= 0 or anchor.shape[1] <= 0:
        raise ValueError(f"anchor_X must be a non-empty 2D matrix; got shape={anchor.shape}")

    obj = _loads_model(delta_bytes)
    try:
        expected_features = int(obj["learner"]["learner_model_param"]["num_feature"])
    except Exception as exc:
        raise ValueError("cannot determine model num_feature") from exc
    if anchor.shape[1] != expected_features:
        raise ValueError(
            f"anchor feature-count mismatch: anchor={anchor.shape[1]} "
            f"model={expected_features}"
        )

    booster = xgb_module.Booster()
    booster.load_model(bytearray(delta_bytes))
    dmat = xgb_module.DMatrix(anchor)
    pred = np.asarray(booster.predict(dmat), dtype=np.float64)
    q = quantize_probabilities_v1(pred, int(scale))
    return q, q_sha256_v1(q)


@dataclass(frozen=True)
class ModelAnchorVerificationResult:
    ok: bool
    delta_sha256: str
    expected_delta_sha256: str
    delta_hash_matches: bool
    submitted_q_sha256: str
    recomputed_q_sha256: str
    same_vector: bool
    reason: str


def verify_model_anchor_link_v1(
    *,
    delta_bytes: bytes,
    expected_delta_sha256: str,
    submitted_q: np.ndarray,
    submitted_q_sha256: str,
    anchor_X: np.ndarray,
    scale: int,
    xgb_module: Any,
) -> ModelAnchorVerificationResult:
    """Verify the complete signed-delta -> submitted-anchor relation.

    The check order is deliberate:
      1) received delta bytes must match the signed model-delta digest;
      2) submitted q bytes must match the signed q digest;
      3) q must equal deterministic recomputation from the exact delta bytes.
    """
    expected_delta = _normalize_sha256_hex(
        expected_delta_sha256, field_name="expected_delta_sha256"
    )
    expected_q = _normalize_sha256_hex(
        submitted_q_sha256, field_name="submitted_q_sha256"
    )

    delta_sha = sha256_hex(delta_bytes)
    delta_hash_matches = delta_sha == expected_delta

    q_sub = np.asarray(submitted_q, dtype=np.int64).reshape(-1)
    submitted_q_actual = q_sha256_v1(q_sub)

    if not delta_hash_matches:
        return ModelAnchorVerificationResult(
            ok=False,
            delta_sha256=delta_sha,
            expected_delta_sha256=expected_delta,
            delta_hash_matches=False,
            submitted_q_sha256=expected_q,
            recomputed_q_sha256="",
            same_vector=False,
            reason="model_delta_sha256_mismatch",
        )

    if submitted_q_actual != expected_q:
        return ModelAnchorVerificationResult(
            ok=False,
            delta_sha256=delta_sha,
            expected_delta_sha256=expected_delta,
            delta_hash_matches=True,
            submitted_q_sha256=expected_q,
            recomputed_q_sha256="",
            same_vector=False,
            reason="submitted_q_sha256_does_not_match_submitted_vector",
        )

    try:
        q_re, q_re_sha = derive_anchor_from_delta_bytes_v1(
            delta_bytes,
            anchor_X,
            scale=scale,
            xgb_module=xgb_module,
        )
    except Exception:
        return ModelAnchorVerificationResult(
            ok=False,
            delta_sha256=delta_sha,
            expected_delta_sha256=expected_delta,
            delta_hash_matches=True,
            submitted_q_sha256=expected_q,
            recomputed_q_sha256="",
            same_vector=False,
            reason="model_anchor_derivation_error",
        )

    same = bool(np.array_equal(q_sub, q_re))
    if not same:
        reason = "model_anchor_derivation_mismatch"
    elif q_re_sha != expected_q:
        reason = "recomputed_anchor_hash_mismatch"
    else:
        reason = "ok"

    return ModelAnchorVerificationResult(
        ok=(reason == "ok"),
        delta_sha256=delta_sha,
        expected_delta_sha256=expected_delta,
        delta_hash_matches=True,
        submitted_q_sha256=expected_q,
        recomputed_q_sha256=q_re_sha,
        same_vector=same,
        reason=reason,
    )


def build_aggregation_replay_record_v1(
    *,
    previous_model_bytes: bytes | None,
    ordered_delta_bytes: Sequence[bytes],
    aggregate_model_bytes: bytes,
) -> tuple[dict[str, Any], str]:
    """Build a canonical replay transcript suitable for later manifest binding."""
    if not ordered_delta_bytes:
        raise ValueError("ordered_delta_bytes must not be empty")

    validate_xgboost_model_structure_v1(aggregate_model_bytes)
    record = {
        "schema": "RSUAggregationReplayRecordV1",
        "aggregation_spec": ALL_TREE_AGGREGATION_SPEC_V1,
        "previous_model_sha256": (
            sha256_hex(previous_model_bytes) if previous_model_bytes else ""
        ),
        "ordered_delta_sha256": [sha256_hex(x) for x in ordered_delta_bytes],
        "ordered_delta_tree_counts": [tree_count(x) for x in ordered_delta_bytes],
        "aggregate_model_sha256": sha256_hex(aggregate_model_bytes),
        "aggregate_tree_count": tree_count(aggregate_model_bytes),
        "model_contract_sha256": model_contract_sha256_v1(aggregate_model_bytes),
    }
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return record, sha256_hex(payload)
