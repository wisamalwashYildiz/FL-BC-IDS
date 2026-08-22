#!/usr/bin/env python3
"""Reviewer 1 Comment 4: model-anchor linkage and aggregation replay validation.

This standalone validation does not require Flower or dp_xgboost. It uses
standard XGBoost to exercise the same JSON model structure and reproduces the
Flower 1.23.0 FedXgbBagging aggregation algorithm for comparison.

It is a controlled structural validation harness. Passing it validates the
linkage/replay mechanisms exercised here; it does not by itself establish that
those checks were integrated into every historical FL-BC-IDS training run.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from FL_IoV_VerifiableAggregation_UtilsV1 import (
    aggregate_all_delta_trees_v1,
    assert_compatible_model_contract_v1,
    build_aggregation_replay_record_v1,
    derive_anchor_from_delta_bytes_v1,
    flower_1_23_aggregate_exact,
    model_contract_sha256_v1,
    num_parallel_tree,
    replay_all_tree_aggregation_v1,
    sha256_hex,
    tree_count,
    verify_model_anchor_link_v1,
)


def canon_bytes(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def booster_bytes(booster: xgb.Booster) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        booster.save_model(path)
        return Path(path).read_bytes()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def make_delta(full: xgb.Booster, start_round: int) -> xgb.Booster:
    return full[start_round: full.num_boosted_rounds()]


def load_booster(raw: bytes) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(bytearray(raw))
    return booster


def train_client_delta(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    local_rounds: int,
    base_bytes: bytes | None = None,
) -> tuple[bytes, bytes, bytes]:
    """Train a client from a supplied common base and return base/full/delta."""
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "eta": 0.15,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "seed": int(seed),
        "num_parallel_tree": 1,
        "tree_method": "hist",
    }

    if base_bytes is None:
        # Controlled common initializer. The second client receives these exact
        # bytes so both clients in the simulated FL round start from one state.
        base = xgb.train(params, dtrain, num_boost_round=1)
    else:
        base = load_booster(base_bytes)

    start_round = base.num_boosted_rounds()
    full = xgb.train(
        params,
        dtrain,
        num_boost_round=int(local_rounds),
        xgb_model=base,
    )
    delta = make_delta(full, start_round)
    return booster_bytes(base), booster_bytes(full), booster_bytes(delta)


def mutate_one_byte(raw: bytes) -> bytes:
    b = bytearray(raw)
    idx = max(0, len(b) // 2)
    b[idx] = (b[idx] + 1) % 256
    return bytes(b)


def mutate_objective_contract(raw: bytes) -> bytes:
    obj = json.loads(bytearray(raw))
    obj["learner"]["objective"]["name"] = "reg:squarederror"
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        default=os.getenv(
            "FLBCIDS_COMMENT4_RESULTS_DIR",
            "artifacts/model_consistency",
        ),
    )
    ap.add_argument("--local-rounds", type=int, default=10)
    ap.add_argument("--anchor-size", type=int, default=64)
    ap.add_argument("--scale", type=int, default=1_000_000)
    args = ap.parse_args()

    if int(args.local_rounds) <= 0:
        raise ValueError("--local-rounds must be positive")
    if int(args.anchor_size) <= 0:
        raise ValueError("--anchor-size must be positive")
    if int(args.scale) <= 0:
        raise ValueError("--scale must be positive")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    X = rng.normal(size=(1200, 8)).astype(np.float32)
    logits = 1.4 * X[:, 0] - 0.9 * X[:, 1] + 0.55 * X[:, 2] - 0.2 * X[:, 3]
    y = (logits + rng.normal(scale=0.45, size=X.shape[0]) > 0).astype(np.int32)
    anchor_X = rng.normal(size=(int(args.anchor_size), X.shape[1])).astype(np.float32)
    probe_X = rng.normal(size=(256, X.shape[1])).astype(np.float32)
    probe_dmatrix = xgb.DMatrix(probe_X)

    base1, full1, delta1 = train_client_delta(
        X[:600],
        y[:600],
        seed=11,
        local_rounds=args.local_rounds,
    )
    _, _, delta2 = train_client_delta(
        X[600:],
        y[600:],
        seed=29,
        local_rounds=args.local_rounds,
        base_bytes=base1,
    )

    tests: list[dict[str, Any]] = []

    def add(
        name: str,
        passed: bool,
        observed: Any,
        expected: Any,
        severity: str,
        note: str,
    ) -> None:
        tests.append(
            {
                "test": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "note": note,
            }
        )

    delta1_sha = sha256_hex(delta1)
    delta2_sha = sha256_hex(delta2)

    # ------------------------------------------------------------------
    # 1) Positive signed-delta -> anchor linkage from exact delta bytes.
    # ------------------------------------------------------------------
    q1, q1_sha = derive_anchor_from_delta_bytes_v1(
        delta1,
        anchor_X,
        scale=args.scale,
        xgb_module=xgb,
    )
    link_ok = verify_model_anchor_link_v1(
        delta_bytes=delta1,
        expected_delta_sha256=delta1_sha,
        submitted_q=q1,
        submitted_q_sha256=q1_sha,
        anchor_X=anchor_X,
        scale=args.scale,
        xgb_module=xgb,
    )
    add(
        "direct_signed_delta_to_anchor_link_positive",
        link_ok.ok,
        link_ok.reason,
        "ok",
        "critical",
        "The exact received delta first matches the signed delta digest, then its anchor is independently recomputed.",
    )

    # ------------------------------------------------------------------
    # 2) Exact reviewer counterexample: valid signature over unrelated
    #    model-delta and anchor hashes. Signature verifies; derivation fails.
    # ------------------------------------------------------------------
    q2, q2_sha = derive_anchor_from_delta_bytes_v1(
        delta2,
        anchor_X,
        scale=args.scale,
        xgb_module=xgb,
    )
    report = {
        "schema": "VehicleSSIReportComment4V1",
        "rsu_id": 1,
        "vehicle_id": 1,
        "round": 1,
        "model_delta_sha256": delta1_sha,
        "q_anchor_sha256": q2_sha,  # deliberately derived from delta2
    }
    report_bytes = canon_bytes(report)
    sk = Ed25519PrivateKey.generate()
    sig = sk.sign(report_bytes)

    signature_verified = True
    try:
        sk.public_key().verify(sig, report_bytes)
    except InvalidSignature:
        signature_verified = False

    unrelated = verify_model_anchor_link_v1(
        delta_bytes=delta1,
        expected_delta_sha256=report["model_delta_sha256"],
        submitted_q=q2,
        submitted_q_sha256=report["q_anchor_sha256"],
        anchor_X=anchor_X,
        scale=args.scale,
        xgb_module=xgb,
    )
    add(
        "reviewer_unrelated_artifacts_signature_attack",
        (
            signature_verified
            and (not unrelated.ok)
            and unrelated.reason == "model_anchor_derivation_mismatch"
        ),
        {
            "signature_verified": signature_verified,
            "derivation_result": unrelated.reason,
        },
        {
            "signature_verified": True,
            "derivation_result": "model_anchor_derivation_mismatch",
        },
        "critical",
        "Directly demonstrates that a valid signature cannot make an unrelated anchor pass the new derivation check.",
    )

    # ------------------------------------------------------------------
    # 3) Signed delta substitution is rejected before deserialization.
    # ------------------------------------------------------------------
    delta1_tampered = mutate_one_byte(delta1)
    substituted = verify_model_anchor_link_v1(
        delta_bytes=delta1_tampered,
        expected_delta_sha256=delta1_sha,
        submitted_q=q1,
        submitted_q_sha256=q1_sha,
        anchor_X=anchor_X,
        scale=args.scale,
        xgb_module=xgb,
    )
    add(
        "signed_delta_substitution_rejected",
        (not substituted.ok) and substituted.reason == "model_delta_sha256_mismatch",
        substituted.reason,
        "model_delta_sha256_mismatch",
        "critical",
        "The exact received model bytes must match the model-delta digest carried by the signed report.",
    )

    # ------------------------------------------------------------------
    # 4) Submitted anchor-vector substitution is rejected by its signed hash.
    # ------------------------------------------------------------------
    q1_modified = q1.copy()
    q1_modified[0] += 1
    q_substitution = verify_model_anchor_link_v1(
        delta_bytes=delta1,
        expected_delta_sha256=delta1_sha,
        submitted_q=q1_modified,
        submitted_q_sha256=q1_sha,
        anchor_X=anchor_X,
        scale=args.scale,
        xgb_module=xgb,
    )
    add(
        "signed_anchor_vector_substitution_rejected",
        (
            (not q_substitution.ok)
            and q_substitution.reason
            == "submitted_q_sha256_does_not_match_submitted_vector"
        ),
        q_substitution.reason,
        "submitted_q_sha256_does_not_match_submitted_vector",
        "critical",
        "The retained q vector must hash to the q digest carried by the signed report.",
    )

    # ------------------------------------------------------------------
    # 5) Base + multi-round delta reconstructs the original post-update model.
    # ------------------------------------------------------------------
    reconstructed_full1 = aggregate_all_delta_trees_v1(base1, delta1)
    pred_original = load_booster(full1).predict(probe_dmatrix)
    pred_recon = load_booster(reconstructed_full1).predict(probe_dmatrix)
    max_abs = float(np.max(np.abs(pred_original - pred_recon)))
    add(
        "all_tree_base_plus_delta_reconstruction",
        max_abs <= 1e-7 and tree_count(reconstructed_full1) == tree_count(full1),
        {
            "max_abs_prediction_difference": max_abs,
            "trees": tree_count(reconstructed_full1),
        },
        {
            "max_abs_prediction_difference": "<=1e-7",
            "trees": tree_count(full1),
        },
        "critical",
        "Confirms the all-tree JSON operation reproduces a native XGBoost continuation from base + sliced delta.",
    )

    # ------------------------------------------------------------------
    # 6) Both client deltas use a compatible XGBoost model contract.
    # ------------------------------------------------------------------
    contract_ok = True
    contract_error = ""
    try:
        assert_compatible_model_contract_v1(delta1, delta2)
    except Exception as exc:
        contract_ok = False
        contract_error = str(exc)
    add(
        "client_delta_model_contract_compatibility",
        contract_ok,
        {
            "delta1_contract": model_contract_sha256_v1(delta1),
            "delta2_contract": model_contract_sha256_v1(delta2),
            "error": contract_error,
        },
        "matching model-contract hashes",
        "critical",
        "Aggregation is allowed only for structurally compatible XGBoost artifacts.",
    )

    # ------------------------------------------------------------------
    # 7) Exact Flower 1.23.0 diagnostic.
    # ------------------------------------------------------------------
    flower_global = flower_1_23_aggregate_exact(None, delta1)
    flower_global = flower_1_23_aggregate_exact(flower_global, delta2)
    expected_all = tree_count(delta1) + tree_count(delta2)
    expected_flower = tree_count(delta1) + num_parallel_tree(delta2)
    observed_flower = tree_count(flower_global)
    multi_tree_delta = tree_count(delta2) > num_parallel_tree(delta2)
    mismatch_present = observed_flower != expected_all

    add(
        "flower_1_23_exact_behavior_reproduced",
        observed_flower == expected_flower,
        observed_flower,
        expected_flower,
        "critical",
        "Reproduces the exact Flower 1.23.0 later-update tree-count behavior used for the contract diagnostic.",
    )
    add(
        "flower_multitree_contract_diagnostic",
        mismatch_present == multi_tree_delta,
        {
            "multi_tree_delta": multi_tree_delta,
            "flower_tree_count": observed_flower,
            "all_tree_intent": expected_all,
            "mismatch_present": mismatch_present,
        },
        {"mismatch_present": multi_tree_delta},
        "critical",
        "For multi-iteration deltas the Flower 1.23.0 one-iteration update contract diverges from the intended all-tree contract.",
    )

    # ------------------------------------------------------------------
    # 8) Corrected deterministic all-tree aggregation includes every tree.
    # ------------------------------------------------------------------
    corrected = replay_all_tree_aggregation_v1([delta1, delta2])
    corrected_trees = tree_count(corrected)
    add(
        "corrected_all_tree_aggregation_tree_count",
        corrected_trees == expected_all,
        corrected_trees,
        expected_all,
        "critical",
        "The corrected replay consumes every tree in every admitted multi-round delta.",
    )

    # ------------------------------------------------------------------
    # 9) Corrected aggregate is a loadable predictive XGBoost model.
    # ------------------------------------------------------------------
    corrected_pred = np.asarray(
        load_booster(corrected).predict(probe_dmatrix),
        dtype=np.float64,
    )
    add(
        "corrected_aggregate_loadable_and_predictive",
        (
            corrected_pred.shape[0] == probe_X.shape[0]
            and bool(np.all(np.isfinite(corrected_pred)))
        ),
        {
            "predictions": int(corrected_pred.shape[0]),
            "all_finite": bool(np.all(np.isfinite(corrected_pred))),
        },
        {
            "predictions": int(probe_X.shape[0]),
            "all_finite": True,
        },
        "critical",
        "Prevents a tree-count-only success from hiding an unusable aggregate artifact.",
    )

    # ------------------------------------------------------------------
    # 10) Replay is byte-stable for identical ordered inputs.
    # ------------------------------------------------------------------
    corrected2 = replay_all_tree_aggregation_v1([delta1, delta2])
    add(
        "aggregation_replay_byte_stability",
        corrected == corrected2,
        sha256_hex(corrected2),
        sha256_hex(corrected),
        "critical",
        "Identical ordered retained artifacts reproduce the exact aggregate bytes and digest.",
    )

    # ------------------------------------------------------------------
    # 11) Reordering admitted deltas changes the artifact digest.
    # ------------------------------------------------------------------
    reordered = replay_all_tree_aggregation_v1([delta2, delta1])
    add(
        "aggregation_order_mismatch_detection",
        sha256_hex(reordered) != sha256_hex(corrected),
        sha256_hex(reordered),
        f"!= {sha256_hex(corrected)}",
        "high",
        "The replay transcript binds the deterministic admitted-client ordering.",
    )

    # ------------------------------------------------------------------
    # 12) Aggregate artifact substitution remains digest-detectable.
    # ------------------------------------------------------------------
    corrected_tampered = mutate_one_byte(corrected)
    add(
        "aggregate_artifact_tamper_detection",
        sha256_hex(corrected_tampered) != sha256_hex(corrected),
        sha256_hex(corrected_tampered),
        f"!= {sha256_hex(corrected)}",
        "high",
        "The persisted aggregate digest detects post-replay output substitution.",
    )

    # ------------------------------------------------------------------
    # 13) Incompatible XGBoost contracts are rejected before aggregation.
    # ------------------------------------------------------------------
    incompatible_delta = mutate_objective_contract(delta2)
    incompatible_rejected = False
    incompatible_reason = ""
    try:
        assert_compatible_model_contract_v1(delta1, incompatible_delta)
    except ValueError as exc:
        incompatible_rejected = True
        incompatible_reason = str(exc)
    add(
        "incompatible_model_contract_rejected",
        incompatible_rejected and "model-contract mismatch" in incompatible_reason,
        incompatible_reason,
        "XGBoost model-contract mismatch",
        "critical",
        "Fails closed instead of concatenating trees from semantically incompatible XGBoost model contracts.",
    )

    # ------------------------------------------------------------------
    # 14) Second federated round: previous aggregate + new ordered deltas.
    # ------------------------------------------------------------------
    _, _, delta1_r2 = train_client_delta(
        X[:600],
        y[:600],
        seed=111,
        local_rounds=args.local_rounds,
        base_bytes=corrected,
    )
    _, _, delta2_r2 = train_client_delta(
        X[600:],
        y[600:],
        seed=129,
        local_rounds=args.local_rounds,
        base_bytes=corrected,
    )
    corrected_round2 = replay_all_tree_aggregation_v1(
        [delta1_r2, delta2_r2],
        previous_model_bytes=corrected,
    )
    expected_round2_trees = (
        corrected_trees + tree_count(delta1_r2) + tree_count(delta2_r2)
    )
    round2_pred = np.asarray(
        load_booster(corrected_round2).predict(probe_dmatrix),
        dtype=np.float64,
    )
    add(
        "second_round_aggregation_replay",
        (
            tree_count(corrected_round2) == expected_round2_trees
            and round2_pred.shape[0] == probe_X.shape[0]
            and bool(np.all(np.isfinite(round2_pred)))
        ),
        {
            "trees": tree_count(corrected_round2),
            "predictions": int(round2_pred.shape[0]),
            "all_finite": bool(np.all(np.isfinite(round2_pred))),
        },
        {
            "trees": expected_round2_trees,
            "predictions": int(probe_X.shape[0]),
            "all_finite": True,
        },
        "critical",
        "Exercises the actual multi-round transition: previous RSU aggregate + ordered new deltas -> next RSU aggregate.",
    )

    # ------------------------------------------------------------------
    # 15) Canonical replay transcript binds previous state, ordered deltas,
    #     result digest, tree counts, and model contract.
    # ------------------------------------------------------------------
    replay_record, replay_record_sha = build_aggregation_replay_record_v1(
        previous_model_bytes=None,
        ordered_delta_bytes=[delta1, delta2],
        aggregate_model_bytes=corrected,
    )
    expected_record_ok = (
        replay_record["ordered_delta_sha256"] == [delta1_sha, delta2_sha]
        and replay_record["aggregate_model_sha256"] == sha256_hex(corrected)
        and len(replay_record_sha) == 64
    )
    add(
        "aggregation_replay_transcript_binding",
        expected_record_ok,
        {
            "record_sha256": replay_record_sha,
            "ordered_delta_sha256": replay_record["ordered_delta_sha256"],
            "aggregate_model_sha256": replay_record["aggregate_model_sha256"],
        },
        {
            "ordered_delta_sha256": [delta1_sha, delta2_sha],
            "aggregate_model_sha256": sha256_hex(corrected),
            "record_sha256_length": 64,
        },
        "critical",
        "Produces the exact canonical transition record that should later be bound into the production RSU manifest/evidence path.",
    )

    summary = {
        "schema": "ReviewerConcern4ValidationV2",
        "validation_scope": "controlled_structural_model_anchor_and_aggregation_replay",
        "controlled_data_generated_in_harness": True,
        "production_path_integration_claimed_by_this_harness": False,
        "local_rounds_per_client_delta": int(args.local_rounds),
        "num_parallel_tree": 1,
        "anchor_size": int(args.anchor_size),
        "anchor_scale": int(args.scale),
        "delta1_tree_count": tree_count(delta1),
        "delta2_tree_count": tree_count(delta2),
        "flower_1_23_tree_count_after_two_deltas": observed_flower,
        "intended_all_tree_count_after_two_deltas": expected_all,
        "corrected_all_tree_count_after_two_deltas": corrected_trees,
        "corrected_second_round_tree_count": tree_count(corrected_round2),
        "aggregation_replay_record_sha256": replay_record_sha,
        "tests_total": len(tests),
        "tests_passed": sum(1 for item in tests if item["passed"]),
        "tests_failed": sum(1 for item in tests if not item["passed"]),
        "tests": tests,
    }

    (out / "comment4_validation_results.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    with (out / "comment4_test_matrix.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test", "passed", "severity", "observed", "expected", "note"],
        )
        writer.writeheader()
        for test in tests:
            row = dict(test)
            row["observed"] = json.dumps(row["observed"], sort_keys=True)
            row["expected"] = json.dumps(row["expected"], sort_keys=True)
            writer.writerow(row)

    (out / "comment4_aggregation_replay_record.json").write_text(
        json.dumps(replay_record, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Reviewer 1 Comment 4 — Controlled Structural Validation",
        "",
        f"- Tests passed: **{summary['tests_passed']}/{summary['tests_total']}**",
        f"- Trees per client delta: **{summary['delta1_tree_count']}** and **{summary['delta2_tree_count']}**",
        f"- Flower 1.23.0 aggregation after both deltas: **{observed_flower} trees**",
        f"- Intended/corrected all-tree replay after both deltas: **{corrected_trees} trees**",
        f"- Corrected replay after a second FL round: **{summary['corrected_second_round_tree_count']} trees**",
        "",
        "## Test matrix",
        "",
        "| Test | Status | Main observation |",
        "|---|---:|---|",
    ]
    for test in tests:
        status = "PASS" if test["passed"] else "FAIL"
        lines.append(f"| `{test['test']}` | **{status}** | {test['note']} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The valid-signature/unrelated-anchor counterexample is explicitly exercised. "
            "The new verifier first requires the received delta bytes to match the signed "
            "model-delta digest, then requires the submitted q vector to match its signed "
            "digest, and finally recomputes q from the exact delta bytes on the fixed anchor set.",
            "",
            "Aggregation correctness is treated separately from Groth16. The corrected replay "
            "uses the exact previous RSU model and exact deterministically ordered admitted "
            "delta artifacts to reconstruct the next XGBoost aggregate and its digest. "
            "Groth16 should remain described as a scoped proof of anchor arithmetic, not as "
            "a proof of the complete XGBoost program.",
            "",
            "Scope note: this controlled harness validates the signed-delta→anchor linkage "
            "and deterministic all-tree aggregation-replay mechanisms on XGBoost artifacts "
            "generated within the harness. It does not, by itself, assert production-path "
            "integration for historical training runs; any such integration claim must be "
            "supported by the separately retained run/evidence artifacts.",
        ]
    )
    (out / "comment4_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    artifact_meta = {
        "delta1_sha256": delta1_sha,
        "delta2_sha256": delta2_sha,
        "q1_sha256": q1_sha,
        "unrelated_q2_sha256": q2_sha,
        "signed_unrelated_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "signed_unrelated_report_signature_b64": base64.b64encode(sig).decode("ascii"),
        "flower_aggregate_sha256": sha256_hex(flower_global),
        "corrected_aggregate_sha256": sha256_hex(corrected),
        "corrected_second_round_aggregate_sha256": sha256_hex(corrected_round2),
        "delta_model_contract_sha256": model_contract_sha256_v1(delta1),
        "aggregation_replay_record_sha256": replay_record_sha,
    }
    (out / "comment4_artifact_digests.json").write_text(
        json.dumps(artifact_meta, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "tests_total",
                    "tests_passed",
                    "tests_failed",
                    "delta1_tree_count",
                    "delta2_tree_count",
                    "flower_1_23_tree_count_after_two_deltas",
                    "corrected_all_tree_count_after_two_deltas",
                    "corrected_second_round_tree_count",
                ]
            },
            indent=2,
        )
    )
    return 0 if summary["tests_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
