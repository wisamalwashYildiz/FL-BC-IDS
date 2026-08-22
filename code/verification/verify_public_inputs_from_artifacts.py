#!/usr/bin/env python3
"""
verify_public_inputs_from_artifacts_runready.py

Run-and-go artifact-reader verifier for RSU and GLOBAL public-input vectors.

Designed for PyCharm use without command-line arguments: open the file and press Run.
You can optionally edit the CONFIG block below, but the script also tries to auto-detect
common run roots.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# User-editable config (safe defaults for the compact security-evidence run)
# ---------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    # Preferred run root. Leave as None to auto-detect, or set explicitly.
    "ROOT_DIR": os.getenv(
        "FLBCIDS_COMPACT_EVIDENCE_DIR",
        os.path.join(
            os.getenv("FLBCIDS_REPO_ROOT", "."),
            "experiments",
            "08_compact_security_evidence",
            "CSECICIDS2018",
            "run_outputs",
        ),
    ),
    "RSU_ID": 2,
    "RSU_ROUND": 2,
    "GLOBAL_ROUND": 2,
    # Output report path. Leave None to save under <ROOT_DIR>/ablation_reports/.
    "OUTPUT_JSON": None,
    # If True, a non-pass result exits with code 1.
    "FAIL_EXIT_CODE": True,
}

DEFAULT_RSU_ORDER = [
    "anchor_id",
    "round_idx",
    "rsu_id",
    "K_used",
    "root_poseidon_field",
    "pins_hash_field",
    "policy_id_field",
    "public_input_order_id_field",
    "r_chal",
    "agg_commit",
]

DEFAULT_GLOBAL_ORDER = [
    "anchor_id",
    "round_idx",
    "global_id",
    "K_used",
    "root_poseidon_field",
    "pins_hash_field",
    "policy_id_field",
    "public_input_order_id_field",
    "r_chal",
    "agg_commit",
]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _as_int(val: Any, default: int = 0) -> int:
    try:
        if val is None or isinstance(val, bool):
            return default
        if isinstance(val, int):
            return val
        s = str(val).strip()
        return int(s) if s else default
    except Exception:
        return default


def _as_str(val: Any) -> str:
    return "" if val is None else str(val)


def _norm_sep(p: str) -> str:
    return p.replace("\\", os.sep).replace("/", os.sep)


def _looks_like_run_root(path: Path) -> bool:
    return path.exists() and path.is_dir() and (path / "zkp_anchor_gating_manifest.json").exists()


def _candidate_roots() -> List[Path]:
    here = Path(__file__).resolve()
    configured = _as_str(CONFIG.get("ROOT_DIR", "")).strip()

    candidates = [
        Path(_norm_sep(configured)) if configured else None,
        here.parent,
        here.parent / "run_outputs",
    ]

    # Search upward for the publication compact-evidence directory without
    # assuming where this verifier script itself is installed.
    for cur in (here.parent, *here.parent.parents):
        candidates.append(
            cur
            / "experiments"
            / "08_compact_security_evidence"
            / "CSECICIDS2018"
            / "run_outputs"
        )

    # Legacy locations remain candidates only so retained historical artifacts
    # with old layout can still be independently checked.
    candidates.extend(
        [
            here.parent / "rsu_outputs_dp",
            here.parent.parent / "rsu_outputs_dp",
            Path(os.getenv("FLBCIDS_DP_OUTPUT_DIR", "artifacts/rsu_outputs_dp")),
        ]
    )
    out: List[Path] = []
    for c in candidates:
        if c is None:
            continue
        try:
            out.append(c.resolve())
        except Exception:
            out.append(c)
    # de-dup while preserving order
    seen = set()
    uniq: List[Path] = []
    for c in out:
        key = str(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _auto_detect_root() -> Path:
    for cand in _candidate_roots():
        if _looks_like_run_root(cand):
            return cand
    msg = ["Could not auto-detect the compact security-evidence run root. Tried:"]
    msg.extend(f"  - {p}" for p in _candidate_roots())
    raise FileNotFoundError("\n".join(msg))


def _resolve_existing_path(root: Path, raw_path: str) -> Optional[Path]:
    s = _as_str(raw_path).strip()
    if not s:
        return None

    candidates: List[Path] = []
    candidates.append(Path(_norm_sep(s)))

    marker = "rsu_outputs_dp"
    normalized = s.replace("\\", "/")
    lowered = normalized.lower()
    marker_pos = lowered.find(marker)
    if marker_pos >= 0:
        # Match the legacy marker case-insensitively but preserve the original
        # suffix case so the remap also works on case-sensitive filesystems.
        suffix = normalized[
            marker_pos + len(marker):
        ].lstrip("/")
        if suffix:
            candidates.append(root / Path(_norm_sep(suffix)))

    candidates.append(root / Path(_norm_sep(s)))

    if len(s) >= 2 and s[1] == ":":
        stripped = s[2:].lstrip("\\/")
        candidates.append(root / Path(_norm_sep(stripped)))

    for cand in candidates:
        try:
            if cand.exists():
                return cand.resolve()
        except Exception:
            continue
    return None


def _coerce_public_inputs(obj: Any) -> List[str]:
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        for key in ("public_inputs", "publicSignals", "public", "inputs"):
            val = obj.get(key)
            if isinstance(val, list):
                return [str(x) for x in val]
    return []


def _first_existing(candidates: Iterable[Optional[Path]]) -> Optional[Path]:
    for cand in candidates:
        if cand is not None and cand.exists():
            return cand
    return None


def _extract_order_from_artifact(proof_obj: Dict[str, Any], default_order: List[str]) -> List[str]:
    for container_key in ("zkp", "payload"):
        cont = proof_obj.get(container_key)
        if isinstance(cont, dict):
            for key in ("public_inputs_order", "public_input_order", "public_input_names"):
                val = cont.get(key)
                if isinstance(val, list) and val:
                    return [str(x) for x in val]
    for key in ("public_inputs_order", "public_input_order", "public_input_names"):
        val = proof_obj.get(key)
        if isinstance(val, list) and val:
            return [str(x) for x in val]
    return list(default_order)


def _extract_by_name_from_artifact(proof_obj: Dict[str, Any]) -> Dict[str, str]:
    for container_key in ("zkp", "payload"):
        cont = proof_obj.get(container_key)
        if isinstance(cont, dict):
            val = cont.get("public_inputs_by_name")
            if isinstance(val, dict):
                return {str(k): str(v) for k, v in val.items()}
    val = proof_obj.get("public_inputs_by_name")
    if isinstance(val, dict):
        return {str(k): str(v) for k, v in val.items()}
    return {}


def _extract_public_section(proof_obj: Dict[str, Any]) -> Dict[str, Any]:
    pub = proof_obj.get("public")
    return pub if isinstance(pub, dict) else {}


def _extract_artifact_public_inputs(proof_obj: Dict[str, Any]) -> List[str]:
    pub = _extract_public_section(proof_obj)
    vals = _coerce_public_inputs(pub)
    if vals:
        return vals
    for container_key in ("zkp", "payload"):
        cont = proof_obj.get(container_key)
        if isinstance(cont, dict):
            vals = _coerce_public_inputs(cont)
            if vals:
                return vals
    return _coerce_public_inputs(proof_obj)


def _sidecar_candidates_for_rsu(root: Path, rsu_id: int, round_idx: int, summary_obj: Dict[str, Any], proof_summary: Dict[str, Any]) -> List[Optional[Path]]:
    rsu_public_map = summary_obj.get("rsu_public_sidecar_by_round", {}) if isinstance(summary_obj, dict) else {}
    return [
        _resolve_existing_path(root, _as_str(proof_summary.get("public_inputs_sidecar_copy_path"))),
        _resolve_existing_path(root, _as_str(proof_summary.get("public_inputs_sidecar_path"))),
        _resolve_existing_path(root, _as_str(rsu_public_map.get(str(round_idx), ""))) if isinstance(rsu_public_map, dict) else None,
        root / "zkp_artifacts" / "anchorsum" / f"rsu_{rsu_id}" / f"round_{round_idx}_public.json",
        root / "zkp_artifacts" / "anchorsum" / f"rsu_{rsu_id}" / f"round_{round_idx}_public_inputs_v1.json",
        root / f"rsu_{rsu_id}" / "zkp_artifacts" / "anchorsum" / f"rsu_{rsu_id}" / f"round_{round_idx}_public.json",
        root / f"rsu_{rsu_id}" / "zkp_artifacts" / "anchorsum" / f"rsu_{rsu_id}" / f"round_{round_idx}_public_inputs_v1.json",
    ]


def _sidecar_candidates_for_global(root: Path, round_idx: int, gate_obj: Dict[str, Any], proof_summary: Dict[str, Any]) -> List[Optional[Path]]:
    return [
        _resolve_existing_path(root, _as_str(proof_summary.get("public_inputs_sidecar_path"))),
        _resolve_existing_path(root, _as_str(gate_obj.get("global_public_inputs_sidecar_path"))),
        root / "zkp_artifacts" / "anchorsum" / "global" / f"round_{round_idx}_public.json",
        root / "zkp_artifacts" / "anchorsum" / "global" / f"round_{round_idx}_public_inputs_v1.json",
    ]


def _default_rsu_manifest_path(root: Path, rsu_id: int, round_idx: int) -> Path:
    return root / f"rsu_{rsu_id}" / "round_manifests" / "rsu" / f"rsu_{rsu_id}" / f"round_{round_idx}.json"


def _default_rsu_proof_summary_path(root: Path, rsu_id: int, round_idx: int) -> Path:
    return root / f"rsu_{rsu_id}" / "zkp_anchor_summaries" / "anchorsum" / f"rsu_{rsu_id}" / f"round_{round_idx}.json"


def _default_global_proof_summary_path(root: Path, round_idx: int) -> Path:
    return root / "zkp_anchor_summaries" / "anchorsum" / "global" / f"round_{round_idx}.json"


def _default_global_manifest_path(root: Path, round_idx: int) -> Path:
    return root / "zkp_round_manifests" / "global" / f"round_{round_idx}.json"


def _default_global_proof_artifact_path(root: Path, round_idx: int) -> Path:
    return root / "zkp_artifacts" / "anchorsum" / "global" / f"round_{round_idx}.json"


def _default_rsu_proof_artifact_path(root: Path, rsu_id: int, round_idx: int) -> Path:
    return root / f"rsu_{rsu_id}" / "zkp_artifacts" / "anchorsum" / f"rsu_{rsu_id}" / f"round_{round_idx}.json"


def _make_check(name: str, left: Any, right: Any) -> Dict[str, Any]:
    return {"name": name, "left": left, "right": right, "ok": str(left) == str(right)}


def _make_vector(order: List[str], mapping: Dict[str, Any]) -> List[str]:
    return [str(mapping.get(key, "")) for key in order]


def _mismatches(order: List[str], expected: List[str], exported: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = max(len(order), len(expected), len(exported))
    for i in range(n):
        name = order[i] if i < len(order) else f"extra_{i}"
        exp = expected[i] if i < len(expected) else ""
        got = exported[i] if i < len(exported) else ""
        if str(exp) != str(got):
            out.append({"index": i, "name": name, "expected": str(exp), "exported": str(got)})
    return out


def verify_rsu_scope(root: Path, rsu_id: int, round_idx: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {"scope": "rsu", "rsu_id": int(rsu_id), "round": int(round_idx), "result": "fail"}

    iov_summary_path = root / f"iov_rsu_{rsu_id}_summary.json"
    proof_summary_path = _default_rsu_proof_summary_path(root, rsu_id, round_idx)
    manifest_path = _default_rsu_manifest_path(root, rsu_id, round_idx)

    iov_summary = _read_json(iov_summary_path) if iov_summary_path.exists() else {}
    proof_summary = _read_json(proof_summary_path) if proof_summary_path.exists() else {}

    manifest_path = _resolve_existing_path(root, _as_str(proof_summary.get("rsu_round_manifest_path"))) or manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"RSU manifest not found: {manifest_path}")
    manifest_obj = _read_json(manifest_path)

    proof_artifact_path = (
        _resolve_existing_path(root, _as_str(proof_summary.get("proof_artifact_path")))
        or _resolve_existing_path(root, _as_str(proof_summary.get("proof_artifact_copy_path")))
        or _default_rsu_proof_artifact_path(root, rsu_id, round_idx)
    )
    if not proof_artifact_path.exists():
        raise FileNotFoundError(f"RSU proof artifact not found: {proof_artifact_path}")
    proof_obj = _read_json(proof_artifact_path)

    sidecar_path = _first_existing(_sidecar_candidates_for_rsu(root, rsu_id, round_idx, iov_summary, proof_summary))
    if sidecar_path is None:
        raise FileNotFoundError(f"No RSU sidecar/public-input file found for rsu={rsu_id} round={round_idx}")
    sidecar_obj = _read_json(sidecar_path)

    order = _extract_order_from_artifact(proof_obj, DEFAULT_RSU_ORDER)
    by_name = _extract_by_name_from_artifact(proof_obj)
    public_sec = _extract_public_section(proof_obj)

    expected_map: Dict[str, Any] = {
        "anchor_id": (
            _as_str(by_name.get("anchor_id"))
            or _as_str(public_sec.get("anchor_id_field"))
            or _as_str(proof_summary.get("anchor_id_field"))
            or _as_str(iov_summary.get("anchor_ctx", {}).get("anchor_id_field"))
        ),
        "round_idx": int(round_idx),
        "rsu_id": int(rsu_id),
        "K_used": _as_int(by_name.get("K_used") or public_sec.get("N_used") or proof_summary.get("N_used"), 0),
        "root_poseidon_field": _as_str(by_name.get("root_poseidon_field") or manifest_obj.get("root_poseidon_field") or proof_summary.get("root_poseidon_field") or public_sec.get("root_poseidon_field")),
        "pins_hash_field": _as_str(by_name.get("pins_hash_field") or proof_summary.get("pins_hash_field") or public_sec.get("pins_hash_field")),
        "policy_id_field": _as_str(by_name.get("policy_id_field") or proof_summary.get("policy_id_field") or public_sec.get("policy_id_field")),
        "public_input_order_id_field": _as_str(by_name.get("public_input_order_id_field") or proof_summary.get("public_input_order_id_field") or public_sec.get("public_input_order_id_field")),
        "r_chal": _as_str(by_name.get("r_chal") or public_sec.get("r_chal")),
        "agg_commit": _as_str(by_name.get("agg_commit") or public_sec.get("agg_commit_field") or proof_summary.get("rsu_commit_field")),
    }

    exported_inputs = _coerce_public_inputs(sidecar_obj) or _extract_artifact_public_inputs(proof_obj)
    expected_inputs = _make_vector(order, expected_map)
    vector_mismatches = _mismatches(order, expected_inputs, exported_inputs)

    checks = [
        _make_check("manifest.root_poseidon_field == summary.root_poseidon_field", manifest_obj.get("root_poseidon_field"), proof_summary.get("root_poseidon_field")),
        _make_check("summary.rsu_commit_field == proof.public.agg_commit_field", proof_summary.get("rsu_commit_field"), public_sec.get("agg_commit_field")),
        _make_check("summary.anchor_id_field == proof.public.anchor_id_field", proof_summary.get("anchor_id_field"), public_sec.get("anchor_id_field")),
        _make_check("summary.pins_hash_field == proof.public.pins_hash_field", proof_summary.get("pins_hash_field"), public_sec.get("pins_hash_field")),
        _make_check("summary.policy_id_field == proof.public.policy_id_field", proof_summary.get("policy_id_field"), public_sec.get("policy_id_field")),
        _make_check("summary.public_input_order_id_field == proof.public.public_input_order_id_field", proof_summary.get("public_input_order_id_field"), public_sec.get("public_input_order_id_field")),
        _make_check("proof.public.N_used == summary.N_used", public_sec.get("N_used"), proof_summary.get("N_used")),
        _make_check("sidecar.public_inputs == proof.public_inputs", exported_inputs, _extract_artifact_public_inputs(proof_obj)),
    ]

    report.update(
        {
            "paths": {
                "iov_rsu_summary": str(iov_summary_path),
                "proof_summary": str(proof_summary_path),
                "manifest": str(manifest_path),
                "proof_artifact": str(proof_artifact_path),
                "sidecar": str(sidecar_path),
            },
            "public_inputs_order": order,
            "expected_public_inputs": expected_inputs,
            "exported_public_inputs": exported_inputs,
            "public_inputs_by_name_from_proof": by_name,
            "expected_public_inputs_by_name": {k: str(v) for k, v in expected_map.items()},
            "consistency_checks": checks,
            "mismatches": vector_mismatches,
            "all_consistency_checks_ok": all(bool(x.get("ok", False)) for x in checks),
        }
    )
    report["result"] = "pass" if (not vector_mismatches and report["all_consistency_checks_ok"]) else "fail"
    return report


def verify_global_scope(root: Path, round_idx: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {"scope": "global", "round": int(round_idx), "result": "fail"}

    gate_path = root / "zkp_anchor_gating_manifest.json"
    proof_summary_path = _default_global_proof_summary_path(root, round_idx)
    global_manifest_path = _default_global_manifest_path(root, round_idx)

    gate_obj = _read_json(gate_path)
    proof_summary = _read_json(proof_summary_path)

    global_manifest_path = _resolve_existing_path(root, _as_str(gate_obj.get("global_round_manifest_path"))) or global_manifest_path
    if not global_manifest_path.exists():
        raise FileNotFoundError(f"GLOBAL round manifest not found: {global_manifest_path}")
    global_manifest_obj = _read_json(global_manifest_path)

    proof_artifact_path = _resolve_existing_path(root, _as_str(proof_summary.get("proof_artifact_path"))) or _default_global_proof_artifact_path(root, round_idx)
    if not proof_artifact_path.exists():
        raise FileNotFoundError(f"GLOBAL proof artifact not found: {proof_artifact_path}")
    proof_obj = _read_json(proof_artifact_path)

    sidecar_path = _first_existing(_sidecar_candidates_for_global(root, round_idx, gate_obj, proof_summary))
    if sidecar_path is None:
        raise FileNotFoundError(f"No GLOBAL sidecar/public-input file found for round={round_idx}")
    sidecar_obj = _read_json(sidecar_path)

    order = _extract_order_from_artifact(proof_obj, DEFAULT_GLOBAL_ORDER)
    by_name = _extract_by_name_from_artifact(proof_obj)
    public_sec = _extract_public_section(proof_obj)
    used_rsus = gate_obj.get("used_rsu_ids", []) if isinstance(gate_obj.get("used_rsu_ids"), list) else []

    expected_map: Dict[str, Any] = {
        "anchor_id": (_as_str(by_name.get("anchor_id")) or _as_str(gate_obj.get("anchor_id_field")) or _as_str(proof_summary.get("anchor_id_field"))),
        "round_idx": int(round_idx),
        "global_id": _as_int(by_name.get("global_id") or proof_summary.get("global_id") or public_sec.get("global_id"), 0),
        "K_used": _as_int(by_name.get("K_used") or len(used_rsus) or gate_obj.get("num_rsus_masked_in"), 0),
        "root_poseidon_field": _as_str(by_name.get("root_poseidon_field") or global_manifest_obj.get("root_poseidon_field") or gate_obj.get("root_poseidon_field") or proof_summary.get("root_poseidon_field")),
        "pins_hash_field": _as_str(by_name.get("pins_hash_field") or global_manifest_obj.get("pins_hash_field") or gate_obj.get("pins_hash_field") or proof_summary.get("pins_hash_field")),
        "policy_id_field": _as_str(by_name.get("policy_id_field") or global_manifest_obj.get("policy_id_field") or gate_obj.get("policy_id_field") or proof_summary.get("policy_id_field")),
        "public_input_order_id_field": _as_str(by_name.get("public_input_order_id_field") or global_manifest_obj.get("public_input_order_id_field") or gate_obj.get("public_input_order_id_field") or proof_summary.get("public_input_order_id_field")),
        "r_chal": _as_str(by_name.get("r_chal") or public_sec.get("r_chal")),
        "agg_commit": _as_str(by_name.get("agg_commit") or public_sec.get("agg_commit_field") or gate_obj.get("global_commit_field") or proof_summary.get("global_commit_field")),
    }

    exported_inputs = _coerce_public_inputs(sidecar_obj) or _extract_artifact_public_inputs(proof_obj)
    expected_inputs = _make_vector(order, expected_map)
    vector_mismatches = _mismatches(order, expected_inputs, exported_inputs)

    checks = [
        _make_check(
            "gate.root_poseidon_field == proof/public root_poseidon_field",
            gate_obj.get("root_poseidon_field"),
            (
                by_name.get("root_poseidon_field")
                or public_sec.get("root_poseidon_field")
                or proof_summary.get("root_poseidon_field")
            ),
        ),
        _make_check("gate.global_commit_field == summary.global_commit_field", gate_obj.get("global_commit_field"), proof_summary.get("global_commit_field")),
        _make_check("len(gate.used_rsu_ids) == expected K_used", len(used_rsus), expected_map["K_used"]),
        _make_check("gate.pins_hash_field == global_manifest.pins_hash_field", gate_obj.get("pins_hash_field"), global_manifest_obj.get("pins_hash_field")),
        _make_check("gate.policy_id_field == global_manifest.policy_id_field", gate_obj.get("policy_id_field"), global_manifest_obj.get("policy_id_field")),
        _make_check("gate.public_input_order_id_field == global_manifest.public_input_order_id_field", gate_obj.get("public_input_order_id_field"), global_manifest_obj.get("public_input_order_id_field")),
        _make_check("sidecar.public_inputs == proof.public_inputs", exported_inputs, _extract_artifact_public_inputs(proof_obj)),
    ]

    report.update(
        {
            "paths": {
                "gating_manifest": str(gate_path),
                "proof_summary": str(proof_summary_path),
                "global_round_manifest": str(global_manifest_path),
                "proof_artifact": str(proof_artifact_path),
                "sidecar": str(sidecar_path),
            },
            "public_inputs_order": order,
            "expected_public_inputs": expected_inputs,
            "exported_public_inputs": exported_inputs,
            "public_inputs_by_name_from_proof": by_name,
            "expected_public_inputs_by_name": {k: str(v) for k, v in expected_map.items()},
            "consistency_checks": checks,
            "mismatches": vector_mismatches,
            "all_consistency_checks_ok": all(bool(x.get("ok", False)) for x in checks),
        }
    )
    report["result"] = "pass" if (not vector_mismatches and report["all_consistency_checks_ok"]) else "fail"
    return report


def build_report(root: Path, rsu_id: int, rsu_round: int, global_round: int) -> Dict[str, Any]:
    rsu_report = verify_rsu_scope(root, rsu_id=rsu_id, round_idx=rsu_round)
    global_report = verify_global_scope(root, round_idx=global_round)
    return {
        "schema": "PublicInputArtifactVerificationReportV1",
        "root_dir": str(root.resolve()),
        "rsu_scope": rsu_report,
        "global_scope": global_report,
        "result": "pass" if rsu_report.get("result") == "pass" and global_report.get("result") == "pass" else "fail",
    }


def main() -> int:
    root = _auto_detect_root()
    rsu_id = int(_as_int(CONFIG.get("RSU_ID"), 2))
    rsu_round = int(_as_int(CONFIG.get("RSU_ROUND"), 2))
    global_round = int(_as_int(CONFIG.get("GLOBAL_ROUND"), 2))

    report = build_report(root=root, rsu_id=rsu_id, rsu_round=rsu_round, global_round=global_round)

    output_json_cfg = _as_str(CONFIG.get("OUTPUT_JSON")).strip()
    output_path = Path(output_json_cfg) if output_json_cfg else root / "ablation_reports" / "public_inputs_artifact_verification_report.json"
    _write_json(output_path, report)

    print("=" * 88)
    print("Artifact public-input verification")
    print(f"Root         : {root}")
    print(f"RSU target   : rsu_id={rsu_id}, round={rsu_round}")
    print(f"GLOBAL target: round={global_round}")
    print(f"Result       : {report.get('result')}")
    print(f"Saved report : {output_path}")
    print("=" * 88)
    print(json.dumps(report, indent=2))

    if CONFIG.get("FAIL_EXIT_CODE", True):
        return 0 if report.get("result") == "pass" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
