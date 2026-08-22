#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-chain ↔ On-chain Dry-Run Simulator (PyCharm-friendly)
What it does:
  1) Loads onchain_export/latest_run_index_v1.json (OnChainExportIndexV1)
  2) Resolves proof/public/verifier/vkey files for:
        - RSU proofs for rounds 1..num_rounds, rsu_ids_sorted
        - GLOBAL proof for global_round.round_idx
  3) Builds Solidity verifyProof() args (a,b,c,input) from proof + public inputs v1
  4) Runs snarkjs groth16 verify for each proof (DEFAULT: ON)
  5) Enforces "swap-detection": compares pinned SHA256 (from index) vs computed SHA256
  6) Enforces BN254 field bounds for public inputs (prevents on-chain revert)
  7) Writes a JSON report to:
        <root_dir>/onchain_export/offchain_onchain_dryrun_report.json
PyCharm behavior:
  - No CLI flags needed.
  - snarkjs verification runs automatically unless you pass --no-snarkjs
  - pin-enforcement is STRICT by default unless you pass --no-strict-pins
"""
from __future__ import annotations
import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
# ------------------------------------------------------------
# Defaults (PyCharm-friendly)
# ------------------------------------------------------------
_DEFAULT_REPO_ROOT = os.getenv("FLBCIDS_REPO_ROOT", ".")
DEFAULT_ROOT_DIR = os.getenv(
    "FLBCIDS_COMPACT_EVIDENCE_DIR",
    os.path.join(
        _DEFAULT_REPO_ROOT,
        "experiments",
        "08_compact_security_evidence",
        "CSECICIDS2018",
        "run_outputs",
    ),
)
DEFAULT_INDEX_PATH = os.path.join(
    DEFAULT_ROOT_DIR,
    "onchain_export",
    "latest_run_index_v1.json",
)
# ✅ By default, snarkjs verification is ON (no CLI required).
SNARKJS_VERIFY_DEFAULT = True
# ------------------------------------------------------------
# BN254 field modulus (alt_bn128 scalar field)
# Public inputs MUST be < FIELD_MODULUS or on-chain verify can revert.
# ------------------------------------------------------------
FIELD_MODULUS_BN254 = int(
    "21888242871839275222246405745257275088548364400416034343698204186575808495617"
)
# ------------------------------------------------------------
# Pins structure (from latest_run_index_v1.json)
# ------------------------------------------------------------
@dataclass
class ExpectedPins:
    # Optional pinned relpaths (from index)
    proof_relpath: str = ""
    public_inputs_v1_relpath: str = ""
    verifier_sol_relpath: str = ""
    vkey_relpath: str = ""
    manifest_relpath: str = ""
    # Optional pinned hashes (from index)
    proof_sha256: str = ""
    public_inputs_v1_sha256: str = ""
    verifier_sol_sha256: str = ""
    vkey_sha256: str = ""
    manifest_sha256: str = ""
# ------------------------------------------------------------
# Bundle structure
# ------------------------------------------------------------
@dataclass
class ProofBundle:
    scope: str  # "RSU" or "GLOBAL"
    rsu_id: int
    round_idx: int
    proof_json: str
    public_inputs_v1_json: str
    snarkjs_public_json: str
    verifier_sol: str
    vkey_json: str
    # ✅ RSU-only: round manifest that contains vehicle SSI verification evidence
    manifest_json: str = ""
    # ✅ NEW: verifier resolution becomes self-auditing in the report
    resolved_verifier_strategy: str = ""
    resolved_verifier_candidates: List[str] = field(default_factory=list)
# ------------------------------------------------------------
# Basic IO helpers
# ------------------------------------------------------------
def _utc_now_iso() -> str:
    dt = _dt.datetime.now(_dt.UTC).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")
def _is_nonempty_file(path: str) -> bool:
    try:
        return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def _atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
def _first_existing(paths: List[str]) -> str:
    for p in paths:
        if _is_nonempty_file(p):
            return p
    return ""
def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or isinstance(x, bool):
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(s)
    except Exception:
        return default
# ------------------------------------------------------------
# Index relpath helpers (pins)
# ------------------------------------------------------------
def _norm_relpath(p: str) -> str:
    s = str(p or "").strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s
def _expand_ellipsis_segments(root_dir: str, rel: str) -> str:
    """
    Expands abbreviated path segments containing '...' by scanning the filesystem.

    Example:
      AggRSU_AnchorSum...0_RB2  -> AggRSU_AnchorSum_M64_K2_RM1_S100000_RC0_RB2
    """
    try:
        rel = _norm_relpath(rel)
        if not rel or "..." not in rel:
            return rel
        parts = [p for p in rel.split("/") if p and p != "."]
        cur = Path(root_dir)
        resolved: List[str] = []
        for seg in parts:
            if "..." not in seg:
                resolved.append(seg)
                cur = cur / seg
                continue
            pre, suf = seg.split("...", 1)
            matches: List[str] = []
            try:
                if cur.exists():
                    for child in cur.iterdir():
                        name = child.name
                        if name.startswith(pre) and name.endswith(suf):
                            matches.append(name)
            except Exception:
                matches = []
            # If exactly one match, replace the abbreviated segment
            if len(matches) == 1:
                seg = matches[0]
            resolved.append(seg)
            cur = cur / seg
        return "/".join(resolved)
    except Exception:
        # If anything goes wrong, fall back to original rel
        return _norm_relpath(rel)
def _abs_from_index_rel(root_dir: str, relpath: str) -> str:
    rel = _norm_relpath(relpath)
    if not rel:
        return ""
    # allow absolute in index (rare but safe)
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    # ✅ NEW: expand abbreviated "..."" segments into real names
    rel = _expand_ellipsis_segments(root_dir, rel)
    return os.path.normpath(os.path.join(root_dir, rel))
def _iter_dicts_recursive(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts_recursive(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from _iter_dicts_recursive(x)
def _get_str(d: dict, *keys: str) -> str:
    for k in keys:
        v = d.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""
def _get_int(d: dict, *keys: str) -> int:
    for k in keys:
        v = d.get(k, None)
        if v is None:
            continue
        try:
            if isinstance(v, bool):
                continue
            return int(str(v).strip())
        except Exception:
            pass
    return 0
def extract_expected_pins_from_index(
    idx: dict,
    scope: str,
    rsu_id: int,
    round_idx: int,
) -> ExpectedPins:
    """
    V1 index-aware pin extractor (OnChainExportIndexV1).
    STRICTLY reads pins from:
      - rsu_rounds[rsu_id].rounds[round_idx]
      - global_round
      - circuits[] (verifier + vkey pins)
    """
    want_scope = str(scope or "").strip().lower()
    want_rsu = int(rsu_id)
    want_round = int(round_idx)
    # -----------------------------
    # 1) Resolve circuit pins (verifier + vkey) from circuits[]
    # -----------------------------
    circuits_list = idx.get("circuits") or []
    rsu_circuit_entry = None
    global_circuit_entry = None
    if isinstance(circuits_list, list):
        # Heuristic: detect RSU vs GLOBAL by name pattern
        for c in circuits_list:
            if not isinstance(c, dict):
                continue
            name = str(c.get("circuit_name") or "").strip()
            if not name:
                continue
            up = name.upper()
            if ("RSU" in up or "AGGRSU" in up) and rsu_circuit_entry is None:
                rsu_circuit_entry = c
            if ("GLOBAL" in up or "AGGGLOBAL" in up) and global_circuit_entry is None:
                global_circuit_entry = c
        # If still ambiguous (rare), fallback: first=rsu second=global
        if rsu_circuit_entry is None and len(circuits_list) >= 1 and isinstance(circuits_list[0], dict):
            rsu_circuit_entry = circuits_list[0]
        if global_circuit_entry is None and len(circuits_list) >= 2 and isinstance(circuits_list[1], dict):
            global_circuit_entry = circuits_list[1]
    # Select circuit pins by scope
    use_circuit = rsu_circuit_entry if want_scope == "rsu" else global_circuit_entry
    verifier_sol_relpath = ""
    verifier_sol_sha256 = ""
    vkey_relpath = ""
    vkey_sha256 = ""
    if isinstance(use_circuit, dict):
        verifier_sol_relpath = str(
            use_circuit.get("verifier_sol_path")
            or use_circuit.get("verifier_sol_relpath")
            or use_circuit.get("verifier_sol")
            or ""
        ).strip()
        verifier_sol_sha256 = str(use_circuit.get("verifier_sol_sha256") or "").strip()
        vkey_relpath = str(
            use_circuit.get("verification_key_path")
            or use_circuit.get("vkey_path")
            or use_circuit.get("verification_key_json_path")
            or ""
        ).strip()
        vkey_sha256 = str(
            use_circuit.get("verification_key_sha256")
            or use_circuit.get("vkey_sha256")
            or ""
        ).strip()
    # -----------------------------
    # 2) Resolve per-round pins for proof + public
    # -----------------------------
    proof_relpath = ""
    proof_sha256 = ""
    public_relpath = ""
    public_sha256 = ""
    manifest_relpath = ""
    manifest_sha256 = ""
    if want_scope == "rsu":
        rsu_rounds = idx.get("rsu_rounds") or {}
        rsu_entry = rsu_rounds.get(str(want_rsu)) or {}
        rounds = rsu_entry.get("rounds") or {}
        r_entry = rounds.get(str(want_round)) or {}
        p = r_entry.get("proof_json") or {}
        pub = r_entry.get("public_inputs_sidecar") or {}
        man = r_entry.get("round_manifest_json") or {}
        proof_relpath = str(p.get("path") or "").strip()
        proof_sha256 = str(p.get("sha256") or "").strip()
        # NOTE: index currently pins "public_inputs_sidecar" (snarkjs public list)
        public_relpath = str(pub.get("path") or "").strip()
        public_sha256 = str(pub.get("sha256") or "").strip()
        # Manifest is present but often empty today → do NOT hard-fail
        manifest_relpath = str(man.get("path") or "").strip()
        manifest_sha256 = str(man.get("sha256") or "").strip()
    else:
        g = idx.get("global_round") or {}
        p = g.get("proof_json") or {}
        pub = g.get("public_inputs_sidecar") or {}
        proof_relpath = str(p.get("path") or "").strip()
        proof_sha256 = str(p.get("sha256") or "").strip()
        public_relpath = str(pub.get("path") or "").strip()
        public_sha256 = str(pub.get("sha256") or "").strip()
    return ExpectedPins(
        proof_relpath=_norm_relpath(proof_relpath),
        public_inputs_v1_relpath=_norm_relpath(public_relpath),
        verifier_sol_relpath=_norm_relpath(verifier_sol_relpath),
        vkey_relpath=_norm_relpath(vkey_relpath),
        manifest_relpath=_norm_relpath(manifest_relpath),
        proof_sha256=proof_sha256,
        public_inputs_v1_sha256=public_sha256,
        verifier_sol_sha256=verifier_sol_sha256,
        vkey_sha256=vkey_sha256,
        manifest_sha256=manifest_sha256,
    )
# ------------------------------------------------------------
# Solidity verifier parsing
# ------------------------------------------------------------
_VERIFIER_INPUT_RE = re.compile(
    r"uint\s*\[\s*(\d+)\s*]\s*(?:calldata|memory)\s*(?:input|_pubSignals)\b",
    flags=re.IGNORECASE,
)
def parse_expected_public_len_from_verifier_sol(verifier_sol_path: str) -> int:
    try:
        txt = Path(verifier_sol_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0
    m = _VERIFIER_INPUT_RE.search(txt)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0
# ------------------------------------------------------------
# Groth16 public input parsing
# ------------------------------------------------------------
def _as_field_int(x: Any) -> int:
    if isinstance(x, bool):
        return 0
    if isinstance(x, int):
        return int(x)
    s = str(x).strip()
    if s == "":
        return 0
    return int(s)
def parse_public_inputs_any(public_json_path: str) -> List[int]:
    """
    Supports:
      - list: [ "123", "456", ... ]
      - dict: { "publicSignals": [...] }
      - dict: { "public_inputs": [...] }
      - dict: { "public_inputs_v1": [...] }
    """
    obj = _load_json(public_json_path)
    if isinstance(obj, dict):
        for k in ("publicSignals", "public_signals", "public_inputs", "public_inputs_v1", "inputs", "input"):
            if k in obj and isinstance(obj[k], list):
                return [_as_field_int(v) for v in obj[k]]
        # fallback: first list-like value
        for v in obj.values():
            if isinstance(v, list):
                return [_as_field_int(x) for x in v]
    if isinstance(obj, list):
        return [_as_field_int(v) for v in obj]
    return []
# ------------------------------------------------------------
# Groth16 proof parsing (snarkjs format → solidity args)
# ------------------------------------------------------------
def _extract_proof_dict_any(obj: Any) -> Dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    if all(k in obj for k in ("pi_a", "pi_b", "pi_c")):
        return obj
    if all(k in obj for k in ("a", "b", "c")):
        return obj
    for key in ("proof", "payload", "zkp", "groth16_proof", "snarkjs_proof"):
        v = obj.get(key)
        if isinstance(v, dict):
            got = _extract_proof_dict_any(v)
            if got is not None:
                return got
    for v in obj.values():
        if isinstance(v, dict):
            got = _extract_proof_dict_any(v)
            if got is not None:
                return got
    return None
_PROOF_PATH_KEYS = (
    "proof_path",
    "proof_json_path",
    "proof_json",
    "snarkjs_proof_json",
    "snarkjs_proof_path",
    "groth16_proof_json",
    "groth16_proof_path",
    "proof_relpath",
)
def _resolve_maybe_relpath(root_dir: str, wrapper_path: str, ref: str) -> str:
    s = str(ref or "").strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    # absolute path
    if os.path.isabs(s):
        p = os.path.normpath(s)
        return p if _is_nonempty_file(p) else ""
    # relative to wrapper dir first
    p1 = os.path.normpath(os.path.join(os.path.dirname(wrapper_path), s))
    if _is_nonempty_file(p1):
        return p1
    # relative to root dir
    p2 = os.path.normpath(os.path.join(root_dir, s))
    if _is_nonempty_file(p2):
        return p2
    return ""
def _find_proof_ref_paths_recursive(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _PROOF_PATH_KEYS and isinstance(v, str) and v.strip():
                out.append(v.strip())
            out.extend(_find_proof_ref_paths_recursive(v))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(_find_proof_ref_paths_recursive(x))
    return out
def _heuristic_guess_proof_files(root_dir: str, wrapper_proof_path: str) -> List[str]:
    p = Path(wrapper_proof_path)
    guesses: List[str] = []
    # round_1.json -> round_1_proof.json
    if p.name.startswith("round_") and p.suffix.lower() == ".json":
        guesses.append(str(p.with_name(p.stem + "_proof.json")))
    # zkp_anchor_summaries/... -> zkp_artifacts/...
    s = str(p).replace("\\", "/")
    if "/zkp_anchor_summaries/" in s:
        mapped = s.replace("/zkp_anchor_summaries/", "/zkp_artifacts/")
        mapped = mapped.replace("/round_", "/round_").replace(".json", "_proof.json")
        guesses.append(os.path.normpath(mapped))
    # fallback: common zkp_artifacts locations
    # (works even if wrapper lives in anchor_summaries)
    for m in (
        "zkp_artifacts/anchorsum",
        "zkp_artifacts\\anchorsum",
    ):
        if m in s:
            guesses.append(os.path.normpath(s))
    # also try: root_dir/zkp_artifacts/anchorsum/**/round_X_proof.json
    try:
        stem = p.stem  # round_1
        if stem.startswith("round_"):
            rnum = stem.split("_", 1)[1]
            guesses.append(os.path.normpath(os.path.join(root_dir, "zkp_artifacts", "anchorsum", "global", f"round_{rnum}_proof.json")))
    except Exception:
        pass
    # unique, existing only
    uniq: List[str] = []
    for g in guesses:
        if g and g not in uniq and _is_nonempty_file(g):
            uniq.append(g)
    return uniq
def _load_real_snarkjs_proof_dict(root_dir: str, proof_json_path: str) -> Dict[str, Any]:
    """
    Returns a dict that contains pi_a/pi_b/pi_c.
    Accepts:
      - direct snarkjs proof JSON
      - wrapper/summary JSON that references the real proof file
      - wrapper where we can guess round_<r>_proof.json
    """
    obj = _load_json(proof_json_path)
    if isinstance(obj, dict):
        extracted = _extract_proof_dict_any(obj)
        if extracted is not None and all(k in extracted for k in ("pi_a", "pi_b", "pi_c")):
            return extracted
        # follow referenced proof path(s)
        refs = _find_proof_ref_paths_recursive(obj)
        for ref in refs:
            resolved = _resolve_maybe_relpath(root_dir, proof_json_path, ref)
            if resolved:
                inner = _load_json(resolved)
                if isinstance(inner, dict):
                    extracted2 = _extract_proof_dict_any(inner) or inner
                    if isinstance(extracted2, dict) and all(k in extracted2 for k in ("pi_a", "pi_b", "pi_c")):
                        return extracted2
    # heuristic guesses
    for g in _heuristic_guess_proof_files(root_dir, proof_json_path):
        inner = _load_json(g)
        if isinstance(inner, dict) and all(k in inner for k in ("pi_a", "pi_b", "pi_c")):
            return inner
        extracted3 = _extract_proof_dict_any(inner) if isinstance(inner, dict) else None
        if isinstance(extracted3, dict) and all(k in extracted3 for k in ("pi_a", "pi_b", "pi_c")):
            return extracted3
    raise ValueError("proof_json missing pi_a/pi_b/pi_c (no embedded proof, no referenced proof, no guessed proof file)")
def parse_groth16_proof_any_to_solidity_args(proof_json_path: str, root_dir: str) -> Dict[str, Any]:
    obj = _load_json(proof_json_path)
    # If wrapper, resolve the real snarkjs proof dict
    if not (isinstance(obj, dict) and (("pi_a" in obj) or ("a" in obj))):
        obj = _load_real_snarkjs_proof_dict(root_dir=root_dir, proof_json_path=proof_json_path)
    # Already solidity format?
    if isinstance(obj, dict) and all(k in obj for k in ("a", "b", "c")):
        a = obj["a"]
        b = obj["b"]
        c = obj["c"]
        if (
            isinstance(a, list) and len(a) >= 2 and
            isinstance(b, list) and len(b) == 2 and
            isinstance(c, list) and len(c) >= 2
        ):
            return {
                "a": [_as_field_int(a[0]), _as_field_int(a[1])],
                "b": [
                    [_as_field_int(b[0][0]), _as_field_int(b[0][1])],
                    [_as_field_int(b[1][0]), _as_field_int(b[1][1])],
                ],
                "c": [_as_field_int(c[0]), _as_field_int(c[1])],
            }
    if not isinstance(obj, dict):
        raise ValueError("proof_json is not a dict")
    pi_a = obj.get("pi_a", None)
    pi_b = obj.get("pi_b", None)
    pi_c = obj.get("pi_c", None)
    if not (isinstance(pi_a, list) and isinstance(pi_b, list) and isinstance(pi_c, list)):
        raise ValueError("proof_json missing pi_a/pi_b/pi_c")
    a = [_as_field_int(pi_a[0]), _as_field_int(pi_a[1])]
    b = [
        [_as_field_int(pi_b[0][1]), _as_field_int(pi_b[0][0])],
        [_as_field_int(pi_b[1][1]), _as_field_int(pi_b[1][0])],
    ]
    c = [_as_field_int(pi_c[0]), _as_field_int(pi_c[1])]
    return {"a": a, "b": b, "c": c}
# ------------------------------------------------------------
# snarkjs verify runner helpers
# ------------------------------------------------------------
def _find_project_root_from_outputs(root_dir: str) -> str:
    """Locate the repository/project root from an arbitrary retained run root."""
    override = os.environ.get("FLBCIDS_REPO_ROOT", "").strip()
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"FLBCIDS_REPO_ROOT is not a directory: {p}"
            )
        return str(p)

    starts = [
        Path(root_dir).resolve(),
        Path(__file__).resolve().parent,
    ]
    for start in starts:
        for cur in (start, *start.parents):
            if (cur / "package.json").is_file():
                return str(cur)
            if (cur / "environment" / "package.json").is_file():
                return str(cur)
            if (cur / "node_modules").is_dir():
                return str(cur)
            if (cur / "environment" / "node_modules").is_dir():
                return str(cur)
            if (cur / "README.md").is_file() and (cur / "code").is_dir():
                return str(cur)

    return str(Path(__file__).resolve().parent)


def _snarkjs_cmd_candidates(project_root: str) -> List[List[str]]:
    """Return portable snarkjs command candidates without assuming repo layout."""
    cands: List[List[str]] = []

    pinned = os.environ.get("ANCHOR_ZKP_SNARKJS_CMD", "").strip()
    if pinned:
        cands.append([pinned])

    pr = Path(project_root).resolve()
    node_module_roots: List[Path] = []

    for raw in (
        os.environ.get("FLBCIDS_NODE_MODULES", "").strip(),
        os.environ.get("ANCHOR_ZKP_NODE_MODULES", "").strip(),
    ):
        if raw:
            node_module_roots.append(Path(raw).expanduser())

    node_module_roots.extend(
        [
            pr / "node_modules",
            pr / "environment" / "node_modules",
        ]
    )

    for nm in node_module_roots:
        bin_dir = nm / ".bin"
        if os.name == "nt":
            for name in ("snarkjs.CMD", "snarkjs.cmd"):
                p = bin_dir / name
                if p.is_file():
                    cands.append([str(p.resolve())])
        else:
            p = bin_dir / "snarkjs"
            if p.is_file():
                cands.append([str(p.resolve())])

    cands.extend(
        [
            ["snarkjs"],
            ["snarkjs.cmd"],
            ["snarkjs.CMD"],
            ["npx", "snarkjs"],
            ["npx.cmd", "snarkjs"],
            ["npx.CMD", "snarkjs"],
        ]
    )

    # De-duplicate while preserving priority.
    unique: List[List[str]] = []
    seen: set[tuple[str, ...]] = set()
    for cmd in cands:
        key = tuple(cmd)
        if key not in seen:
            seen.add(key)
            unique.append(cmd)
    return unique
def _snarkjs_public_as_list_file(public_json_path: str) -> Tuple[str, bool]:
    """
    snarkjs groth16 verify expects public signals as a JSON list.
    If input is an object sidecar, convert to list temp file.
    """
    obj = _load_json(public_json_path)
    if isinstance(obj, list):
        return public_json_path, False
    if isinstance(obj, dict):
        if isinstance(obj.get("publicSignals"), list):
            pub_list = obj["publicSignals"]
        elif isinstance(obj.get("public_inputs"), list):
            pub_list = obj["public_inputs"]
        elif isinstance(obj.get("public_inputs_v1"), list):
            pub_list = obj["public_inputs_v1"]
        else:
            pub_list = None
            for v in obj.values():
                if isinstance(v, list):
                    pub_list = v
                    break
            if pub_list is None:
                raise ValueError(f"public json has no list-like public signals: {public_json_path}")
        fd, tmp_path = tempfile.mkstemp(prefix="snarkjs_public_", suffix=".json")
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(pub_list, f, indent=0)
        return tmp_path, True
    raise ValueError(f"unsupported public json type for snarkjs: {type(obj)}")
def _snarkjs_proof_as_file(proof_json_path: str, root_dir: str) -> Tuple[str, bool]:
    """
    snarkjs groth16 verify expects a dict containing pi_a/pi_b/pi_c.
    If wrapper/artifact, resolve proof via _load_real_snarkjs_proof_dict and write temp.
    """
    obj = _load_json(proof_json_path)
    if isinstance(obj, dict) and all(k in obj for k in ("pi_a", "pi_b", "pi_c")):
        return proof_json_path, False
    extracted = None
    try:
        extracted = _load_real_snarkjs_proof_dict(root_dir=root_dir, proof_json_path=proof_json_path)
    except Exception:
        extracted = _extract_proof_dict_any(obj)
    if extracted is None or not all(k in extracted for k in ("pi_a", "pi_b", "pi_c")):
        raise ValueError(f"proof json has no snarkjs pi_a/pi_b/pi_c structure: {proof_json_path}")
    fd, tmp_path = tempfile.mkstemp(prefix="snarkjs_proof_", suffix=".json")
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=0)
    return tmp_path, True
def try_run_snarkjs_verify(
    project_root: str,
    vkey_json: str,
    public_json: str,
    proof_json: str,
    timeout_sec: int = 120,
) -> Tuple[bool, str, str]:
    logs: List[str] = []
    cmd_used = "none"
    for base_cmd in _snarkjs_cmd_candidates(project_root):
        cmd = list(base_cmd) + ["groth16", "verify", vkey_json, public_json, proof_json]
        pretty = " ".join(cmd)
        try:
            r = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
            )
            out = (r.stdout or "").strip()
            logs.append(f"[{base_cmd[0]}] rc={r.returncode}\n{out}\n")
            if r.returncode == 0:
                cmd_used = base_cmd[0]
                return True, "\n".join(logs).strip(), cmd_used
        except FileNotFoundError:
            logs.append(f"[{pretty}] FileNotFoundError: command not found")
        except subprocess.TimeoutExpired:
            logs.append(f"[{pretty}] TimeoutExpired: exceeded {timeout_sec}s")
        except Exception as exc:
            logs.append(f"[{pretty}] Exception: {exc}")
    return False, "\n".join(logs).strip(), cmd_used
# ------------------------------------------------------------
# Bundle resolution (prefers pinned paths first)
# ------------------------------------------------------------
# ------------------------------------------------------------
# RSU round manifest (Vehicle SSI evidence)
# ------------------------------------------------------------
def resolve_rsu_round_manifest(
    root_dir: str,
    rsu_id: int,
    round_idx: int,
) -> str:
    """
    Resolves:
      rsu_outputs_dp/rsu_<id>/round_manifests/rsu/rsu_<id>/round_<r>.json
    """
    root = Path(root_dir)
    return _first_existing([
        str(root / f"rsu_{rsu_id}" / "round_manifests" / "rsu" / f"rsu_{rsu_id}" / f"round_{round_idx}.json"),
        # fallback (in case a different exporter layout is used later)
        str(root / "round_manifests" / "rsu" / f"rsu_{rsu_id}" / f"round_{round_idx}.json"),
    ])
def parse_vehicle_ssi_evidence_from_rsu_manifest(manifest_json_path: str) -> Dict[str, Any]:
    """
    Extracts Vehicle SSI verification evidence from the RSU round manifest.
    Expected structure (your uploaded files match this):
      manifest["extra"]["ssi_verify_total|ok|fail"]
      manifest["extra"]["records"][i]["record"]["ssi_*"]
    """
    out: Dict[str, Any] = {
        "manifest_path": manifest_json_path,
        "ssi_verify_total": 0,
        "ssi_verify_ok": 0,
        "ssi_verify_fail": 0,
        "records_count": 0,
        "vehicle_records": [],
        "ok": None,
        "errors": [],
    }
    if not _is_nonempty_file(manifest_json_path):
        out["ok"] = None
        out["errors"].append(f"manifest missing/empty: {manifest_json_path!r}")
        return out
    try:
        man = _load_json(manifest_json_path)
    except Exception as exc:
        out["ok"] = None
        out["errors"].append(f"failed to load manifest json: {exc}")
        return out
    extra = man.get("extra") or {}
    out["ssi_verify_total"] = _safe_int(extra.get("ssi_verify_total"), 0)
    out["ssi_verify_ok"] = _safe_int(extra.get("ssi_verify_ok"), 0)
    out["ssi_verify_fail"] = _safe_int(extra.get("ssi_verify_fail"), 0)
    records = extra.get("records") or []
    if not isinstance(records, list):
        records = []
    out["records_count"] = len(records)
    bad = 0
    vehicle_rows: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rr = rec.get("record") or {}
        if not isinstance(rr, dict):
            rr = {}
        vehicle_rows.append({
            "vehicle_id": _safe_int(rr.get("vehicle_id"), 0),
            "ssi_did": str(rr.get("ssi_did") or "").strip(),
            "ssi_verify_ok": _safe_int(rr.get("ssi_verify_ok"), 0),
            "ssi_report_sha256_hex": str(rr.get("ssi_report_sha256_hex") or "").strip(),
            "ssi_pubkey_b64": str(rr.get("ssi_pubkey_b64") or "").strip(),
            "ssi_sig_b64": str(rr.get("ssi_sig_b64") or "").strip(),
        })
        if _safe_int(rr.get("ssi_verify_ok"), 0) != 1:
            bad += 1
    out["vehicle_records"] = vehicle_rows
    # ✅ "OK" means: totals exist + no fails + every record says verify_ok==1
    if out["ssi_verify_total"] > 0:
        out["ok"] = (
            out["ssi_verify_fail"] == 0
            and out["ssi_verify_ok"] == out["ssi_verify_total"]
            and bad == 0
        )
    else:
        # If total is 0, we cannot claim success (no vehicles checked)
        out["ok"] = False
        out["errors"].append("ssi_verify_total is 0 (no vehicle verification evidence)")
    return out
def _resolve_verifier_sol_strict_with_evidence(
    root_dir: str,
    circuit_name: str,
    pins: ExpectedPins | None,
) -> Tuple[str, str, List[str]]:
    """
    Returns:
      (verifier_sol_path, resolved_verifier_strategy, resolved_verifier_candidates)
    resolved_verifier_strategy ∈:
      - "pinned_relpath"
      - "common_name"
      - "sha256_match"
      - "missing"
    """
    root = Path(root_dir)
    circuit_dir = root / "zkp_artifacts" / "anchorsum" / str(circuit_name)
    # 1) Use pinned relpath if valid
    if pins and pins.verifier_sol_relpath:
        p_abs = _abs_from_index_rel(root_dir, pins.verifier_sol_relpath)
        if _is_nonempty_file(p_abs):
            return p_abs, "pinned_relpath", [p_abs]

    # 2) Common names
    common_candidates = [
        str(circuit_dir / f"Verifier_{circuit_name}.sol"),
        str(circuit_dir / "Verifier.sol"),
        str(circuit_dir / "verifier.sol"),
    ]
    found = _first_existing(common_candidates)
    if found:
        return found, "common_name", common_candidates
    # 3) SHA-based lookup (strongest: supports filename changes safely)
    want_sha = (pins.verifier_sol_sha256.strip().lower() if (pins and pins.verifier_sol_sha256) else "")
    scanned: List[str] = []
    if want_sha and circuit_dir.exists():
        try:
            for sol in sorted(circuit_dir.glob("*.sol")):
                sp = str(sol)
                scanned.append(sp)
                if _is_nonempty_file(sp):
                    if _sha256_file(sp).strip().lower() == want_sha:
                        return sp, "sha256_match", scanned
        except Exception:
            pass
    return "", "missing", (scanned if scanned else common_candidates)
def _resolve_verifier_sol_strict(
    root_dir: str,
    circuit_name: str,
    pins: ExpectedPins | None,
) -> str:
    p, _, _ = _resolve_verifier_sol_strict_with_evidence(root_dir, circuit_name, pins)
    return p
def resolve_rsu_bundle(
    root_dir: str,
    rsu_circuit: str,
    rsu_id: int,
    round_idx: int,
    pins: ExpectedPins | None = None,
) -> ProofBundle:
    root = Path(root_dir)
    pinned_proof = _abs_from_index_rel(root_dir, pins.proof_relpath if pins else "")
    pinned_pubv1 = _abs_from_index_rel(root_dir, pins.public_inputs_v1_relpath if pins else "")
    pinned_vkey = _abs_from_index_rel(root_dir, pins.vkey_relpath if pins else "")
    central_base = root / "zkp_artifacts" / "anchorsum" / f"rsu_{rsu_id}"
    local_base = root / f"rsu_{rsu_id}" / "zkp_artifacts" / "anchorsum" / f"rsu_{rsu_id}"
    proof_path = _first_existing([
        pinned_proof,
        str(local_base / f"round_{round_idx}_proof.json"),
        str(central_base / f"round_{round_idx}_proof.json"),
        str(local_base / f"round_{round_idx}.json"),
        str(central_base / f"round_{round_idx}.json"),
    ])
    pub_v1_path = _first_existing([
        pinned_pubv1,
        str(local_base / f"round_{round_idx}_public_inputs_v1.json"),
        str(central_base / f"round_{round_idx}_public_inputs_v1.json"),
        str(local_base / f"round_{round_idx}_public.json"),
        str(central_base / f"round_{round_idx}_public.json"),
    ])
    snark_pub_path = _first_existing([
        str(central_base / f"round_{round_idx}_public.json"),
        str(local_base / f"round_{round_idx}_public.json"),
    ])
    verifier_sol, v_strategy, v_candidates = _resolve_verifier_sol_strict_with_evidence(
        root_dir=root_dir,
        circuit_name=str(rsu_circuit),
        pins=pins,
    )
    vkey_json = _first_existing([
        pinned_vkey,
        str(root / "zkp_artifacts" / "anchorsum" / rsu_circuit / "verification_key.json"),
        str(root / "zkp_artifacts" / "anchorsum" / rsu_circuit / "vkey.json"),
    ])
    # ✅ NEW: RSU round manifest contains Vehicle SSI verification evidence
    manifest_path = resolve_rsu_round_manifest(root_dir=root_dir, rsu_id=int(rsu_id), round_idx=int(round_idx))
    return ProofBundle(
        scope="RSU",
        rsu_id=int(rsu_id),
        round_idx=int(round_idx),
        proof_json=proof_path,
        public_inputs_v1_json=pub_v1_path,
        snarkjs_public_json=snark_pub_path,
        verifier_sol=verifier_sol,
        vkey_json=vkey_json,
        manifest_json=manifest_path,
        resolved_verifier_strategy=v_strategy,
        resolved_verifier_candidates=v_candidates,
    )
def resolve_global_bundle(
    root_dir: str,
    global_circuit: str,
    round_idx: int,
    pins: ExpectedPins | None = None,
) -> ProofBundle:
    root = Path(root_dir)
    base = root / "zkp_artifacts" / "anchorsum" / "global"
    pinned_proof = _abs_from_index_rel(root_dir, pins.proof_relpath if pins else "")
    pinned_pubv1 = _abs_from_index_rel(root_dir, pins.public_inputs_v1_relpath if pins else "")
    pinned_vkey = _abs_from_index_rel(root_dir, pins.vkey_relpath if pins else "")
    proof_path = _first_existing([
        pinned_proof,
        str(base / f"round_{round_idx}_proof.json"),
        str(base / f"round_{round_idx}.json"),
    ])
    pub_v1_path = _first_existing([
        pinned_pubv1,
        str(base / f"round_{round_idx}_public_inputs_v1.json"),
        str(base / f"round_{round_idx}_public.json"),
    ])
    snark_pub_path = _first_existing([
        str(base / f"round_{round_idx}_public.json"),
    ])
    verifier_sol, v_strategy, v_candidates = _resolve_verifier_sol_strict_with_evidence(
        root_dir=root_dir,
        circuit_name=str(global_circuit),
        pins=pins,
    )
    vkey_json = _first_existing([
        pinned_vkey,
        str(root / "zkp_artifacts" / "anchorsum" / global_circuit / "verification_key.json"),
        str(root / "zkp_artifacts" / "anchorsum" / global_circuit / "vkey.json"),
    ])
    return ProofBundle(
        scope="GLOBAL",
        rsu_id=0,
        round_idx=int(round_idx),
        proof_json=proof_path,
        public_inputs_v1_json=pub_v1_path,
        snarkjs_public_json=snark_pub_path,
        verifier_sol=verifier_sol,
        vkey_json=vkey_json,
        manifest_json="",  # GLOBAL has no per-vehicle manifest
        resolved_verifier_strategy=v_strategy,
        resolved_verifier_candidates=v_candidates,
    )
# ------------------------------------------------------------
# Core checker (files + parse + pins compare + BN254 safety + snarkjs verify)
# ------------------------------------------------------------
def check_bundle(
    root_dir: str,
    bundle: ProofBundle,
    do_snarkjs: bool,
    pins: ExpectedPins | None = None,
    strict_pins: bool = True,
) -> Dict[str, Any]:
    errors: List[str] = []
    # ✅ NEW: vehicle SSI evidence does NOT break proof verification,
    # it is additional 3-layer evidence, so we keep it separate from "errors"
    warnings: List[str] = []
    vehicle_ssi: Dict[str, Any] = {}
    vehicle_ssi_ok: bool | None = None
    # -----------------------------
    # 1) File existence checks
    # -----------------------------
    if not _is_nonempty_file(bundle.proof_json):
        errors.append(f"missing/empty proof_json: {bundle.proof_json!r}")
    if not _is_nonempty_file(bundle.public_inputs_v1_json):
        errors.append(f"missing/empty public_inputs_v1_json: {bundle.public_inputs_v1_json!r}")
    if not _is_nonempty_file(bundle.verifier_sol):
        errors.append(f"missing/empty verifier_sol: {bundle.verifier_sol!r}")
    if not _is_nonempty_file(bundle.vkey_json):
        errors.append(f"missing/empty vkey_json: {bundle.vkey_json!r}")
    # -----------------------------
    # 2) Expected public input length from Verifier.sol
    # -----------------------------
    pub_len_expected = (
        parse_expected_public_len_from_verifier_sol(bundle.verifier_sol)
        if _is_nonempty_file(bundle.verifier_sol)
        else 0
    )
    # -----------------------------
    # 3) Parse public inputs v1
    # -----------------------------
    pub_inputs: List[int] = []
    if _is_nonempty_file(bundle.public_inputs_v1_json):
        try:
            pub_inputs = parse_public_inputs_any(bundle.public_inputs_v1_json)
        except Exception as exc:
            errors.append(f"failed to parse public_inputs_v1_json: {exc}")
    pub_len_actual = len(pub_inputs)
    if _is_nonempty_file(bundle.verifier_sol) and pub_len_expected <= 0:
        errors.append(
            "could not parse fixed public-input length from verifier.sol; "
            "refusing to infer a Solidity calldata contract"
        )
    elif pub_len_actual != pub_len_expected:
        errors.append(
            f"public inputs length mismatch ({pub_len_actual} != {pub_len_expected})"
        )
    # -----------------------------
    # 4) BN254 field safety check
    # -----------------------------
    if pub_inputs:
        for i, v in enumerate(pub_inputs):
            try:
                if v < 0 or v >= FIELD_MODULUS_BN254:
                    errors.append(f"public input out of field: idx={i} value={v}")
                    break
            except Exception:
                errors.append(f"public input invalid type: idx={i} value={v!r}")
                break
    # -----------------------------
    # 5) Parse proof → solidity args
    # -----------------------------
    solidity_args: Dict[str, Any] = {}
    if _is_nonempty_file(bundle.proof_json):
        try:
            solidity_args = parse_groth16_proof_any_to_solidity_args(
                proof_json_path=bundle.proof_json,
                root_dir=root_dir,
            )
            solidity_args["input"] = pub_inputs
        except Exception as exc:
            errors.append(f"failed to parse proof_json into solidity args: {exc}")
    # -----------------------------
    # 6) snarkjs verification
    # -----------------------------
    snarkjs_ok = None
    snarkjs_log = ""
    snarkjs_cmd_used = "disabled"
    if do_snarkjs:
        snarkjs_cmd_used = "none"
        if not _is_nonempty_file(bundle.vkey_json):
            snarkjs_ok = False
            snarkjs_log = "vkey_json missing/empty (cannot run snarkjs verify)"
        else:
            # prefer snarkjs_public_json if available else v1 list/object
            snark_public_to_use = (
                bundle.snarkjs_public_json
                if _is_nonempty_file(bundle.snarkjs_public_json)
                else bundle.public_inputs_v1_json
            )
            if not _is_nonempty_file(snark_public_to_use):
                snarkjs_ok = False
                snarkjs_log = f"snarkjs public json missing/empty: {snark_public_to_use!r}"
            else:
                public_list_path = snark_public_to_use
                created_tmp_public = False
                created_tmp_proof = False
                proof_path_to_use = bundle.proof_json
                try:
                    public_list_path, created_tmp_public = _snarkjs_public_as_list_file(snark_public_to_use)
                    proof_path_to_use, created_tmp_proof = _snarkjs_proof_as_file(bundle.proof_json, root_dir=root_dir)
                    project_root = _find_project_root_from_outputs(root_dir)
                    ok, log, used = try_run_snarkjs_verify(
                        project_root=project_root,
                        vkey_json=bundle.vkey_json,
                        public_json=public_list_path,
                        proof_json=proof_path_to_use,
                    )
                    snarkjs_ok = bool(ok)
                    snarkjs_log = log
                    snarkjs_cmd_used = used
                except Exception as exc:
                    snarkjs_ok = False
                    snarkjs_log = f"snarkjs exception: {exc}"
                    snarkjs_cmd_used = "none"
                finally:
                    if created_tmp_public:
                        try:
                            os.remove(public_list_path)
                        except Exception:
                            pass
                    if created_tmp_proof:
                        try:
                            os.remove(proof_path_to_use)
                        except Exception:
                            pass
    # -----------------------------
    # 7) SHA256 for key files
    # -----------------------------
    sha_block: Dict[str, str] = {}
    if _is_nonempty_file(bundle.proof_json):
        sha_block["proof_sha256"] = _sha256_file(bundle.proof_json)
    if _is_nonempty_file(bundle.public_inputs_v1_json):
        sha_block["public_inputs_v1_sha256"] = _sha256_file(bundle.public_inputs_v1_json)
    if _is_nonempty_file(bundle.verifier_sol):
        sha_block["verifier_sol_sha256"] = _sha256_file(bundle.verifier_sol)
    if _is_nonempty_file(bundle.vkey_json):
        sha_block["vkey_sha256"] = _sha256_file(bundle.vkey_json)
    if _is_nonempty_file(bundle.snarkjs_public_json):
        sha_block["snarkjs_public_sha256"] = _sha256_file(bundle.snarkjs_public_json)
    # ✅ NEW: hash the RSU round manifest too (Vehicle SSI evidence carrier)
    if bundle.scope.upper() == "RSU" and _is_nonempty_file(bundle.manifest_json):
        sha_block["manifest_sha256"] = _sha256_file(bundle.manifest_json)
    # -----------------------------
    # 8) PINS CHECK: compare pinned SHA256 vs computed SHA256
    # -----------------------------
    pins_expected: Dict[str, str] = {}
    pins_relpaths: Dict[str, str] = {}
    pins_mismatches: List[str] = []
    if pins:
        if pins.proof_relpath:
            pins_relpaths["proof_relpath"] = pins.proof_relpath
        if pins.public_inputs_v1_relpath:
            pins_relpaths["public_inputs_v1_relpath"] = pins.public_inputs_v1_relpath
        if pins.verifier_sol_relpath:
            pins_relpaths["verifier_sol_relpath"] = pins.verifier_sol_relpath
        if pins.vkey_relpath:
            pins_relpaths["vkey_relpath"] = pins.vkey_relpath
        if pins.proof_sha256:
            pins_expected["proof_sha256"] = pins.proof_sha256
        if pins.public_inputs_v1_sha256:
            pins_expected["public_inputs_v1_sha256"] = pins.public_inputs_v1_sha256
        if pins.verifier_sol_sha256:
            pins_expected["verifier_sol_sha256"] = pins.verifier_sol_sha256
        if pins.vkey_sha256:
            pins_expected["vkey_sha256"] = pins.vkey_sha256
    # ------------------------------------------------------------
    # STRICT PINS REQUIREMENT: every bundle must have pins in index
    # ------------------------------------------------------------
    if strict_pins and (pins is None or (not pins_expected and not pins_relpaths)):
        errors.append("strict_pins enabled but no pins found in index for this bundle")
    # ✅ Do NOT require manifest pins yet (index currently leaves them empty)
    # Manifests are filesystem-resolved evidence carriers for SSI only.
    def _cmp(name: str, expected: str, actual: str):
        if not expected:
            return
        if not actual:
            pins_mismatches.append(f"{name} mismatch (expected {expected} != actual <missing>)")
            return
        if expected.lower() != actual.lower():
            pins_mismatches.append(f"{name} mismatch (expected {expected} != actual {actual})")
    if pins_expected:
        _cmp("proof_sha256", pins_expected.get("proof_sha256", ""), sha_block.get("proof_sha256", ""))
        _cmp("public_inputs_v1_sha256", pins_expected.get("public_inputs_v1_sha256", ""), sha_block.get("public_inputs_v1_sha256", ""))
        _cmp("verifier_sol_sha256", pins_expected.get("verifier_sol_sha256", ""), sha_block.get("verifier_sol_sha256", ""))
        _cmp("vkey_sha256", pins_expected.get("vkey_sha256", ""), sha_block.get("vkey_sha256", ""))
    def _same_path(a: str, b: str) -> bool:
        try:
            return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))
        except Exception:
            return False
    # strict: if relpaths exist but resolved files missing OR resolved != pinned, fail
    if strict_pins and pins_relpaths:
        # proof.json
        if pins.proof_relpath:
            p_abs = _abs_from_index_rel(root_dir, pins.proof_relpath)
            if not _is_nonempty_file(p_abs):
                errors.append(f"pinned proof_relpath missing/empty: {pins.proof_relpath!r} -> {p_abs!r}")
            elif not _same_path(bundle.proof_json, p_abs):
                errors.append(
                    f"resolved proof_json differs from pinned proof_relpath: resolved={bundle.proof_json!r} pinned={p_abs!r}")
        # public_inputs_v1.json
        if pins.public_inputs_v1_relpath:
            p_abs = _abs_from_index_rel(root_dir, pins.public_inputs_v1_relpath)
            if not _is_nonempty_file(p_abs):
                errors.append(
                    f"pinned public_inputs_v1_relpath missing/empty: {pins.public_inputs_v1_relpath!r} -> {p_abs!r}")
            elif not _same_path(bundle.public_inputs_v1_json, p_abs):
                errors.append(
                    f"resolved public_inputs differs from pinned: resolved={bundle.public_inputs_v1_json!r} pinned={p_abs!r}")
        # Verifier.sol (special-case: allow filename changes if SHA matches)
        if pins.verifier_sol_relpath:
            p_abs = _abs_from_index_rel(root_dir, pins.verifier_sol_relpath)
            pinned_exists = _is_nonempty_file(p_abs)
            sha_expected = (pins.verifier_sol_sha256 or "").strip().lower()
            sha_actual = (sha_block.get("verifier_sol_sha256") or "").strip().lower()
            sha_matches = bool(sha_expected and sha_actual and sha_expected == sha_actual)
            if not pinned_exists:
                if sha_matches and bundle.resolved_verifier_strategy == "sha256_match":
                    warnings.append(
                        f"pinned verifier_sol_relpath missing but sha256 matches resolved "
                        f"(strategy={bundle.resolved_verifier_strategy})"
                    )
                else:
                    errors.append(
                        f"pinned verifier_sol_relpath missing/empty: {pins.verifier_sol_relpath!r} -> {p_abs!r}"
                    )
            else:
                if _same_path(bundle.verifier_sol, p_abs):
                    # Explicitly record "perfect match"
                    # (useful in audits / debugging renamed verifier outputs)
                    pass_msg = (
                        f"verifier_sol path matches pinned exactly "
                        f"(strategy={bundle.resolved_verifier_strategy})"
                    )
                    # keep it as a warning-like audit breadcrumb (but not an error)
                    warnings.append(pass_msg)
                else:
                    if sha_matches:
                        warnings.append(
                            f"resolved verifier_sol path differs but sha256 matches pinned "
                            f"(strategy={bundle.resolved_verifier_strategy})"
                        )
                    else:
                        errors.append(
                            f"resolved verifier_sol differs from pinned: resolved={bundle.verifier_sol!r} pinned={p_abs!r}"
                        )
        # verification_key.json
        if pins.vkey_relpath:
            p_abs = _abs_from_index_rel(root_dir, pins.vkey_relpath)
            if not _is_nonempty_file(p_abs):
                errors.append(f"pinned vkey_relpath missing/empty: {pins.vkey_relpath!r} -> {p_abs!r}")
            elif not _same_path(bundle.vkey_json, p_abs):
                errors.append(f"resolved vkey_json differs from pinned: resolved={bundle.vkey_json!r} pinned={p_abs!r}")
    pins_ok = (len(pins_mismatches) == 0)
    if strict_pins and pins_expected and not pins_ok:
        for m in pins_mismatches:
            errors.append(m)
    # -----------------------------
    # 9) Final OK status
    # -----------------------------
    ok_files = (len(errors) == 0)
    ok_strict = ok_files and ((snarkjs_ok is True) if do_snarkjs else True)
    # -----------------------------
    # 9) Vehicle SSI evidence (RSU manifests)
    # -----------------------------
    if bundle.scope.upper() == "RSU":
        if _is_nonempty_file(bundle.manifest_json):
            vehicle_ssi = parse_vehicle_ssi_evidence_from_rsu_manifest(bundle.manifest_json)
            vehicle_ssi_ok = bool(vehicle_ssi.get("ok")) if vehicle_ssi.get("ok") is not None else None
        else:
            vehicle_ssi = {
                "manifest_path": bundle.manifest_json,
                "ok": None,
                "errors": [f"manifest missing/empty: {bundle.manifest_json!r}"],
            }
            vehicle_ssi_ok = None
            warnings.append(f"RSU manifest missing/empty (no vehicle SSI evidence): {bundle.manifest_json!r}")
    # ✅ NEW: “3-layer OK” definition:
    # - GLOBAL: proof-only is enough
    # - RSU: proof + vehicle SSI evidence
    ok_3layer = bool(ok_strict)
    if bundle.scope.upper() == "RSU":
        ok_3layer = bool(ok_strict) and (vehicle_ssi_ok is True)
    return {
        "scope": bundle.scope,
        "rsu_id": int(bundle.rsu_id),
        "round_idx": int(bundle.round_idx),
        "proof_json": bundle.proof_json,
        "public_inputs_v1_json": bundle.public_inputs_v1_json,
        "snarkjs_public_json": bundle.snarkjs_public_json,
        "verifier_sol": bundle.verifier_sol,
        "vkey_json": bundle.vkey_json,
        "manifest_json": bundle.manifest_json,
        "ok": bool(ok_files),
        "ok_strict": bool(ok_strict),
        "ok_3layer": bool(ok_3layer),
        "errors": errors,
        "warnings": warnings,
        "pub_len_expected": int(pub_len_expected),
        "pub_len_actual": int(pub_len_actual),
        "snarkjs_verify_ok": snarkjs_ok if do_snarkjs else None,
        "snarkjs_verify_log": snarkjs_log if do_snarkjs else "",
        "snarkjs_cmd_used": snarkjs_cmd_used if do_snarkjs else "disabled",
        "solidity_args": solidity_args if solidity_args else {},
        "sha256": sha_block,
        # pins evidence
        "pins_relpaths": pins_relpaths,
        "pins_expected": pins_expected,
        "pins_ok": bool(pins_ok) if (pins_expected or pins_relpaths) else None,
        "pins_mismatches": pins_mismatches,
        "strict_pins": bool(strict_pins),
        # ✅ NEW: Vehicle SSI evidence summary (from manifest)
        "vehicle_ssi": vehicle_ssi if vehicle_ssi else {},
        "vehicle_ssi_ok": vehicle_ssi_ok,
        # ✅ NEW: self-auditing verifier resolution (important for renamed Verifier.sol)
        "resolved_verifier_strategy": bundle.resolved_verifier_strategy,
        "resolved_verifier_candidates": list(bundle.resolved_verifier_candidates),
    }
# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT_DIR, help="Compact security-evidence run root")
    ap.add_argument("--index", default=DEFAULT_INDEX_PATH, help="OnChainExportIndexV1 JSON path")
    ap.add_argument(
        "--no-snarkjs",
        action="store_true",
        help="Disable snarkjs groth16 verify (default is ON for PyCharm runs)",
    )
    ap.add_argument(
        "--no-strict-pins",
        action="store_true",
        help="Disable strict pinned-sha enforcement (default is strict)",
    )
    args = ap.parse_args()
    strict_pins = (not bool(args.no_strict_pins))
    def _derive_root_from_index(index_path: str) -> str:
        p = Path(index_path).resolve()
        # .../rsu_outputs_dp/onchain_export/latest_run_index_v1.json
        return str(p.parent.parent)
    root_dir = os.path.normpath(str(args.root))
    index_path = os.path.normpath(str(args.index))
    if _is_nonempty_file(index_path):
        root_dir = _derive_root_from_index(index_path)
    else:
        index_path = os.path.join(root_dir, "onchain_export", "latest_run_index_v1.json")
    do_snarkjs = bool(SNARKJS_VERIFY_DEFAULT) and (not bool(args.no_snarkjs))
    print("\n==============================")
    print(" OFF-CHAIN ON-CHAIN DRY RUN ")
    print("==============================")
    print(f"root_dir               : {root_dir}")
    print(f"index_path             : {index_path}")
    print(f"snarkjs_verify         : {'ON' if do_snarkjs else 'OFF'}")
    print(f"strict_pins            : {'ON' if strict_pins else 'OFF'}")
    if not _is_nonempty_file(index_path):
        print(f"\n[FATAL] index file missing: {index_path}")
        return 2
    idx = _load_json(index_path)
    topology = idx.get("topology") or {}
    num_rounds = _safe_int(topology.get("num_rounds"), _safe_int(idx.get("num_rounds"), 0))
    num_rsus = _safe_int(topology.get("num_rsus"), _safe_int(idx.get("num_rsus"), 0))
    rsu_vehicle_ids = topology.get("rsu_vehicle_ids") or {}
    rsu_ids_sorted = topology.get("rsu_ids_sorted") or sorted(
        [_safe_int(k, 0) for k in rsu_vehicle_ids.keys() if str(k).strip()]
    )
    # circuits may be list or dict depending on exporter version
    circuits_raw = idx.get("circuits") or {}
    rsu_circuit = ""
    global_circuit = ""
    if isinstance(circuits_raw, list):
        c_map: Dict[str, Dict[str, Any]] = {}
        for entry in circuits_raw:
            if isinstance(entry, dict):
                scope = str(entry.get("scope") or entry.get("kind") or "").strip().lower()
                if scope:
                    c_map[scope] = entry
        rsu_circuit = str(
            (c_map.get("rsu") or {}).get("circuit")
            or (c_map.get("rsu") or {}).get("circuit_name")
            or ""
        ).strip()
        global_circuit = str(
            (c_map.get("global") or {}).get("circuit")
            or (c_map.get("global") or {}).get("circuit_name")
            or ""
        ).strip()
    elif isinstance(circuits_raw, dict):
        rsu_circuit = str(circuits_raw.get("rsu") or "").strip()
        global_circuit = str(circuits_raw.get("global") or "").strip()
    # fallback safe defaults
    if not rsu_circuit:
        rsu_circuit = "AggRSU_AnchorSum_M64_K2_RM1_S100000_RC0_RB2"
    if not global_circuit:
        global_circuit = "AggGlobal_AnchorSum_M64_K2_RM2_S100000_RC0_RB2"
    global_round = idx.get("global_round") or {}
    global_round_idx = _safe_int(global_round.get("round_idx"), num_rounds)
    used_rsu_ids = global_round.get("used_rsu_ids") or rsu_ids_sorted
    # ------------------------------------------------------------
    # Normalize RSU ids to ints (index may store strings)
    # ------------------------------------------------------------
    def _as_int_list(xs: Any) -> List[int]:
        if not isinstance(xs, list):
            return []
        out: List[int] = []
        for x in xs:
            v = _safe_int(x, 0)
            if v > 0:
                out.append(v)
        # stable unique sorted
        return sorted(set(out))
    rsu_ids_sorted = _as_int_list(rsu_ids_sorted)
    used_rsu_ids = _as_int_list(used_rsu_ids)
    print("\n------------------------------")
    print("INDEX SUMMARY")
    print("------------------------------")
    print(f"num_rounds             : {num_rounds}")
    print(f"num_rsus               : {num_rsus}")
    print(f"rsu_ids_sorted         : {list(rsu_ids_sorted)}")
    print(f"GLOBAL round_idx       : {global_round_idx}")
    print(f"GLOBAL used_rsu_ids     : {list(used_rsu_ids)}")
    print(f"RSU circuit            : {rsu_circuit}")
    print(f"GLOBAL circuit         : {global_circuit}")
    print("")
    checks: List[Dict[str, Any]] = []
    print("------------------------------")
    print("RSU proof bundles")
    print("------------------------------")
    rsu_rounds_map = idx.get("rsu_rounds") or {}
    for rid_str in sorted(rsu_rounds_map.keys(), key=lambda x: _safe_int(x, 0)):
        rid = _safe_int(rid_str, 0)
        if rid <= 0:
            continue
        rounds_map = (rsu_rounds_map.get(rid_str) or {}).get("rounds") or {}
        round_ids_sorted = sorted(rounds_map.keys(), key=lambda x: _safe_int(x, 0))
        for r_str in round_ids_sorted:
            r = _safe_int(r_str, 0)
            if r <= 0:
                continue
            pins = extract_expected_pins_from_index(idx, scope="rsu", rsu_id=rid, round_idx=r)
            bundle = resolve_rsu_bundle(root_dir, rsu_circuit, rid, r, pins=pins)
            res = check_bundle(
                root_dir=root_dir,
                bundle=bundle,
                do_snarkjs=do_snarkjs,
                pins=pins,
                strict_pins=strict_pins,
            )
            checks.append(res)
            proof_name = os.path.basename(res.get("proof_json") or "") or "?"
            in_len = int(res.get("pub_len_actual") or 0)
            pins_status = ""
            if res.get("pins_expected") or res.get("pins_relpaths"):
                pins_status = f" | PINS={'OK' if res.get('pins_ok') else 'FAIL'}"
            snark_status = ""
            if do_snarkjs:
                snark_ok = res.get("snarkjs_verify_ok")
                snark_status = f" | SNARKJS={'OK' if snark_ok else 'FAIL'}"
            veh_status = ""
            if res.get("scope") == "RSU":
                vssi = res.get("vehicle_ssi") or {}
                if res.get("vehicle_ssi_ok") is True:
                    veh_status = f" | VEH_SSI=OK({int(vssi.get('ssi_verify_ok', 0))}/{int(vssi.get('ssi_verify_total', 0))})"
                elif res.get("vehicle_ssi_ok") is False:
                    veh_status = " | VEH_SSI=FAIL"
                else:
                    veh_status = " | VEH_SSI=UNKNOWN"
            if res.get("ok_3layer", False):
                print(
                    f"[OK]   RSU {rid} round {r} -> inputs={in_len} proof={proof_name}{pins_status}{snark_status}{veh_status}")
            elif res.get("ok_strict", False):
                print(
                    f"[WARN] RSU {rid} round {r} -> inputs={in_len} proof={proof_name}{pins_status}{snark_status}{veh_status}")
            else:
                print(
                    f"[FAIL] RSU {rid} round {r} -> inputs={in_len} proof={proof_name}{pins_status}{snark_status}{veh_status}")
                for e in res.get("errors", [])[:4]:
                    print(f"       - {e}")
    print("\n------------------------------")
    print("GLOBAL proof bundle")
    print("------------------------------")
    pins_g = extract_expected_pins_from_index(idx, scope="global", rsu_id=0, round_idx=global_round_idx)
    bundle_g = resolve_global_bundle(root_dir, global_circuit, global_round_idx, pins=pins_g)
    res_g = check_bundle(
        root_dir=root_dir,
        bundle=bundle_g,
        do_snarkjs=do_snarkjs,
        pins=pins_g,
        strict_pins=strict_pins,
    )
    checks.append(res_g)
    proof_name_g = os.path.basename(res_g.get("proof_json") or "") or "?"
    in_len_g = int(res_g.get("pub_len_actual") or 0)
    pins_status_g = ""
    if res_g.get("pins_expected") or res_g.get("pins_relpaths"):
        pins_status_g = f" | PINS={'OK' if res_g.get('pins_ok') else 'FAIL'}"
    snark_status_g = ""
    if do_snarkjs:
        snark_ok_g = res_g.get("snarkjs_verify_ok")
        snark_status_g = f" | SNARKJS={'OK' if snark_ok_g else 'FAIL'}"
    if res_g.get("ok_strict", False):
        print(
            f"[OK]   GLOBAL round {global_round_idx} -> inputs={in_len_g} proof={proof_name_g}{pins_status_g}{snark_status_g} | VEH_SSI=N/A")
    else:
        print(
            f"[FAIL] GLOBAL round {global_round_idx} -> inputs={in_len_g} proof={proof_name_g}{pins_status_g}{snark_status_g} | VEH_SSI=N/A")
        for e in res_g.get("errors", [])[:4]:
            print(f"       - {e}")
    report = {
        "generated_utc": _utc_now_iso(),
        "root_dir": root_dir,
        "index_path": index_path,
        "snarkjs_verify": bool(do_snarkjs),
        "strict_pins": bool(strict_pins),
        "field_modulus_bn254": str(FIELD_MODULUS_BN254),
        "topology": {
            "num_rounds": num_rounds,
            "num_rsus": num_rsus,
            "rsu_ids_sorted": rsu_ids_sorted,
            "rsu_vehicle_ids": rsu_vehicle_ids,
            "vehicles_per_rsu": topology.get("vehicles_per_rsu"),
        },
        "circuits": {
            "rsu": rsu_circuit,
            "global": global_circuit,
        },
        "global_round": {
            "round_idx": global_round_idx,
            "used_rsu_ids": used_rsu_ids,
        },
        "checks": checks,
        "summary": {
            "ok_files_count": sum(1 for x in checks if x.get("ok")),
            "ok_strict_count": sum(1 for x in checks if x.get("ok_strict")),
            "ok_3layer_count": sum(1 for x in checks if x.get("ok_3layer")),
            "vehicle_ssi_ok_count": sum(
                1 for x in checks if x.get("scope") == "RSU" and x.get("vehicle_ssi_ok") is True),
            "vehicle_ssi_fail_count": sum(
                1 for x in checks if x.get("scope") == "RSU" and x.get("vehicle_ssi_ok") is False),
            "pins_ok_count": sum(1 for x in checks if x.get("pins_expected") and x.get("pins_ok") is True),
            "pins_fail_count": sum(1 for x in checks if x.get("pins_expected") and x.get("pins_ok") is False),
            "total": len(checks),
        },
    }
    out_path = os.path.join(root_dir, "onchain_export", "offchain_onchain_dryrun_report.json")
    _atomic_write_json(out_path, report)
    print("\n------------------------------")
    print("DRY RUN REPORT WRITTEN")
    print("------------------------------")
    print(out_path)
    print("\nDone.\n")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
