# On-Chain-RSU-GlobalServer-SSI-Verification_V10.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ON-CHAIN RSU + GLOBAL + SSI VERIFICATION (Transaction-Based, Thesis-Grade)
What it does (NO DRY-RUN, NO SNARKJS):
  1) Loads onchain_export/latest_run_index_v1.json (OnChainExportIndexV1)
  2) Resolves proof/public/verifier/vkey files for all RSU rounds + GLOBAL round
  3) Parses Solidity verifyProof() args (a,b,c,input)
  4) Enforces STRICT swap-detection pins + BN254 field bounds (prevents wasting gas)
  5) Deploys/reuses:
        - RSU Verifier contract
        - GLOBAL Verifier contract
        - ProofRegistryV1 contract (verifies inside tx, stores results, emits events)
  6) Submits tx for each RSU proof + GLOBAL proof:
        - submitAndVerifyRSU_V1(...)
        - submitAndVerifyGLOBAL_V1(...)
  7) Finalizes run on-chain:
        - finalizeRunV1(runId)
  8) Writes JSON report:
        <root_dir>/onchain_export/onchain_verification_report.json
IMPORTANT:
  - This script performs REAL on-chain verification and costs gas.
  - Results become immutable evidence via ProofVerifiedV1 / RunFinalizedV1 events.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import time
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from eth_account import Account  # type: ignore
# ------------------------------------------------------------
# Credentials / sender safety
# ------------------------------------------------------------
# Secrets are read only at execution time from environment/.env. They are never
# embedded in this publication source and are not accepted as CLI arguments.
EXPECTED_SENDER_ADDRESS = os.getenv(
    "SEPOLIA_EXPECTED_ADDRESS",
    "",
).strip()

# ------------------------------------------------------------
# Defaults (publication-repository friendly)
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
# Sepolia defaults
DEFAULT_CHAIN_ID = 11155111
DEFAULT_RPC_ENV_KEYS = ("SEPOLIA_RPC_URL", "RPC_URL", "WEB3_RPC_URL")
DEFAULT_PRIVKEY_ENV_KEYS = ("SEPOLIA_PRIVATE_KEY", "PRIVATE_KEY", "WALLET_PRIVATE_KEY")
# Optional workflow toggles (thesis default = submit everything)
SUBMIT_GLOBAL_ONLY = False       # True => submit only GLOBAL proof (cheaper)
DO_FINALIZE_RUN = True           # finalizeRunV1(runId) after submissions
WAIT_FOR_RECEIPTS = True         # wait until each tx is included/mined
RECEIPT_TIMEOUT_SEC = 300

# Reviewer Comment 12: consensus-finality measurement.
# Finality is measured separately from transaction receipt/inclusion.
WAIT_FOR_FINALITY = True
FINALITY_TIMEOUT_SEC = 1200
FINALITY_POLL_INTERVAL_SEC = 3.0

# The current publication workflow is intentionally serial:
# one sender submits one proof transaction and waits for its receipt
# before the next proof transaction is submitted.
SUBMISSION_PATTERN = "SERIAL_SINGLE_SENDER_RECEIPT_WAIT"
MAX_INFLIGHT_SUBMISSIONS = 1

# Print a consolidated reviewer-facing Sepolia report to the console.
PRINT_REVIEWER_METRICS = True

# ✅ Thesis-grade evidence: RSU submission must include pinned RSU manifest
REQUIRE_RSU_MANIFEST = True
# Deployment caches
DEPLOYED_VERIFIERS_CACHE = "deployed_verifiers_sepolia.json"
DEPLOYED_REGISTRY_CACHE = "deployed_registry_sepolia.json"
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
    proof_relpath: str = ""
    public_inputs_v1_relpath: str = ""
    verifier_sol_relpath: str = ""
    vkey_relpath: str = ""
    manifest_relpath: str = ""

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
    verifier_sol: str
    vkey_json: str
    # RSU-only
    manifest_json: str = ""
    # Verifier resolution evidence (debug/audit)
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
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _file_size_bytes(path: str) -> int:
    try:
        return int(os.path.getsize(path)) if path and os.path.exists(path) else 0
    except Exception:
        return 0

def _tx_data_size_bytes(tx: Dict[str, Any]) -> int:
    try:
        data = tx.get("data", b"")
        if isinstance(data, str):
            s = data[2:] if data.startswith("0x") else data
            return len(s) // 2
        if isinstance(data, (bytes, bytearray, memoryview)):
            return len(data)
    except Exception:
        pass
    return 0
def _atomic_write_json(path: str, obj: Any) -> None:
    """
    Atomic JSON writer that is safe for Web3 objects (HexBytes/bytes) and other
    non-JSON-native types.
    - bytes / bytearray / memoryview / HexBytes -> "0x..." hex string
    - Path -> str(path)
    - set / tuple -> list
    - dict keys that are not strings -> converted to strings (bytes -> hex)
    """
    def _to_hex0x(b: bytes) -> str:
        return "0x" + b.hex()
    def _sanitize(x: Any) -> Any:
        # --- bytes-like values (common from Web3 tx hashes, bytes32, etc.) ---
        if isinstance(x, (bytes, bytearray, memoryview)):
            return _to_hex0x(bytes(x))
        # --- HexBytes (web3) ---
        try:
            from hexbytes import HexBytes  # type: ignore
            if isinstance(x, HexBytes):
                return _to_hex0x(bytes(x))
        except Exception:
            pass
        # --- Paths ---
        if isinstance(x, Path):
            return str(x)
        # --- Basic JSON-native types ---
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        # --- Containers ---
        if isinstance(x, dict):
            out: Dict[str, Any] = {}
            for k, v in x.items():
                # JSON requires string keys
                if isinstance(k, (bytes, bytearray, memoryview)):
                    kk = _to_hex0x(bytes(k))
                else:
                    kk = str(k)
                out[kk] = _sanitize(v)
            return out
        if isinstance(x, (list, tuple)):
            return [_sanitize(v) for v in x]
        if isinstance(x, set):
            return [_sanitize(v) for v in sorted(x, key=lambda t: str(t))]
        # --- Web3 AttributeDict / objects that behave like dict ---
        if hasattr(x, "items") and callable(getattr(x, "items")):
            try:
                return _sanitize(dict(x.items()))
            except Exception:
                pass
        # --- Fallback: string representation (never crash JSON writer) ---
        return str(x)
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    safe_obj = _sanitize(obj)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(safe_obj, f, indent=2, ensure_ascii=False)
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
# Dotenv (simple loader, no dependencies)
# ------------------------------------------------------------
def _load_dotenv_simple(path: str) -> None:
    """
    Loads KEY=VALUE pairs into os.environ if not already set.
    Safe for local usage; ignores malformed lines.
    """
    try:
        if not _is_nonempty_file(path):
            return
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        for line in txt.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and (k not in os.environ):
                os.environ[k] = v
    except Exception:
        return
def _get_env_first(keys: Tuple[str, ...]) -> str:
    for k in keys:
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return ""
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
            if len(matches) == 1:
                seg = matches[0]
            resolved.append(seg)
            cur = cur / seg
        return "/".join(resolved)
    except Exception:
        return _norm_relpath(rel)
def _abs_from_index_rel(root_dir: str, relpath: str) -> str:
    rel = _norm_relpath(relpath)
    if not rel:
        return ""
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    rel = _expand_ellipsis_segments(root_dir, rel)
    return os.path.normpath(os.path.join(root_dir, rel))
def extract_expected_pins_from_index(
    idx: dict,
    scope: str,
    rsu_id: int,
    round_idx: int,
) -> ExpectedPins:
    """
    Reads pins from:
      - rsu_rounds[rsu_id].rounds[round_idx]
      - global_round
      - circuits[] (verifier + vkey pins)
    """
    want_scope = str(scope or "").strip().lower()
    want_rsu = int(rsu_id)
    want_round = int(round_idx)
    circuits_list = idx.get("circuits") or []
    rsu_circuit_entry = None
    global_circuit_entry = None
    if isinstance(circuits_list, list):
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
        if rsu_circuit_entry is None and len(circuits_list) >= 1 and isinstance(circuits_list[0], dict):
            rsu_circuit_entry = circuits_list[0]
        if global_circuit_entry is None and len(circuits_list) >= 2 and isinstance(circuits_list[1], dict):
            global_circuit_entry = circuits_list[1]
    use_circuit = rsu_circuit_entry if want_scope == "rsu" else global_circuit_entry
    def _path_sha_from_any(v: Any) -> Tuple[str, str]:
        # Supports:
        #  - "path/to/file"
        #  - {"path": "...", "sha256": "..."}
        #  - {"relpath": "...", "sha256": "..."}
        if isinstance(v, str):
            return v.strip(), ""
        if isinstance(v, dict):
            p = str(v.get("path") or v.get("relpath") or v.get("file") or "").strip()
            s = str(v.get("sha256") or v.get("hash") or v.get("sha") or "").strip()
            return p, s
        return "", ""
    def _pin_path_sha(ent: dict, path_keys: Tuple[str, ...], sha_keys: Tuple[str, ...]) -> Tuple[str, str]:
        if not isinstance(ent, dict):
            return "", ""
        # If a field is stored as {"path":..., "sha256":...}
        for k in path_keys:
            if k in ent:
                p, s = _path_sha_from_any(ent.get(k))
                if p:
                    # try to locate sha in dict payload first, else in sibling keys
                    if not s:
                        for sk in sha_keys:
                            if sk in ent and str(ent.get(sk) or "").strip():
                                s = str(ent.get(sk) or "").strip()
                                break
                    return p, s
        # Fall back to sibling sha keys if any
        p2 = ""
        for k2 in path_keys:
            if k2 in ent and isinstance(ent.get(k2), str) and str(ent.get(k2) or "").strip():
                p2 = str(ent.get(k2) or "").strip()
                break
        s2 = ""
        for sk2 in sha_keys:
            if sk2 in ent and str(ent.get(sk2) or "").strip():
                s2 = str(ent.get(sk2) or "").strip()
                break
        return p2, s2
    verifier_sol_relpath = ""
    verifier_sol_sha256 = ""
    vkey_relpath = ""
    vkey_sha256 = ""
    if isinstance(use_circuit, dict):
        verifier_sol_relpath, verifier_sol_sha256 = _pin_path_sha(
            use_circuit,
            path_keys=("verifier_sol", "verifier_sol_path", "verifier_sol_relpath", "verifier"),
            sha_keys=("verifier_sol_sha256", "verifier_sha256", "verifierSolSha256"),
        )
        vkey_relpath, vkey_sha256 = _pin_path_sha(
            use_circuit,
            path_keys=("verification_key", "verification_key_path", "verification_key_json_path", "vkey", "vkey_path"),
            sha_keys=("verification_key_sha256", "vkey_sha256", "vkeySha256"),
        )
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
        public_relpath = str(pub.get("path") or "").strip()
        public_sha256 = str(pub.get("sha256") or "").strip()
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
# Solidity verifier parsing (public input length)
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
        for v in obj.values():
            if isinstance(v, list):
                return [_as_field_int(x) for x in v]
    if isinstance(obj, list):
        return [_as_field_int(v) for v in obj]
    return []
# ------------------------------------------------------------
# Groth16 proof parsing (snarkjs format → solidity args)
# ------------------------------------------------------------
def _extract_proof_dict_any(obj: Any) -> Optional[Dict[str, Any]]:
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
    if os.path.isabs(s):
        p = os.path.normpath(s)
        return p if _is_nonempty_file(p) else ""
    p1 = os.path.normpath(os.path.join(os.path.dirname(wrapper_path), s))
    if _is_nonempty_file(p1):
        return p1
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
    if p.name.startswith("round_") and p.suffix.lower() == ".json":
        guesses.append(str(p.with_name(p.stem + "_proof.json")))
    s = str(p).replace("\\", "/")
    if "/zkp_anchor_summaries/" in s:
        mapped = s.replace("/zkp_anchor_summaries/", "/zkp_artifacts/")
        mapped = mapped.replace(".json", "_proof.json")
        guesses.append(os.path.normpath(mapped))
    try:
        stem = p.stem
        if stem.startswith("round_"):
            rnum = stem.split("_", 1)[1]
            guesses.append(
                os.path.normpath(
                    os.path.join(root_dir, "zkp_artifacts", "anchorsum", "global", f"round_{rnum}_proof.json")
                )
            )
    except Exception:
        pass
    uniq: List[str] = []
    for g in guesses:
        if g and g not in uniq and _is_nonempty_file(g):
            uniq.append(g)
    return uniq
def _load_real_snarkjs_proof_dict(root_dir: str, proof_json_path: str) -> Dict[str, Any]:
    obj = _load_json(proof_json_path)
    if isinstance(obj, dict):
        extracted = _extract_proof_dict_any(obj)
        if extracted is not None and all(k in extracted for k in ("pi_a", "pi_b", "pi_c")):
            return extracted
        refs = _find_proof_ref_paths_recursive(obj)
        for ref in refs:
            resolved = _resolve_maybe_relpath(root_dir, proof_json_path, ref)
            if resolved:
                inner = _load_json(resolved)
                if isinstance(inner, dict):
                    extracted2 = _extract_proof_dict_any(inner) or inner
                    if isinstance(extracted2, dict) and all(k in extracted2 for k in ("pi_a", "pi_b", "pi_c")):
                        return extracted2
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
    if not (isinstance(obj, dict) and (("pi_a" in obj) or ("a" in obj))):
        obj = _load_real_snarkjs_proof_dict(root_dir=root_dir, proof_json_path=proof_json_path)
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
# RSU round manifest (Vehicle SSI evidence)
# ------------------------------------------------------------
def resolve_rsu_round_manifest(root_dir: str, rsu_id: int, round_idx: int) -> str:
    root = Path(root_dir)
    return _first_existing([
        str(root / f"rsu_{rsu_id}" / "round_manifests" / "rsu" / f"rsu_{rsu_id}" / f"round_{round_idx}.json"),
        str(root / "round_manifests" / "rsu" / f"rsu_{rsu_id}" / f"round_{round_idx}.json"),
    ])
def parse_vehicle_ssi_evidence_from_rsu_manifest(manifest_json_path: str) -> Dict[str, Any]:
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
    if out["ssi_verify_total"] > 0:
        out["ok"] = (
            out["ssi_verify_fail"] == 0
            and out["ssi_verify_ok"] == out["ssi_verify_total"]
            and bad == 0
        )
    else:
        out["ok"] = False
        out["errors"].append("ssi_verify_total is 0 (no vehicle verification evidence)")
    return out
# ------------------------------------------------------------
# Verifier.sol resolution (pinned → common → sha-match)
# ------------------------------------------------------------
def _dedup_paths_keep_order(paths: List[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        key = os.path.normcase(os.path.normpath(str(p)))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
def _candidate_anchorsum_dirs_v1(root_dir: str, circuit_name: str) -> List[Path]:
    """
    Returns all plausible directories where Verifier.sol / vkey.json may exist.
    Matches both:
      - zkp_artifacts/anchorsum/<circuit>
      - circuits_generated/zkp_artifacts/anchorsum/<circuit>
      - *_js variants
    """
    root = Path(root_dir)
    c = str(circuit_name or "").strip()
    bases = [
        root / "zkp_artifacts" / "anchorsum",
        root / "circuits_generated" / "zkp_artifacts" / "anchorsum",
        root / "circuits_generated" / "anchorsum",
        root / "onchain_export" / "zkp_artifacts" / "anchorsum",
    ]
    dirs: List[Path] = []
    for b in bases:
        if c:
            dirs.append(b / c)
            dirs.append(b / f"{c}_js")
            dirs.append(b / f"{c}-js")
        dirs.append(b)
    # Only keep existing dirs first (but keep non-existing too for evidence)
    existing = [d for d in dirs if d.exists() and d.is_dir()]
    missing = [d for d in dirs if not (d.exists() and d.is_dir())]
    return _dedup_paths_keep_order(existing + missing)
def _sha256_match_in_dirs_v1(dirs: List[Path], want_sha256: str, glob_pat: str) -> str:
    """
    Searches (non-recursive then recursive) for any file matching want_sha256.
    """
    target = str(want_sha256 or "").strip().lower()
    if not target:
        return ""
    for d in dirs:
        try:
            if d.exists() and d.is_dir():
                # 1) non-recursive scan
                for p in sorted(d.glob(glob_pat)):
                    sp = str(p)
                    if _is_nonempty_file(sp) and _sha256_file(sp).strip().lower() == target:
                        return sp
                # 2) recursive scan (bounded)
                hits = 0
                for p in sorted(d.rglob(glob_pat)):
                    hits += 1
                    if hits > 300:
                        break
                    sp = str(p)
                    if _is_nonempty_file(sp) and _sha256_file(sp).strip().lower() == target:
                        return sp
        except Exception:
            continue
    return ""
def _resolve_verifier_sol_strict_with_evidence(
    root_dir: str,
    circuit_name: str,
    pins: Optional[ExpectedPins],
) -> Tuple[str, str, List[str]]:
    # 1) Pinned relpath always wins
    if pins and pins.verifier_sol_relpath:
        p_abs = _abs_from_index_rel(root_dir, pins.verifier_sol_relpath)
        if _is_nonempty_file(p_abs):
            return p_abs, "pinned_relpath", [p_abs]
    # 2) Search common locations across multiple candidate dirs (like dry-run resolution)
    dirs = _candidate_anchorsum_dirs_v1(root_dir, circuit_name)
    common_candidates: List[str] = []
    for d in dirs:
        cn = str(circuit_name or "").strip()
        if cn:
            common_candidates.append(str(d / f"Verifier_{cn}.sol"))
        common_candidates.append(str(d / "Verifier.sol"))
        common_candidates.append(str(d / "verifier.sol"))
    found = _first_existing(common_candidates)
    if found:
        return found, "common_name_multidir", common_candidates
    # 3) SHA256 match scan (strongest fallback)
    want_sha = (str(pins.verifier_sol_sha256 or "").strip().lower() if pins else "")
    sha_match = _sha256_match_in_dirs_v1(dirs, want_sha, "*.sol")
    if sha_match:
        return sha_match, "sha256_match_multidir", common_candidates
    # 4) Missing
    return "", "missing", common_candidates
def _resolve_vkey_json_strict_with_evidence(
    root_dir: str,
    circuit_name: str,
    pins: Optional[ExpectedPins],
) -> Tuple[str, str, List[str]]:
    # 1) Pinned relpath wins
    if pins and pins.vkey_relpath:
        p_abs = _abs_from_index_rel(root_dir, pins.vkey_relpath)
        if _is_nonempty_file(p_abs):
            return p_abs, "pinned_relpath", [p_abs]
    # 2) Common candidates in multiple dirs
    dirs = _candidate_anchorsum_dirs_v1(root_dir, circuit_name)
    common_candidates: List[str] = []
    for d in dirs:
        common_candidates.append(str(d / "verification_key.json"))
        common_candidates.append(str(d / "vkey.json"))
    found = _first_existing(common_candidates)
    if found:
        return found, "common_name_multidir", common_candidates
    # 3) SHA match (if provided)
    want_sha = (str(pins.vkey_sha256 or "").strip().lower() if pins else "")
    sha_match = _sha256_match_in_dirs_v1(dirs, want_sha, "*.json")
    if sha_match:
        return sha_match, "sha256_match_multidir", common_candidates
    return "", "missing", common_candidates
# ------------------------------------------------------------
# Bundle resolution (prefers pinned paths first)
# ------------------------------------------------------------
def resolve_rsu_bundle(
    root_dir: str,
    rsu_circuit: str,
    rsu_id: int,
    round_idx: int,
    pins: Optional[ExpectedPins] = None,
) -> ProofBundle:
    root = Path(root_dir)
    pinned_proof = _abs_from_index_rel(root_dir, pins.proof_relpath if pins else "")
    pinned_pubv1 = _abs_from_index_rel(root_dir, pins.public_inputs_v1_relpath if pins else "")
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
    verifier_sol, v_strategy, v_candidates = _resolve_verifier_sol_strict_with_evidence(
        root_dir=root_dir,
        circuit_name=str(rsu_circuit),
        pins=pins,
    )
    vkey_json, _, _ = _resolve_vkey_json_strict_with_evidence(
        root_dir=root_dir,
        circuit_name=str(rsu_circuit),
        pins=pins,
    )
    pinned_manifest = _abs_from_index_rel(root_dir, pins.manifest_relpath if pins else "")
    fallback_manifest = resolve_rsu_round_manifest(root_dir=root_dir, rsu_id=int(rsu_id), round_idx=int(round_idx))
    manifest_path = _first_existing([pinned_manifest, fallback_manifest])
    return ProofBundle(
        scope="RSU",
        rsu_id=int(rsu_id),
        round_idx=int(round_idx),
        proof_json=proof_path,
        public_inputs_v1_json=pub_v1_path,
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
    pins: Optional[ExpectedPins] = None,
) -> ProofBundle:
    root = Path(root_dir)
    base = root / "zkp_artifacts" / "anchorsum" / "global"
    pinned_proof = _abs_from_index_rel(root_dir, pins.proof_relpath if pins else "")
    pinned_pubv1 = _abs_from_index_rel(root_dir, pins.public_inputs_v1_relpath if pins else "")
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
    verifier_sol, v_strategy, v_candidates = _resolve_verifier_sol_strict_with_evidence(
        root_dir=root_dir,
        circuit_name=str(global_circuit),
        pins=pins,
    )
    vkey_json, _, _ = _resolve_vkey_json_strict_with_evidence(
        root_dir=root_dir,
        circuit_name=str(global_circuit),
        pins=pins,
    )
    return ProofBundle(
        scope="GLOBAL",
        rsu_id=0,
        round_idx=int(round_idx),
        proof_json=proof_path,
        public_inputs_v1_json=pub_v1_path,
        verifier_sol=verifier_sol,
        vkey_json=vkey_json,
        manifest_json="",
        resolved_verifier_strategy=v_strategy,
        resolved_verifier_candidates=v_candidates,
    )
# ------------------------------------------------------------
# Strict pin enforcement + BN254 safety + calldata builder (NO SNARKJS)
# ------------------------------------------------------------
def prepare_bundle_for_onchain(
    root_dir: str,
    bundle: ProofBundle,
    pins: Optional[ExpectedPins],
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    # Existence
    if not _is_nonempty_file(bundle.proof_json):
        errors.append(f"missing/empty proof_json: {bundle.proof_json!r}")
    if not _is_nonempty_file(bundle.public_inputs_v1_json):
        errors.append(f"missing/empty public_inputs_v1_json: {bundle.public_inputs_v1_json!r}")
    if not _is_nonempty_file(bundle.verifier_sol):
        errors.append(f"missing/empty verifier_sol: {bundle.verifier_sol!r}")
    if not _is_nonempty_file(bundle.vkey_json):
        errors.append(f"missing/empty vkey_json: {bundle.vkey_json!r}")
    # ✅ RSU must have manifest when strict evidence is enabled
    if bundle.scope.upper() == "RSU" and bool(globals().get("REQUIRE_RSU_MANIFEST", True)):
        if not _is_nonempty_file(bundle.manifest_json):
            errors.append(f"missing/empty RSU manifest_json: {bundle.manifest_json!r}")
    pub_len_expected = (
        parse_expected_public_len_from_verifier_sol(bundle.verifier_sol)
        if _is_nonempty_file(bundle.verifier_sol)
        else 0
    )
    pub_inputs: List[int] = []
    if _is_nonempty_file(bundle.public_inputs_v1_json):
        try:
            pub_inputs = parse_public_inputs_any(bundle.public_inputs_v1_json)
        except Exception as exc:
            errors.append(f"failed to parse public_inputs_v1_json: {exc}")
    pub_len_actual = len(pub_inputs)
    if pub_len_expected > 0 and pub_len_actual != pub_len_expected:
        errors.append(f"public inputs length mismatch ({pub_len_actual} != {pub_len_expected})")
    # BN254 safety
    if pub_inputs:
        for i, v in enumerate(pub_inputs):
            try:
                if v < 0 or v >= FIELD_MODULUS_BN254:
                    errors.append(f"public input out of field: idx={i} value={v}")
                    break
            except Exception:
                errors.append(f"public input invalid type: idx={i} value={v!r}")
                break
    # Parse proof -> a,b,c
    solidity_args: Dict[str, Any] = {}
    if _is_nonempty_file(bundle.proof_json):
        try:
            solidity_args = parse_groth16_proof_any_to_solidity_args(bundle.proof_json, root_dir=root_dir)
            solidity_args["input"] = pub_inputs
        except Exception as exc:
            errors.append(f"failed to parse proof_json into solidity args: {exc}")
    # Compute SHA256 for evidence + pins comparison
    sha_block: Dict[str, str] = {}
    if _is_nonempty_file(bundle.proof_json):
        sha_block["proof_sha256"] = _sha256_file(bundle.proof_json)
    if _is_nonempty_file(bundle.public_inputs_v1_json):
        sha_block["public_inputs_v1_sha256"] = _sha256_file(bundle.public_inputs_v1_json)
    if _is_nonempty_file(bundle.verifier_sol):
        sha_block["verifier_sol_sha256"] = _sha256_file(bundle.verifier_sol)
    if _is_nonempty_file(bundle.vkey_json):
        sha_block["vkey_sha256"] = _sha256_file(bundle.vkey_json)
    if bundle.scope.upper() == "RSU" and _is_nonempty_file(bundle.manifest_json):
        sha_block["manifest_sha256"] = _sha256_file(bundle.manifest_json)
    # STRICT pins MUST exist + MUST match
    pins_expected: Dict[str, str] = {}
    pins_relpaths: Dict[str, str] = {}
    if pins:
        if pins.proof_relpath:
            pins_relpaths["proof_relpath"] = pins.proof_relpath
        if pins.public_inputs_v1_relpath:
            pins_relpaths["public_inputs_v1_relpath"] = pins.public_inputs_v1_relpath
        if pins.verifier_sol_relpath:
            pins_relpaths["verifier_sol_relpath"] = pins.verifier_sol_relpath
        if pins.vkey_relpath:
            pins_relpaths["vkey_relpath"] = pins.vkey_relpath
        # ✅ RSU manifest pins (strict swap-detection)
        if pins.manifest_relpath:
            pins_relpaths["manifest_relpath"] = pins.manifest_relpath
        if pins.proof_sha256:
            pins_expected["proof_sha256"] = pins.proof_sha256
        if pins.public_inputs_v1_sha256:
            pins_expected["public_inputs_v1_sha256"] = pins.public_inputs_v1_sha256
        if pins.verifier_sol_sha256:
            pins_expected["verifier_sol_sha256"] = pins.verifier_sol_sha256
        if pins.vkey_sha256:
            pins_expected["vkey_sha256"] = pins.vkey_sha256
        # ✅ RSU manifest sha pin
        if pins.manifest_sha256:
            pins_expected["manifest_sha256"] = pins.manifest_sha256
    if not pins_expected and not pins_relpaths:
        errors.append("STRICT pins required, but no pins found in index for this bundle")
    # Enforce thesis-grade completeness: require all sha256 pins
    required_sha = [
        "proof_sha256",
        "public_inputs_v1_sha256",
        "verifier_sol_sha256",
        "vkey_sha256",
    ]
    if bundle.scope.upper() == "RSU" and bool(globals().get("REQUIRE_RSU_MANIFEST", True)):
        required_sha.append("manifest_sha256")
    missing_sha = [k for k in required_sha if not str(pins_expected.get(k, "") or "").strip()]
    if missing_sha:
        errors.append("missing required pins sha256 fields: " + ", ".join(missing_sha))
    def _cmp(name: str, expected: str, actual: str):
        if not expected:
            return
        if not actual:
            errors.append(f"{name} mismatch (expected {expected} != actual <missing>)")
            return
        if expected.lower() != actual.lower():
            errors.append(f"{name} mismatch (expected {expected} != actual {actual})")
    _cmp("proof_sha256", pins_expected.get("proof_sha256", ""), sha_block.get("proof_sha256", ""))
    _cmp("public_inputs_v1_sha256", pins_expected.get("public_inputs_v1_sha256", ""),
         sha_block.get("public_inputs_v1_sha256", ""))
    _cmp("verifier_sol_sha256", pins_expected.get("verifier_sol_sha256", ""), sha_block.get("verifier_sol_sha256", ""))
    _cmp("vkey_sha256", pins_expected.get("vkey_sha256", ""), sha_block.get("vkey_sha256", ""))
    # ✅ Strict RSU manifest pins check
    if bundle.scope.upper() == "RSU":
        _cmp("manifest_sha256", pins_expected.get("manifest_sha256", ""), sha_block.get("manifest_sha256", ""))
    # SSI evidence (RSU only) = layer-3 evidence (does not block tx unless you want)
    vehicle_ssi: Dict[str, Any] = {}
    vehicle_ssi_ok: Optional[bool] = None
    if bundle.scope.upper() == "RSU":
        if _is_nonempty_file(bundle.manifest_json):
            vehicle_ssi = parse_vehicle_ssi_evidence_from_rsu_manifest(bundle.manifest_json)
            vehicle_ssi_ok = bool(vehicle_ssi.get("ok")) if vehicle_ssi.get("ok") is not None else None
        else:
            vehicle_ssi_ok = None
            warnings.append(f"RSU manifest missing/empty (no vehicle SSI evidence): {bundle.manifest_json!r}")
    ok = (len(errors) == 0)
    return {
        "ok": bool(ok),
        "errors": errors,
        "warnings": warnings,
        "scope": bundle.scope,
        "rsu_id": int(bundle.rsu_id),
        "round_idx": int(bundle.round_idx),
        "proof_json": bundle.proof_json,
        "public_inputs_v1_json": bundle.public_inputs_v1_json,
        "verifier_sol": bundle.verifier_sol,
        "vkey_json": bundle.vkey_json,
        "manifest_json": bundle.manifest_json,
        "pub_len_expected": int(pub_len_expected),
        "pub_len_actual": int(pub_len_actual),
        "solidity_args": solidity_args,
        "sha256": sha_block,
        "pins_relpaths": pins_relpaths,
        "pins_expected": pins_expected,
        "vehicle_ssi": vehicle_ssi,
        "vehicle_ssi_ok": vehicle_ssi_ok,
        "resolved_verifier_strategy": bundle.resolved_verifier_strategy,
        "resolved_verifier_candidates": list(bundle.resolved_verifier_candidates),
    }
# ------------------------------------------------------------
# Web3 (on-chain submit)
# ------------------------------------------------------------
def _require_web3():
    try:
        from web3 import Web3
        return Web3
    except Exception as exc:
        raise RuntimeError(
            "web3 is required for on-chain mode. Install with: pip install web3 eth-account"
        ) from exc
def _inject_poa_middleware_if_available(w3):
    # Safe injection; does nothing if not present
    try:
        from web3.middleware import geth_poa_middleware
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    except Exception:
        pass
def _bytes32_from_hex_sha256(hex_sha: str) -> str:
    s = str(hex_sha or "").strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) != 64:
        raise ValueError(f"sha256 hex must be 32 bytes (64 hex chars), got len={len(s)} value={hex_sha!r}")
    return "0x" + s
def _is_contract_address(w3, addr: str) -> bool:
    try:
        if not addr or not isinstance(addr, str) or not addr.startswith("0x"):
            return False
        code = w3.eth.get_code(addr)
        return bool(code and len(code) > 0)
    except Exception:
        return False
def _get_nonce(w3, addr: str) -> int:
    return int(w3.eth.get_transaction_count(addr, "pending"))
def _wait_receipt(w3, tx_hash, timeout_sec: int):
    try:
        return w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout_sec)
    except Exception as exc:
        # Web3 may raise TimeExhausted (or provider timeout)
        raise RuntimeError(f"tx not mined within timeout={timeout_sec}s hash={tx_hash.hex()} error={exc}")
def _web3_is_connected(w3) -> bool:
    """
    Web3 version-compatible connectivity check.

    Web3.py v6/v7 uses:
        w3.is_connected()

    Older Web3.py releases used:
        w3.isConnected()
    """
    try:
        method = getattr(w3, "is_connected")
        if callable(method):
            return bool(method())
    except (AttributeError, TypeError):
        pass
    except Exception:
        pass

    try:
        method = getattr(w3, "isConnected")
        if callable(method):
            return bool(method())
    except (AttributeError, TypeError):
        pass
    except Exception:
        pass

    return False
def _signed_raw_tx_bytes(signed) -> bytes:
    raw = getattr(signed, "rawTransaction", None)
    if raw is None:
        raw = getattr(signed, "raw_transaction", None)
    if raw is None:
        raise RuntimeError("signed tx missing rawTransaction/raw_transaction (web3 version mismatch)")
    return raw
def _process_event_receipt_any(ev, receipt):
    """
    Web3 version-compatible event-receipt decoder.

    Newer Web3.py:
        process_receipt(...)

    Older Web3.py:
        processReceipt(...)
    """
    if receipt is None:
        return []

    try:
        method = getattr(ev, "process_receipt")
        if callable(method):
            return method(receipt)
    except (AttributeError, TypeError):
        pass

    try:
        method = getattr(ev, "processReceipt")
        if callable(method):
            return method(receipt)
    except (AttributeError, TypeError):
        pass

    return []
def _get_balance_wei(w3, addr: str) -> int:
    return int(w3.eth.get_balance(addr))
def _fmt_eth(w3, wei_amt: int) -> str:
    try:
        return str(w3.from_wei(int(wei_amt), "ether"))
    except Exception:
        return f"{wei_amt} wei"
def _receipt_fee_wei(receipt: Any) -> int:
    """
    Returns gasUsed * effectiveGasPrice (EIP-1559) or gasUsed * gasPrice (legacy).
    Works with dict receipts and AttributeDict receipts.
    """
    if receipt is None:
        return 0
    try:
        if isinstance(receipt, dict):
            gas_used = int(receipt.get("gasUsed", 0) or 0)
            eff = receipt.get("effectiveGasPrice", None)
            if eff is None:
                eff = receipt.get("gasPrice", 0)
            return gas_used * int(eff or 0)
        # AttributeDict / web3 receipt object
        gas_used = int(getattr(receipt, "gasUsed", 0) or 0)
        eff = getattr(receipt, "effectiveGasPrice", None)
        if eff is None:
            eff = getattr(receipt, "gasPrice", 0)
        return gas_used * int(eff or 0)
    except Exception:
        return 0

# ------------------------------------------------------------
# Reviewer Comment 12: descriptive statistics
# ------------------------------------------------------------
def _percentile_nearest_rank(values: List[float], percentile: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0

    p = min(100.0, max(0.0, float(percentile)))

    if p <= 0.0:
        return float(vals[0])
    if p >= 100.0:
        return float(vals[-1])

    rank = int(math.ceil((p / 100.0) * len(vals)))
    rank = max(1, min(rank, len(vals)))
    return float(vals[rank - 1])


def _describe_values(values: List[float]) -> Dict[str, Any]:
    vals = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]

    if not vals:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "stddev_population": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
        }

    return {
        "n": int(len(vals)),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "stddev_population": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "p95": float(_percentile_nearest_rank(vals, 95.0)),
    }


# ------------------------------------------------------------
# Reviewer Comment 12: observed failure rate + 95% Wilson CI
# ------------------------------------------------------------
def _wilson_interval_95(failures: int, trials: int) -> Tuple[float, float]:
    """
    Wilson score interval for an observed binomial failure proportion.

    Important:
      This describes the observed transaction sample.
      It is NOT proof of an intrinsic Sepolia failure probability.
    """
    n = int(trials)
    k = int(failures)

    if n <= 0:
        return 0.0, 0.0

    k = max(0, min(k, n))
    p = float(k) / float(n)

    z = 1.959963984540054
    z2 = z * z

    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt(
            (p * (1.0 - p) / n)
            + (z2 / (4.0 * n * n))
        )
        / denom
    )

    return (
        float(max(0.0, center - margin)),
        float(min(1.0, center + margin)),
    )


# ------------------------------------------------------------
# Reviewer Comment 12: Ethereum finality measurement
# ------------------------------------------------------------
def _get_finalized_block_number(w3) -> int:
    """
    Returns the latest consensus-finalized block number.

    Uses the standard Ethereum JSON-RPC 'finalized' block tag.
    Returns 0 if the RPC/provider does not expose the tag.
    """
    try:
        blk = w3.eth.get_block("finalized")

        if isinstance(blk, dict):
            return int(blk.get("number", 0) or 0)

        return int(getattr(blk, "number", 0) or 0)

    except Exception:
        return 0


def _attach_finality_measurements(
    w3,
    rows: List[Dict[str, Any]],
    timeout_sec: int,
    poll_interval_sec: float,
) -> Dict[str, Any]:
    """
    Waits collectively for all already-mined transactions in `rows`
    to become consensus-finalized.

    This function is called once after publication transactions are mined,
    avoiding a separate full finality wait after every individual tx.
    """
    eligible_indices: List[int] = []

    for idx, row in enumerate(rows):
        block_no = _safe_int(row.get("block_number"), 0)
        receipt_epoch = float(row.get("receipt_observed_epoch_sec", 0.0) or 0.0)

        if block_no > 0 and receipt_epoch > 0.0:
            row["finality_measured"] = False
            row["finalized"] = False
            row["finalized_head_block_number"] = None
            row["finality_observed_epoch_sec"] = None
            row["receipt_to_finality_sec"] = None
            row["tx_start_to_finality_sec"] = None
            eligible_indices.append(idx)

    if not eligible_indices:
        return {
            "supported": False,
            "eligible_transactions": 0,
            "finalized_transactions": 0,
            "timed_out_transactions": 0,
            "error": "No mined transactions with receipt timestamps were available.",
        }

    first_probe = _get_finalized_block_number(w3)

    if first_probe <= 0:
        return {
            "supported": False,
            "eligible_transactions": len(eligible_indices),
            "finalized_transactions": 0,
            "timed_out_transactions": len(eligible_indices),
            "error": "RPC provider did not expose a usable 'finalized' block.",
        }

    pending = set(eligible_indices)
    t_start_wait = time.monotonic()
    t_deadline = t_start_wait + float(timeout_sec)
    last_progress_print = 0.0

    target_blocks = [
        int(rows[idx].get("block_number") or 0)
        for idx in eligible_indices
    ]

    highest_target_block = max(target_blocks) if target_blocks else 0

    while pending and time.monotonic() < t_deadline:
        finalized_head = _get_finalized_block_number(w3)
        now_mono = time.monotonic()

        if finalized_head > 0:
            observed_epoch = time.time()

            completed_now: List[int] = []

            for idx in pending:
                row = rows[idx]
                block_no = int(row.get("block_number") or 0)

                if finalized_head >= block_no:
                    receipt_epoch = float(
                        row.get("receipt_observed_epoch_sec", 0.0) or 0.0
                    )
                    start_epoch = float(
                        row.get("tx_start_epoch_sec", 0.0) or 0.0
                    )

                    row["finality_measured"] = True
                    row["finalized"] = True
                    row["finalized_head_block_number"] = int(finalized_head)
                    row["finality_observed_epoch_sec"] = float(observed_epoch)
                    row["receipt_to_finality_sec"] = float(
                        max(0.0, observed_epoch - receipt_epoch)
                    )

                    if start_epoch > 0.0:
                        row["tx_start_to_finality_sec"] = float(
                            max(0.0, observed_epoch - start_epoch)
                        )

                    completed_now.append(idx)

            for idx in completed_now:
                pending.discard(idx)

        # Print progress every ~15 seconds instead of appearing frozen.
        if (
                last_progress_print == 0.0
                or (now_mono - last_progress_print) >= 15.0
        ):
            elapsed = now_mono - t_start_wait
            remaining_timeout = max(0.0, t_deadline - now_mono)

            print(
                f"[FINALITY PROGRESS] "
                f"finalized_head={finalized_head} "
                f"highest_tx_block={highest_target_block} "
                f"blocks_remaining={max(0, highest_target_block - finalized_head)} "
                f"pending={len(pending)}/{len(eligible_indices)} "
                f"elapsed_sec={elapsed:.1f} "
                f"timeout_remaining_sec={remaining_timeout:.1f}"
            )

            last_progress_print = now_mono

        if pending:
            time.sleep(max(0.25, float(poll_interval_sec)))

    for idx in pending:
        rows[idx]["finality_measured"] = True
        rows[idx]["finalized"] = False

    return {
        "supported": True,
        "eligible_transactions": int(len(eligible_indices)),
        "finalized_transactions": int(len(eligible_indices) - len(pending)),
        "timed_out_transactions": int(len(pending)),
        "timeout_sec": int(timeout_sec),
        "poll_interval_sec": float(poll_interval_sec),
    }


# ------------------------------------------------------------
# Reviewer Comment 12: deployed-contract runtime fingerprint
# ------------------------------------------------------------
def _contract_runtime_fingerprint(w3, address: str) -> Dict[str, Any]:
    try:
        code = bytes(w3.eth.get_code(address))

        return {
            "address": str(address),
            "runtime_code_present": bool(len(code) > 0),
            "runtime_code_size_bytes": int(len(code)),
            "runtime_code_sha256": _sha256_bytes(code) if code else "",
        }

    except Exception as exc:
        return {
            "address": str(address),
            "runtime_code_present": False,
            "runtime_code_size_bytes": 0,
            "runtime_code_sha256": "",
            "error": str(exc),
        }

# ------------------------------------------------------------
# Solidity compilation (Verifier.sol + ProofRegistryV1.sol)
# ------------------------------------------------------------
DEFAULT_SOLC_VERSION = "0.8.20"
def _compile_with_solcx(source_code: str, solc_version: str = "") -> Dict[str, Any]:
    """
    Returns:
      {
        "contracts": {
           "<name>": {"abi": [...], "bytecode": "0x..."}
        }
      }
    ✅ Behavior:
    1) Ensures solc exists + selects it inside THIS process
    2) Tries normal solcx.compile_source() first
    3) If it fails with "Stack too deep", retries using compile_standard(viaIR=True)
    """
    try:
        import solcx  # type: ignore
    except Exception as exc:
        raise RuntimeError("Solidity compiler missing. Install: pip install py-solc-x") from exc
    # ------------------------------------------------------------
    # Decide which solc version to use
    # ------------------------------------------------------------
    use_ver = (str(solc_version or "").strip() or DEFAULT_SOLC_VERSION).strip()
    # ------------------------------------------------------------
    # Ensure solc is installed + selected for THIS run
    # ------------------------------------------------------------
    try:
        installed_versions = solcx.get_installed_solc_versions()
        installed_set = {str(v) for v in installed_versions} if isinstance(installed_versions, list) else set()

        if use_ver not in installed_set:
            solcx.install_solc(use_ver)
        solcx.set_solc_version(use_ver)
    except Exception:
        # If this fails, the compile step will throw a meaningful error below.
        pass
    # ------------------------------------------------------------
    # Helper: normalize output to {"contracts": {name: {abi, bytecode}}}
    # ------------------------------------------------------------
    def _normalize_out() -> Dict[str, Any]:
        return {"contracts": {}}
    # ------------------------------------------------------------
    # Attempt #1: Normal compilation (fast path)
    # ------------------------------------------------------------
    try:
        compiled = solcx.compile_source(
            source_code,
            output_values=["abi", "bin"],
            optimize=True,
        )
        out: Dict[str, Any] = _normalize_out()
        for k, v in compiled.items():
            name = str(k).split(":")[-1]
            abi = v.get("abi")
            bin_ = v.get("bin")
            if isinstance(abi, list) and isinstance(bin_, str) and bin_:
                out["contracts"][name] = {"abi": abi, "bytecode": "0x" + bin_}
        if not out["contracts"]:
            raise RuntimeError("solcx compile succeeded but produced no contracts")

        return out
    except Exception as exc:
        msg = str(exc)
        # Only fallback to viaIR when Stack too deep happens
        if "Stack too deep" not in msg:
            raise RuntimeError(f"solcx compile failed: {exc}") from exc
    # ------------------------------------------------------------
    # Attempt #2: viaIR compilation (fixes Stack too deep)
    # ------------------------------------------------------------
    try:
        input_json: Dict[str, Any] = {
            "language": "Solidity",
            "sources": {
                "Input.sol": {
                    "content": source_code
                }
            },
            "settings": {
                "optimizer": {
                    "enabled": True,
                    "runs": 200
                },
                "viaIR": True,
                "outputSelection": {
                    "*": {
                        "*": [
                            "abi",
                            "evm.bytecode.object"
                        ]
                    }
                }
            }
        }
        compiled_std = solcx.compile_standard(input_json)
        out: Dict[str, Any] = _normalize_out()
        contracts_block = compiled_std.get("contracts") or {}
        if not isinstance(contracts_block, dict):
            raise RuntimeError("compile_standard returned unexpected format: missing contracts")
        for src_name, src_contracts in contracts_block.items():
            if not isinstance(src_contracts, dict):
                continue
            for cname, cdata in src_contracts.items():
                if not isinstance(cdata, dict):
                    continue
                abi = cdata.get("abi")
                evm = cdata.get("evm") or {}
                bytecode_obj = ""
                if isinstance(evm, dict):
                    bytecode = evm.get("bytecode") or {}
                    if isinstance(bytecode, dict):
                        bytecode_obj = str(bytecode.get("object") or "")
                if isinstance(abi, list) and isinstance(bytecode_obj, str) and bytecode_obj:
                    out["contracts"][str(cname)] = {"abi": abi, "bytecode": "0x" + bytecode_obj}
        if not out["contracts"]:
            raise RuntimeError("viaIR compile succeeded but produced no contracts")
        return out
    except Exception as exc:
        raise RuntimeError(f"solcx viaIR compile failed: {exc}") from exc
def _pick_contract_by_method(compiled: Dict[str, Any], method_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Picks a contract whose ABI includes a function named method_name.
    """
    contracts = compiled.get("contracts") or {}
    if not isinstance(contracts, dict) or not contracts:
        raise RuntimeError("compiled contracts missing/empty")
    for name, meta in contracts.items():
        abi = meta.get("abi")
        if isinstance(abi, list):
            for entry in abi:
                if isinstance(entry, dict) and entry.get("type") == "function" and entry.get("name") == method_name:
                    return str(name), meta
    # fallback: pick last contract
    last_name = list(contracts.keys())[-1]
    return last_name, contracts[last_name]
def _compile_solidity_file(path: str, prefer_solc_version: str = "") -> Dict[str, Any]:
    src = Path(path).read_text(encoding="utf-8", errors="ignore")
    return _compile_with_solcx(src, solc_version=prefer_solc_version)
def _write_registry_sol_file(out_path: str, pub_len: int, pragma_hint: str = "") -> None:
    """
    Writes ProofRegistryV1.sol as a self-documenting artifact in onchain_export/.
    Solidity is kept broadly compatible (>=0.6.11 <0.9.0).
    """
    pragma_line = "pragma solidity >=0.6.11 <0.9.0;"
    if pragma_hint:
        # do not hard-force a caret version, keep compatible range
        pragma_line = "pragma solidity >=0.6.11 <0.9.0;"
    sol = f"""{pragma_line}
interface IVerifierFixedInput {{
    function verifyProof(
        uint[2] calldata a,
        uint[2][2] calldata b,
        uint[2] calldata c,
        uint[{pub_len}] calldata input
    ) external view returns (bool r);
}}
contract ProofRegistryV1 {{
    // Scope enum (0 = NONE, 1 = RSU, 2 = GLOBAL)
    uint8 public constant SCOPE_RSU = 1;
    uint8 public constant SCOPE_GLOBAL = 2;
    struct ProofRecordV1 {{
        uint8 scope;
        bytes32 runId;
        uint32 rsuId;
        uint32 roundIdx;
        bool verifiedOk;
        bytes32 proofSha256;
        bytes32 publicInputsSha256;
        bytes32 vkeySha256;
        bytes32 verifierSolSha256;
        bytes32 manifestSha256;
        address submitter;
        uint256 blockNumber;
        uint256 timestamp;
    }}
    mapping(bytes32 => ProofRecordV1) public proofRecords;
    mapping(bytes32 => uint256) public runSubmitted;
    mapping(bytes32 => uint256) public runVerifiedOk;
    mapping(bytes32 => bool) public runFinalized;
    mapping(bytes32 => bytes32) public globalProofKey;
    IVerifierFixedInput public rsuVerifier;
    IVerifierFixedInput public globalVerifier;
    event ProofVerifiedV1(
        bytes32 indexed proofKey,
        uint8 indexed scope,
        bytes32 indexed runId,
        uint32 rsuId,
        uint32 roundIdx,
        bool verifiedOk,
        bytes32 proofSha256,
        bytes32 publicInputsSha256,
        bytes32 vkeySha256,
        bytes32 verifierSolSha256,
        bytes32 manifestSha256,
        address submitter
    );
    event RunFinalizedV1(
        bytes32 indexed runId,
        bytes32 globalProofKey,
        bool globalVerifiedOk,
        uint256 totalSubmitted,
        uint256 totalVerifiedOk
    );
    constructor(address rsuVerifierAddr, address globalVerifierAddr) {{
        rsuVerifier = IVerifierFixedInput(rsuVerifierAddr);
        globalVerifier = IVerifierFixedInput(globalVerifierAddr);
    }}
    function _proofKey(
        uint8 scope,
        bytes32 runId,
        uint32 rsuId,
        uint32 roundIdx,
        bytes32 proofSha256,
        bytes32 publicInputsSha256,
        bytes32 vkeySha256,
        bytes32 verifierSolSha256,
        bytes32 manifestSha256
    ) internal pure returns (bytes32) {{
        return keccak256(
            abi.encodePacked(
                scope, runId, rsuId, roundIdx,
                proofSha256, publicInputsSha256,
                vkeySha256, verifierSolSha256, manifestSha256
            )
        );
    }}
    function submitAndVerifyRSU_V1(
        bytes32 runId,
        uint32 rsuId,
        uint32 roundIdx,
        uint[2] calldata a,
        uint[2][2] calldata b,
        uint[2] calldata c,
        uint[{pub_len}] calldata input,
        bytes32 proofSha256,
        bytes32 publicInputsSha256,
        bytes32 vkeySha256,
        bytes32 verifierSolSha256,
        bytes32 manifestSha256
    ) external returns (bytes32 proofKey, bool ok) {{
        proofKey = _proofKey(
            SCOPE_RSU, runId, rsuId, roundIdx,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256
        );
        require(proofRecords[proofKey].scope == 0, "already submitted");
        ok = rsuVerifier.verifyProof(a, b, c, input);
        proofRecords[proofKey] = ProofRecordV1(
            SCOPE_RSU, runId, rsuId, roundIdx, ok,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256,
            msg.sender, block.number, block.timestamp
        );
        runSubmitted[runId] += 1;
        if (ok) {{
            runVerifiedOk[runId] += 1;
        }}
        emit ProofVerifiedV1(
            proofKey, SCOPE_RSU, runId, rsuId, roundIdx, ok,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256,
            msg.sender
        );
        return (proofKey, ok);
    }}
    function submitAndVerifyGLOBAL_V1(
        bytes32 runId,
        uint32 roundIdx,
        uint[2] calldata a,
        uint[2][2] calldata b,
        uint[2] calldata c,
        uint[{pub_len}] calldata input,
        bytes32 proofSha256,
        bytes32 publicInputsSha256,
        bytes32 vkeySha256,
        bytes32 verifierSolSha256
    ) external returns (bytes32 proofKey, bool ok) {{
        bytes32 manifestSha256 = bytes32(0);
        proofKey = _proofKey(
            SCOPE_GLOBAL, runId, 0, roundIdx,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256
        );
        require(proofRecords[proofKey].scope == 0, "already submitted");
        ok = globalVerifier.verifyProof(a, b, c, input);
        proofRecords[proofKey] = ProofRecordV1(
            SCOPE_GLOBAL, runId, 0, roundIdx, ok,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256,
            msg.sender, block.number, block.timestamp
        );
        globalProofKey[runId] = proofKey;
        runSubmitted[runId] += 1;
        if (ok) {{
            runVerifiedOk[runId] += 1;
        }}
        emit ProofVerifiedV1(
            proofKey, SCOPE_GLOBAL, runId, 0, roundIdx, ok,
            proofSha256, publicInputsSha256, vkeySha256, verifierSolSha256, manifestSha256,
            msg.sender
        );
        return (proofKey, ok);
    }}
    function finalizeRunV1(bytes32 runId) external returns (bool globalOk) {{
        require(!runFinalized[runId], "already finalized");
        bytes32 gkey = globalProofKey[runId];
        require(gkey != bytes32(0), "missing global proof");
        globalOk = proofRecords[gkey].verifiedOk;
        runFinalized[runId] = true;
        emit RunFinalizedV1(
            runId, gkey, globalOk, runSubmitted[runId], runVerifiedOk[runId]
        );
        return globalOk;
    }}
}}
"""
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(sol, encoding="utf-8")
# ------------------------------------------------------------
# Deployment cache helpers
# ------------------------------------------------------------
def _load_cache_json(path: str) -> Dict[str, Any]:
    try:
        if _is_nonempty_file(path):
            obj = _load_json(path)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    return {}
def _save_cache_json(path: str, obj: Dict[str, Any]) -> None:
    _atomic_write_json(path, obj)
# ------------------------------------------------------------
# Deploy contracts
# ------------------------------------------------------------
def _deploy_contract(
    w3,
    account,
    abi,
    bytecode: str,
    constructor_args: list,
    nonce: int,
    chain_id: int
) -> Tuple[str, str, Dict[str, Any], int]:
    """
    Deploys contract, returns (tx_hash_hex, contract_address, receipt_dict, next_nonce)
    """
    t0 = time.perf_counter()

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract.constructor(*constructor_args).build_transaction({
        "from": account.address,
        "nonce": nonce,
        "chainId": chain_id,
    })
    t_build = time.perf_counter()

    # Gas + fees
    try:
        est = int(w3.eth.estimate_gas(tx))
        tx["gas"] = int(est * 12 // 10 + 50000)
    except Exception:
        tx["gas"] = 7_000_000

    # EIP-1559
    try:
        pending = w3.eth.get_block("pending")
        base_fee = 0
        try:
            base_fee = int(getattr(pending, "baseFeePerGas", 0) or 0)
        except Exception:
            base_fee = int(pending.get("baseFeePerGas", 0) or 0) if isinstance(pending, dict) else 0
        if base_fee <= 0:
            base_fee = int(w3.eth.gas_price)
        priority = int(w3.to_wei(2, "gwei"))
        tx["type"] = 2
        tx["maxPriorityFeePerGas"] = priority
        tx["maxFeePerGas"] = int(base_fee * 2 + priority)
    except Exception:
        pass

    signed = account.sign_transaction(tx)
    t_sign = time.perf_counter()

    raw_tx = _signed_raw_tx_bytes(signed)
    tx_hash = w3.eth.send_raw_transaction(raw_tx)
    t_send = time.perf_counter()

    tx_hash_hex = tx_hash.hex()
    tx_telemetry = {
        "build_latency_sec": float(t_build - t0),
        "sign_latency_sec": float(t_sign - t_build),
        "send_latency_sec": float(t_send - t_sign),
        "receipt_wait_latency_sec": 0.0,
        "tx_total_latency_sec": float(t_send - t0),
        "calldata_size_bytes": int(_tx_data_size_bytes(tx)),
        "signed_tx_size_bytes": int(len(raw_tx)),
    }

    if WAIT_FOR_RECEIPTS:
        receipt = _wait_receipt(w3, tx_hash, RECEIPT_TIMEOUT_SEC)
        t_receipt = time.perf_counter()

        receipt_dict = dict(receipt) if receipt is not None else {}
        tx_telemetry["receipt_wait_latency_sec"] = float(t_receipt - t_send)
        tx_telemetry["tx_total_latency_sec"] = float(t_receipt - t0)
        receipt_dict["_tx_telemetry"] = tx_telemetry

        addr = getattr(receipt, "contractAddress", None)
        if not addr and isinstance(receipt, dict):
            addr = receipt.get("contractAddress")
        if not addr:
            try:
                addr = receipt.get("contractAddress")  # AttributeDict supports get()
            except Exception:
                addr = None
        if not addr:
            raise RuntimeError(f"deployment tx mined but contractAddress empty: {tx_hash_hex}")
        return tx_hash_hex, str(addr), receipt_dict, nonce + 1

    raise RuntimeError("WAIT_FOR_RECEIPTS=False is not supported for deployment (need contract address)")
def _send_contract_tx(w3, account, fn_call, nonce: int, chain_id: int) -> Tuple[str, Optional[dict], int]:
    """
    Sends a contract function transaction, returns
    (tx_hash_hex, receipt_dict_or_none, next_nonce).

    Reviewer Comment 12:
      Also records local wall-clock epochs so receipt and consensus-finality
      latencies can be measured separately.
    """
    t0 = time.perf_counter()
    tx_start_epoch_sec = time.time()

    tx = fn_call.build_transaction({
        "from": account.address,
        "nonce": nonce,
        "chainId": chain_id,
    })
    t_build = time.perf_counter()

    try:
        est = int(w3.eth.estimate_gas(tx))
        tx["gas"] = int(est * 12 // 10 + 50000)
    except Exception:
        tx["gas"] = 9_000_000

    try:
        pending = w3.eth.get_block("pending")
        base_fee = 0
        try:
            base_fee = int(getattr(pending, "baseFeePerGas", 0) or 0)
        except Exception:
            base_fee = int(pending.get("baseFeePerGas", 0) or 0) if isinstance(pending, dict) else 0
        if base_fee <= 0:
            base_fee = int(w3.eth.gas_price)
        priority = int(w3.to_wei(2, "gwei"))
        tx["type"] = 2
        tx["maxPriorityFeePerGas"] = priority
        tx["maxFeePerGas"] = int(base_fee * 2 + priority)
    except Exception:
        pass

    signed = account.sign_transaction(tx)
    t_sign = time.perf_counter()

    raw_tx = _signed_raw_tx_bytes(signed)
    tx_hash = w3.eth.send_raw_transaction(raw_tx)
    t_send = time.perf_counter()
    broadcast_epoch_sec = time.time()

    tx_hash_hex = tx_hash.hex()
    tx_telemetry = {
        "build_latency_sec": float(t_build - t0),
        "sign_latency_sec": float(t_sign - t_build),
        "send_latency_sec": float(t_send - t_sign),
        "receipt_wait_latency_sec": 0.0,
        "tx_total_latency_sec": float(t_send - t0),

        "tx_start_epoch_sec": float(tx_start_epoch_sec),
        "broadcast_epoch_sec": float(broadcast_epoch_sec),
        "receipt_observed_epoch_sec": None,

        "calldata_size_bytes": int(_tx_data_size_bytes(tx)),
        "signed_tx_size_bytes": int(len(raw_tx)),
    }

    if WAIT_FOR_RECEIPTS:
        receipt = _wait_receipt(w3, tx_hash, RECEIPT_TIMEOUT_SEC)
        t_receipt = time.perf_counter()
        receipt_observed_epoch_sec = time.time()

        receipt_dict = dict(receipt)
        tx_telemetry["receipt_wait_latency_sec"] = float(t_receipt - t_send)
        tx_telemetry["tx_total_latency_sec"] = float(t_receipt - t0)
        tx_telemetry["receipt_observed_epoch_sec"] = float(
            receipt_observed_epoch_sec
        )

        receipt_dict["_tx_telemetry"] = tx_telemetry
        return tx_hash_hex, receipt_dict, nonce + 1

    return tx_hash_hex, None, nonce + 1
# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT_DIR, help="Compact security-evidence run root")
    ap.add_argument("--index", default=DEFAULT_INDEX_PATH, help="OnChainExportIndexV1 JSON path")
    ap.add_argument("--chain-id", default=str(DEFAULT_CHAIN_ID), help="Chain ID (default Sepolia 11155111)")
    args = ap.parse_args()
    # Load local .env if present under root_dir/onchain_export/.env
    root_dir = os.path.normpath(str(args.root))
    index_path = os.path.normpath(str(args.index))
    dotenv_path = os.path.join(root_dir, "onchain_export", ".env")
    _load_dotenv_simple(dotenv_path)
    # Derive root from index if index exists
    def _derive_root_from_index(index_path_: str) -> str:
        p = Path(index_path_).resolve()
        return str(p.parent.parent)
    if _is_nonempty_file(index_path):
        root_dir = _derive_root_from_index(index_path)
    else:
        index_path = os.path.join(root_dir, "onchain_export", "latest_run_index_v1.json")
    if not _is_nonempty_file(index_path):
        print(f"[FATAL] index file missing: {index_path}")
        return 2
    idx = _load_json(index_path)
    # ------------------------------------------------------------
    # Run identity (thesis-grade):
    #   index_sha256  = stable identity of the exported index file
    #   run_id_bytes32 = unique per execution (prevents "already submitted")
    # Scheme:
    #   run_id_sha256 = sha256("ONCHAIN_RUN_ID_V1|" + index_bytes + "|" + salt32)
    # ------------------------------------------------------------
    index_bytes = Path(index_path).read_bytes()
    # Stable fingerprint of the exported index (same if file is the same)
    index_sha256 = _sha256_bytes(index_bytes)
    # Unique run salt (32 bytes) -> new run_id each execution
    run_salt_bytes = os.urandom(32)
    run_salt_hex = run_salt_bytes.hex()
    run_id_material = b"ONCHAIN_RUN_ID_V1|" + index_bytes + b"|" + run_salt_bytes
    run_id_sha256 = _sha256_bytes(run_id_material)
    run_id_bytes32 = "0x" + run_id_sha256  # bytes32 hex string
    # Topology
    topology = idx.get("topology") or {}
    num_rounds = _safe_int(topology.get("num_rounds"), _safe_int(idx.get("num_rounds"), 0))
    num_rsus = _safe_int(topology.get("num_rsus"), _safe_int(idx.get("num_rsus"), 0))
    rsu_vehicle_ids = topology.get("rsu_vehicle_ids") or {}
    rsu_ids_sorted = topology.get("rsu_ids_sorted") or sorted(
        [_safe_int(k, 0) for k in rsu_vehicle_ids.keys() if str(k).strip()]
    )
    def _as_int_list(xs: Any) -> List[int]:
        if not isinstance(xs, list):
            return []
        out: List[int] = []
        for x in xs:
            v = _safe_int(x, 0)
            if v > 0:
                out.append(v)
        return sorted(set(out))
    rsu_ids_sorted = _as_int_list(rsu_ids_sorted)
    # Circuits
    circuits_raw = idx.get("circuits") or []
    rsu_circuit = ""
    global_circuit = ""
    def _get_circuit_name(ent: Any) -> str:
        if not isinstance(ent, dict):
            return ""
        return str(
            ent.get("circuit_name")
            or ent.get("circuit")
            or ent.get("name")
            or ""
        ).strip()
    if isinstance(circuits_raw, list):
        rsu_ent = None
        glob_ent = None
        for ent in circuits_raw:
            name = _get_circuit_name(ent)
            if not name:
                continue
            up = name.upper()
            if rsu_ent is None and ("AGGRSU" in up or "RSU" in up):
                rsu_ent = ent
            if glob_ent is None and ("AGGGLOBAL" in up or "GLOBAL" in up):
                glob_ent = ent
        # Safe fallbacks: first/second entries if patterns missing
        if rsu_ent is None and len(circuits_raw) >= 1 and isinstance(circuits_raw[0], dict):
            rsu_ent = circuits_raw[0]
        if glob_ent is None:
            if len(circuits_raw) >= 2 and isinstance(circuits_raw[1], dict):
                glob_ent = circuits_raw[1]
            elif rsu_ent is not None:
                glob_ent = rsu_ent
        rsu_circuit = _get_circuit_name(rsu_ent)
        global_circuit = _get_circuit_name(glob_ent)
    elif isinstance(circuits_raw, dict):
        # Supports older index formats
        rsu_circuit = str(circuits_raw.get("rsu") or circuits_raw.get("rsu_circuit") or "").strip()
        global_circuit = str(circuits_raw.get("global") or circuits_raw.get("global_circuit") or "").strip()
    # Final hard fallback only if index truly missing circuits
    if not rsu_circuit:
        rsu_circuit = "AggRSU_AnchorSum_M64_K2_RM1_S100000_RC0_RB2"
    if not global_circuit:
        global_circuit = "AggGlobal_AnchorSum_M64_K2_RM2_S100000_RC0_RB2"
    global_round = idx.get("global_round") or {}
    global_round_idx = _safe_int(global_round.get("round_idx"), num_rounds)
    used_rsu_ids = _as_int_list(global_round.get("used_rsu_ids") or rsu_ids_sorted)
    print("\n=======================================")
    print(" ON-CHAIN RSU + GLOBAL + SSI VERIFIER ")
    print("=======================================")
    print(f"root_dir               : {root_dir}")
    print(f"index_path             : {index_path}")
    print(f"index_sha256           : 0x{index_sha256}")
    print(f"run_salt_hex           : 0x{run_salt_hex}")
    print(f"run_id_bytes32         : {run_id_bytes32}")
    print(f"num_rounds             : {num_rounds}")
    print(f"num_rsus               : {num_rsus}")
    print(f"rsu_ids_sorted         : {list(rsu_ids_sorted)}")
    print(f"GLOBAL round_idx       : {global_round_idx}")
    print(f"GLOBAL used_rsu_ids    : {list(used_rsu_ids)}")
    print(f"RSU circuit            : {rsu_circuit}")
    print(f"GLOBAL circuit         : {global_circuit}")
    print(f"SUBMIT_GLOBAL_ONLY     : {SUBMIT_GLOBAL_ONLY}")
    print(f"DO_FINALIZE_RUN        : {DO_FINALIZE_RUN}")
    print("")
    # Web3 init
    Web3 = _require_web3()
    # Read credentials only after the optional local .env has been loaded.
    # Environment variables prevent RPC/private-key material from leaking into
    # command history or being embedded in the publication source.
    rpc_url = _get_env_first(DEFAULT_RPC_ENV_KEYS)
    privkey = _get_env_first(DEFAULT_PRIVKEY_ENV_KEYS)
    # normalize private key to 0x-prefixed hex
    if privkey and isinstance(privkey, str) and not privkey.startswith("0x"):
        privkey = "0x" + privkey
    chain_id = _safe_int(args.chain_id, DEFAULT_CHAIN_ID)
    if not rpc_url:
        print("[FATAL] Missing RPC URL. Set SEPOLIA_RPC_URL (or RPC_URL/WEB3_RPC_URL).")
        return 3
    if not privkey:
        print("[FATAL] Missing private key. Set SEPOLIA_PRIVATE_KEY (or another supported private-key environment variable).")
        return 3
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
    _inject_poa_middleware_if_available(w3)
    if not _web3_is_connected(w3):
        print("[FATAL] Could not connect to RPC.")
        return 4
    try:
        net_chain_id = int(w3.eth.chain_id)
    except Exception as exc:
        print(f"[FATAL] Could not read RPC chain ID: {exc}")
        return 4
    if net_chain_id != int(chain_id):
        print(f"[FATAL] Chain ID mismatch: RPC={net_chain_id} expected={chain_id}")
        return 4
    account = Account.from_key(privkey)

    # Optional sender lock. Set SEPOLIA_EXPECTED_ADDRESS when reproducing a
    # specific historical sender; leave unset for an independent verifier using
    # their own funded Sepolia account.
    if EXPECTED_SENDER_ADDRESS:
        try:
            expected_addr = Web3.to_checksum_address(EXPECTED_SENDER_ADDRESS)
        except Exception as exc:
            print(f"[FATAL] Invalid SEPOLIA_EXPECTED_ADDRESS: {exc}")
            return 99
        if account.address.lower() != expected_addr.lower():
            print(
                f"[FATAL] Wallet mismatch: derived={account.address} "
                f"expected={expected_addr}"
            )
            return 99
    print(f"[WALLET] address={account.address}")
    nonce = _get_nonce(w3, account.address)
    print(f"[WALLET] starting nonce={nonce}")
    # ------------------------------
    # ETH Balance + Fee Tracking
    # ------------------------------
    balance_start_wei = _get_balance_wei(w3, account.address)
    print(f"[WALLET] balance_start_eth={_fmt_eth(w3, balance_start_wei)}")
    total_fee_wei = 0
    fee_breakdown: List[Dict[str, Any]] = []
    # Prepare onchain_export paths
    onchain_export_dir = os.path.join(root_dir, "onchain_export")
    Path(onchain_export_dir).mkdir(parents=True, exist_ok=True)
    # Resolve verifier.sol (RSU + GLOBAL) and compile/deploy/reuse
    rsu_rounds_map = idx.get("rsu_rounds") or {}
    def _first_int_key(d: Any, default: int) -> int:
        if not isinstance(d, dict) or not d:
            return default
        vals = []
        for k in d.keys():
            v = _safe_int(k, 0)
            if v > 0:
                vals.append(v)
        return min(vals) if vals else default
    first_rsu_id = _first_int_key(rsu_rounds_map, default=1)
    first_round_map = ((rsu_rounds_map.get(str(first_rsu_id)) or {}).get("rounds") or {})
    first_round_idx = _first_int_key(first_round_map, default=1)
    rsu_pins_any = extract_expected_pins_from_index(
        idx=idx,
        scope="rsu",
        rsu_id=int(first_rsu_id),
        round_idx=int(first_round_idx),
    )
    rsu_verifier_sol, _, _ = _resolve_verifier_sol_strict_with_evidence(root_dir, rsu_circuit, rsu_pins_any)
    global_pins_any = extract_expected_pins_from_index(idx, scope="global", rsu_id=0, round_idx=global_round_idx)
    global_verifier_sol, _, _ = _resolve_verifier_sol_strict_with_evidence(root_dir, global_circuit, global_pins_any)
    if not _is_nonempty_file(rsu_verifier_sol):
        print(f"[FATAL] RSU verifier.sol missing: {rsu_verifier_sol!r}")
        return 5
    if not _is_nonempty_file(global_verifier_sol):
        print(f"[FATAL] GLOBAL verifier.sol missing: {global_verifier_sol!r}")
        return 5
    # Determine expected pub input length from verifier.sol (should be 10)
    rsu_pub_len = parse_expected_public_len_from_verifier_sol(rsu_verifier_sol)
    glob_pub_len = parse_expected_public_len_from_verifier_sol(global_verifier_sol)
    # REQUIRED: derive the calldata contract from both generated verifiers.
    # Never silently assume a public-input count: a wrong fixed array size would
    # compile a registry that does not represent the retained proof contract.
    if rsu_pub_len <= 0:
        print("[FATAL] Could not parse RSU verifier public-input length.")
        return 55
    if glob_pub_len <= 0:
        print("[FATAL] Could not parse GLOBAL verifier public-input length.")
        return 55
    if rsu_pub_len != glob_pub_len:
        print(
            f"[FATAL] RSU pub_len ({rsu_pub_len}) != "
            f"GLOBAL pub_len ({glob_pub_len})"
        )
        return 55
    pub_len = int(rsu_pub_len)
    # Load deployment caches
    ver_cache_path = os.path.join(onchain_export_dir, DEPLOYED_VERIFIERS_CACHE)
    reg_cache_path = os.path.join(onchain_export_dir, DEPLOYED_REGISTRY_CACHE)
    ver_cache = _load_cache_json(ver_cache_path)
    reg_cache = _load_cache_json(reg_cache_path)
   # Compile verifiers if needed
    def _need_redeploy_verifier(cache_obj: dict, scope_key: str, want_verifier_sha: str, want_vkey_sha: str) -> bool:
        ent = cache_obj.get(scope_key) or {}
        addr = str(ent.get("address") or "").strip()
        if not addr or not _is_contract_address(w3, addr):
            return True
        if str(ent.get("verifier_sol_sha256") or "").strip().lower() != str(want_verifier_sha or "").strip().lower():
            return True
        if str(ent.get("vkey_sha256") or "").strip().lower() != str(want_vkey_sha or "").strip().lower():
            return True
        return False
    rsu_verifier_sha_expected = str(rsu_pins_any.verifier_sol_sha256 or "").strip()
    rsu_vkey_sha_expected = str(rsu_pins_any.vkey_sha256 or "").strip()
    global_verifier_sha_expected = str(global_pins_any.verifier_sol_sha256 or "").strip()
    global_vkey_sha_expected = str(global_pins_any.vkey_sha256 or "").strip()
    # RSU verifier
    if _need_redeploy_verifier(ver_cache, "rsu", rsu_verifier_sha_expected, rsu_vkey_sha_expected):
        print("\n[DEPLOY] RSU verifier contract (compiling Verifier.sol)...")
        compiled = _compile_solidity_file(rsu_verifier_sol, prefer_solc_version="")
        _, meta = _pick_contract_by_method(compiled, "verifyProof")
        txh, addr, dep_rcpt, nonce = _deploy_contract(
            w3=w3,
            account=account,
            abi=meta["abi"],
            bytecode=meta["bytecode"],
            constructor_args=[],
            nonce=nonce,
            chain_id=chain_id,
        )
        fee_wei = _receipt_fee_wei(dep_rcpt)
        total_fee_wei += fee_wei
        fee_breakdown.append({
            "kind": "DEPLOY_VERIFIER_RSU",
            "tx_hash": txh,
            "gas_used": int(dep_rcpt.get("gasUsed", 0) or 0),
            "effective_gas_price": int(dep_rcpt.get("effectiveGasPrice", dep_rcpt.get("gasPrice", 0)) or 0),
            "fee_wei": str(fee_wei),
            "fee_eth": _fmt_eth(w3, fee_wei),
            "calldata_size_bytes": int((dep_rcpt or {}).get("_tx_telemetry", {}).get("calldata_size_bytes", 0)),
            "signed_tx_size_bytes": int((dep_rcpt or {}).get("_tx_telemetry", {}).get("signed_tx_size_bytes", 0)),
            "build_latency_sec": float((dep_rcpt or {}).get("_tx_telemetry", {}).get("build_latency_sec", 0.0)),
            "sign_latency_sec": float((dep_rcpt or {}).get("_tx_telemetry", {}).get("sign_latency_sec", 0.0)),
            "send_latency_sec": float((dep_rcpt or {}).get("_tx_telemetry", {}).get("send_latency_sec", 0.0)),
            "receipt_wait_latency_sec": float((dep_rcpt or {}).get("_tx_telemetry", {}).get("receipt_wait_latency_sec", 0.0)),
            "tx_total_latency_sec": float((dep_rcpt or {}).get("_tx_telemetry", {}).get("tx_total_latency_sec", 0.0)),
        })
        print(f"[DEPLOY] RSU verifier deployed: {addr} tx={txh} fee_eth={_fmt_eth(w3, fee_wei)}")
        ver_cache["rsu"] = {
            "address": addr,
            "deploy_tx": txh,
            "verifier_sol_sha256": rsu_verifier_sha_expected or _sha256_file(rsu_verifier_sol),
            "vkey_sha256": rsu_vkey_sha_expected,
            "circuit_name": rsu_circuit,
            "chain_id": chain_id,
            "deployer": account.address,
            "updated_utc": _utc_now_iso(),
        }
        _save_cache_json(ver_cache_path, ver_cache)
    else:
        print("\n[REUSE] RSU verifier contract from cache")
        print(f"        address={ver_cache.get('rsu', {}).get('address')}")
    # GLOBAL verifier
    if _need_redeploy_verifier(ver_cache, "global", global_verifier_sha_expected, global_vkey_sha_expected):
        print("\n[DEPLOY] GLOBAL verifier contract (compiling Verifier.sol)...")
        compiled = _compile_solidity_file(global_verifier_sol, prefer_solc_version="")
        _, meta = _pick_contract_by_method(compiled, "verifyProof")
        txh, addr, dep_rcpt, nonce = _deploy_contract(
            w3=w3,
            account=account,
            abi=meta["abi"],
            bytecode=meta["bytecode"],
            constructor_args=[],
            nonce=nonce,
            chain_id=chain_id,
        )
        fee_wei = _receipt_fee_wei(dep_rcpt)
        total_fee_wei += int(fee_wei)
        fee_breakdown.append({
            "kind": "DEPLOY_VERIFIER_GLOBAL",
            "tx_hash": txh,
            "gas_used": int(dep_rcpt.get("gasUsed", 0) or 0),
            "effective_gas_price": int(dep_rcpt.get("effectiveGasPrice", dep_rcpt.get("gasPrice", 0)) or 0),
            "fee_wei": str(fee_wei),
            "fee_eth": _fmt_eth(w3, int(fee_wei)),
        })
        print(f"[DEPLOY] GLOBAL verifier deployed: {addr} tx={txh} fee_eth={_fmt_eth(w3, int(fee_wei))}")
        ver_cache["global"] = {
            "address": addr,
            "deploy_tx": txh,
            "verifier_sol_sha256": global_verifier_sha_expected or _sha256_file(global_verifier_sol),
            "vkey_sha256": global_vkey_sha_expected,
            "circuit_name": global_circuit,
            "chain_id": chain_id,
            "deployer": account.address,
            "updated_utc": _utc_now_iso(),
        }
        _save_cache_json(ver_cache_path, ver_cache)
    else:
        print("\n[REUSE] GLOBAL verifier contract from cache")
        print(f"        address={ver_cache.get('global', {}).get('address')}")
    rsu_verifier_addr = str((ver_cache.get("rsu") or {}).get("address") or "").strip()
    global_verifier_addr = str((ver_cache.get("global") or {}).get("address") or "").strip()
    if not _is_contract_address(w3, rsu_verifier_addr):
        print("[FATAL] RSU verifier address invalid / no code.")
        return 6
    if not _is_contract_address(w3, global_verifier_addr):
        print("[FATAL] GLOBAL verifier address invalid / no code.")
        return 6
    # Deploy/reuse ProofRegistryV1
    def _need_redeploy_registry(cache_obj: dict) -> bool:
        addr = str(cache_obj.get("address") or "").strip()
        if not addr or not _is_contract_address(w3, addr):
            return True
        if str(cache_obj.get("rsu_verifier") or "").strip().lower() != rsu_verifier_addr.lower():
            return True
        if str(cache_obj.get("global_verifier") or "").strip().lower() != global_verifier_addr.lower():
            return True
        return False
    registry_sol_path = os.path.join(onchain_export_dir, "ProofRegistryV1.sol")
    _write_registry_sol_file(registry_sol_path, pub_len=pub_len, pragma_hint="")
    if _need_redeploy_registry(reg_cache):
        print("\n[DEPLOY] ProofRegistryV1 (compiling + deploying)...")
        compiled = _compile_solidity_file(registry_sol_path, prefer_solc_version="")
        # Pick contract by name if present
        contracts = compiled.get("contracts") or {}
        if "ProofRegistryV1" in contracts:
            meta = contracts["ProofRegistryV1"]
        else:
            _, meta = _pick_contract_by_method(compiled, "submitAndVerifyRSU_V1")
        txh, addr, dep_rcpt, nonce = _deploy_contract(
            w3=w3,
            account=account,
            abi=meta["abi"],
            bytecode=meta["bytecode"],
            constructor_args=[rsu_verifier_addr, global_verifier_addr],
            nonce=nonce,
            chain_id=chain_id,
        )
        fee_wei = _receipt_fee_wei(dep_rcpt)
        total_fee_wei += int(fee_wei)
        fee_breakdown.append({
            "kind": "DEPLOY_REGISTRY",
            "tx_hash": txh,
            "gas_used": int(dep_rcpt.get("gasUsed", 0) or 0),
            "effective_gas_price": int(dep_rcpt.get("effectiveGasPrice", dep_rcpt.get("gasPrice", 0)) or 0),
            "fee_wei": str(fee_wei),
            "fee_eth": _fmt_eth(w3, int(fee_wei)),
        })
        print(f"[DEPLOY] ProofRegistryV1 deployed: {addr} tx={txh} fee_eth={_fmt_eth(w3, int(fee_wei))}")
        reg_cache = {
            "address": addr,
            "deploy_tx": txh,
            "rsu_verifier": rsu_verifier_addr,
            "global_verifier": global_verifier_addr,
            "chain_id": chain_id,
            "deployer": account.address,
            "updated_utc": _utc_now_iso(),
        }
        _save_cache_json(reg_cache_path, reg_cache)
    else:
        print("\n[REUSE] ProofRegistryV1 from cache")
        print(f"        address={reg_cache.get('address')}")
    registry_addr = str(reg_cache.get("address") or "").strip()
    if not _is_contract_address(w3, registry_addr):
        print("[FATAL] ProofRegistryV1 address invalid / no code.")
        return 7
    # Load registry ABI from compilation for event decoding
    compiled_reg = _compile_solidity_file(registry_sol_path, prefer_solc_version="")
    reg_contract_meta = (compiled_reg.get("contracts") or {}).get("ProofRegistryV1")
    if not reg_contract_meta:
        _, reg_contract_meta = _pick_contract_by_method(compiled_reg, "submitAndVerifyRSU_V1")
    registry_addr_checksum = Web3.to_checksum_address(registry_addr)
    registry = w3.eth.contract(address=registry_addr_checksum, abi=reg_contract_meta["abi"])
    # Build submission jobs from index
    jobs: List[Dict[str, Any]] = []
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
            jobs.append({"scope": "RSU", "rsu_id": rid, "round_idx": r})
    jobs.append({"scope": "GLOBAL", "rsu_id": 0, "round_idx": global_round_idx})
    # Deterministic ordering: RSU first, then GLOBAL, by (rsu_id, round_idx)
    def _job_key(j: dict):
        scope = str(j.get("scope") or "").upper()
        scope_ord = 0 if scope == "RSU" else 1
        return (scope_ord, int(j.get("rsu_id") or 0), int(j.get("round_idx") or 0))
    jobs = sorted(jobs, key=_job_key)
    # On-chain submissions
    submissions: List[Dict[str, Any]] = []
    print("\n------------------------------")
    print("ON-CHAIN SUBMISSIONS")
    print("------------------------------")
    for j in jobs:
        scope = str(j["scope"]).upper()
        rsu_id = int(j["rsu_id"])
        rnd = int(j["round_idx"])
        if SUBMIT_GLOBAL_ONLY and scope == "RSU":
            continue
        pins = extract_expected_pins_from_index(idx, scope=("rsu" if scope == "RSU" else "global"), rsu_id=rsu_id, round_idx=rnd)
        if scope == "RSU":
            bundle = resolve_rsu_bundle(root_dir, rsu_circuit, rsu_id, rnd, pins=pins)
        else:
            bundle = resolve_global_bundle(root_dir, global_circuit, rnd, pins=pins)
        prep = prepare_bundle_for_onchain(root_dir=root_dir, bundle=bundle, pins=pins)
        if not prep.get("ok"):
            print(f"[SKIP] {scope} rsu={rsu_id} round={rnd} -> NOT SAFE TO SUBMIT (pins/bounds/files)")
            for e in prep.get("errors", [])[:6]:
                print(f"       - {e}")
            submissions.append({
                "scope": scope,
                "rsu_id": rsu_id,
                "round_idx": rnd,
                "submitted": False,
                "errors": prep.get("errors", []),
                "warnings": prep.get("warnings", []),
                "prep": prep,
            })
            continue
        proof_json_size_bytes = _file_size_bytes(prep.get("proof_json", ""))
        public_inputs_v1_json_size_bytes = _file_size_bytes(prep.get("public_inputs_v1_json", ""))
        verifier_sol_size_bytes = _file_size_bytes(prep.get("verifier_sol", ""))
        vkey_json_size_bytes = _file_size_bytes(prep.get("vkey_json", ""))
        manifest_json_size_bytes = _file_size_bytes(prep.get("manifest_json", ""))

        solargs = prep.get("solidity_args") or {}
        a = solargs.get("a")
        b = solargs.get("b")
        c = solargs.get("c")
        inp = solargs.get("input")
        sha = prep.get("sha256") or {}
        proof_sha = _bytes32_from_hex_sha256(sha.get("proof_sha256", ""))
        pub_sha = _bytes32_from_hex_sha256(sha.get("public_inputs_v1_sha256", ""))
        vkey_sha = _bytes32_from_hex_sha256(sha.get("vkey_sha256", ""))
        ver_sha = _bytes32_from_hex_sha256(sha.get("verifier_sol_sha256", ""))
        manifest_sha = "0x" + ("00" * 32)
        if scope == "RSU" and sha.get("manifest_sha256"):
            try:
                manifest_sha = _bytes32_from_hex_sha256(sha.get("manifest_sha256", ""))
            except Exception:
                manifest_sha = "0x" + ("00" * 32)
        try:
            if scope == "RSU":
                fn = registry.functions.submitAndVerifyRSU_V1(
                    run_id_bytes32,
                    int(rsu_id),
                    int(rnd),
                    a, b, c, inp,
                    proof_sha,
                    pub_sha,
                    vkey_sha,
                    ver_sha,
                    manifest_sha,
                )
            else:
                fn = registry.functions.submitAndVerifyGLOBAL_V1(
                    run_id_bytes32,
                    int(rnd),
                    a, b, c, inp,
                    proof_sha,
                    pub_sha,
                    vkey_sha,
                    ver_sha,
                )
            txh, receipt, nonce = _send_contract_tx(w3, account, fn, nonce=nonce, chain_id=chain_id)
            mined = (receipt is not None)
            status = int(receipt.get("status", 0)) if receipt else None
            block_no = int(receipt.get("blockNumber", 0)) if receipt else None
            gas_used = int(receipt.get("gasUsed", 0)) if receipt else None
            # ✅ Tx fee tracking (per proof submission)
            fee_wei = _receipt_fee_wei(receipt) if receipt else 0
            total_fee_wei += int(fee_wei)
            tx_telemetry = dict((receipt or {}).get("_tx_telemetry", {})) if receipt else {}

            fee_breakdown.append({
                "kind": f"SUBMIT_{scope}",
                "scope": scope,
                "rsu_id": rsu_id,
                "round_idx": rnd,
                "tx_hash": txh,
                "gas_used": gas_used,
                "effective_gas_price": int(
                    receipt.get("effectiveGasPrice", receipt.get("gasPrice", 0)) or 0) if receipt else 0,
                "fee_wei": str(fee_wei),
                "fee_eth": _fmt_eth(w3, int(fee_wei)),
                "calldata_size_bytes": int(tx_telemetry.get("calldata_size_bytes", 0)),
                "signed_tx_size_bytes": int(tx_telemetry.get("signed_tx_size_bytes", 0)),
                "build_latency_sec": float(tx_telemetry.get("build_latency_sec", 0.0)),
                "sign_latency_sec": float(tx_telemetry.get("sign_latency_sec", 0.0)),
                "send_latency_sec": float(tx_telemetry.get("send_latency_sec", 0.0)),
                "receipt_wait_latency_sec": float(tx_telemetry.get("receipt_wait_latency_sec", 0.0)),
                "tx_total_latency_sec": float(tx_telemetry.get("tx_total_latency_sec", 0.0)),
            })
            # Decode events if possible
            evs: List[Dict[str, Any]] = []
            verified_ok = None
            proof_key = None
            try:
                processed = _process_event_receipt_any(registry.events.ProofVerifiedV1(), receipt) if receipt else []
                for ev in processed:
                    args_ = dict(ev.get("args") or {})
                    evs.append({
                        "event": "ProofVerifiedV1",
                        "proofKey": args_.get("proofKey"),
                        "scope": int(args_.get("scope")) if args_.get("scope") is not None else None,
                        "runId": args_.get("runId"),
                        "rsuId": int(args_.get("rsuId")) if args_.get("rsuId") is not None else None,
                        "roundIdx": int(args_.get("roundIdx")) if args_.get("roundIdx") is not None else None,
                        "verifiedOk": bool(args_.get("verifiedOk")) if args_.get("verifiedOk") is not None else None,
                        "submitter": args_.get("submitter"),
                    })
                    verified_ok = bool(args_.get("verifiedOk")) if args_.get("verifiedOk") is not None else verified_ok
                    proof_key = args_.get("proofKey") or proof_key
            except Exception:
                pass
            print(
                f"[TX] {scope} rsu={rsu_id} round={rnd} -> tx={txh} status={status} verified={verified_ok} fee_eth={_fmt_eth(w3, int(fee_wei))}"
            )
            # ------------------------------------------------------------
            # ✅ Thesis-grade on-chain evidence:
            # 1) Print decoded ProofVerifiedV1 event fields
            # 2) Read-back the stored proofRecords(proofKey) from contract storage
            # ------------------------------------------------------------
            if evs:
                for ev in evs:
                    if ev.get("event") == "ProofVerifiedV1":
                        print(
                            f"[EVENT] ProofVerifiedV1 scope={ev.get('scope')} runId={ev.get('runId')} "
                            f"rsuId={ev.get('rsuId')} roundIdx={ev.get('roundIdx')} "
                            f"verifiedOk={ev.get('verifiedOk')} proofKey={ev.get('proofKey')} submitter={ev.get('submitter')}"
                        )
            # Read-back from chain storage (strongest proof)
            onchain_record = None
            block_ts = None
            try:
                if proof_key is not None:
                    onchain_record = registry.functions.proofRecords(proof_key).call()
            except Exception:
                onchain_record = None
            if onchain_record is not None:
                try:
                    # Web3 may return a dict-like or tuple-like struct
                    rec_scope = onchain_record[0] if isinstance(onchain_record, (list, tuple)) else onchain_record.get(
                        "scope")
                    rec_verified = onchain_record[4] if isinstance(onchain_record,
                                                                   (list, tuple)) else onchain_record.get("verifiedOk")
                    rec_block = onchain_record[11] if isinstance(onchain_record, (list, tuple)) else onchain_record.get(
                        "blockNumber")
                    rec_ts = onchain_record[12] if isinstance(onchain_record, (list, tuple)) else onchain_record.get(
                        "timestamp")
                    block_ts = int(rec_ts) if rec_ts is not None else None
                    print(
                        f"[ONCHAIN] proofRecords[proofKey] => scope={rec_scope} verifiedOk={rec_verified} "
                        f"blockNumber={rec_block} timestamp={rec_ts}"
                    )
                except Exception:
                    print(f"[ONCHAIN] proofRecords[proofKey] fetched (raw) => {onchain_record!r}")
            else:
                if status == 1 and verified_ok is True:
                    print("[WARNING] Event says verifiedOk=True, but proofRecords read-back failed (RPC/ABI issue?)")
                if block_no is not None:
                    try:
                        blk = w3.eth.get_block(block_no)
                        if isinstance(blk, dict):
                            block_ts = int(blk.get("timestamp", 0) or 0)
                        else:
                            block_ts = int(getattr(blk, "timestamp", 0) or 0)
                    except Exception:
                        block_ts = None
            submissions.append({
                "scope": scope,
                "rsu_id": rsu_id,
                "round_idx": rnd,
                "submitted": True,
                "tx_hash": txh,
                "mined": mined,
                "status": status,
                "block_number": block_no,
                "block_timestamp": block_ts,
                "gas_used": gas_used,
                "tx_fee_wei": str(fee_wei),
                "tx_fee_eth": _fmt_eth(w3, int(fee_wei)),
                "verified_ok": verified_ok,
                "proof_key": proof_key,
                "proof_json_size_bytes": int(proof_json_size_bytes),
                "public_inputs_v1_json_size_bytes": int(public_inputs_v1_json_size_bytes),
                "verifier_sol_size_bytes": int(verifier_sol_size_bytes),
                "vkey_json_size_bytes": int(vkey_json_size_bytes),
                "manifest_json_size_bytes": int(manifest_json_size_bytes),
                "calldata_size_bytes": int(tx_telemetry.get("calldata_size_bytes", 0)),
                "signed_tx_size_bytes": int(tx_telemetry.get("signed_tx_size_bytes", 0)),
                "build_latency_sec": float(tx_telemetry.get("build_latency_sec", 0.0)),
                "sign_latency_sec": float(tx_telemetry.get("sign_latency_sec", 0.0)),
                "send_latency_sec": float(tx_telemetry.get("send_latency_sec", 0.0)),
                "receipt_wait_latency_sec": float(
                    tx_telemetry.get("receipt_wait_latency_sec", 0.0)
                ),
                "tx_total_latency_sec": float(
                    tx_telemetry.get("tx_total_latency_sec", 0.0)
                ),

                "tx_start_epoch_sec": float(
                    tx_telemetry.get("tx_start_epoch_sec", 0.0) or 0.0
                ),
                "broadcast_epoch_sec": float(
                    tx_telemetry.get("broadcast_epoch_sec", 0.0) or 0.0
                ),
                "receipt_observed_epoch_sec": float(
                    tx_telemetry.get("receipt_observed_epoch_sec", 0.0) or 0.0
                ),

                "events": evs,
                "prep": prep,
            })
        except Exception as exc:
            print(f"[FAIL] {scope} rsu={rsu_id} round={rnd} -> exception: {exc}")
            submissions.append({
                "scope": scope,
                "rsu_id": rsu_id,
                "round_idx": rnd,
                "submitted": False,
                "errors": [str(exc)],
                "prep": prep,
            })
            continue
    # Finalize run
    finalize_tx: Dict[str, Any] = {}
    if DO_FINALIZE_RUN:
        try:
            fn = registry.functions.finalizeRunV1(run_id_bytes32)
            txh, receipt, nonce = _send_contract_tx(w3, account, fn, nonce=nonce, chain_id=chain_id)
            status = int(receipt.get("status", 0)) if receipt else None
            block_no = int(receipt.get("blockNumber", 0)) if receipt else None
            block_ts = None
            if block_no is not None:
                try:
                    blk = w3.eth.get_block(block_no)
                    if isinstance(blk, dict):
                        block_ts = int(blk.get("timestamp", 0) or 0)
                    else:
                        block_ts = int(getattr(blk, "timestamp", 0) or 0)
                except Exception:
                    block_ts = None
            gas_used = int(receipt.get("gasUsed", 0)) if receipt else None
            # ✅ Tx fee tracking (finalizeRunV1)
            fee_wei = _receipt_fee_wei(receipt) if receipt else 0
            total_fee_wei += int(fee_wei)
            tx_telemetry = dict((receipt or {}).get("_tx_telemetry", {})) if receipt else {}

            fee_breakdown.append({
                "kind": "FINALIZE_RUN",
                "tx_hash": txh,
                "gas_used": gas_used,
                "effective_gas_price": int(
                    receipt.get("effectiveGasPrice", receipt.get("gasPrice", 0)) or 0) if receipt else 0,
                "fee_wei": str(fee_wei),
                "fee_eth": _fmt_eth(w3, int(fee_wei)),
                "calldata_size_bytes": int(tx_telemetry.get("calldata_size_bytes", 0)),
                "signed_tx_size_bytes": int(tx_telemetry.get("signed_tx_size_bytes", 0)),
                "build_latency_sec": float(tx_telemetry.get("build_latency_sec", 0.0)),
                "sign_latency_sec": float(tx_telemetry.get("sign_latency_sec", 0.0)),
                "send_latency_sec": float(tx_telemetry.get("send_latency_sec", 0.0)),
                "receipt_wait_latency_sec": float(tx_telemetry.get("receipt_wait_latency_sec", 0.0)),
                "tx_total_latency_sec": float(tx_telemetry.get("tx_total_latency_sec", 0.0)),
            })
            evs = []
            global_ok = None
            try:
                processed = _process_event_receipt_any(registry.events.RunFinalizedV1(), receipt) if receipt else []
                for ev in processed:
                    args_ = dict(ev.get("args") or {})
                    global_ok = bool(args_.get("globalVerifiedOk")) if args_.get(
                        "globalVerifiedOk") is not None else None
                    evs.append({
                        "event": "RunFinalizedV1",
                        "runId": args_.get("runId"),
                        "globalProofKey": args_.get("globalProofKey"),
                        "globalVerifiedOk": global_ok,
                        "totalSubmitted": int(args_.get("totalSubmitted", 0)),
                        "totalVerifiedOk": int(args_.get("totalVerifiedOk", 0)),
                    })
            except Exception:
                pass
            finalize_tx = {
                "submitted": True,
                "tx_hash": txh,
                "status": status,
                "block_number": block_no,
                "block_timestamp": block_ts,
                "gas_used": gas_used,
                "tx_fee_wei": str(fee_wei),
                "tx_fee_eth": _fmt_eth(w3, int(fee_wei)),
                "calldata_size_bytes": int(tx_telemetry.get("calldata_size_bytes", 0)),
                "signed_tx_size_bytes": int(tx_telemetry.get("signed_tx_size_bytes", 0)),
                "build_latency_sec": float(tx_telemetry.get("build_latency_sec", 0.0)),
                "sign_latency_sec": float(tx_telemetry.get("sign_latency_sec", 0.0)),
                "send_latency_sec": float(tx_telemetry.get("send_latency_sec", 0.0)),
                "receipt_wait_latency_sec": float(
                    tx_telemetry.get("receipt_wait_latency_sec", 0.0)
                ),
                "tx_total_latency_sec": float(
                    tx_telemetry.get("tx_total_latency_sec", 0.0)
                ),

                "tx_start_epoch_sec": float(
                    tx_telemetry.get("tx_start_epoch_sec", 0.0) or 0.0
                ),
                "broadcast_epoch_sec": float(
                    tx_telemetry.get("broadcast_epoch_sec", 0.0) or 0.0
                ),
                "receipt_observed_epoch_sec": float(
                    tx_telemetry.get("receipt_observed_epoch_sec", 0.0) or 0.0
                ),

                "global_verified_ok": global_ok,
                "events": evs,
            }
            print(f"\n[FINALIZE] tx={txh} status={status} global_ok={global_ok} fee_eth={_fmt_eth(w3, int(fee_wei))}")
            # ------------------------------------------------------------
            # ✅ Thesis-grade finalize evidence:
            # 1) Print RunFinalizedV1 event
            # 2) Read-back runFinalized(runId) + counts from chain
            # ------------------------------------------------------------
            if evs:
                for ev in evs:
                    if ev.get("event") == "RunFinalizedV1":
                        print(
                            f"[EVENT] RunFinalizedV1 runId={ev.get('runId')} globalProofKey={ev.get('globalProofKey')} "
                            f"globalVerifiedOk={ev.get('globalVerifiedOk')} totalSubmitted={ev.get('totalSubmitted')} "
                            f"totalVerifiedOk={ev.get('totalVerifiedOk')}"
                        )
            try:
                finalized_flag = bool(registry.functions.runFinalized(run_id_bytes32).call())
                total_submitted = int(registry.functions.runSubmitted(run_id_bytes32).call())
                total_ok = int(registry.functions.runVerifiedOk(run_id_bytes32).call())
                print(
                    f"[ONCHAIN] runFinalized(runId)={finalized_flag} runSubmitted={total_submitted} runVerifiedOk={total_ok}"
                )
            except Exception:
                pass
        except Exception as exc:
            finalize_tx = {"submitted": False, "error": str(exc)}
            print(f"\n[FINALIZE FAIL] {exc}")

    # ------------------------------------------------------------
    # Reviewer Comment 12: consensus-finality evaluation
    # ------------------------------------------------------------
    publication_finality_rows: List[Dict[str, Any]] = [
        s
        for s in submissions
        if s.get("submitted")
           and s.get("mined")
           and _safe_int(s.get("block_number"), 0) > 0
    ]

    if finalize_tx.get("submitted") and _safe_int(finalize_tx.get("block_number"), 0) > 0:
        publication_finality_rows.append(finalize_tx)

    if WAIT_FOR_FINALITY:
        print("\n------------------------------")
        print("CONSENSUS FINALITY")
        print("------------------------------")
        print(
            f"[FINALITY] Waiting for {len(publication_finality_rows)} "
            f"publication transaction(s) to become finalized..."
        )

        finality_measurement = _attach_finality_measurements(
            w3=w3,
            rows=publication_finality_rows,
            timeout_sec=FINALITY_TIMEOUT_SEC,
            poll_interval_sec=FINALITY_POLL_INTERVAL_SEC,
        )

        print(
            "[FINALITY] "
            f"supported={finality_measurement.get('supported')} "
            f"eligible={finality_measurement.get('eligible_transactions')} "
            f"finalized={finality_measurement.get('finalized_transactions')} "
            f"timed_out={finality_measurement.get('timed_out_transactions')}"
        )

        if finality_measurement.get("error"):
            print(f"[FINALITY] note={finality_measurement.get('error')}")

    else:
        finality_measurement = {
            "supported": False,
            "eligible_transactions": len(publication_finality_rows),
            "finalized_transactions": 0,
            "timed_out_transactions": 0,
            "error": "Finality measurement disabled by configuration.",
        }
    # ------------------------------
    # Wallet final balance (end) + summary print
    # ------------------------------
    balance_end_wei = _get_balance_wei(w3, account.address)
    spent_wei = int(balance_start_wei) - int(balance_end_wei)
    print("\n------------------------------")
    print("WALLET COST SUMMARY")
    print("------------------------------")
    print(f"balance_start_eth      : {_fmt_eth(w3, balance_start_wei)}")
    print(f"balance_end_eth        : {_fmt_eth(w3, balance_end_wei)}")
    print(f"spent_eth (balance Δ)  : {_fmt_eth(w3, spent_wei)}")
    print(f"spent_eth (receipts)   : {_fmt_eth(w3, total_fee_wei)}")
    # sanity note: balance delta can differ slightly if txs pending/extra transfers happen
    if spent_wei >= 0 and abs(spent_wei - total_fee_wei) > 0:
        print(f"[NOTE] balanceΔ != receiptsSum (wei): {spent_wei - total_fee_wei}")
    # Build report
    submitted_rows = [s for s in submissions if s.get("submitted")]
    submitted_count = len(submitted_rows)

    total_calldata_size_bytes = int(sum(int(s.get("calldata_size_bytes", 0) or 0) for s in submitted_rows))
    total_proof_json_size_bytes = int(sum(int(s.get("proof_json_size_bytes", 0) or 0) for s in submissions))
    total_public_inputs_size_bytes = int(sum(int(s.get("public_inputs_v1_json_size_bytes", 0) or 0) for s in submissions))
    total_verifier_sol_size_bytes = int(sum(int(s.get("verifier_sol_size_bytes", 0) or 0) for s in submissions))
    total_vkey_json_size_bytes = int(sum(int(s.get("vkey_json_size_bytes", 0) or 0) for s in submissions))
    total_manifest_json_size_bytes = int(sum(int(s.get("manifest_json_size_bytes", 0) or 0) for s in submissions))

    gas_rows = [row for row in fee_breakdown if row.get("gas_used") is not None]
    gas_row_count = len(gas_rows)
    total_gas_used = int(sum(int(row.get("gas_used", 0) or 0) for row in gas_rows))
    avg_gas_used_per_tx = (
        float(total_gas_used) / float(gas_row_count)
        if gas_row_count > 0 else 0.0
    )

    # ------------------------------------------------------------
    # Reviewer Comment 12: transaction-latency distributions
    # ------------------------------------------------------------
    tx_total_latency_values = [
        float(s.get("tx_total_latency_sec", 0.0) or 0.0)
        for s in submitted_rows
    ]

    receipt_wait_latency_values = [
        float(s.get("receipt_wait_latency_sec", 0.0) or 0.0)
        for s in submitted_rows
    ]

    tx_total_latency_stats = _describe_values(tx_total_latency_values)
    receipt_wait_latency_stats = _describe_values(receipt_wait_latency_values)

    # Preserve old scalar fields for backward compatibility.
    avg_tx_total_latency_sec = float(tx_total_latency_stats["mean"])
    avg_receipt_wait_latency_sec = float(receipt_wait_latency_stats["mean"])

    # ------------------------------------------------------------
    # Reviewer Comment 12: finality-latency distributions
    # ------------------------------------------------------------
    receipt_to_finality_values = [
        float(s.get("receipt_to_finality_sec"))
        for s in submitted_rows
        if s.get("finalized") is True
           and s.get("receipt_to_finality_sec") is not None
    ]

    tx_start_to_finality_values = [
        float(s.get("tx_start_to_finality_sec"))
        for s in submitted_rows
        if s.get("finalized") is True
           and s.get("tx_start_to_finality_sec") is not None
    ]

    receipt_to_finality_stats = _describe_values(
        receipt_to_finality_values
    )

    tx_start_to_finality_stats = _describe_values(
        tx_start_to_finality_values
    )

    # ------------------------------------------------------------
    # Reviewer Comment 12: observed transaction reliability
    # ------------------------------------------------------------
    tx_success_count = sum(
        1
        for s in submitted_rows
        if s.get("mined") is True and s.get("status") == 1
    )

    tx_failure_count = max(
        0,
        int(submitted_count) - int(tx_success_count)
    )

    observed_tx_failure_rate = (
        float(tx_failure_count) / float(submitted_count)
        if submitted_count > 0
        else 0.0
    )

    failure_ci_low, failure_ci_high = _wilson_interval_95(
        failures=tx_failure_count,
        trials=submitted_count,
    )

    # ------------------------------------------------------------
    # Proof-verification result is distinct from transaction success.
    # ------------------------------------------------------------
    proof_verified_ok_count = sum(
        1 for s in submitted_rows if s.get("verified_ok") is True
    )

    proof_verified_fail_count = sum(
        1 for s in submitted_rows if s.get("verified_ok") is False
    )

    proof_verification_failure_rate = (
        float(proof_verified_fail_count) / float(submitted_count)
        if submitted_count > 0
        else 0.0
    )

    # ------------------------------------------------------------
    # Reviewer Comment 12: deployed-contract runtime fingerprints
    # ------------------------------------------------------------
    contract_runtime_fingerprints = {
        "rsu_verifier": _contract_runtime_fingerprint(
            w3, rsu_verifier_addr
        ),
        "global_verifier": _contract_runtime_fingerprint(
            w3, global_verifier_addr
        ),
        "proof_registry": _contract_runtime_fingerprint(
            w3, registry_addr
        ),
    }

    report = {
        "generated_utc": _utc_now_iso(),
        "mode": "ON_CHAIN_TRANSACTION_VERIFICATION",
        "root_dir": root_dir,
        "index_path": index_path,
        "index_sha256": index_sha256,
        "run_id_sha256": run_id_sha256,
        "run_salt_hex": run_salt_hex,
        "run_id_scheme": "sha256('ONCHAIN_RUN_ID_V1|' + index_bytes + '|' + salt32)",
        "run_id_bytes32": run_id_bytes32,
        "chain_id": chain_id,
        "rpc_url_used": "[masked]",
        "wallet_address": account.address,
        "field_modulus_bn254": str(FIELD_MODULUS_BN254),
        "wallet_costs": {
            "balance_start_wei": str(balance_start_wei),
            "balance_start_eth": _fmt_eth(w3, balance_start_wei),
            "balance_end_wei": str(balance_end_wei),
            "balance_end_eth": _fmt_eth(w3, balance_end_wei),
            "spent_total_wei_balance_delta": str(spent_wei),
            "spent_total_eth_balance_delta": _fmt_eth(w3, spent_wei),
            "spent_total_wei_by_receipts": str(total_fee_wei),
            "spent_total_eth_by_receipts": _fmt_eth(w3, total_fee_wei),
            "fee_breakdown": fee_breakdown,
        },
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
        "contracts": {
            "rsu_verifier": rsu_verifier_addr,
            "global_verifier": global_verifier_addr,
            "proof_registry": registry_addr,

            "runtime_fingerprints": contract_runtime_fingerprints,

            "security_scope": {
                "formal_security_audit_claimed": False,
                "formal_verification_claimed": False,
                "functional_onchain_verification_exercised": True,
                "onchain_storage_readback_exercised": True,
                "duplicate_proof_key_guard_present": True,
                "single_finalization_guard_present": True,
                "global_proof_required_before_finalization": True,
            },
        },
        "deployment_caches": {
            "verifiers_cache_path": ver_cache_path,
            "registry_cache_path": reg_cache_path,
        },
        "network_evaluation": {
            "network": "Sepolia",
            "submission_pattern": SUBMISSION_PATTERN,
            "max_inflight_submissions": int(MAX_INFLIGHT_SUBMISSIONS),

            "concurrent_independent_submitters_tested": False,

            "finality_measurement": finality_measurement,

            "interpretation": (
                "Transaction receipt/inclusion and consensus finality are "
                "measured separately. The current publication run is serial "
                "and does not claim independent multi-sender concurrency."
            ),
        },

        "submissions": submissions,
        "finalize": finalize_tx,
        "summary": {
            "total_jobs": len([j for j in jobs if not (SUBMIT_GLOBAL_ONLY and str(j["scope"]).upper() == "RSU")]),
            "submitted_count": sum(1 for s in submissions if s.get("submitted")),
            "mined_count": sum(1 for s in submissions if s.get("submitted") and s.get("mined")),
            "tx_success_count": sum(1 for s in submissions if s.get("submitted") and s.get("status") == 1),
            "verified_ok_count": sum(1 for s in submissions if s.get("verified_ok") is True),
            "verified_fail_count": sum(1 for s in submissions if s.get("verified_ok") is False),
            "skipped_unsafe_count": sum(
                1 for s in submissions if not s.get("submitted") and s.get("prep", {}).get("errors")),
            "vehicle_ssi_ok_count": sum(
                1 for s in submissions if s.get("scope") == "RSU" and s.get("prep", {}).get("vehicle_ssi_ok") is True),
            "vehicle_ssi_fail_count": sum(
                1 for s in submissions if s.get("scope") == "RSU" and s.get("prep", {}).get("vehicle_ssi_ok") is False),
            "total_calldata_size_bytes": int(total_calldata_size_bytes),
            "total_proof_json_size_bytes": int(total_proof_json_size_bytes),
            "total_public_inputs_v1_json_size_bytes": int(total_public_inputs_size_bytes),
            "total_verifier_sol_size_bytes": int(total_verifier_sol_size_bytes),
            "total_vkey_json_size_bytes": int(total_vkey_json_size_bytes),
            "total_manifest_json_size_bytes": int(total_manifest_json_size_bytes),
            "total_gas_used": int(total_gas_used),
            "avg_gas_used_per_tx": float(avg_gas_used_per_tx),
            # Backward-compatible means
            "avg_tx_total_latency_sec": float(avg_tx_total_latency_sec),
            "avg_receipt_wait_latency_sec": float(avg_receipt_wait_latency_sec),

            # Full transaction-latency distributions
            "tx_total_latency_sec": tx_total_latency_stats,
            "receipt_wait_latency_sec": receipt_wait_latency_stats,

            # Consensus-finality distributions
            "receipt_to_finality_sec": receipt_to_finality_stats,
            "tx_start_to_finality_sec": tx_start_to_finality_stats,

            # Observed transaction reliability
            "tx_failure_count": int(tx_failure_count),
            "observed_tx_failure_rate": float(observed_tx_failure_rate),
            "observed_tx_failure_rate_wilson95_low": float(failure_ci_low),
            "observed_tx_failure_rate_wilson95_high": float(failure_ci_high),

            # Proof-verification reliability
            "proof_verified_ok_count": int(proof_verified_ok_count),
            "proof_verified_fail_count": int(proof_verified_fail_count),
            "proof_verification_failure_rate": float(
                proof_verification_failure_rate
            ),

            # Submission/concurrency semantics
            "submission_pattern": SUBMISSION_PATTERN,
            "max_inflight_submissions": int(MAX_INFLIGHT_SUBMISSIONS),
            "concurrent_independent_submitters_tested": False,
        },
    }
    out_path = os.path.join(root_dir, "onchain_export", "onchain_verification_report.json")
    _atomic_write_json(out_path, report)
    print("\n------------------------------")
    print("ON-CHAIN REPORT WRITTEN")
    print("------------------------------")
    print(out_path)

    # ============================================================
    # Reviewer Comment 12: console-ready Sepolia result summary
    # ============================================================
    if PRINT_REVIEWER_METRICS:
        txs = tx_total_latency_stats
        receipts = receipt_wait_latency_stats
        fin_receipt = receipt_to_finality_stats
        fin_total = tx_start_to_finality_stats

        print("\n")
        print("=" * 78)
        print(" REVIEWER COMMENT 12 - SEPOLIA BLOCKCHAIN EVALUATION")
        print("=" * 78)

        print("\n[NETWORK AND SUBMISSION MODEL]")
        print(f"network                         : Sepolia")
        print(f"chain_id                        : {chain_id}")
        print(f"submission_pattern              : {SUBMISSION_PATTERN}")
        print(f"max_inflight_submissions        : {MAX_INFLIGHT_SUBMISSIONS}")
        print("independent_submitter_concurrency: NOT TESTED")
        print(
            "validator_assumption            : "
            "Sepolia uses a permissioned validator set"
        )

        print("\n[TRANSACTION OUTCOMES]")
        print(f"proof_transactions_broadcast    : {submitted_count}")
        print(f"proof_transactions_successful   : {tx_success_count}")
        print(f"proof_transactions_failed       : {tx_failure_count}")
        print(
            f"observed_tx_failure_rate        : "
            f"{observed_tx_failure_rate:.6f}"
        )
        print(
            f"failure_rate_95pct_Wilson_CI    : "
            f"[{failure_ci_low:.6f}, {failure_ci_high:.6f}]"
        )
        print(f"proofs_verified_ok              : {proof_verified_ok_count}")
        print(f"proofs_verified_fail            : {proof_verified_fail_count}")
        print(
            f"proof_verification_failure_rate : "
            f"{proof_verification_failure_rate:.6f}"
        )

        print("\n[TRANSACTION TOTAL LATENCY - seconds]")
        print(f"n                               : {txs['n']}")
        print(f"mean                            : {txs['mean']:.6f}")
        print(f"median                          : {txs['median']:.6f}")
        print(f"stddev_population               : {txs['stddev_population']:.6f}")
        print(f"min                             : {txs['min']:.6f}")
        print(f"max                             : {txs['max']:.6f}")
        print(f"p95                             : {txs['p95']:.6f}")

        print("\n[RECEIPT / INCLUSION WAIT LATENCY - seconds]")
        print(f"n                               : {receipts['n']}")
        print(f"mean                            : {receipts['mean']:.6f}")
        print(f"median                          : {receipts['median']:.6f}")
        print(
            f"stddev_population               : "
            f"{receipts['stddev_population']:.6f}"
        )
        print(f"min                             : {receipts['min']:.6f}")
        print(f"max                             : {receipts['max']:.6f}")
        print(f"p95                             : {receipts['p95']:.6f}")

        print("\n[CONSENSUS FINALITY]")
        print(
            f"finality_rpc_supported          : "
            f"{finality_measurement.get('supported')}"
        )
        print(
            f"eligible_transactions           : "
            f"{finality_measurement.get('eligible_transactions')}"
        )
        print(
            f"finalized_transactions          : "
            f"{finality_measurement.get('finalized_transactions')}"
        )
        print(
            f"finality_timeouts               : "
            f"{finality_measurement.get('timed_out_transactions')}"
        )

        print("\n[RECEIPT -> FINALITY LATENCY - seconds]")
        print(f"n                               : {fin_receipt['n']}")
        print(f"mean                            : {fin_receipt['mean']:.6f}")
        print(f"median                          : {fin_receipt['median']:.6f}")
        print(
            f"stddev_population               : "
            f"{fin_receipt['stddev_population']:.6f}"
        )
        print(f"min                             : {fin_receipt['min']:.6f}")
        print(f"max                             : {fin_receipt['max']:.6f}")
        print(f"p95                             : {fin_receipt['p95']:.6f}")

        print("\n[TX START -> FINALITY LATENCY - seconds]")
        print(f"n                               : {fin_total['n']}")
        print(f"mean                            : {fin_total['mean']:.6f}")
        print(f"median                          : {fin_total['median']:.6f}")
        print(
            f"stddev_population               : "
            f"{fin_total['stddev_population']:.6f}"
        )
        print(f"min                             : {fin_total['min']:.6f}")
        print(f"max                             : {fin_total['max']:.6f}")
        print(f"p95                             : {fin_total['p95']:.6f}")

        print("\n[ON-CHAIN FOOTPRINT]")
        print(f"total_gas_used                  : {total_gas_used}")
        print(f"avg_gas_used_per_tx             : {avg_gas_used_per_tx:.3f}")
        print(f"total_calldata_bytes            : {total_calldata_size_bytes}")
        print(
            f"total_ETH_by_receipts           : "
            f"{_fmt_eth(w3, total_fee_wei)}"
        )

        print("\n[DEPLOYED CONTRACT RUNTIME FINGERPRINTS]")
        for contract_name, fp in contract_runtime_fingerprints.items():
            print(
                f"{contract_name:31s}: "
                f"bytes={fp.get('runtime_code_size_bytes')} "
                f"sha256={fp.get('runtime_code_sha256')}"
            )

        print("\n[CONTRACT SECURITY INTERPRETATION]")
        print("functional on-chain verification: YES")
        print("on-chain storage read-back       : YES")
        print("duplicate proof-key guard        : YES")
        print("single-finalization guard        : YES")
        print("global proof before finalization : YES")
        print("formal smart-contract audit      : NO")
        print("formal contract verification     : NO")

        print("\n[IMPORTANT INTERPRETATION]")
        print(
            "Failure rate above is the OBSERVED rate in this publication sample; "
            "it is not claimed as Sepolia's intrinsic failure probability."
        )
        print(
            "Receipt latency measures transaction inclusion/mining. "
            "Consensus finality is measured separately using the finalized head."
        )
        print(
            "The current experiment uses serial single-sender publication. "
            "Independent multi-sender concurrency is not claimed."
        )

        print("=" * 78)

    print("\nDone.\n")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())