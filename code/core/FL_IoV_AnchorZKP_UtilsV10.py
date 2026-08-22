# FL_IoV_AnchorZKP_UtilsV10.py
from __future__ import annotations
import base64
import hashlib
import inspect
import json
import logging
import math
import os
os.environ["PYSNARK_BACKEND"] = "snarkjs"
import shutil
import time
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union, List
import subprocess
import numpy as np

# NumPy 2.x compatibility shim for legacy libraries (e.g., dp_xgboost)
# Must be placed BEFORE importing dp_xgboost/xgboost.
if not hasattr(np, "cfloat"):
    np.cfloat = np.complex128
if not hasattr(np, "cdouble"):
    np.cdouble = np.complex128
if not hasattr(np, "clongdouble"):
    np.clongdouble = getattr(np, "complex256", np.complex128)

import pandas as pd
# -----------------------------------------------------------------------------#
# Canonical Spec V1 (frozen encodings + hash-to-field)
# -----------------------------------------------------------------------------#
try:
    from FL_IoV_CanonicalSpecV10 import (
        BN254_PRIME as _BN254_PRIME,
        ANCHOR_SPEC_V1,
        sha256_hex as _spec_sha256_hex,
        sha256_digest as _spec_sha256_digest,
        canon_json_bytes_v1 as _spec_canon_json_bytes_v1,
        sha256_to_field as _spec_sha256_to_field,
        sha256_hex_to_field as _spec_sha256_hex_to_field,
        assert_sha256_hex_str_v1 as _spec_assert_sha256_hex_str_v1,
        assert_field as _spec_assert_field,
        build_client_update_record_v1 as _spec_build_client_update_record_v1,
        # NEW (plan evidence): SSI Poseidon preimage definition fingerprint
        ssi_preimage_def_sha256_v1 as _spec_ssi_preimage_def_sha256_v1,
        ssi_preimage_def_field_bn254_v1 as _spec_ssi_preimage_def_field_bn254_v1,
    )
except Exception as e:
    raise ImportError(
        "[AnchorZKP] Missing/failed import: FL_IoV_CanonicalSpecV10.py.\n"
        "AnchorZKP must use the frozen encodings + SSI preimage fingerprint.\n"
        "Expected at least: BN254_PRIME, ANCHOR_SPEC_V1, sha256_hex, sha256_digest, canon_json_bytes_v1,\n"
        "sha256_to_field, sha256_hex_to_field, assert_sha256_hex_str_v1, assert_field,\n"
        "ssi_preimage_def_sha256_v1, ssi_preimage_def_field_bn254_v1."
    ) from e
# -----------------------------------------------------------------------------#
# Logging
# -----------------------------------------------------------------------------#
LOGGER = logging.getLogger("AnchorZKP.V10")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# -----------------------------------------------------------------------------#
# Field prime (BN254 / alt_bn128 scalar field)
# -----------------------------------------------------------------------------#
BN128_PRIME: int = int(_BN254_PRIME)
# -----------------------------------------------------------------------------#
# NEW (plan evidence): SSI Poseidon preimage definition fingerprint
# Logged at import time so driver + Ray actors emit the same evidence line.
# -----------------------------------------------------------------------------#
SSI_PREIMAGE_DEF_SHA256_V1: str = _spec_assert_sha256_hex_str_v1(
    str(_spec_ssi_preimage_def_sha256_v1()),
    allow_empty=False,
    field_name="ssi_preimage_def_sha256_v1",
)
SSI_PREIMAGE_DEF_FIELD_BN254_V1: int = int(_spec_ssi_preimage_def_field_bn254_v1())
SSI_PREIMAGE_DEF_FIELD_BN254_V1 = int(_spec_assert_field(int(SSI_PREIMAGE_DEF_FIELD_BN254_V1) % int(BN128_PRIME)))
if int(SSI_PREIMAGE_DEF_FIELD_BN254_V1) in (0, 1):
    raise ValueError(
        f"[ZKP][SSI] ssi_preimage_def_field_bn254_v1 must be non-trivial (not 0/1), got {int(SSI_PREIMAGE_DEF_FIELD_BN254_V1)}"
    )
# Log via BOTH the module logger and root logger so Ray workers reliably emit it.
LOGGER.info(
    "[ZKP][SSI] ssi_preimage_def_sha256_v1=%s ssi_preimage_def_field_bn254_v1=%s",
    SSI_PREIMAGE_DEF_SHA256_V1,
    str(int(SSI_PREIMAGE_DEF_FIELD_BN254_V1)),
)
logging.info(
    "[ZKP][SSI] ssi_preimage_def_sha256_v1=%s ssi_preimage_def_field_bn254_v1=%s",
    SSI_PREIMAGE_DEF_SHA256_V1,
    str(int(SSI_PREIMAGE_DEF_FIELD_BN254_V1)),
)
# -----------------------------------------------------------------------------#
# Commitment "mix" constants (MUST match Circom + Python)
# -----------------------------------------------------------------------------#
MIX_C1 = 1315423911
MIX_C2 = 2654435761
MIX_C3 = 97531
MIX_C4_ROOT = 1580030173  # binds root_poseidon_field (Phase-2)
# NEW: bind pins/policy/public-input-order into the ZK statement (Plan requirement)
MIX_C5_PINS  = 3266489917
MIX_C6_POLICY = 668265263
MIX_C7_ORDER  = 374761393
# -----------------------------------------------------------------------------#
# Defaults / paths
# -----------------------------------------------------------------------------#
DEFAULT_SCALE: int = int(ANCHOR_SPEC_V1.SCALE)
DEFAULT_ANCHOR_VERSION: str = str(ANCHOR_SPEC_V1.anchor_version)
DEFAULT_SCHEMA: str = "AnchorZKPArtifactV8b"  # frozen artifact schema identifier; retained for proof compatibility
ENV_PROJECT_ROOT = "ANCHOR_ZKP_PROJECT_ROOT"
def _detect_project_root() -> Path:
    # 1) Explicit override (recommended for Ray/Windows and publication archives)
    override = (
        os.getenv(ENV_PROJECT_ROOT, "").strip()
        or os.getenv("FLBCIDS_REPO_ROOT", "").strip()
    )
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"[AnchorZKP] configured project root is not a directory: {p}"
            )
        return p

    # 2) Search upward.  The publication archive stores package.json under
    #    environment/, whereas the original development tree kept it at root.
    starts = [Path.cwd(), Path(__file__).resolve().parent]
    for start in starts:
        cur = start.resolve()
        for _ in range(0, 25):
            if (cur / "package.json").is_file():
                return cur
            if (cur / "environment" / "package.json").is_file():
                return cur
            if (cur / "node_modules" / "circomlib").is_dir():
                return cur
            if (cur / "environment" / "node_modules" / "circomlib").is_dir():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent

    # 3) Fallback
    return Path(__file__).resolve().parent
PROJECT_ROOT = _detect_project_root()
MODULE_ROOT = PROJECT_ROOT  # keep your name, but make it stable
LOGGER.info("[AnchorZKP] PROJECT_ROOT=%s (set %s to override)", PROJECT_ROOT, ENV_PROJECT_ROOT)
# Keep the names if you want, but they must be resolved against MODULE_ROOT
DEFAULT_GENERATED_DIRNAME: str = "circuits_generated/anchorsum"
DEFAULT_ARTIFACTS_DIRNAME: str = "zkp_artifacts/anchorsum"
# -----------------------------------------------------------------------------#
# Env toggles
# -----------------------------------------------------------------------------#
SELECTION_MODE_V15: str = "pubmask"  # V15 single-mode: PUBMASK only
ENV_DEBUG_INLINE_VECTORS = "ANCHOR_ZKP_DEBUG_INLINE_VECTORS"  # "1" => include debug vectors in artifacts
ENV_KEEP_RUN_DIR = "ANCHOR_ZKP_KEEP_RUN_DIR"                  # "1" => keep temp run dirs
ENV_VERBOSE_PIPELINE = "ANCHOR_ZKP_VERBOSE_PIPELINE"
# If "1": NEVER auto-compile/setup inside the per-proof pipeline.
# The driver must precompile and distribute pins (pre_dir must already be complete).
ENV_STRICT_PRECOMPILE = "ANCHOR_ZKP_STRICT_PRECOMPILE"
# If "1": log pin/policy/order evidence on each prove_verify_* call.
ENV_LOG_ZK_PINS = "ANCHOR_ZKP_LOG_PINS"
# CLI / paths (standalone pipeline)
ENV_CIRCOM_CMD   = "ANCHOR_ZKP_CIRCOM_CMD"     # default: "circom"
ENV_SNARKJS_CMD  = "ANCHOR_ZKP_SNARKJS_CMD"    # default: "snarkjs"
ENV_NODE_CMD     = "ANCHOR_ZKP_NODE_CMD"       # default: "node"
ENV_PTAU_PATH    = "ANCHOR_ZKP_PTAU_PATH"      # default: "<this_file_dir>/powersOfTau28_hez_final_20.ptau"
# NEW: semicolon-separated list of library roots to pass as `circom -l <dir>`
ENV_CIRCOM_LINK_LIBS = "ANCHOR_ZKP_CIRCOM_LINK_LIBS"
def ssi_preimage_fingerprint_v1() -> Dict[str, str]:
    """
    Plan evidence helper:
    returns the frozen SSI Poseidon preimage definition fingerprint values that must
    be consistent across driver + RSU actors + global stage.
    """
    return {
        "ssi_preimage_def_sha256_v1": str(SSI_PREIMAGE_DEF_SHA256_V1),
        "ssi_preimage_def_field_bn254_v1": str(int(SSI_PREIMAGE_DEF_FIELD_BN254_V1)),
    }
def _sha256_to_nontrivial_field_v1(msg: str, prime: int) -> int:
    v = int.from_bytes(hashlib.sha256(msg.encode("utf-8")).digest(), "big") % int(prime)
    if v in (0, 1):
        v = 2
    return v
def _which_str(cmd: Union[str, Path, bytes]) -> Optional[str]:
    # Normalize PathLike -> (str|bytes) explicitly
    raw = os.fspath(cmd)
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    if isinstance(raw, (bytes, bytearray)) and len(raw) == 0:
        return None
    return shutil.which(raw)  # raw is str|bytes here
def _which_variants(base: Union[str, Path]) -> Optional[str]:
    b = str(base)
    direct = (
        _which_str(b)
        or _which_str(b + ".exe")
        or _which_str(b + ".cmd")
        or _which_str(b + ".bat")
    )
    if direct:
        return direct

    node_module_roots: List[Path] = []
    for raw in (
        os.getenv("FLBCIDS_NODE_MODULES", "").strip(),
        os.getenv("ANCHOR_ZKP_NODE_MODULES", "").strip(),
    ):
        if raw:
            node_module_roots.append(Path(raw).expanduser())

    node_module_roots.extend(
        [
            PROJECT_ROOT / "node_modules",
            PROJECT_ROOT / "environment" / "node_modules",
        ]
    )

    for nm in node_module_roots:
        bin_dir = nm / ".bin"
        for suffix in ("", ".exe", ".cmd", ".bat"):
            candidate = bin_dir / f"{b}{suffix}"
            if candidate.is_file():
                return str(candidate.resolve())
    return None
# --- AnchorZKP: pin tool binaries for Ray workers (Windows) ---
def _pin_cmd(env_name: str, base: str) -> None:
    if os.environ.get(env_name, "").strip():
        return
    exe = _which_variants(base)
    if exe:
        os.environ[env_name] = exe
_pin_cmd("ANCHOR_ZKP_CIRCOM_CMD", "circom")
_pin_cmd("ANCHOR_ZKP_SNARKJS_CMD", "snarkjs")
_pin_cmd("ANCHOR_ZKP_NODE_CMD", "node")
# Optional: show the resolved tools once (helps debugging)
logging.info("[ZKP] tools: circom=%s snarkjs=%s node=%s",
             os.environ.get("ANCHOR_ZKP_CIRCOM_CMD"),
             os.environ.get("ANCHOR_ZKP_SNARKJS_CMD"),
             os.environ.get("ANCHOR_ZKP_NODE_CMD"))
# artifact retention (debug)
ENV_KEEP_PUBLIC_JSON = "ANCHOR_ZKP_KEEP_PUBLIC_JSON"  # "1" keep public.json
ENV_KEEP_WITNESS     = "ANCHOR_ZKP_KEEP_WITNESS"      # "1" keep witness.wtns
ENV_KEEP_INPUT_JSON  = "ANCHOR_ZKP_KEEP_INPUT_JSON"   # "1" keep input.json
# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _ensure_dir(p: Union[str, Path]) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp
def _now_ms() -> int:
    return int(time.time() * 1000)
def _sha256_hex(b: bytes) -> str:
    return _spec_sha256_hex(b)
def _sha256_bytes(b: bytes) -> bytes:
    return _spec_sha256_digest(b)
def _json_canon(obj: Any) -> str:
    # Canonical JSON for this project's schemas (stable key order, no whitespace, forbid NaN/Inf)
    return _spec_canon_json_bytes_v1(obj).decode("utf-8")
def _hash_to_field(tag: str, payload: bytes, prime: int = BN128_PRIME) -> int:
    # Domain-separated hash-to-field (freeze base hash->field in CanonicalSpecV1)
    return int(_spec_sha256_to_field(tag.encode("utf-8") + b"|" + payload, prime=int(prime)))
def _map_to_field_nontrivial(x: int, prime: int = BN128_PRIME) -> int:
    if prime <= 5:
        raise ValueError("prime too small")
    return (x % (prime - 3)) + 2  # in [2..prime-2]
def public_input_order_id_field_v1(*, level: str, prime: int = BN128_PRIME) -> int:
    # Exclude the order-id itself to avoid self-reference.
    if level == "rsu":
        order = "anchor_id,round_idx,rsu_id,K_used,root_poseidon_field,pins_hash_field,policy_id_field,r_chal,agg_commit"
    elif level == "global":
        order = "anchor_id,round_idx,global_id,K_used,root_poseidon_field,pins_hash_field,policy_id_field,r_chal,agg_commit"
    else:
        raise ValueError("level must be 'rsu' or 'global'")
    h = _hash_to_field(f"PUBLIC_INPUT_ORDER_V1|{level}", order.encode("utf-8"), prime=int(prime))
    return _map_to_field_nontrivial(int(h), prime=int(prime))
def _ceil_log2(n: int) -> int:
    if n <= 1:
        return 0
    return int(math.ceil(math.log2(n)))
def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, (bool, np.bool_)):
            return int(bool(x))
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return default
            return int(x)
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return default
            return int(float(s)) if ("." in s or "e" in s.lower()) else int(s)
        return int(x)
    except Exception:
        return default
def _bool01(x: Any) -> int:
    return 1 if bool(_safe_int(x, 0)) else 0
def _as_2d_float32(X: Any) -> np.ndarray:
    arr = X if isinstance(X, np.ndarray) else np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array; got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)
def _as_1d_int64_le(v: Any, name: str) -> np.ndarray:
    arr = np.asarray(v)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape={arr.shape}")
    return np.asarray(arr, dtype=np.dtype("<i8"))
def _env_on(name: str) -> bool:
    return os.getenv(name, "0").strip() == "1"
def sha256_file(path: Union[str, Path]) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def _acquire_lockfile(lock_path: Path, *, timeout_sec: float = 900.0, poll_sec: float = 0.25) -> None:
    """
    Cross-process lock using atomic create (works well on Windows for local FS).
    """
    lock_path = Path(lock_path)
    t0 = time.time()
    while True:
        try:
            # atomic create
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"pid={os.getpid()} time={time.time()}".encode("utf-8"))
            finally:
                os.close(fd)
            return
        except FileExistsError:
            if (time.time() - t0) >= float(timeout_sec):
                raise TimeoutError(f"[ZKP] Timeout acquiring lock: {lock_path}")
            time.sleep(float(poll_sec))
def _release_lockfile(lock_path: Path) -> None:
    lock_path = Path(lock_path)
    try:
        lock_path.unlink(missing_ok=True)
    except TypeError:
        if lock_path.exists():
            lock_path.unlink()
def _expected_precompile_paths(pre_dir: Path, base_name: str) -> Dict[str, Path]:
    pre_dir = Path(pre_dir)
    return {
        "r1cs": pre_dir / f"{base_name}.r1cs",
        "sym": pre_dir / f"{base_name}.sym",
        "wasm": pre_dir / f"{base_name}_js" / f"{base_name}.wasm",
        "gen_witness": pre_dir / f"{base_name}_js" / "generate_witness.js",
        "zkey": pre_dir / "circuit_0000.zkey",
        "vkey": pre_dir / "verification_key.json",
        "verifier_sol": pre_dir / f"Verifier_{base_name}.sol",
    }
def _precompile_groth16_for_circuit(
    *,
    circom_file: Path,
    project_root: Path,
    pre_dir: Path,
    overwrite: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Compile + Groth16 setup once per circuit into `pre_dir`.
    Produces: <base>.r1cs, <base>_js/*, circuit_0000.zkey, verification_key.json
    """
    circom_cmd = _cmd_from_env(ENV_CIRCOM_CMD, "circom")
    snarkjs_cmd = _cmd_from_env(ENV_SNARKJS_CMD, "snarkjs")
    ptau_path = _resolve_ptau(project_root)
    base_name = Path(circom_file).stem
    pre_dir = _ensure_dir(pre_dir)
    paths = _expected_precompile_paths(pre_dir, base_name)
    if (not overwrite) and all(p.exists() for p in paths.values()):
        return {
            "ok": True,
            "precompiled": True,
            "pre_dir": str(pre_dir),
            "circuit": base_name,
            "circuit_path": str(Path(circom_file).resolve()),
            "circuit_sha256": sha256_file(circom_file),
            "r1cs_sha256": sha256_file(paths["r1cs"]),
            "zkey_sha256": sha256_file(paths["zkey"]),
            "vkey_sha256": sha256_file(paths["vkey"]),
        }
    lock_path = pre_dir / ".precompile.lock"
    _acquire_lockfile(lock_path)
    try:
        # re-check under lock
        if (not overwrite) and all(p.exists() for p in paths.values()):
            return {
                "ok": True,
                "precompiled": True,
                "pre_dir": str(pre_dir),
                "circuit": base_name,
                "circuit_path": str(Path(circom_file).resolve()),
                "circuit_sha256": sha256_file(circom_file),
                "r1cs_sha256": sha256_file(paths["r1cs"]),
                "zkey_sha256": sha256_file(paths["zkey"]),
                "vkey_sha256": sha256_file(paths["vkey"]),
            }
        if overwrite and pre_dir.exists():
            for child in pre_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    try:
                        child.unlink(missing_ok=True)
                    except TypeError:
                        if child.exists():
                            child.unlink()
        # 1) circom compile -> pre_dir
        link_dirs = _zkp_guess_circom_link_dirs(Path(circom_file), project_root)
        compile_argv: List[str] = [circom_cmd, str(Path(circom_file).resolve())]
        for libdir in link_dirs:
            compile_argv += ["-l", str(libdir)]
        compile_argv += ["--r1cs", "--wasm", "--sym", "--O1", "-o", str(pre_dir)]
        LOGGER.info("[ZKP][precompile] circom link dirs: %s", [str(p) for p in link_dirs])
        _run_cmd(
            compile_argv,
            cwd=project_root,
            label="precompile-circom",
            verbose=verbose,
            project_root=project_root,
        )
        # 2) groth16 setup -> pre_dir
        r1cs_name = f"{base_name}.r1cs"
        _run_cmd(
            [snarkjs_cmd, "groth16", "setup", r1cs_name, str(ptau_path), "circuit_0000.zkey"],
            cwd=pre_dir,
            label="precompile-snarkjs-setup",
            verbose=verbose,
            project_root=project_root,
        )
        _run_cmd(
            [snarkjs_cmd, "zkey", "export", "verificationkey", "circuit_0000.zkey", "verification_key.json"],
            cwd=pre_dir,
            label="precompile-snarkjs-export-vkey",
            verbose=verbose,
            project_root=project_root,
        )
        # ✅ NEW: export Solidity verifier into pre_dir
        _run_cmd(
            [snarkjs_cmd, "zkey", "export", "solidityverifier", "circuit_0000.zkey", f"Verifier_{base_name}.sol"],
            cwd=pre_dir,
            label="precompile-snarkjs-export-solidityverifier",
            verbose=verbose,
            project_root=project_root,
        )
        # Validate outputs
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            raise RuntimeError(f"[ZKP][precompile] Missing outputs in {pre_dir}: {missing}")
        return {
            "ok": True,
            "precompiled": True,
            "pre_dir": str(pre_dir),
            "circuit": base_name,
            "circuit_path": str(Path(circom_file).resolve()),
            "circuit_sha256": sha256_file(circom_file),
            "r1cs_sha256": sha256_file(paths["r1cs"]),
            "zkey_sha256": sha256_file(paths["zkey"]),
            "vkey_sha256": sha256_file(paths["vkey"]),
        }
    finally:
        _release_lockfile(lock_path)
def precompile_anchorsum_groth16(
    *,
    cfg: "AnchorZKPConfig",
    spec: "CircuitSpec",
    overwrite_circuit: bool = False,
    overwrite_precompile: bool = False,
    verbose: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Public API: ensure the circuit exists in cfg.generated_circuits_dir AND its Groth16 artifacts
    exist in cfg.artifacts_dir/<circuit_stem>/ so your V15 pin enforcement can resolve them early.
    V15 hardening: always ensure Circom emits <circuit_stem>.sym (requires --sym).
    """
    if verbose is None:
        verbose = bool(_env_on(ENV_VERBOSE_PIPELINE))
    circuit_path = write_anchorsum_circuit(
        spec,
        cfg.generated_circuits_dir,
        overwrite=bool(overwrite_circuit),
    )
    project_root = _find_project_root_for_circom(Path(circuit_path))
    pre_dir = _ensure_dir(Path(cfg.artifacts_dir) / Path(circuit_path).stem)
    out = _precompile_groth16_for_circuit(
        circom_file=Path(circuit_path),
        project_root=Path(project_root),
        pre_dir=Path(pre_dir),
        overwrite=bool(overwrite_precompile),
        verbose=bool(verbose),
    )
    # ------------------------------------------------------------------
    # ✅ Ensure .sym exists (Circom only emits it if compiled with --sym).
    # If the underlying pipeline didn't generate it, do a fast Circom compile
    # into the same pre_dir with --sym enabled.
    # ------------------------------------------------------------------
    sym_path = Path(pre_dir) / f"{Path(circuit_path).stem}.sym"
    if not sym_path.exists():
        # Prefer your configured env var (matches your logs), else rely on PATH.
        circom_cmd = os.environ.get("ANCHOR_ZKP_CIRCOM_CMD", "circom")
        cmd = [
            str(circom_cmd),
            str(Path(circuit_path)),
            "--r1cs",
            "--wasm",
            "--sym",
            "--output",
            str(Path(pre_dir)),
        ]
        if bool(verbose):
            subprocess.run(cmd, cwd=str(project_root), check=True)
        else:
            subprocess.run(
                cmd,
                cwd=str(project_root),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        if not sym_path.exists():
            raise RuntimeError(f"[ZKP][precompile] circom did not produce .sym: {str(sym_path)}")
    # Make paths explicit in returned metadata for downstream code/audits.
    if isinstance(out, dict):
        out["circuit_path"] = str(circuit_path)
        out["pre_dir"] = str(pre_dir)  # where .r1cs/.wasm/.sym and Groth16 artifacts live
        out["sym_path"] = str(sym_path)
    return out
def _validate_cap(actual: int, cap: int, what: str) -> None:
    if cap <= 0:
        raise ValueError(f"{what} cap must be > 0, got {cap}")
    if actual < 0:
        raise ValueError(f"{what} actual must be >= 0, got {actual}")
    if actual > cap:
        raise ValueError(
            f"{what} exceeds cap: actual={actual} > cap={cap}. "
            f"Increase cap (compile-time KMAX) or reduce participants."
        )
def _require_nontrivial_field(x: int, *, name: str, prime: int) -> int:
    v = int(_spec_assert_field(int(x) % int(prime)))
    if v == 0 or v == 1:
        raise ValueError(f"{name} must be a non-trivial field element (not 0 or 1), got {v}")
    return v
# -----------------------------------------------------------------------------#
# Circom/snarkjs JSON bigint safety
# -----------------------------------------------------------------------------#
JS_SAFE_INT: int = 2**53 - 1  # max safe integer in JavaScript
def _circom_scalar(x: Any) -> Union[int, str]:
    """
    Circom/snarkjs inputs are typically read via Node.js JSON parsing.
    Any integer outside JS safe range may lose precision unless encoded as a decimal string.
    """
    v = int(x)
    return v if (-JS_SAFE_INT <= v <= JS_SAFE_INT) else str(v)
# -----------------------------------------------------------------------------#
# Base64(zlib) vector encoding (Flower-safe)
# -----------------------------------------------------------------------------#
def compress_b64(raw: bytes, level: int = 6) -> str:
    return base64.b64encode(zlib.compress(raw, level=level)).decode("ascii")
def decompress_b64(b64: str) -> bytes:
    return zlib.decompress(base64.b64decode(b64.encode("ascii")))
def encode_int_vector_b64(v_int64: np.ndarray) -> Tuple[str, str]:
    v = _as_1d_int64_le(v_int64, "encode_int_vector_b64.v_int64")
    raw = v.tobytes(order="C")
    return compress_b64(raw), _sha256_hex(raw)
def decode_int_vector_b64(b64: str, expected_len: int) -> np.ndarray:
    raw = decompress_b64(b64)
    if len(raw) % 8 != 0:
        raise ValueError("Decoded bytes length is not a multiple of 8 (int64)")
    arr = np.frombuffer(raw, dtype=np.dtype("<i8"))
    if arr.size != int(expected_len):
        raise ValueError(f"Decoded vector length mismatch: got {arr.size}, expected {expected_len}")
    return arr.copy()
# -----------------------------------------------------------------------------#
# Anchors: build/load + anchor_id_field
# -----------------------------------------------------------------------------#
@dataclass(frozen=True)
class AnchorMeta:
    schema: str
    anchor_version: str
    M: int
    SCALE: int
    seed: int
    dtype: str
    shape: Tuple[int, int]
    sha256_anchor_bytes: str
    root_poseidon_field: int  # NEW
    created_ms: int
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "anchor_version": self.anchor_version,
            "M": int(self.M),
            "SCALE": int(self.SCALE),
            "seed": int(self.seed),
            "dtype": self.dtype,
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "sha256_anchor_bytes": self.sha256_anchor_bytes,
            # store as decimal string (JS/Node safety)
            "root_poseidon_field": str(int(self.root_poseidon_field)),  # NEW
            "created_ms": int(self.created_ms),
        }
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AnchorMeta":
        shape = d.get("shape")
        if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
            raise ValueError("AnchorMeta.shape must be [rows, cols]")
        rpf_raw = d.get("root_poseidon_field", 0)  # NEW (may be missing in old files)
        try:
            rpf = int(str(rpf_raw).strip()) if rpf_raw not in (None, "") else 0
        except Exception:
            rpf = 0
        return AnchorMeta(
            schema=str(d["schema"]),
            anchor_version=str(d["anchor_version"]),
            M=int(d["M"]),
            SCALE=int(d["SCALE"]),
            seed=int(d["seed"]),
            dtype=str(d["dtype"]),
            shape=(int(shape[0]), int(shape[1])),
            sha256_anchor_bytes=str(d["sha256_anchor_bytes"]),
            root_poseidon_field=int(rpf),  # NEW
            created_ms=int(d["created_ms"]),
        )
def compute_anchor_id_field(meta: Union["AnchorMeta", Dict[str, Any]], prime: int = BN128_PRIME) -> int:
    md = meta.to_dict() if isinstance(meta, AnchorMeta) else dict(meta)
    md.pop("created_ms", None)  # make id stable across rebuilds
    md.pop("root_poseidon_field", None)  # keep anchor_id stable across versions
    canon = _json_canon(md).encode("utf-8")
    return _hash_to_field("ANCHOR_META", canon, prime=prime)
def compute_anchor_root_poseidon_field(anchor_bytes: bytes, *, prime: int = BN128_PRIME) -> int:
    """
    Deterministic, Poseidon-friendly *field element* derived from anchor bytes.
    We intentionally keep this self-contained (no circomlibjs dependency) and map away from {0,1}
    so it binds non-trivially when mixed into commitments.
    """
    h = _hash_to_field("ANCHOR_ROOT_POSEIDON_FIELD", anchor_bytes, prime=prime)
    return _map_to_field_nontrivial(int(h), prime=prime)
def build_anchor_set(
    X_source: Any,
    M: int,
    seed: int,
    out_dir: Union[str, Path],
    *,
    SCALE: int = DEFAULT_SCALE,
    anchor_version: str = DEFAULT_ANCHOR_VERSION,
    overwrite: bool = False,
) -> Tuple[np.ndarray, AnchorMeta, int]:
    out_dir = _ensure_dir(out_dir)
    anchor_path = out_dir / "anchor_X.npy"
    meta_path = out_dir / "anchor_meta.json"
    if (anchor_path.exists() or meta_path.exists()) and not overwrite:
        raise FileExistsError(f"Anchor files exist in {out_dir}. Use overwrite=True to rebuild.")
    X = _as_2d_float32(X_source)
    if X.shape[0] < int(M):
        raise ValueError(f"Not enough rows to sample M={M} anchors (rows={X.shape[0]}).")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(X.shape[0], size=int(M), replace=False)
    X_anchor = X[idx].astype(np.float32, copy=False)
    raw = X_anchor.tobytes(order="C")
    root_poseidon_field = compute_anchor_root_poseidon_field(raw, prime=BN128_PRIME)  # NEW
    meta = AnchorMeta(
        schema="AnchorMetaV1",
        anchor_version=str(anchor_version),
        M=int(M),
        SCALE=int(SCALE),
        seed=int(seed),
        dtype=str(X_anchor.dtype),
        shape=(int(X_anchor.shape[0]), int(X_anchor.shape[1])),
        sha256_anchor_bytes=_sha256_hex(raw),
        root_poseidon_field=int(root_poseidon_field),  # NEW
        created_ms=_now_ms(),
    )
    np.save(anchor_path, X_anchor)
    meta_path.write_text(_json_canon(meta.to_dict()), encoding="utf-8")
    anchor_id_field = compute_anchor_id_field(meta)
    return X_anchor, meta, anchor_id_field
def load_anchor_set(anchor_dir: Union[str, Path]) -> Tuple[np.ndarray, AnchorMeta, int]:
    anchor_dir = Path(anchor_dir)
    anchor_path = anchor_dir / "anchor_X.npy"
    meta_path = anchor_dir / "anchor_meta.json"
    if not anchor_path.exists():
        raise FileNotFoundError(f"Missing: {anchor_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")
    X_anchor = np.load(anchor_path).astype(np.float32, copy=False)
    meta = AnchorMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
    raw = X_anchor.tobytes(order="C")
    if _sha256_hex(raw) != meta.sha256_anchor_bytes:
        raise ValueError("Anchor sha256 mismatch (anchor_X.npy vs anchor_meta.json)")
    if int(X_anchor.shape[0]) != int(meta.M):
        raise ValueError(f"Anchor M mismatch: file={X_anchor.shape[0]} meta={meta.M}")
    expected_root = compute_anchor_root_poseidon_field(raw, prime=BN128_PRIME)  # NEW
    # If missing in old JSON, fill it in-memory and warn (then you can rebuild with overwrite to persist).
    if int(getattr(meta, "root_poseidon_field", 0) or 0) == 0:
        LOGGER.warning(
            "[AnchorZKP] anchor_meta.json missing root_poseidon_field; computed=%s. "
            "Rebuild anchors with overwrite=True to persist it.",
            str(int(expected_root)),
        )
        meta = replace(meta, root_poseidon_field=int(expected_root))
    else:
        if int(meta.root_poseidon_field) != int(expected_root):
            raise ValueError("Anchor root_poseidon_field mismatch (anchor_X.npy vs anchor_meta.json)")
    anchor_id_field = compute_anchor_id_field(meta)
    return X_anchor, meta, anchor_id_field
# -----------------------------------------------------------------------------#
# Quantization
# -----------------------------------------------------------------------------#
def quantize_proba_to_int(
    p: Union[np.ndarray, Sequence[float]],
    *,
    SCALE: int = DEFAULT_SCALE,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D proba; got shape={arr.shape}")
    arr = np.clip(arr, clip_min, clip_max)
    return np.rint(arr * float(SCALE)).astype(np.dtype("<i8"))
# -----------------------------------------------------------------------------#
# Circom circuit writer (AnchorSum)
# -----------------------------------------------------------------------------#
@dataclass(frozen=True)
class CircuitSpec:
    level: str  # "rsu" or "global"
    M: int
    KMAX: int          # NMAX for rsu, RMAX for global
    SCALE: int
    enable_range_checks: bool
    row_scale_mult: int = 1  # rsu: 1, global: NMAX (max vehicles per RSU)
    # ---- V15 hardening: explicit public input order (matches Circom main { public [ ... ] }) ----
    public_inputs: Tuple[str, ...] = field(init=False, repr=False)
    public_input_names: Tuple[str, ...] = field(init=False, repr=False)
    public_signals: Tuple[str, ...] = field(init=False, repr=False)
    def __post_init__(self) -> None:
        if self.level == "rsu":
            pi = (
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
            )
        elif self.level == "global":
            pi = (
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
            )
        else:
            raise ValueError("CircuitSpec.level must be 'rsu' or 'global'")
        object.__setattr__(self, "public_inputs", pi)
        object.__setattr__(self, "public_input_names", pi)
        object.__setattr__(self, "public_signals", pi)
    @property
    def filename(self) -> str:
        tag = "AggRSU" if self.level == "rsu" else "AggGlobal"
        rc = "RC1" if self.enable_range_checks else "RC0"
        # Disambiguate only when needed (keeps RSU filenames stable)
        rm = f"_RM{int(self.row_scale_mult)}"
        # RB2 = root-binding + pins/policy/order binding (Phase-2 plan compliance)
        return f"{tag}_AnchorSum_M{self.M}_K{self.KMAX}{rm}_S{self.SCALE}_{rc}_RB2.circom"
def _circom_code_anchorsum(spec: CircuitSpec) -> str:
    if spec.level not in ("rsu", "global"):
        raise ValueError("spec.level must be 'rsu' or 'global'")
    entity_name = "rsu_id" if spec.level == "rsu" else "global_id"
    row_mult = int(getattr(spec, "row_scale_mult", 1))
    if row_mult <= 0:
        raise ValueError(f"row_scale_mult must be >= 1, got {row_mult}")
    max_Q = int(spec.KMAX) * row_mult * int(spec.SCALE)
    nbits = max(32, _ceil_log2(max_Q + 2) + 1)
    # Hard safety cap for inlined LessThan/Num2Bits (Circom standard constraint)
    if nbits > 252:
        raise ValueError(
            f"NBITS too large for inlined comparator: NBITS={nbits} (>252). "
            f"Reduce KMAX/SCALE or change the range-check design."
        )
    # Must match Python poly_commit_with_mix
    C1 = MIX_C1
    C2 = MIX_C2
    C3 = MIX_C3
    C4 = MIX_C4_ROOT  # binds root_poseidon_field (Phase-2)
    lines: list[str] = []
    lines.append("pragma circom 2.1.6;")
    lines.append("")
    if spec.enable_range_checks:
        lines.append("// Self-contained comparator primitives (inlined; no circomlib dependency)")
        lines.append("")
        lines.append("template Num2Bits(n) {")
        lines.append("    signal input in;")
        lines.append("    signal output out[n];")
        lines.append("    var lc1 = 0;")
        lines.append("    var e2 = 1;")
        lines.append("    for (var i = 0; i < n; i++) {")
        lines.append("        out[i] <-- (in >> i) & 1;")
        lines.append("        out[i] * (out[i] - 1) === 0;")
        lines.append("        lc1 += out[i] * e2;")
        lines.append("        e2 = e2 + e2;")
        lines.append("    }")
        lines.append("    lc1 === in;")
        lines.append("}")
        lines.append("")
        lines.append("template LessThan(n) {")
        lines.append("    // Requires non-negative inputs within bit-width; safe here because q/Q are >= 0")
        lines.append("    assert(n <= 252);")
        lines.append("    signal input in[2];")
        lines.append("    signal output out;")
        lines.append("    component n2b = Num2Bits(n+1);")
        lines.append("    n2b.in <== in[0] + (1 << n) - in[1];")
        lines.append("    out <== 1 - n2b.out[n];")
        lines.append("}")
        lines.append("")
        lines.append("template AssertLeq(nBits) {")
        lines.append("    signal input x;")
        lines.append("    signal input bound; // inclusive")
        lines.append("    component lt = LessThan(nBits);")
        lines.append("    lt.in[0] <== x;")
        lines.append("    lt.in[1] <== bound + 1;")
        lines.append("    lt.out === 1;")
        lines.append("}")
        lines.append("")
    lines.append("template AnchorSum(M, KMAX, SCALE, NBITS) {")
    lines.append("    // Private")
    lines.append("    signal input q[KMAX][M];")
    lines.append("    signal input mask[KMAX];")
    lines.append("")
    lines.append("    // Public")
    lines.append("    signal input anchor_id;")
    lines.append("    signal input round_idx;")
    lines.append(f"    signal input {entity_name};")
    lines.append("    signal input K_used;")
    lines.append("    signal input root_poseidon_field;")
    lines.append("    signal input pins_hash_field;")
    lines.append("    signal input policy_id_field;")
    lines.append("    signal input public_input_order_id_field;")
    lines.append("    signal input r_chal;")
    lines.append("    signal input agg_commit;")
    lines.append("")
    lines.append("    // Row multiplier for range checks (rsu=1, global=NMAX)")
    lines.append(f"    var row_mult = {row_mult};")
    lines.append("")
    lines.append("    // mask is boolean + compute K_used")
    lines.append("    var ku = 0;")
    lines.append("    for (var i = 0; i < KMAX; i++) {")
    lines.append("        mask[i] * (mask[i] - 1) === 0;")
    lines.append("        ku += mask[i];")
    lines.append("    }")
    lines.append("    signal ku_sig;")
    lines.append("    ku_sig <== ku;")
    lines.append("    ku_sig === K_used;")
    lines.append("")
    lines.append("    // Q[d] = sum_i mask[i] * q[i][d]  (R1CS-safe: chained steps)")
    lines.append("    signal Q[M];")
    lines.append("    signal qsum[M][KMAX + 1];")
    lines.append("    for (var d = 0; d < M; d++) {")
    lines.append("        qsum[d][0] <== 0;")
    lines.append("        for (var i = 0; i < KMAX; i++) {")
    lines.append("            // qsum[d][i+1] = qsum[d][i] + (mask[i] * q[i][d])")
    lines.append("            qsum[d][i + 1] <== qsum[d][i] + (mask[i] * q[i][d]);")
    lines.append("        }")
    lines.append("        Q[d] <== qsum[d][KMAX];")
    lines.append("    }")
    lines.append("")
    if spec.enable_range_checks:
        lines.append("    // Range checks (unused q rows must be zero-padded in Python)")
        lines.append("    component q_le[KMAX][M];")
        lines.append("    for (var i = 0; i < KMAX; i++) {")
        lines.append("        for (var d = 0; d < M; d++) {")
        lines.append("            q_le[i][d] = AssertLeq(NBITS);")
        lines.append("            q_le[i][d].x <== q[i][d];")
        lines.append("            q_le[i][d].bound <== row_mult * SCALE;")
        lines.append("        }")
        lines.append("    }")
        lines.append("")
        lines.append("    component Q_le[M];")
        lines.append("    for (var d = 0; d < M; d++) {")
        lines.append("        Q_le[d] = AssertLeq(NBITS);")
        lines.append("        Q_le[d].x <== Q[d];")
        lines.append("        Q_le[d].bound <== KMAX * row_mult * SCALE;")
        lines.append("    }")
        lines.append("")
    lines.append("    // poly = Horner(Q, r_chal)  (R1CS-safe: chained steps)")
    lines.append("    signal poly_steps[M + 1];")
    lines.append("    poly_steps[0] <== 0;")
    lines.append("    for (var t = 0; t < M; t++) {")
    lines.append("        poly_steps[t + 1] <== (poly_steps[t] * r_chal) + Q[M - 1 - t];")
    lines.append("    }")
    lines.append("    signal poly;")
    lines.append("    poly <== poly_steps[M];")
    lines.append("")
    lines.append("    signal mix;")
    lines.append(
        f"    mix <== anchor_id"
        f" + round_idx*{C1}"
        f" + {entity_name}*{C2}"
        f" + K_used*{C3}"
        f" + root_poseidon_field*{C4}"
        f" + pins_hash_field*{MIX_C5_PINS}"
        f" + policy_id_field*{MIX_C6_POLICY}"
        f" + public_input_order_id_field*{MIX_C7_ORDER};"
    )
    lines.append("")
    lines.append("    signal commit_check;")
    lines.append("    commit_check <== poly + mix;")
    lines.append("    commit_check === agg_commit;")
    lines.append("}")
    lines.append("")
    lines.append(
        f"component main {{ public [anchor_id, round_idx, {entity_name}, K_used, root_poseidon_field, pins_hash_field, policy_id_field, public_input_order_id_field, r_chal, agg_commit] }} = "
        f"AnchorSum({int(spec.M)}, {int(spec.KMAX)}, {int(spec.SCALE)}, {nbits});"
    )
    lines.append("")
    return "\n".join(lines)
def write_anchorsum_circuit(spec: CircuitSpec, out_dir: Union[str, Path], *, overwrite: bool = True) -> Path:
    out_dir = _ensure_dir(out_dir)
    out_path = (out_dir / spec.filename).resolve()
    if out_path.exists() and not overwrite:
        return out_path
    code = _circom_code_anchorsum(spec)
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(code, encoding="utf-8")
    tmp_path.replace(out_path)  # atomic on Windows for same-volume replace
    return out_path
# -----------------------------------------------------------------------------#
# Commitment + challenge (must match circuit)
# -----------------------------------------------------------------------------#
def derive_r_chal(
    *,
    level: str,
    anchor_id_field: int,
    round_idx: int,
    entity_id: int,
    M: int,
    SCALE: int,
    prime: int = BN128_PRIME,
) -> int:
    payload = f"{level}|{anchor_id_field}|{round_idx}|{entity_id}|{M}|{SCALE}".encode("utf-8")
    h = int.from_bytes(_sha256_bytes(b"ANCHOR_ZKP_R|" + payload), "big")
    return _map_to_field_nontrivial(h, prime=prime)
def poly_commit_with_mix(
    Q: np.ndarray,
    *,
    r_chal: int,
    anchor_id_field: int,
    round_idx: int,
    entity_id: int,
    K_used: int,
    root_poseidon_field: int,
    pins_hash_field: int,
    policy_id_field: int,
    public_input_order_id_field: int,
    prime: int = BN128_PRIME,
) -> int:
    q = _as_1d_int64_le(Q, "poly_commit_with_mix.Q")
    acc = 0
    r = int(r_chal) % prime
    for x in q[::-1]:
        acc = (acc * r + (int(x) % prime)) % prime
    mix = (
        int(anchor_id_field)
        + int(round_idx) * MIX_C1
        + int(entity_id) * MIX_C2
        + int(K_used) * MIX_C3
        + (int(root_poseidon_field) % prime) * MIX_C4_ROOT
        + (int(pins_hash_field) % prime) * MIX_C5_PINS
        + (int(policy_id_field) % prime) * MIX_C6_POLICY
        + (int(public_input_order_id_field) % prime) * MIX_C7_ORDER
    ) % prime
    return (acc + mix) % prime
def _sha256_vec(v: np.ndarray) -> Tuple[str, bytes]:
    arr = _as_1d_int64_le(v, "_sha256_vec.v")
    raw = arr.tobytes(order="C")
    return _sha256_hex(raw), raw
# -----------------------------------------------------------------------------#
# Groth16 pipeline adapter (no static import => fewer IDE "unresolved" errors)
# -----------------------------------------------------------------------------#
def _slugify(s: str, max_len: int = 80) -> str:
    # Windows-safe directory slug
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug[:max_len] if len(slug) > max_len else slug
def _cmd_from_env(env_name: str, default: str) -> str:
    v = os.getenv(env_name, "").strip()
    if v:
        return str(v)
    cand = _which_variants(default)
    return cand if cand else default
def _run_cmd(
    argv: Sequence[str],
    *,
    cwd: Path,
    label: str,
    verbose: bool,
    project_root: Optional[Path] = None,
) -> None:
    argv = list(map(str, argv))
    env = os.environ.copy()
    # Add runtime Node bins (important when cwd != project_root).
    node_bins: List[Path] = []
    for raw in (
        os.getenv("FLBCIDS_NODE_MODULES", "").strip(),
        os.getenv("ANCHOR_ZKP_NODE_MODULES", "").strip(),
    ):
        if raw:
            node_bins.append(Path(raw).expanduser() / ".bin")

    for root in [project_root, Path(cwd)]:
        if root is None:
            continue
        node_bins.append(Path(root) / "node_modules" / ".bin")
        node_bins.append(Path(root) / "environment" / "node_modules" / ".bin")

    for nm_bin in node_bins:
        if nm_bin.is_dir():
            env["PATH"] = str(nm_bin) + os.pathsep + env.get("PATH", "")
    # Windows: if the resolved command is a .cmd/.bat, run it via cmd.exe /c
    if os.name == "nt" and argv:
        a0 = argv[0].lower()
        if a0.endswith(".cmd") or a0.endswith(".bat"):
            argv = ["cmd.exe", "/c"] + argv
    if verbose:
        LOGGER.info("[ZKP] CMD (%s): %s", label, " ".join(argv))
        LOGGER.info("[ZKP] CWD (%s): %s", label, cwd)
        LOGGER.info("[ZKP] PATH head (%s): %s", label, env.get("PATH", "")[:300])
    try:
        p = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"[ZKP] Missing executable while running ({label}). "
            f"Tried argv[0]={argv[0]!r}. "
            f"Set ANCHOR_ZKP_CIRCOM_CMD / ANCHOR_ZKP_SNARKJS_CMD / ANCHOR_ZKP_NODE_CMD to absolute paths."
        ) from e
    if verbose and p.stdout:
        LOGGER.info("[ZKP] OUT (%s):\n%s", label, p.stdout)
    if p.returncode != 0:
        msg = p.stdout[-4000:] if p.stdout else ""
        raise RuntimeError(f"[ZKP] Command failed ({label}), rc={p.returncode}\n{msg}")
def _find_project_root_for_circom(circuit_path: Path) -> Path:
    """
    Find a stable project root for Circom compilation.
    We prefer a directory that contains:
      - node_modules/circomlib   (npm install circomlib)
      - circomlib                (manual vendoring)
    """
    starts = [
        circuit_path.resolve().parent,
        Path(__file__).resolve().parent,  # <-- strong fallback (where this utils file lives)
        Path.cwd(),
    ]
    for start in starts:
        cur = start
        for _ in range(0, 25):  # deeper + more robust on Windows layouts
            if (cur / "node_modules" / "circomlib").is_dir():
                return cur
            if (cur / "circomlib").is_dir():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
    # final fallback: this module directory (better than the circuit directory)
    return Path(__file__).resolve().parent
def _resolve_ptau(project_root: Path) -> Path:
    override = os.getenv(ENV_PTAU_PATH, "").strip()
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"[ZKP] Ptau not found via {ENV_PTAU_PATH}: {p}")
        return p
    # default: alongside this file
    p = Path(__file__).resolve().parent / "powersOfTau28_hez_final_20.ptau"
    if p.exists():
        return p
    # fallback: project root
    p2 = project_root / "powersOfTau28_hez_final_20.ptau"
    if p2.exists():
        return p2
    raise FileNotFoundError(
        "[ZKP] powersOfTau28_hez_final_20.ptau not found. "
        f"Place it next to this file OR set {ENV_PTAU_PATH}."
    )
def _zkp_guess_circom_link_dirs(circom_file: Path, project_root: Path) -> List[Path]:
    """
    Best-effort discovery of Circom library roots for `circom -l <dir>`.
    We want directories that *contain* either:
      - <dir>/circomlib/...
      - <dir>/node_modules/circomlib/...
    In practice, most projects use: -l <project_root>/node_modules
    """
    candidates: List[Path] = []

    # Publication/runtime dependency roots.
    for raw in (
        os.getenv("FLBCIDS_NODE_MODULES", "").strip(),
        os.getenv("ANCHOR_ZKP_NODE_MODULES", "").strip(),
    ):
        if raw:
            p = Path(raw).expanduser()
            if p.is_dir():
                candidates.append(p)

    env_nm = project_root / "environment" / "node_modules"
    if env_nm.is_dir():
        candidates.append(env_nm)

    # Optional override (semicolon-separated on Windows, colon-separated on Linux/macOS)
    env = (
            os.getenv(ENV_CIRCOM_LINK_LIBS, "").strip()
            or os.getenv("CIRCOM_LINK_LIBS", "").strip()
            or os.getenv("CIRCOM_LIBRARY_PATH", "").strip()
            or os.getenv("CIRCOM_INCLUDE_PATH", "").strip()
            or ""
    )
    if env.strip():
        for part in env.split(os.pathsep):
            p = Path(part.strip())
            if p.exists():
                candidates.append(p)
    def probe(start: Path) -> None:
        cur = start
        for _ in range(0, 12):
            nm = cur / "node_modules" / "circomlib"
            if nm.is_dir():
                candidates.append(cur / "node_modules")
                return
            cl = cur / "circomlib"
            if cl.is_dir():
                candidates.append(cur)
                return
            if cur.parent == cur:
                return
            cur = cur.parent
    probe(circom_file.parent)
    probe(project_root)
    probe(Path(__file__).resolve().parent)  # <-- NEW: module location
    probe(Path.cwd())                      # <-- NEW: runtime working dir
    # De-dup while preserving order
    out: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p.resolve())
        if key not in seen and p.exists():
            out.append(p)
            seen.add(key)
    return out
def _zkp_run_groth16_pipeline(
    *,
    label: str,
    circuit_filename: str,
    input_data: Dict[str, object],
    overwrite_zkp_dir: bool = True,
    zkp_dir: Optional[str] = None,
    verbose: bool = False,
    precompiled_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    Groth16 pipeline with optional precompiled artifacts reuse.
    If `precompiled_dir` is provided (recommended), we:
      - precompile (compile + groth16 setup + export vkey + export solidity verifier) once into precompiled_dir
      - for each proof run: generate witness + prove + verify in the per-run work_dir using the precompiled zkey/vkey
    If `precompiled_dir` is None, we default to:
      <project_root>/<DEFAULT_ARTIFACTS_DIRNAME>/<base_name>/
    """
    circom_cmd = _cmd_from_env(ENV_CIRCOM_CMD, "circom")
    snarkjs_cmd = _cmd_from_env(ENV_SNARKJS_CMD, "snarkjs")
    node_cmd = _cmd_from_env(ENV_NODE_CMD, "node")
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    def _expected_precompile_paths(pre_dir: Path, base: str) -> Dict[str, Path]:
        wasm_dir = pre_dir / f"{base}_js"
        return {
            "r1cs": pre_dir / f"{base}.r1cs",
            "sym": pre_dir / f"{base}.sym",
            "wasm_dir": wasm_dir,
            "wasm": wasm_dir / f"{base}.wasm",
            "gen_witness": wasm_dir / "generate_witness.js",
            "zkey": pre_dir / "circuit_0000.zkey",
            "vkey": pre_dir / "verification_key.json",
            "verifier_sol": pre_dir / f"Verifier_{base}.sol",
        }

    def _is_precompiled_ok(pre_dir: Path, base: str) -> bool:
        paths = _expected_precompile_paths(pre_dir, base)
        required = ["r1cs", "sym", "wasm", "gen_witness", "zkey", "vkey", "verifier_sol"]
        return all(paths[k].exists() for k in required)
    def _acquire_lock(lock_path: Path, *, timeout_sec: int = 180) -> Optional[int]:
        """
        Simple cross-process lock using exclusive-create.
        Returns an OS fd if acquired, else None (timeout).
        """
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, f"pid={os.getpid()} ts={time.time()}\n".encode("utf-8", errors="ignore"))
                return fd
            except FileExistsError:
                time.sleep(1.0)
            except Exception:
                time.sleep(0.25)
        return None
    def _release_lock(lock_path: Path, fd: Optional[int]) -> None:
        try:
            if fd is not None:
                os.close(fd)
        except Exception:
            pass
        try:
            lock_path.unlink(missing_ok=True)  # py3.8+: emulate below if needed
        except TypeError:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass
    # --- resolve circuit file ---
    cf = Path(circuit_filename)
    candidates: List[Path] = []
    if cf.is_absolute():
        circom_file = cf.resolve()
        candidates = [circom_file]
    else:
        candidates = [
            (Path.cwd() / cf).resolve(),
            (PROJECT_ROOT / cf).resolve(),
            (Path(__file__).resolve().parent / cf).resolve(),
        ]
        circom_file = next((p for p in candidates if p.exists()), candidates[0])
    if not circom_file.exists():
        tried = [str(p) for p in candidates]
        raise FileNotFoundError(
            f"[ZKP] Circuit file not found.\n"
            f"  given={circuit_filename!r}\n"
            f"  tried={tried}\n"
            f"Set {ENV_PROJECT_ROOT} to your repo root to stabilize path resolution."
        )
    project_root = _find_project_root_for_circom(circom_file)
    ptau_path = _resolve_ptau(project_root)
    base_name = circom_file.stem
    # --- per-run working dir ---
    if zkp_dir is not None:
        work_dir = Path(zkp_dir).resolve()
    else:
        safe_label = _slugify(label)
        work_dir = (project_root / "zkp_runs" / f"{base_name}_{safe_label}").resolve()
    if work_dir.exists() and overwrite_zkp_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    _ensure_dir(work_dir)
    # --- precompiled artifacts dir (setup once per circuit) ---
    if precompiled_dir is None:
        pre_dir = (project_root / DEFAULT_ARTIFACTS_DIRNAME / base_name).resolve()
    else:
        pre_dir = Path(precompiled_dir).resolve()
    _ensure_dir(pre_dir)
    lock_path = pre_dir / ".precompile.lock"
    lock_fd: Optional[int] = None
    circuit_sha256 = ""
    r1cs_sha256 = ""
    zkey_sha256 = ""
    vkey_sha256 = ""
    verifier_sol_pre_sha256 = ""
    t0 = time.time()
    # --- precompile if needed (compile + setup + vkey + verifier) ---
    if not _is_precompiled_ok(pre_dir, base_name):
        lock_fd = _acquire_lock(lock_path, timeout_sec=180)
        if lock_fd is None:
            # If we can't lock, we still try to proceed with whatever is there.
            # If another worker is precompiling, files should appear soon; otherwise we will fail loudly below.
            LOGGER.warning("[ZKP] Could not acquire precompile lock: %s", str(lock_path))
        else:
            try:
                # Re-check under lock (another worker may have finished between checks)
                if not _is_precompiled_ok(pre_dir, base_name):
                    tmp_pre = pre_dir.parent / f"{pre_dir.name}.tmp.{os.getpid()}.{_now_ms()}"
                    if tmp_pre.exists():
                        shutil.rmtree(tmp_pre, ignore_errors=True)
                    _ensure_dir(tmp_pre)
                    # 1) Compile circom -> tmp_pre (r1cs/wasm/sym)
                    link_dirs = _zkp_guess_circom_link_dirs(circom_file, project_root)
                    compile_argv: List[str] = [circom_cmd, str(circom_file)]
                    for libdir in link_dirs:
                        compile_argv += ["-l", str(libdir)]
                    compile_argv += ["--r1cs", "--wasm", "--sym", "--O1", "-o", str(tmp_pre)]
                    LOGGER.info("[ZKP] circom link dirs: %s", [str(p) for p in link_dirs])
                    if not link_dirs:
                        LOGGER.warning(
                            "[ZKP] No circom link dirs found; includes like 'circomlib/...' may fail. "
                            "Set CIRCOM_LINK_LIBS/ANCHOR_ZKP_CIRCOM_LINK_LIBS or ensure node_modules/circomlib is reachable."
                        )
                    _run_cmd(
                        compile_argv,
                        cwd=project_root,
                        label="circom-compile(precompile)",
                        verbose=verbose,
                        project_root=project_root,
                    )
                    # 2) Groth16 setup -> tmp_pre
                    r1cs_name = f"{base_name}.r1cs"
                    zkey_name = "circuit_0000.zkey"
                    vkey_name = "verification_key.json"
                    _run_cmd(
                        [snarkjs_cmd, "groth16", "setup", r1cs_name, str(ptau_path), zkey_name],
                        cwd=tmp_pre,
                        label="snarkjs-setup(precompile)",
                        verbose=verbose,
                        project_root=project_root,
                    )
                    _run_cmd(
                        [snarkjs_cmd, "zkey", "export", "verificationkey", zkey_name, vkey_name],
                        cwd=tmp_pre,
                        label="snarkjs-export-vkey(precompile)",
                        verbose=verbose,
                        project_root=project_root,
                    )
                    # 3) Export solidity verifier ONCE (deterministic given zkey)
                    verifier_sol_name = f"Verifier_{base_name}.sol"
                    _run_cmd(
                        [snarkjs_cmd, "zkey", "export", "solidityverifier", zkey_name, verifier_sol_name],
                        cwd=tmp_pre,
                        label="snarkjs-export-solidityverifier(precompile)",
                        verbose=verbose,
                        project_root=project_root,
                    )
                    # Atomically publish tmp_pre -> pre_dir (best-effort on Windows)
                    if pre_dir.exists():
                        shutil.rmtree(pre_dir, ignore_errors=True)
                    try:
                        tmp_pre.replace(pre_dir)
                    except Exception:
                        # fallback: copytree then delete
                        shutil.copytree(tmp_pre, pre_dir, dirs_exist_ok=True)
                        shutil.rmtree(tmp_pre, ignore_errors=True)
            finally:
                _release_lock(lock_path, lock_fd)
                lock_fd = None
    # --- assert precompiled outputs exist (or fail loudly) ---
    paths = _expected_precompile_paths(pre_dir, base_name)
    if not _is_precompiled_ok(pre_dir, base_name):
        raise RuntimeError(
            "[ZKP] Precompile artifacts missing/incomplete after attempted precompile.\n"
            f"  pre_dir={pre_dir}\n"
            f"  missing_any_of={[k for k, p in paths.items() if k in ('r1cs','wasm','gen_witness','zkey','vkey') and not p.exists()]}"
        )
    # --- compute hashes for pins/auditing ---
    try:
        circuit_sha256 = _sha256_file(circom_file)
    except Exception:
        circuit_sha256 = ""
    try:
        r1cs_sha256 = _sha256_file(paths["r1cs"])
    except Exception:
        r1cs_sha256 = ""
    try:
        zkey_sha256 = _sha256_file(paths["zkey"])
    except Exception:
        zkey_sha256 = ""
    try:
        vkey_sha256 = _sha256_file(paths["vkey"])
    except Exception:
        vkey_sha256 = ""
    try:
        if paths["verifier_sol"].exists():
            verifier_sol_pre_sha256 = _sha256_file(paths["verifier_sol"])
    except Exception:
        verifier_sol_pre_sha256 = ""
    # --- per-run: witness + prove + verify using precompiled zkey/vkey ---
    input_json_path = work_dir / "input.json"
    witness_path = work_dir / "witness.wtns"
    proof_name = "proof.json"
    public_name = "public.json"
    proof_path = work_dir / proof_name
    public_path = work_dir / public_name
    input_json_path.write_text(_json_canon(input_data), encoding="utf-8")
    _run_cmd(
        [node_cmd, str(paths["gen_witness"]), str(paths["wasm"]), str(input_json_path), str(witness_path)],
        cwd=work_dir,
        label="node-generate-witness",
        verbose=verbose,
        project_root=project_root,
    )
    _run_cmd(
        [snarkjs_cmd, "groth16", "prove", str(paths["zkey"]), str(witness_path), proof_name, public_name],
        cwd=work_dir,
        label="snarkjs-prove",
        verbose=verbose,
        project_root=project_root,
    )
    _run_cmd(
        [snarkjs_cmd, "groth16", "verify", str(paths["vkey"]), public_name, proof_name],
        cwd=work_dir,
        label="snarkjs-verify",
        verbose=verbose,
        project_root=project_root,
    )
    # read public inputs (MUST exist after prove; fail loud if missing/invalid)
    if not public_path.exists():
        raise RuntimeError(f"[ZKP] public.json missing after prove: {public_path}")
    try:
        public_inputs = json.loads(public_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"[ZKP] Failed to parse public.json: {public_path}") from e
    if not isinstance(public_inputs, list) or len(public_inputs) == 0:
        raise RuntimeError(
            f"[ZKP] public.json has unexpected type/empty: type={type(public_inputs).__name__} len={len(public_inputs) if isinstance(public_inputs, list) else 'NA'}"
        )
    # ------------------------------------------------------------------
    # ✅ NEW (on-chain export requirement):
    # Always persist a stable public-input sidecar, even if public.json is deleted.
    # ------------------------------------------------------------------
    public_sidecar_path = work_dir / "public_inputs_v1.json"
    public_sidecar_obj = {
        "schema": "PublicInputsSidecarV1",
        "public_inputs": [str(x) for x in public_inputs],
    }
    tmp_sidecar = public_sidecar_path.with_suffix(public_sidecar_path.suffix + f".tmp.{os.getpid()}")
    tmp_sidecar.write_text(_json_canon(public_sidecar_obj), encoding="utf-8")
    tmp_sidecar.replace(public_sidecar_path)  # atomic on same volume
    public_sidecar_sha256 = sha256_file(public_sidecar_path)
    # keep old behavior: provide a verifier.sol in the run dir (copy from pre_dir if possible)
    verifier_sol = work_dir / f"Verifier_{base_name}.sol"
    try:
        if paths["verifier_sol"].exists():
            shutil.copyfile(paths["verifier_sol"], verifier_sol)
        else:
            _run_cmd(
                [snarkjs_cmd, "zkey", "export", "solidityverifier", str(paths["zkey"]), verifier_sol.name],
                cwd=work_dir,
                label="snarkjs-export-solidityverifier",
                verbose=verbose,
                project_root=project_root,
            )
    except Exception:
        _run_cmd(
            [snarkjs_cmd, "zkey", "export", "solidityverifier", str(paths["zkey"]), verifier_sol.name],
            cwd=work_dir,
            label="snarkjs-export-solidityverifier",
            verbose=verbose,
            project_root=project_root,
        )
    # optional cleanup
    if not _env_on(ENV_KEEP_INPUT_JSON):
        try:
            input_json_path.unlink(missing_ok=True)
        except TypeError:
            if input_json_path.exists():
                input_json_path.unlink()
    if not _env_on(ENV_KEEP_WITNESS):
        try:
            witness_path.unlink(missing_ok=True)
        except TypeError:
            if witness_path.exists():
                witness_path.unlink()
    if not _env_on(ENV_KEEP_PUBLIC_JSON):
        try:
            public_path.unlink(missing_ok=True)
        except TypeError:
            if public_path.exists():
                public_path.unlink()
    t1 = time.time()
    elapsed_ms = int(round((t1 - t0) * 1000))
    return {
        "ok": True,
        "verified": True,
        "label": label,
        "circuit": base_name,
        "project_root": str(project_root),
        "zkp_dir": str(work_dir),
        "precompiled_dir": str(pre_dir),
        "ptau_path": str(ptau_path),
        "proof_path": str(proof_path),
        "verification_key": str(paths["vkey"]),
        "public_inputs": public_inputs,
        # ✅ NEW (on-chain export stable public inputs file)
        "public_sidecar_path": str(public_sidecar_path),
        "public_sidecar_sha256": str(public_sidecar_sha256),
        "verifier_sol": str(verifier_sol),
        "circuit_sha256": str(circuit_sha256),
        "r1cs_sha256": str(r1cs_sha256),
        "zkey_sha256": str(zkey_sha256),
        "vkey_sha256": str(vkey_sha256),
        "pre_verifier_sol_sha256": str(verifier_sol_pre_sha256),
        "elapsed_ms": int(elapsed_ms),
        "elapsed_sec": str(round((t1 - t0), 6)),
    }
def _call_pipeline_filtered(pipeline_fn: Callable[..., Any], **kwargs: Any) -> Any:
    try:
        sig = inspect.signature(pipeline_fn)
        accepted: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if k in sig.parameters:
                accepted[k] = v
        return pipeline_fn(**accepted)
    except (TypeError, ValueError):
        core: Dict[str, Any] = {}
        for k in (
            "label",
            "circuit_filename",
            "input_data",
            "overwrite_zkp_dir",
            "zkp_dir",
            "verbose",
            "precompiled_dir",
        ):
            if k in kwargs and kwargs[k] is not None:
                core[k] = kwargs[k]
        return pipeline_fn(**core)
def _extract_ok(res: Any) -> bool:
    if isinstance(res, dict):
        for key in ("ok", "verified", "success"):
            if key in res:
                return bool(res[key])
        if "result" in res and isinstance(res["result"], dict):
            return _extract_ok(res["result"])
        return False
    return False
def _norm_public_inputs_list_v1(v: Any, *, prime: int) -> Optional[List[int]]:
    if not isinstance(v, list) or len(v) == 0:
        return None
    out: List[int] = []
    for x in v:
        try:
            xi = int(str(x).strip())
        except Exception:
            return None
        out.append(int(xi) % int(prime))
    return out
def _ensure_public_inputs_present_v1(
    *,
    res: Any,
    expected_public_inputs: Sequence[int],
    public_input_names: Sequence[str],
    prime: int,
    label: str,
) -> Dict[str, Any]:
    """
    Guarantee res["public_inputs"] exists (stable order), and provide a by-name mapping.
    If the pipeline returned public_inputs, verify it matches expected (soundness).
    """
    if isinstance(res, dict):
        out: Dict[str, Any] = res
    else:
        out = {"raw": str(res)}
    expected_norm: List[int] = [int(x) % int(prime) for x in expected_public_inputs]
    expected_str: List[str] = [str(int(x) % int(prime)) for x in expected_public_inputs]
    got_any = out.get("public_inputs", None)
    got_norm = _norm_public_inputs_list_v1(got_any, prime=int(prime))
    if got_norm is None:
        # Inject expected so downstream NEVER sees KeyError for 'public_inputs'
        out["public_inputs"] = expected_str
    else:
        # Pipeline produced public_inputs; enforce exact match to prevent silent divergence
        if len(got_norm) != len(expected_norm) or any(a != b for a, b in zip(got_norm, expected_norm)):
            raise RuntimeError(
                "[ZKP] public_inputs mismatch (soundness failure).\n"
                f"  label={label}\n"
                f"  expected={expected_str}\n"
                f"  got={[str(int(x) % int(prime)) for x in got_norm]}\n"
                f"  names={list(public_input_names)}"
            )
    # Always add stable metadata helpers
    out["public_inputs_order"] = list(public_input_names)
    out["public_inputs_by_name"] = {
        str(public_input_names[i]): str(out["public_inputs"][i])
        for i in range(min(len(public_input_names), len(out["public_inputs"])))
    }
    return out
def _log_anchor_verify_outcome(
    *,
    level: str,
    ok: bool,
    label: str,
    round_idx: int,
    entity_id: int,
    used: int,
    commit_field: int,
    q_sha256: str,
) -> None:
    # Keep logs compact and audit-friendly: identifiers + hash, no raw vectors
    if ok:
        logging.info(
            "[ZKP][%s] ✅ AnchorSum verification SUCCEEDED | id=%d round=%d used=%d commit=%d Q_sha256=%s | %s",
            level, int(entity_id), int(round_idx), int(used), int(commit_field), str(q_sha256), str(label)
        )
    else:
        logging.error(
            "[ZKP][%s] ❌ AnchorSum verification FAILED | id=%d round=%d used=%d commit=%d Q_sha256=%s | %s",
            level, int(entity_id), int(round_idx), int(used), int(commit_field), str(q_sha256), str(label)
        )
# -----------------------------------------------------------------------------#
# Artifact schema
# -----------------------------------------------------------------------------#
@dataclass
class AnchorZKPArtifact:
    schema: str
    level: str
    ok: bool
    created_ms: int
    meta: Dict[str, Any]
    public: Dict[str, Any]
    hashes: Dict[str, Any]
    payload: Dict[str, Any]
    zkp: Dict[str, Any]
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "level": self.level,
            "ok": bool(self.ok),
            "created_ms": int(self.created_ms),
            "meta": self.meta,
            "public": self.public,
            "hashes": self.hashes,
            "payload": self.payload,
            "zkp": self.zkp,
        }
def _atomic_write_text(path: Union[str, Path], text: str) -> None:
    p = Path(path)
    _ensure_dir(p.parent)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}.{_now_ms()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(p))
def write_json(path: Union[str, Path], obj: Any) -> None:
    _atomic_write_text(path, _json_canon(obj))
# -----------------------------------------------------------------------------#
# Config
# -----------------------------------------------------------------------------#
@dataclass(frozen=True)
class AnchorZKPConfig:
    M: int = int(ANCHOR_SPEC_V1.M)
    SCALE: int = int(ANCHOR_SPEC_V1.SCALE)
    prime: int = BN128_PRIME
    enable_range_checks: bool = False
    generated_circuits_dir: Path = field(default_factory=lambda: Path(DEFAULT_GENERATED_DIRNAME))
    artifacts_dir: Path = field(default_factory=lambda: Path(DEFAULT_ARTIFACTS_DIRNAME))
    run_root_dir: Optional[Path] = None
    def spec_rsu(self, NMAX: int) -> CircuitSpec:
        return CircuitSpec(
            level="rsu",
            M=int(self.M),
            KMAX=int(NMAX),
            SCALE=int(self.SCALE),
            enable_range_checks=bool(self.enable_range_checks),
            row_scale_mult=1,
        )
    def spec_global(self, RMAX: int, NMAX: int) -> CircuitSpec:
        return CircuitSpec(
            level="global",
            M=int(self.M),
            KMAX=int(RMAX),
            SCALE=int(self.SCALE),
            enable_range_checks=bool(self.enable_range_checks),
            row_scale_mult=int(NMAX),  # RSU rows can be up to NMAX*SCALE
        )
    def __post_init__(self) -> None:
        base = PROJECT_ROOT
        def norm(p: Optional[Path]) -> Optional[Path]:
            if p is None:
                return None
            p = Path(p)
            return (p if p.is_absolute() else (base / p)).resolve()
        object.__setattr__(self, "generated_circuits_dir", norm(self.generated_circuits_dir))
        object.__setattr__(self, "artifacts_dir", norm(self.artifacts_dir))
        object.__setattr__(self, "run_root_dir", norm(self.run_root_dir))
# -----------------------------------------------------------------------------#
# Padding + aggregation
# -----------------------------------------------------------------------------#
def _normalize_mask_list(n: int, mask_list: Optional[Sequence[int]]) -> Sequence[int]:
    if mask_list is None:
        return [1] * n
    if len(mask_list) < n:
        # pad missing with 1s (common case: caller passes fewer masks)
        return list(mask_list) + [1] * (n - len(mask_list))
    if len(mask_list) > n:
        # allow, but trim to n
        return list(mask_list[:n])
    return mask_list
def _pad_q_and_mask(
    q_list: Sequence[np.ndarray],
    mask_list: Optional[Sequence[int]],
    *,
    KMAX: int,
    M: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(q_list)
    _validate_cap(n, int(KMAX), "Participants (q_list length)")
    mask_list2 = _normalize_mask_list(n, mask_list)
    q_mat = np.zeros((int(KMAX), int(M)), dtype=np.dtype("<i8"))
    mask = np.zeros((int(KMAX),), dtype=np.dtype("<i8"))
    used = 0
    for i in range(n):
        m = _bool01(mask_list2[i])
        mask[i] = m
        if m == 1:
            q = _as_1d_int64_le(q_list[i], f"q_list[{i}]")
            if q.size != int(M):
                raise ValueError(f"q_list[{i}] length mismatch: got {q.size}, expected {M}")
            q_mat[i, :] = q
            used += 1
    # rows i>=n remain zero; masks remain 0
    return q_mat, mask, used
def _sum_masked_rows(q_mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if q_mat.ndim != 2 or mask.ndim != 1:
        raise ValueError("Invalid shapes for q_mat/mask")
    if q_mat.shape[0] != mask.shape[0]:
        raise ValueError("q_mat rows and mask length mismatch")
    return (q_mat * mask.reshape(-1, 1)).sum(axis=0).astype(np.dtype("<i8"))
# -----------------------------------------------------------------------------#
# Flower-metrics helpers (still no Flower imports)
# -----------------------------------------------------------------------------#
def make_vehicle_anchor_metrics(
    *,
    anchor_version: str,
    anchor_id_field: int,
    M: int,
    SCALE: int,
    q_anchor: np.ndarray,
    dp_round_json_sha256: str = "",
) -> Dict[str, Any]:
    q = _as_1d_int64_le(q_anchor, "make_vehicle_anchor_metrics.q_anchor")
    if q.size != int(M):
        raise ValueError(f"q_anchor length mismatch: got {q.size}, expected {M}")
    b64, sha = encode_int_vector_b64(q)
    dp_round_json_sha256 = _spec_assert_sha256_hex_str_v1(
        str(dp_round_json_sha256 or ""),
        allow_empty=True,
        field_name="dp_round_json_sha256",
    )
    return {
        "anchor_version": str(anchor_version),
        "anchor_id_field": str(int(anchor_id_field)),
        "anchor_M": int(M),
        "anchor_SCALE": int(SCALE),
        "q_anchor_b64": b64,
        "q_anchor_sha256": sha,
        "dp_round_json_sha256": str(dp_round_json_sha256),
    }
def build_client_update_record_sha256(
    *,
    round_idx: int,
    rsu_id: int,
    vehicle_id: int,
    anchor_id_field: int,
    q_anchor_sha256: str,
    dp_round_json_sha256: str = "",
) -> str:
    """Return sha256(record_bytes) for ClientUpdateRecordV1 (canonical bytes per Spec V1)."""
    q_anchor_sha256 = _spec_assert_sha256_hex_str_v1(
        str(q_anchor_sha256),
        allow_empty=False,
        field_name="q_anchor_sha256",
    )
    dp_round_json_sha256 = _spec_assert_sha256_hex_str_v1(
        str(dp_round_json_sha256 or ""),
        allow_empty=True,
        field_name="dp_round_json_sha256",
    )
    _b, rec_sha = _spec_build_client_update_record_v1(
        round_idx=int(round_idx),
        rsu_id=int(rsu_id),
        vehicle_id=int(vehicle_id),
        anchor_version=str(ANCHOR_SPEC_V1.anchor_version),
        anchor_id_field=int(anchor_id_field),
        anchor_M=int(ANCHOR_SPEC_V1.M),
        anchor_SCALE=int(ANCHOR_SPEC_V1.SCALE),
        q_anchor_sha256=str(q_anchor_sha256),
        dp_round_json_sha256=str(dp_round_json_sha256),
    )
    return str(rec_sha)
def decode_vehicle_anchor_metrics_with_dp_sha(
    metrics: Dict[str, Any],
    *,
    expected_anchor_id_field: int,
    expected_M: int,
    expected_SCALE: int,
) -> Tuple[np.ndarray, str]:
    aid_raw = metrics.get("anchor_id_field", None)
    try:
        aid = int(str(aid_raw).strip())
    except Exception as e:
        raise ValueError(f"Missing/invalid anchor_id_field in metrics: {aid_raw!r}") from e
    if aid != int(expected_anchor_id_field):
        raise ValueError("anchor_id_field mismatch in metrics")
    if _safe_int(metrics.get("anchor_M")) != int(expected_M):
        raise ValueError("anchor_M mismatch in metrics")
    if _safe_int(metrics.get("anchor_SCALE")) != int(expected_SCALE):
        raise ValueError("anchor_SCALE mismatch in metrics")
    b64 = metrics.get("q_anchor_b64")
    if not isinstance(b64, str) or not b64.strip():
        raise ValueError("Missing q_anchor_b64 in metrics")
    q = decode_int_vector_b64(b64, expected_len=int(expected_M))
    sha_expected = metrics.get("q_anchor_sha256")
    if isinstance(sha_expected, str) and sha_expected.strip():
        raw = q.tobytes(order="C")
        if _sha256_hex(raw) != sha_expected:
            raise ValueError("q_anchor sha256 mismatch")
    dp_sha_raw = str(metrics.get("dp_round_json_sha256", "") or "").strip()
    dp_sha = _spec_assert_sha256_hex_str_v1(
        dp_sha_raw,
        allow_empty=True,
        field_name="dp_round_json_sha256",
    )
    return q, str(dp_sha)
def decode_vehicle_anchor_metrics(
    metrics: Dict[str, Any],
    *,
    expected_anchor_id_field: int,
    expected_M: int,
    expected_SCALE: int,
) -> np.ndarray:
    q, _dp_sha = decode_vehicle_anchor_metrics_with_dp_sha(
        metrics,
        expected_anchor_id_field=int(expected_anchor_id_field),
        expected_M=int(expected_M),
        expected_SCALE=int(expected_SCALE),
    )
    return q
# -----------------------------------------------------------------------------#
# RSU prove+verify
# -----------------------------------------------------------------------------#
def _require_pubmask_only_v15(cfg: Any, *, where: str) -> str:
    sel = str(getattr(cfg, "selection_mode", "pubmask") or "pubmask").strip().lower()
    if not sel:
        sel = "pubmask"
    if sel != "pubmask":
        raise RuntimeError(f"[AnchorZKP][V15] {where}: selection_mode must be 'pubmask' (got {sel!r})")
    return "pubmask"
def prove_verify_rsu_anchor_sum(
    *,
    cfg: AnchorZKPConfig,
    anchor_id_field: int,
    round_idx: int,
    rsu_id: int,
    root_poseidon_field: int,
    pins_hash_field: int,
    policy_id_field: int,
    public_input_order_id_field: int,
    q_list: Sequence[np.ndarray],
    mask_list: Optional[Sequence[int]],
    NMAX: int,
    out_dir: Optional[Union[str, Path]] = None,
    write_artifact: bool = True,
    include_Q_b64: bool = True,
    debug_inline_vectors: Optional[bool] = None,
    pipeline_fn: Optional[Callable[..., Any]] = None,
    overwrite_circuit: bool = True,
    overwrite_zkp_dir: bool = False,
    anchor_version: str = DEFAULT_ANCHOR_VERSION,
) -> AnchorZKPArtifact:
    if pipeline_fn is None:
        pipeline_fn = _zkp_run_groth16_pipeline
    if debug_inline_vectors is None:
        debug_inline_vectors = _env_on(ENV_DEBUG_INLINE_VECTORS)
    # validate cap early and loudly
    _validate_cap(len(q_list), int(NMAX), "Vehicles-per-RSU (N)")
    spec = cfg.spec_rsu(int(NMAX))
    circuit_path = write_anchorsum_circuit(
        spec,
        cfg.generated_circuits_dir,
        overwrite=overwrite_circuit,
    )
    LOGGER.info("[AnchorZKP] circuit_path=%s exists=%s", circuit_path, circuit_path.exists())
    # ✅ circuit hashing (evidence + pinning)
    circuit_sha256 = str(sha256_file(str(circuit_path)))
    q_mat, mask, used = _pad_q_and_mask(q_list, mask_list, KMAX=int(NMAX), M=int(cfg.M))
    Q = _sum_masked_rows(q_mat, mask)
    r_chal = derive_r_chal(
        level="rsu",
        anchor_id_field=int(anchor_id_field),
        round_idx=int(round_idx),
        entity_id=int(rsu_id),
        M=int(cfg.M),
        SCALE=int(cfg.SCALE),
        prime=int(cfg.prime),
    )
    prime = int(cfg.prime)
    # ✅ PUBMASK-only invariant (V15)
    sel_val = _require_pubmask_only_v15(cfg, where="RSU prove_verify_rsu_anchor_sum")
    range_checks_bit = 1 if bool(getattr(cfg, "enable_range_checks", False)) else 0
    root_poseidon_field = _require_nontrivial_field(
        int(root_poseidon_field),
        name="root_poseidon_field",
        prime=prime,
    )
    pins_hash_field_int = int(pins_hash_field) if pins_hash_field is not None else 0
    if pins_hash_field_int in (0, 1):
        pins_hash_field_int = _sha256_to_nontrivial_field_v1(
            "pins_v1|rsu_anchorsum|"
            f"selection={sel_val}|"
            f"M={int(cfg.M)}|SCALE={int(cfg.SCALE)}|NMAX={int(NMAX)}|"
            f"enable_range_checks={range_checks_bit}|"
            f"ssi_preimage_def_sha256_v1={SSI_PREIMAGE_DEF_SHA256_V1}|"
            f"ssi_preimage_def_field_bn254_v1={int(SSI_PREIMAGE_DEF_FIELD_BN254_V1)}",
            prime,
        )
    pins_hash_field = _require_nontrivial_field(
        pins_hash_field_int,
        name="pins_hash_field",
        prime=prime,
    )
    policy_id_field_int = int(policy_id_field) if policy_id_field is not None else 0
    if policy_id_field_int in (0, 1):
        policy_id_field_int = _sha256_to_nontrivial_field_v1(
            "policy_v1|rsu_anchorsum|"
            f"selection={sel_val}|"
            f"M={int(cfg.M)}|SCALE={int(cfg.SCALE)}|"
            f"enable_range_checks={range_checks_bit}",
            prime,
        )
    policy_id_field = _require_nontrivial_field(
        policy_id_field_int,
        name="policy_id_field",
        prime=prime,
    )
    public_input_order_id_field_int = (
        int(public_input_order_id_field) if public_input_order_id_field is not None else 0
    )
    if public_input_order_id_field_int in (0, 1):
        public_input_order_id_field_int = int(public_input_order_id_field_v1(level="rsu", prime=prime))
    public_input_order_id_field = _require_nontrivial_field(
        public_input_order_id_field_int,
        name="public_input_order_id_field",
        prime=prime,
    )
    commit = poly_commit_with_mix(
        Q,
        r_chal=int(r_chal),
        anchor_id_field=int(anchor_id_field),
        round_idx=int(round_idx),
        entity_id=int(rsu_id),
        K_used=int(used),
        root_poseidon_field=int(root_poseidon_field),
        pins_hash_field=int(pins_hash_field),
        policy_id_field=int(policy_id_field),
        public_input_order_id_field=int(public_input_order_id_field),
        prime=int(cfg.prime),
    )
    Q_sha, Q_raw = _sha256_vec(Q)
    Q_b64 = compress_b64(Q_raw) if include_Q_b64 else ""
    q_json = [[_circom_scalar(int(x)) for x in row] for row in q_mat.tolist()]
    mask_json = [_circom_scalar(int(x)) for x in mask.tolist()]
    input_data = {
        "q": q_json,
        "mask": mask_json,
        "anchor_id": _circom_scalar(int(anchor_id_field) % int(cfg.prime)),
        "round_idx": _circom_scalar(int(round_idx) % int(cfg.prime)),
        "rsu_id": _circom_scalar(int(rsu_id) % int(cfg.prime)),
        "K_used": _circom_scalar(int(used) % int(cfg.prime)),
        "root_poseidon_field": _circom_scalar(int(root_poseidon_field) % int(cfg.prime)),
        "pins_hash_field": _circom_scalar(int(pins_hash_field) % int(cfg.prime)),
        "policy_id_field": _circom_scalar(int(policy_id_field) % int(cfg.prime)),
        "public_input_order_id_field": _circom_scalar(int(public_input_order_id_field) % int(cfg.prime)),
        "r_chal": _circom_scalar(int(r_chal) % int(cfg.prime)),
        "agg_commit": _circom_scalar(int(commit) % int(cfg.prime)),
    }
    run_dir: Optional[Path] = None
    if cfg.run_root_dir is not None:
        run_dir = _ensure_dir(cfg.run_root_dir) / f"rsu_{rsu_id}" / f"round_{round_idx}"
        _ensure_dir(run_dir)
    label = f"RSU_AnchorSum|rsu={rsu_id}|r={round_idx}|M={cfg.M}|KMAX={NMAX}|S={cfg.SCALE}"
    if run_dir is not None and Path(run_dir).exists():
        overwrite_zkp_dir = True
    precompiled_dir = str((Path(cfg.artifacts_dir) / Path(circuit_path).stem).resolve())
    res = _call_pipeline_filtered(
        pipeline_fn,
        label=label,
        circuit_filename=str(circuit_path),
        input_data=input_data,
        overwrite_zkp_dir=bool(overwrite_zkp_dir),
        zkp_dir=str(run_dir) if run_dir is not None else None,
        verbose=bool(_env_on(ENV_VERBOSE_PIPELINE)),
        precompiled_dir=precompiled_dir,
    )
    expected_public_inputs = [
        int(anchor_id_field) % int(cfg.prime),
        int(round_idx) % int(cfg.prime),
        int(rsu_id) % int(cfg.prime),
        int(used) % int(cfg.prime),
        int(root_poseidon_field) % int(cfg.prime),
        int(pins_hash_field) % int(cfg.prime),
        int(policy_id_field) % int(cfg.prime),
        int(public_input_order_id_field) % int(cfg.prime),
        int(r_chal) % int(cfg.prime),
        int(commit) % int(cfg.prime),
    ]
    res = _ensure_public_inputs_present_v1(
        res=res,
        expected_public_inputs=expected_public_inputs,
        public_input_names=spec.public_inputs,
        prime=int(cfg.prime),
        label=label,
    )
    ok = _extract_ok(res)
    _log_anchor_verify_outcome(
        level="RSU",
        ok=ok,
        label=label,
        round_idx=int(round_idx),
        entity_id=int(rsu_id),
        used=int(used),
        commit_field=int(commit),
        q_sha256=str(Q_sha),
    )
    payload: Dict[str, Any] = {"Q_rsu_sha256": Q_sha, "Q_rsu_b64": Q_b64}
    if debug_inline_vectors:
        payload["debug_q_mat"] = q_mat.tolist()
        payload["debug_mask"] = mask.tolist()
        payload["debug_Q_rsu"] = Q.tolist()
    # ✅ Evidence: include SSI preimage fingerprint constants (if available in module)
    ssi_fp: Dict[str, str] = {}
    try:
        ssi_fp = dict(ssi_preimage_fingerprint_v1())
    except Exception:
        ssi_fp = {}
    art = AnchorZKPArtifact(
        schema=DEFAULT_SCHEMA,
        level="rsu",
        ok=ok,
        created_ms=_now_ms(),
        meta={
            "anchor_version": str(anchor_version),
            "selection_mode": str(sel_val),
            "M": int(cfg.M),
            "SCALE": int(cfg.SCALE),
            "KMAX": int(NMAX),
            "enable_range_checks": bool(cfg.enable_range_checks),
            "ssi_preimage_def_sha256_v1": str(ssi_fp.get("ssi_preimage_def_sha256_v1", "")),
            "ssi_preimage_def_field_bn254_v1": str(ssi_fp.get("ssi_preimage_def_field_bn254_v1", "")),
        },
        public={
            "anchor_id_field": str(int(anchor_id_field)),
            "round_idx": int(round_idx),
            "rsu_id": int(rsu_id),
            "N_used": int(used),
            "root_poseidon_field": str(int(root_poseidon_field)),
            "pins_hash_field": str(int(pins_hash_field)),
            "policy_id_field": str(int(policy_id_field)),
            "public_input_order_id_field": str(int(public_input_order_id_field)),
            "r_chal": int(r_chal),
            "agg_commit_field": int(commit),
        },
        hashes={"Q_rsu_sha256": str(Q_sha), "circuit_sha256": str(circuit_sha256)},
        payload=payload,
        zkp={
            "label": label,
            "circuit_path": str(circuit_path),
            "circuit_sha256": str(circuit_sha256),
            "public_inputs": list(res.get("public_inputs", [])),
            "public_inputs_order": list(res.get("public_inputs_order", [])),
            "public_inputs_by_name": dict(res.get("public_inputs_by_name", {})),
            "result": res if isinstance(res, dict) else {"raw": str(res)},
        },
    )
    out_root = cfg.artifacts_dir if out_dir is None else Path(out_dir)
    out_root = _ensure_dir(out_root)
    # ------------------------------------------------------------------
    # ✅ Persist stable proof/public sidecars + verifier for on-chain export (V1)
    # ------------------------------------------------------------------
    stable_dir = _ensure_dir(out_root / f"rsu_{int(rsu_id)}")
    stable_proof_path = stable_dir / f"round_{int(round_idx)}_proof.json"
    stable_public_v1_path = stable_dir / f"round_{int(round_idx)}_public_inputs_v1.json"
    # ✅ NEW: persist Solidity verifier next to proof/public
    stable_verifier_sol_path = stable_dir / f"Verifier_{Path(circuit_path).stem}.sol"
    proof_sha256 = ""
    public_v1_sha256 = ""
    verifier_sol_sha256 = ""
    if run_dir is not None and Path(run_dir).exists():
        # proof.json
        src_proof = Path(run_dir) / "proof.json"
        if src_proof.exists():
            shutil.copy2(str(src_proof), str(stable_proof_path))
            proof_sha256 = str(sha256_file(str(stable_proof_path)))
        # Prefer stable v1 sidecar if produced by pipeline, else fall back to public.json
        src_public_v1 = Path(run_dir) / "public_inputs_v1.json"
        src_public = Path(run_dir) / "public.json"
        if src_public_v1.exists():
            shutil.copy2(str(src_public_v1), str(stable_public_v1_path))
            public_v1_sha256 = str(sha256_file(str(stable_public_v1_path)))
        elif src_public.exists():
            shutil.copy2(str(src_public), str(stable_public_v1_path))
            public_v1_sha256 = str(sha256_file(str(stable_public_v1_path)))
        # ✅ NEW: copy Solidity verifier to stable location
        src_verifier_sol = Path(run_dir) / f"Verifier_{Path(circuit_path).stem}.sol"
        if src_verifier_sol.exists():
            shutil.copy2(str(src_verifier_sol), str(stable_verifier_sol_path))
            verifier_sol_sha256 = str(sha256_file(str(stable_verifier_sol_path)))
    # Store stable paths/hashes in the artifact (no protocol change)
    if not isinstance(art.payload, dict):
        art.payload = {}
    if not isinstance(art.hashes, dict):
        art.hashes = {}
    art.payload["proof_json_path"] = str(stable_proof_path)
    art.payload["public_inputs_v1_json_path"] = str(stable_public_v1_path)
    if proof_sha256:
        art.hashes["proof_sha256"] = str(proof_sha256)
    if public_v1_sha256:
        art.hashes["public_inputs_v1_sha256"] = str(public_v1_sha256)
    # ✅ NEW: persist verifier path + sha256 into artifact
    if verifier_sol_sha256:
        art.hashes["verifier_sol_sha256"] = str(verifier_sol_sha256)
        art.payload["verifier_sol_path"] = str(stable_verifier_sol_path)
    if write_artifact:
        path = stable_dir / f"round_{int(round_idx)}.json"
        write_json(path, art.to_dict())
    if run_dir is not None and not _env_on(ENV_KEEP_RUN_DIR):
        shutil.rmtree(run_dir, ignore_errors=True)
    return art
def prove_verify_rsu_anchor_sum_from_meta(
    *,
    cfg: AnchorZKPConfig,
    meta: AnchorMeta,
    anchor_id_field: int,
    round_idx: int,
    rsu_id: int,
    root_poseidon_field: int,
    pins_hash_field: int,
    policy_id_field: int,
    public_input_order_id_field: int,
    q_list: Sequence[np.ndarray],
    mask_list: Optional[Sequence[int]],
    NMAX: int,
    **kwargs: Any,
) -> AnchorZKPArtifact:
    art = prove_verify_rsu_anchor_sum(
        cfg=cfg,
        anchor_id_field=anchor_id_field,
        anchor_version=meta.anchor_version,
        round_idx=round_idx,
        rsu_id=rsu_id,
        root_poseidon_field=int(root_poseidon_field),
        pins_hash_field=int(pins_hash_field),
        policy_id_field=int(policy_id_field),
        public_input_order_id_field=int(public_input_order_id_field),
        q_list=q_list,
        mask_list=mask_list,
        NMAX=NMAX,
        **kwargs,
    )
    public_inputs: List[str] = _extract_public_inputs_from_artifact_v1(art)
    out_dir = str(kwargs.get("out_dir") or getattr(cfg, "artifacts_dir", "") or "")
    stable_dir = os.path.join(out_dir, f"rsu_{int(rsu_id)}")
    stable_public_path = os.path.join(stable_dir, f"round_{int(round_idx)}_public_inputs_v1.json")
    artifact_json_path = os.path.join(stable_dir, f"round_{int(round_idx)}.json")
    if not isinstance(getattr(art, "public", None), dict):
        setattr(art, "public", {})
    if not isinstance(getattr(art, "payload", None), dict):
        setattr(art, "payload", {})
    art.public["public_inputs"] = list(public_inputs)
    art.payload["public_inputs"] = list(public_inputs)
    art.payload["public_inputs_v1_json_path"] = str(stable_public_path)
    os.makedirs(stable_dir, exist_ok=True)
    write_json(Path(stable_public_path), list(public_inputs))
    try:
        if not isinstance(getattr(art, "hashes", None), dict):
            setattr(art, "hashes", {})
        art.hashes["public_inputs_v1_sha256"] = str(sha256_file(str(stable_public_path)))
    except Exception:
        pass
    if os.path.exists(artifact_json_path):
        write_json(Path(artifact_json_path), art.to_dict())
    return art
# -----------------------------------------------------------------------------#
# Global prove+verify
# -----------------------------------------------------------------------------#
def prove_verify_global_anchor_sum(
    *,
    cfg: AnchorZKPConfig,
    spec: Any = None,  # ✅ accept spec forwarded by from_meta or newer call-sites
    anchor_id_field: int,
    round_idx: int,
    global_id: int,
    root_poseidon_field: int,
    pins_hash_field: int,
    policy_id_field: int,
    public_input_order_id_field: int,
    Q_rsu_list: Sequence[np.ndarray],
    mask_list: Optional[Sequence[int]],
    RMAX: int,
    NMAX: int,
    out_dir: Optional[Union[str, Path]] = None,
    write_artifact: bool = True,
    include_Q_b64: bool = True,
    debug_inline_vectors: Optional[bool] = None,
    pipeline_fn: Optional[Callable[..., Any]] = None,
    overwrite_circuit: bool = True,
    overwrite_zkp_dir: bool = False,
    anchor_version: str = DEFAULT_ANCHOR_VERSION,
) -> AnchorZKPArtifact:
    if pipeline_fn is None:
        pipeline_fn = _zkp_run_groth16_pipeline
    if debug_inline_vectors is None:
        debug_inline_vectors = _env_on(ENV_DEBUG_INLINE_VECTORS)
    _validate_cap(len(Q_rsu_list), int(RMAX), "RSUs (R)")
    # ✅ use caller-provided spec if present; otherwise build it
    if spec is None:
        spec = cfg.spec_global(int(RMAX), int(NMAX))
    circuit_path = write_anchorsum_circuit(
        spec,
        cfg.generated_circuits_dir,
        overwrite=overwrite_circuit,
    )
    LOGGER.info("[AnchorZKP] circuit_path=%s exists=%s", circuit_path, circuit_path.exists())
    # ✅ circuit hashing (evidence + pinning)
    circuit_sha256 = str(sha256_file(str(circuit_path)))
    q_mat, mask, used = _pad_q_and_mask(Q_rsu_list, mask_list, KMAX=int(RMAX), M=int(cfg.M))
    Qg = _sum_masked_rows(q_mat, mask)
    r_chal = derive_r_chal(
        level="global",
        anchor_id_field=int(anchor_id_field),
        round_idx=int(round_idx),
        entity_id=int(global_id),
        M=int(cfg.M),
        SCALE=int(cfg.SCALE),
        prime=int(cfg.prime),
    )
    prime = int(cfg.prime)
    # ✅ PUBMASK-only invariant (V15)
    sel_val = _require_pubmask_only_v15(cfg, where="GLOBAL prove_verify_global_anchor_sum")
    range_checks_bit = 1 if bool(getattr(cfg, "enable_range_checks", False)) else 0
    root_poseidon_field = _require_nontrivial_field(
        int(root_poseidon_field),
        name="root_poseidon_field",
        prime=prime,
    )
    pins_hash_field_int = int(pins_hash_field) if pins_hash_field is not None else 0
    if pins_hash_field_int in (0, 1):
        pins_hash_field_int = _sha256_to_nontrivial_field_v1(
            "pins_v1|global_anchorsum|"
            f"selection={sel_val}|"
            f"M={int(cfg.M)}|SCALE={int(cfg.SCALE)}|RMAX={int(RMAX)}|NMAX={int(NMAX)}|"
            f"enable_range_checks={range_checks_bit}|"
            f"ssi_preimage_def_sha256_v1={SSI_PREIMAGE_DEF_SHA256_V1}|"
            f"ssi_preimage_def_field_bn254_v1={int(SSI_PREIMAGE_DEF_FIELD_BN254_V1)}",
            prime,
        )
    pins_hash_field = _require_nontrivial_field(
        pins_hash_field_int,
        name="pins_hash_field",
        prime=prime,
    )
    policy_id_field_int = int(policy_id_field) if policy_id_field is not None else 0
    if policy_id_field_int in (0, 1):
        policy_id_field_int = _sha256_to_nontrivial_field_v1(
            "policy_v1|global_anchorsum|"
            f"selection={sel_val}|"
            f"M={int(cfg.M)}|SCALE={int(cfg.SCALE)}|RMAX={int(RMAX)}|NMAX={int(NMAX)}|"
            f"enable_range_checks={range_checks_bit}",
            prime,
        )
    policy_id_field = _require_nontrivial_field(
        policy_id_field_int,
        name="policy_id_field",
        prime=prime,
    )
    public_input_order_id_field_int = (
        int(public_input_order_id_field) if public_input_order_id_field is not None else 0
    )
    if public_input_order_id_field_int in (0, 1):
        public_input_order_id_field_int = int(public_input_order_id_field_v1(level="global", prime=prime))
    public_input_order_id_field = _require_nontrivial_field(
        public_input_order_id_field_int,
        name="public_input_order_id_field",
        prime=prime,
    )
    commit = poly_commit_with_mix(
        Qg,
        r_chal=int(r_chal),
        anchor_id_field=int(anchor_id_field),
        round_idx=int(round_idx),
        entity_id=int(global_id),
        K_used=int(used),
        root_poseidon_field=int(root_poseidon_field),
        pins_hash_field=int(pins_hash_field),
        policy_id_field=int(policy_id_field),
        public_input_order_id_field=int(public_input_order_id_field),
        prime=int(cfg.prime),
    )
    Q_sha, Q_raw = _sha256_vec(Qg)
    Q_b64 = compress_b64(Q_raw) if include_Q_b64 else ""
    q_json = [[_circom_scalar(int(x)) for x in row] for row in q_mat.tolist()]
    mask_json = [_circom_scalar(int(x)) for x in mask.tolist()]
    input_data = {
        "q": q_json,
        "mask": mask_json,
        "anchor_id": _circom_scalar(int(anchor_id_field) % int(cfg.prime)),
        "round_idx": _circom_scalar(int(round_idx) % int(cfg.prime)),
        "global_id": _circom_scalar(int(global_id) % int(cfg.prime)),
        "K_used": _circom_scalar(int(used) % int(cfg.prime)),
        "root_poseidon_field": _circom_scalar(int(root_poseidon_field) % int(cfg.prime)),
        "pins_hash_field": _circom_scalar(int(pins_hash_field) % int(cfg.prime)),
        "policy_id_field": _circom_scalar(int(policy_id_field) % int(cfg.prime)),
        "public_input_order_id_field": _circom_scalar(int(public_input_order_id_field) % int(cfg.prime)),
        "r_chal": _circom_scalar(int(r_chal) % int(cfg.prime)),
        "agg_commit": _circom_scalar(int(commit) % int(cfg.prime)),
    }
    run_dir: Optional[Path] = None
    if cfg.run_root_dir is not None:
        run_dir = _ensure_dir(cfg.run_root_dir) / f"global_{global_id}" / f"round_{round_idx}"
        _ensure_dir(run_dir)
    label = f"GLOBAL_AnchorSum|g={global_id}|r={round_idx}|M={cfg.M}|RMAX={RMAX}|NMAX={NMAX}|S={cfg.SCALE}"
    if run_dir is not None and Path(run_dir).exists():
        overwrite_zkp_dir = True
    precompiled_dir = str((Path(cfg.artifacts_dir) / Path(circuit_path).stem).resolve())
    res = _call_pipeline_filtered(
        pipeline_fn,
        label=label,
        circuit_filename=str(circuit_path),
        input_data=input_data,
        overwrite_zkp_dir=bool(overwrite_zkp_dir),
        zkp_dir=str(run_dir) if run_dir is not None else None,
        verbose=bool(_env_on(ENV_VERBOSE_PIPELINE)),
        precompiled_dir=precompiled_dir,
    )
    expected_public_inputs = [
        int(anchor_id_field) % int(cfg.prime),
        int(round_idx) % int(cfg.prime),
        int(global_id) % int(cfg.prime),
        int(used) % int(cfg.prime),
        int(root_poseidon_field) % int(cfg.prime),
        int(pins_hash_field) % int(cfg.prime),
        int(policy_id_field) % int(cfg.prime),
        int(public_input_order_id_field) % int(cfg.prime),
        int(r_chal) % int(cfg.prime),
        int(commit) % int(cfg.prime),
    ]
    res = _ensure_public_inputs_present_v1(
        res=res,
        expected_public_inputs=expected_public_inputs,
        public_input_names=spec.public_inputs,
        prime=int(cfg.prime),
        label=label,
    )
    ok = _extract_ok(res)
    _log_anchor_verify_outcome(
        level="GLOBAL",
        ok=ok,
        label=label,
        round_idx=int(round_idx),
        entity_id=int(global_id),
        used=int(used),
        commit_field=int(commit),
        q_sha256=str(Q_sha),
    )
    payload: Dict[str, Any] = {"Q_global_sha256": Q_sha, "Q_global_b64": Q_b64}
    if debug_inline_vectors:
        payload["debug_Q_rsu_mat"] = q_mat.tolist()
        payload["debug_mask"] = mask.tolist()
        payload["debug_Q_global"] = Qg.tolist()
    # ✅ Evidence: include SSI preimage fingerprint constants (if available in module)
    ssi_fp: Dict[str, str] = {}
    try:
        ssi_fp = dict(ssi_preimage_fingerprint_v1())
    except Exception:
        ssi_fp = {}
    art = AnchorZKPArtifact(
        schema=DEFAULT_SCHEMA,
        level="global",
        ok=ok,
        created_ms=_now_ms(),
        meta={
            "anchor_version": str(anchor_version),
            "selection_mode": str(sel_val),
            "M": int(cfg.M),
            "SCALE": int(cfg.SCALE),
            "RMAX": int(RMAX),
            "NMAX": int(NMAX),
            "row_scale_mult": int(NMAX),
            "enable_range_checks": bool(cfg.enable_range_checks),
            "ssi_preimage_def_sha256_v1": str(ssi_fp.get("ssi_preimage_def_sha256_v1", "")),
            "ssi_preimage_def_field_bn254_v1": str(ssi_fp.get("ssi_preimage_def_field_bn254_v1", "")),
        },
        public={
            "anchor_id_field": str(int(anchor_id_field)),
            "round_idx": int(round_idx),
            "global_id": int(global_id),
            "R_used": int(used),
            "root_poseidon_field": str(int(root_poseidon_field)),
            "pins_hash_field": str(int(pins_hash_field)),
            "policy_id_field": str(int(policy_id_field)),
            "public_input_order_id_field": str(int(public_input_order_id_field)),
            "r_chal": int(r_chal),
            "global_commit_field": int(commit),
        },
        hashes={"Q_global_sha256": str(Q_sha), "circuit_sha256": str(circuit_sha256)},
        payload=payload,
        zkp={
            "label": label,
            "circuit_path": str(circuit_path),
            "circuit_sha256": str(circuit_sha256),
            "public_inputs": list(res.get("public_inputs", [])),
            "public_inputs_order": list(res.get("public_inputs_order", [])),
            "public_inputs_by_name": dict(res.get("public_inputs_by_name", {})),
            "result": res if isinstance(res, dict) else {"raw": str(res)},
        },
    )
    out_root = cfg.artifacts_dir if out_dir is None else Path(out_dir)
    out_root = _ensure_dir(out_root)
    # ------------------------------------------------------------------
    # ✅ Persist stable proof/public sidecars + verifier for on-chain export (V1)
    # ------------------------------------------------------------------
    stable_dir = _ensure_dir(out_root / "global")
    stable_proof_path = stable_dir / f"round_{int(round_idx)}_proof.json"
    stable_public_v1_path = stable_dir / f"round_{int(round_idx)}_public_inputs_v1.json"
    # ✅ NEW: persist Solidity verifier next to proof/public
    stable_verifier_sol_path = stable_dir / f"Verifier_{Path(circuit_path).stem}.sol"
    proof_sha256 = ""
    public_v1_sha256 = ""
    verifier_sol_sha256 = ""
    if run_dir is not None and Path(run_dir).exists():
        # proof.json
        src_proof = Path(run_dir) / "proof.json"
        if src_proof.exists():
            shutil.copy2(str(src_proof), str(stable_proof_path))
            proof_sha256 = str(sha256_file(str(stable_proof_path)))
        # public inputs sidecar v1 preferred, else public.json
        src_public_v1 = Path(run_dir) / "public_inputs_v1.json"
        src_public = Path(run_dir) / "public.json"
        if src_public_v1.exists():
            shutil.copy2(str(src_public_v1), str(stable_public_v1_path))
            public_v1_sha256 = str(sha256_file(str(stable_public_v1_path)))
        elif src_public.exists():
            shutil.copy2(str(src_public), str(stable_public_v1_path))
            public_v1_sha256 = str(sha256_file(str(stable_public_v1_path)))
        # ✅ NEW: copy Solidity verifier to stable location
        src_verifier_sol = Path(run_dir) / f"Verifier_{Path(circuit_path).stem}.sol"
        if src_verifier_sol.exists():
            shutil.copy2(str(src_verifier_sol), str(stable_verifier_sol_path))
            verifier_sol_sha256 = str(sha256_file(str(stable_verifier_sol_path)))
    if not isinstance(art.payload, dict):
        art.payload = {}
    if not isinstance(art.hashes, dict):
        art.hashes = {}
    art.payload["proof_json_path"] = str(stable_proof_path)
    art.payload["public_inputs_v1_json_path"] = str(stable_public_v1_path)
    if proof_sha256:
        art.hashes["proof_sha256"] = str(proof_sha256)
    if public_v1_sha256:
        art.hashes["public_inputs_v1_sha256"] = str(public_v1_sha256)
    # ✅ NEW: persist verifier path + sha256 into artifact
    if verifier_sol_sha256:
        art.hashes["verifier_sol_sha256"] = str(verifier_sol_sha256)
        art.payload["verifier_sol_path"] = str(stable_verifier_sol_path)
    if write_artifact:
        path = stable_dir / f"round_{int(round_idx)}.json"
        write_json(path, art.to_dict())
    if run_dir is not None and not _env_on(ENV_KEEP_RUN_DIR):
        shutil.rmtree(run_dir, ignore_errors=True)
    return art
def _coerce_public_inputs_list_v1(obj: Any) -> List[str]:
    """
    Accepts:
      - list/tuple of ints/strings
      - dict with 'publicSignals' / 'public_signals' / 'public_inputs'
      - anything else -> []
    Returns: list[str] (decimal strings) for stability across tooling.
    """
    if obj is None:
        return []
    if isinstance(obj, dict):
        for k in ("public_inputs", "publicSignals", "public_signals"):
            if k in obj:
                obj = obj.get(k)
                break
    if isinstance(obj, (list, tuple)):
        return [str(x) for x in obj]
    return []
def _try_load_public_inputs_json_v1(path: str) -> List[str]:
    if not path or not isinstance(path, str) or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _coerce_public_inputs_list_v1(data)
    except Exception:
        return []
def _extract_public_inputs_from_artifact_v1(art: Any) -> List[str]:
    """
    Tries, in order:
      1) art.public keys
      2) art.payload keys
      3) art.zkp keys
      4) JSON files referenced in payload
      5) artifact_path JSON content
    """
    public_obj = getattr(art, "public", None)
    payload_obj = getattr(art, "payload", None)
    zkp_obj = getattr(art, "zkp", None)
    if isinstance(public_obj, dict):
        out = _coerce_public_inputs_list_v1(public_obj)
        if out:
            return out
    if isinstance(payload_obj, dict):
        out = _coerce_public_inputs_list_v1(payload_obj)
        if out:
            return out
        for k in (
            "public_inputs_json_path",
            "public_json_path",
            "public_path",
            "publicSignalsPath",
            "public_signals_path",
        ):
            out = _try_load_public_inputs_json_v1(str(payload_obj.get(k, "") or ""))
            if out:
                return out
        artifact_path = str(payload_obj.get("artifact_path", "") or "")
        if artifact_path and os.path.exists(artifact_path):
            try:
                with open(artifact_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                out = _coerce_public_inputs_list_v1(data)
                if out:
                    return out
                if isinstance(data, dict):
                    out = _coerce_public_inputs_list_v1(data.get("public"))
                    if out:
                        return out
            except Exception:
                pass
    if isinstance(zkp_obj, dict):
        out = _coerce_public_inputs_list_v1(zkp_obj)
        if out:
            return out
    return []
def prove_verify_global_anchor_sum_from_meta(
    *,
    cfg: AnchorZKPConfig,
    meta: AnchorMeta,
    spec: Any = None,  # ✅ accept spec passed by newer call-sites
    anchor_id_field: int,
    round_idx: int,
    global_id: int,
    root_poseidon_field: int,
    pins_hash_field: int,
    policy_id_field: int,
    public_input_order_id_field: int,
    Q_rsu_list: Sequence[np.ndarray],
    mask_list: Optional[Sequence[int]],
    RMAX: int,
    NMAX: int,
    **kwargs: Any,
) -> AnchorZKPArtifact:
    # ✅ If caller provided a spec, forward it. If not, prove fn will build its own.
    if spec is not None:
        kwargs["spec"] = spec
    art = prove_verify_global_anchor_sum(
        cfg=cfg,
        anchor_id_field=anchor_id_field,
        anchor_version=meta.anchor_version,
        round_idx=round_idx,
        global_id=global_id,
        root_poseidon_field=int(root_poseidon_field),
        pins_hash_field=int(pins_hash_field),
        policy_id_field=int(policy_id_field),
        public_input_order_id_field=int(public_input_order_id_field),
        Q_rsu_list=Q_rsu_list,
        mask_list=mask_list,
        RMAX=RMAX,
        NMAX=NMAX,
        **kwargs,
    )
    # ------------------------------------------------------------------
    # ✅ Make `public_inputs` ALWAYS present + written to a stable V1 sidecar
    # ------------------------------------------------------------------
    public_inputs: List[str] = _extract_public_inputs_from_artifact_v1(art)
    out_dir = str(kwargs.get("out_dir") or getattr(cfg, "artifacts_dir", "") or "")
    stable_dir = os.path.join(out_dir, "global")
    stable_public_path = os.path.join(
        stable_dir, f"round_{int(round_idx)}_public_inputs_v1.json"
    )
    # Ensure dict containers exist (never allow KeyError paths downstream)
    if not isinstance(getattr(art, "public", None), dict):
        setattr(art, "public", {})
    if not isinstance(getattr(art, "payload", None), dict):
        setattr(art, "payload", {})
    if not isinstance(getattr(art, "hashes", None), dict):
        setattr(art, "hashes", {})
    # Always set the key (even if empty) so downstream code never KeyErrors
    art.public["public_inputs"] = list(public_inputs)
    art.payload["public_inputs"] = list(public_inputs)
    # ✅ Stable V1 sidecar pointer (preferred for on-chain export)
    art.payload["public_inputs_v1_json_path"] = str(stable_public_path)
    # (Optional compatibility alias — safe to keep older readers working)
    art.payload["public_inputs_json_path"] = str(stable_public_path)
    # Write stable file deterministically (canonical JSON via write_json)
    os.makedirs(stable_dir, exist_ok=True)
    write_json(Path(stable_public_path), list(public_inputs))
    # Hash the stable sidecar (for OnChainExportIndexV1 anchoring)
    try:
        art.hashes["public_inputs_v1_sha256"] = str(sha256_file(str(stable_public_path)))
    except Exception:
        pass
    # Rewrite the artifact JSON if it already exists (keep file stable)
    artifact_json_path = os.path.join(stable_dir, f"round_{int(round_idx)}.json")
    if os.path.exists(artifact_json_path):
        write_json(Path(artifact_json_path), art.to_dict())
    return art


# -----------------------------------------------------------------------------#
# Round orchestrator: variable RSUs + variable vehicles per RSU
# -----------------------------------------------------------------------------#
def prove_verify_round_anchor_sum(
    *,
    cfg: AnchorZKPConfig,
    meta: AnchorMeta,
    anchor_id_field: int,
    round_idx: int,
    rsu_q_by_id: Mapping[int, Sequence[np.ndarray]],
    rsu_root_poseidon_by_id: Mapping[int, int],
    global_root_poseidon_field: int,
    rsu_pins_hash_field: int,
    global_pins_hash_field: int,
    policy_id_field: int,
    public_input_order_id_field: int,
    rsu_mask_by_id: Optional[Mapping[int, Sequence[int]]] = None,
    NMAX: int,
    RMAX: int,
    global_id: int = 0,
    rsu_active_mask: Optional[Mapping[int, int]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    write_artifact: bool = True,
    include_Q_b64: bool = True,
    debug_inline_vectors: Optional[bool] = None,
    pipeline_fn: Optional[Callable[..., Any]] = None,
    overwrite_circuit: bool = True,
    overwrite_zkp_dir: bool = False,
) -> Tuple[list[AnchorZKPArtifact], AnchorZKPArtifact]:
    """
    One-call orchestration:
      - Prove/verify each RSU sum (variable vehicles per RSU, capped by NMAX)
      - Prove/verify global sum across RSUs (variable RSUs, capped by RMAX)
    rsu_root_poseidon_by_id:
      - REQUIRED mapping {rsu_id: root_poseidon_field_for_that_RSU_round}
    global_root_poseidon_field:
      - REQUIRED root to bind the global proof for this round.
    rsu_active_mask:
      - Optional dict {rsu_id: 0/1} to drop RSUs at global level.
      - If None, all RSUs in rsu_q_by_id are included.
    """
    if not include_Q_b64:
        raise ValueError(
            "prove_verify_round_anchor_sum requires include_Q_b64=True "
            "because it must decode RSU Q vectors and pass them to the global proof."
        )
    rsu_ids = sorted(int(k) for k in rsu_q_by_id.keys())
    _validate_cap(len(rsu_ids), int(RMAX), "RSUs (R)")
    rsu_arts: list[AnchorZKPArtifact] = []
    Q_list: list[np.ndarray] = []
    global_mask: list[int] = []
    for rsu_id in rsu_ids:
        q_list = rsu_q_by_id[rsu_id]
        m_list = None if rsu_mask_by_id is None else rsu_mask_by_id.get(rsu_id, None)
        # Fail fast if root missing (KeyError is desired)
        root_rsu = int(rsu_root_poseidon_by_id[rsu_id])
        art = prove_verify_rsu_anchor_sum_from_meta(
            cfg=cfg,
            meta=meta,
            anchor_id_field=anchor_id_field,
            round_idx=round_idx,
            rsu_id=rsu_id,
            root_poseidon_field=root_rsu,
            pins_hash_field=int(rsu_pins_hash_field),
            policy_id_field=int(policy_id_field),
            public_input_order_id_field=int(public_input_order_id_field),
            q_list=q_list,
            mask_list=m_list,
            NMAX=int(NMAX),
            out_dir=out_dir,
            write_artifact=write_artifact,
            include_Q_b64=include_Q_b64,
            debug_inline_vectors=debug_inline_vectors,
            pipeline_fn=pipeline_fn,
            overwrite_circuit=overwrite_circuit,
            overwrite_zkp_dir=overwrite_zkp_dir,
        )
        rsu_arts.append(art)
        Q = decode_int_vector_b64(art.payload["Q_rsu_b64"], expected_len=int(cfg.M))
        Q_list.append(Q)
        if rsu_active_mask is None:
            global_mask.append(1)
        else:
            global_mask.append(_bool01(rsu_active_mask.get(rsu_id, 0)))
    g_art = prove_verify_global_anchor_sum_from_meta(
        cfg=cfg,
        meta=meta,
        anchor_id_field=anchor_id_field,
        round_idx=round_idx,
        global_id=int(global_id),
        root_poseidon_field=int(global_root_poseidon_field),
        pins_hash_field=int(global_pins_hash_field),
        policy_id_field=int(policy_id_field),
        public_input_order_id_field=int(public_input_order_id_field),
        Q_rsu_list=Q_list,
        mask_list=global_mask,
        RMAX=int(RMAX),
        NMAX=int(NMAX),
        out_dir=out_dir,
        write_artifact=write_artifact,
        include_Q_b64=include_Q_b64,
        debug_inline_vectors=debug_inline_vectors,
        pipeline_fn=pipeline_fn,
        overwrite_circuit=overwrite_circuit,
        overwrite_zkp_dir=overwrite_zkp_dir,
    )
    return rsu_arts, g_art
# -----------------------------------------------------------------------------#
# Self-test (no Flower): variable RSUs + variable vehicles-per-RSU
# -----------------------------------------------------------------------------#
def self_test_anchorzkp(
    *,
    tmp_root: Union[str, Path] = "tmp_anchorzkp_v8a",
    M: int = 32,
    NMAX: int = 4,                 # cap: max vehicles per RSU (compile-time KMAX)
    RMAX: int = 5,                 # cap: max RSUs (compile-time KMAX)
    SCALE: int = DEFAULT_SCALE,
    enable_range_checks: bool = False,
    num_rsus: int = 3,             # actual RSUs this test run (<= RMAX)
    veh_counts: Optional[Sequence[int]] = None,  # per-RSU vehicles (each <= NMAX)
) -> Dict[str, Any]:
    tmp_root = _ensure_dir(tmp_root)
    gen_dir = _ensure_dir(tmp_root / "circuits" / "generated")
    art_dir = _ensure_dir(tmp_root / "artifacts")
    meta = AnchorMeta(
        schema="AnchorMetaV1",
        anchor_version=DEFAULT_ANCHOR_VERSION,
        M=int(M),
        SCALE=int(SCALE),
        seed=123,
        dtype="float32",
        shape=(int(M), 8),
        sha256_anchor_bytes=_sha256_hex(b"dummy_anchor_bytes"),
        root_poseidon_field=123456,  # NEW (dummy; self-test passes real per-round roots separately)
        created_ms=_now_ms(),
    )
    anchor_id = compute_anchor_id_field(meta)
    cfg = AnchorZKPConfig(
        M=int(M),
        SCALE=int(SCALE),
        enable_range_checks=bool(enable_range_checks),
        generated_circuits_dir=gen_dir,
        artifacts_dir=art_dir,
        run_root_dir=tmp_root / "run",
    )
    num_rsus = int(num_rsus)
    _validate_cap(num_rsus, int(RMAX), "RSUs (R)")
    if num_rsus < 1:
        raise ValueError("num_rsus must be >= 1")
    if veh_counts is None:
        veh_counts = [1 + (i % int(NMAX)) for i in range(num_rsus)]
    if len(veh_counts) != num_rsus:
        raise ValueError("veh_counts length must equal num_rsus")
    for c in veh_counts:
        if int(c) < 1 or int(c) > int(NMAX):
            raise ValueError(f"Each veh_counts[i] must be in [1..NMAX], got {veh_counts} vs NMAX={NMAX}")
    rng = np.random.default_rng(7)
    rsu_q_by_id: Dict[int, list[np.ndarray]] = {}
    for rsu_id in range(1, num_rsus + 1):
        nveh = int(veh_counts[rsu_id - 1])
        rsu_q_by_id[rsu_id] = [
            rng.integers(0, SCALE + 1, size=M, dtype=np.int64)
            for _ in range(nveh)
        ]
    # --- NEW: dummy Poseidon roots for self-test (Phase-2 wiring) ---
    # In the real pipeline, these will come from FL_IoV_MerkleSSI_UtilsV8b.py manifests.
    rsu_root_poseidon_by_id = {int(rsu_id): 1000 + int(rsu_id) for rsu_id in rsu_q_by_id.keys()}
    global_root_poseidon_field = 999999
    rsu_arts, g_art = prove_verify_round_anchor_sum(
        cfg=cfg,
        meta=meta,
        anchor_id_field=anchor_id,
        round_idx=1,
        rsu_q_by_id=rsu_q_by_id,
        rsu_root_poseidon_by_id=rsu_root_poseidon_by_id,
        global_root_poseidon_field=global_root_poseidon_field,
        rsu_pins_hash_field=111111,
        global_pins_hash_field=222222,
        policy_id_field=333333,
        public_input_order_id_field=int(public_input_order_id_field_v1(level="rsu")),
        rsu_mask_by_id=None,
        NMAX=int(NMAX),
        RMAX=int(RMAX),
        global_id=0,
        out_dir=art_dir,
        write_artifact=True,
        include_Q_b64=True,
    )
    return {
        "anchor_id_field": int(anchor_id),
        "rsu_ok": [bool(a.ok) for a in rsu_arts],
        "global_ok": bool(g_art.ok),
        "num_rsus": int(num_rsus),
        "veh_counts": [int(x) for x in veh_counts],
        "paths": {"generated_dir": str(gen_dir), "artifacts_dir": str(art_dir)},
    }
__all__ = [
    "BN128_PRIME",
    "DEFAULT_SCALE",
    "DEFAULT_ANCHOR_VERSION",
    "sha256_file",
    "precompile_anchorsum_groth16",
    "AnchorMeta",
    "AnchorZKPConfig",
    "CircuitSpec",
    "build_anchor_set",
    "load_anchor_set",
    "compute_anchor_id_field",
    "quantize_proba_to_int",
    "compress_b64",
    "decompress_b64",
    "encode_int_vector_b64",
    "decode_int_vector_b64",
    "make_vehicle_anchor_metrics",
    "build_client_update_record_sha256",
    "decode_vehicle_anchor_metrics_with_dp_sha",
    "decode_vehicle_anchor_metrics",
    "derive_r_chal",
    "poly_commit_with_mix",
    "write_anchorsum_circuit",
    "prove_verify_rsu_anchor_sum",
    "prove_verify_global_anchor_sum",
    "prove_verify_rsu_anchor_sum_from_meta",
    "prove_verify_global_anchor_sum_from_meta",
    "prove_verify_round_anchor_sum",
    "self_test_anchorzkp",
]
if __name__ == "__main__":
    out = self_test_anchorzkp()
    print(json.dumps(out, indent=2))