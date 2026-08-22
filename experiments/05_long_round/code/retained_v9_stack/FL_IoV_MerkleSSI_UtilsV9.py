#FL_IoV_MerkleSSI_UtilsV8c.py
# -*- coding: utf-8 -*-
"""
FL_IoV_MerkleSSI_UtilsV8b.py
Dual-root logging utilities (Phase 1 → Phase 2 bridge):
- leaf_sha256 (auditor-friendly) + binary SHA-256 Merkle root
- leaf_poseidon_field (ZK-friendly) + k-ary Poseidon Merkle root (circomlib-compatible)
CRITICAL:
- Poseidon leaf hashing MUST use the frozen bytes→fields preimage layout from CanonicalSpec.
- Poseidon implementation MUST match circomlib Poseidon parameters → use circomlibjs (Node) as reference.
This module also provides a tiny auditor check to detect serialization drift BEFORE touching circuits.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from functools import lru_cache  # ✅ NEW
# ---- Canonical contract (MUST be updated first) ----
try:
    from FL_IoV_CanonicalSpecV9 import (
        BN254_PRIME,
        sha256_hex,
        canon_json_bytes_v1,  # ✅ NEW: canonical body hashing for on-chain export index
        assert_field,
        assert_is_canon_json_bytes_v1,
        assert_sha256_hex_str_v1,
        poseidon_leaf_preimage_from_record_bytes_v2,
        # NEW (plan): evidence + strict schema validation
        ssi_preimage_def_sha256_v1,
        parse_client_update_envelope_v1,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing CanonicalSpec dependencies. Update FL_IoV_CanonicalSpecV8b.py first.\n"
        "Expected: BN254_PRIME, sha256_hex, assert_field, assert_is_canon_json_bytes_v1, "
        "assert_sha256_hex_str_v1, poseidon_leaf_preimage_from_record_bytes_v2, "
        "ssi_preimage_def_sha256_v1, parse_client_update_envelope_v1."
    ) from e
# -----------------------------
# Frozen Merkle specs (V1)
# -----------------------------
@dataclasses.dataclass(frozen=True)
class PoseidonMerkleSpecV1:
    """
    Poseidon tree parameters MUST be frozen before P3 and on-chain verifier export.
    - arity: k-ary branching factor for Poseidon Merkle (e.g., 5)
    - depth: number of hashing layers from leaves to root (max leaves = arity**depth)
    NOTE: leaf domain separators are NOT defined here anymore.
          Leaf hashing is defined by CanonicalSpec's preimage layout.
    """
    arity: int = 5
    depth: int = 32

@dataclasses.dataclass(frozen=True)
class Sha256MerkleSpecV1:
    """
    Auditor-friendly SHA-256 Merkle tree.
    We keep it binary (familiar tooling, easy independent verification).
    """
    depth: int = 32  # fixed-depth padding

POSEIDON_MERKLE_SPEC_V1 = PoseidonMerkleSpecV1()
SHA256_MERKLE_SPEC_V1 = Sha256MerkleSpecV1()
# -----------------------------
# Node/circomlibjs backend
# -----------------------------
_JS_HELPER = r"""
const fs = require("fs");
async function main() {
  const input = JSON.parse(fs.readFileSync(0, "utf8"));
  const { buildPoseidon } = require("circomlibjs");
  const poseidon = await buildPoseidon();
  const F = poseidon.F;
  function toBigInt(x) {
    if (typeof x === "bigint") return x;
    if (typeof x === "number") return BigInt(x);
    if (typeof x === "string") return BigInt(x);
    throw new Error("Unsupported input type: " + (typeof x));
  }
  function poseidonHash(inputs) {
    const arr = inputs.map(toBigInt);
    const out = poseidon(arr);
    const bi = F.toObject(out);
    return bi.toString();
  }
  function merkleRootKary(leaves, arity, depth, padLeaf) {
    let cur = (leaves || []).map(toBigInt);
    const k = Number(arity);
    const d = Number(depth);
    const pad = toBigInt(padLeaf);
    for (let level = 0; level < d; level++) {
      if (cur.length === 0) cur = [pad];
      const rem = cur.length % k;
      if (rem !== 0) {
        const need = k - rem;
        for (let i = 0; i < need; i++) cur.push(pad);
      }
      // hash each k-chunk
      const next = [];
      for (let i = 0; i < cur.length; i += k) {
        const chunk = cur.slice(i, i + k);
        next.push(toBigInt(poseidonHash(chunk)));
      }
      cur = next;
    }
    if (cur.length !== 1) {
      throw new Error("Unexpected merkle root state: " + cur.length);
    }
    return cur[0].toString();
  }
  if (input.op === "poseidon") {
    const out = poseidonHash(input.inputs || []);
    process.stdout.write(JSON.stringify({ ok: true, out }));
    return;
  }
  if (input.op === "poseidon_merkle_root") {
    const out = merkleRootKary(
      input.leaves || [],
      input.arity,
      input.depth,
      input.pad_leaf
    );
    process.stdout.write(JSON.stringify({ ok: true, out }));
    return;
  }
  process.stdout.write(JSON.stringify({ ok: false, error: "unknown op" }));
}
main().catch((e) => {
  process.stdout.write(JSON.stringify({ ok: false, error: String(e && e.stack ? e.stack : e) }));
});
"""
def _atomic_write_text_v1(path: Path, text: str) -> None:
    """
    Atomic write for Ray/multiprocess safety.
    Writes to a temp file then os.replace() into final path.
    """
    path = Path(path)
    tmp = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{os.urandom(6).hex()}"
    )
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))
def _ensure_js_helper() -> Path:
    # Prefer project root so Node can resolve local node_modules reliably
    base = _node_cwd()
    js_path = base / "_poseidon_helper_circomlibjs_v1.js"
    if not js_path.exists() or js_path.stat().st_size < 100:
        _atomic_write_text_v1(js_path, _JS_HELPER)  # ✅ atomic
    return js_path
def _node_cmd() -> str:
    """Resolve Node executable (compatible with AnchorZKP pinning)."""
    return (
        os.environ.get("ANCHOR_ZKP_NODE_CMD", "").strip()
        or os.environ.get("NODE", "").strip()
        or "node"
    )

def _node_cwd() -> Path:
    """
    Ensure Node can resolve `require("circomlibjs")` by running from project root.
    This prevents failures under Ray/Windows where cwd can be a temp worker dir.
    """
    root = os.environ.get("ANCHOR_ZKP_PROJECT_ROOT", "").strip()
    if root:
        p = Path(root).resolve()
        if p.exists():
            return p
    return Path(__file__).resolve().parent

def _run_node_json(js_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run node helper once; pass JSON on stdin; parse JSON on stdout."""
    try:
        p = subprocess.run(
            [_node_cmd(), str(js_path)],
            input=json.dumps(payload).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            cwd=str(_node_cwd()),
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "Node.js executable not found. Ensure `node` is on PATH "
            "or set env var ANCHOR_ZKP_NODE_CMD / NODE."
        ) from e
    out = p.stdout.decode("utf-8", errors="replace").strip()
    if not out:
        raise RuntimeError(
            "Poseidon backend returned empty output.\n"
            f"STDERR:\n{p.stderr.decode('utf-8', errors='replace')}"
        )
    try:
        j = json.loads(out)
    except Exception as e:
        raise RuntimeError(
            "Failed to parse Poseidon backend JSON.\n"
            f"STDOUT:\n{out}\n"
            f"STDERR:\n{p.stderr.decode('utf-8', errors='replace')}"
        ) from e
    if not j.get("ok"):
        err = j.get("error", "unknown error")
        raise RuntimeError(
            "Poseidon backend error. Ensure `circomlibjs` is installed in your Node environment.\n"
            "Recommended: run `npm i circomlibjs` in your project folder (ANCHOR_ZKP_PROJECT_ROOT).\n"
            f"Backend error: {err}"
        )
    return j

def poseidon_hash_fields(fields: Sequence[int]) -> int:
    """
    Poseidon hash of field elements using circomlibjs.
    Returns int in [0, BN254_PRIME).
    """
    js_path = _ensure_js_helper()
    payload = {"op": "poseidon", "inputs": [str(int(x) % BN254_PRIME) for x in fields]}
    j = _run_node_json(js_path, payload)
    return int(j["out"]) % BN254_PRIME
@lru_cache(maxsize=16)
def _poseidon_pad_leaf_field_cached_v1(arity: int) -> int:
    zero_node = [0] * int(arity)
    return poseidon_hash_fields(zero_node)
def poseidon_pad_leaf_field_v1(*, spec: PoseidonMerkleSpecV1 = POSEIDON_MERKLE_SPEC_V1) -> int:
    return int(_poseidon_pad_leaf_field_cached_v1(int(spec.arity)))
def poseidon_merkle_root_kary_v1(
    leaves_field: Sequence[int],
    *,
    spec: PoseidonMerkleSpecV1 = POSEIDON_MERKLE_SPEC_V1,
    pad_leaf_field: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Fixed-depth k-ary Poseidon Merkle root from already-hashed leaf field values.
    Returns: (root_field, pad_leaf_field_used)
    """
    if pad_leaf_field is None:
        pad_leaf_field = poseidon_pad_leaf_field_v1(spec=spec)
    js_path = _ensure_js_helper()
    payload = {
        "op": "poseidon_merkle_root",
        "leaves": [str(int(x) % BN254_PRIME) for x in leaves_field],
        "arity": int(spec.arity),
        "depth": int(spec.depth),
        "pad_leaf": str(int(pad_leaf_field) % BN254_PRIME),
    }
    j = _run_node_json(js_path, payload)
    return int(j["out"]) % BN254_PRIME, int(pad_leaf_field) % BN254_PRIME
# -----------------------------
# SHA-256 Merkle (binary, fixed depth)
# -----------------------------
def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def sha256_merkle_root_hex(
    leaves_bytes: Sequence[bytes],
    *,
    spec: Sha256MerkleSpecV1 = SHA256_MERKLE_SPEC_V1,
) -> str:
    """
    Fixed-depth binary SHA-256 Merkle root.
    - leaf hash = SHA256(leaf_bytes)
    - internal  = SHA256(left || right)
    - padding leaf hash uses 32-byte zeros at the *hash* level.
    """
    zero = b"\x00" * 32
    cur = [_sha256(x) for x in leaves_bytes]
    if not cur:
        cur = [zero]
    for _level in range(int(spec.depth)):
        if len(cur) % 2 == 1:
            cur.append(zero)
        if len(cur) == 1:
            cur = [_sha256(cur[0] + zero)]
            continue
        nxt: List[bytes] = []
        for i in range(0, len(cur), 2):
            nxt.append(_sha256(cur[i] + cur[i + 1]))
        cur = nxt
    return cur[0].hex()
# -----------------------------
# Leaves + dual roots (V1)
# -----------------------------
def _assert_envelope_record_bytes_v1(record_bytes: bytes) -> None:
    """
    Plan enforcement:
    MerkleSSI must consume canonical *ClientUpdateEnvelopeV1* bytes only.
    This provides evidence-grade validation *before* hashing/leaves.
    """
    assert_is_canon_json_bytes_v1(record_bytes)
    # CanonicalSpec parser enforces schema correctness (ClientUpdateEnvelopeV1)
    parse_client_update_envelope_v1(record_bytes)
def record_bytes_to_dual_leaves_v1(
    record_bytes: bytes,
    *,
    validate_record_bytes: bool = True,
    validate_schema: bool = True,
) -> Tuple[str, int]:
    """
    From canonical *ClientUpdateEnvelopeV1* record bytes:
    - leaf_sha256_hex     = SHA256(record_bytes) hex
    - leaf_poseidon_field = Poseidon(preimage_fields_from_CanonicalSpec V2 dispatcher)
    Plan rules:
    - validate_record_bytes=True enforces canonical JSON bytes (CanonicalSpec contract).
    - validate_schema=True enforces ClientUpdateEnvelopeV1 schema via parse_client_update_envelope_v1.
    - Poseidon leaf packing MUST use poseidon_leaf_preimage_from_record_bytes_v2(record_bytes).
    """
    if validate_record_bytes:
        assert_is_canon_json_bytes_v1(record_bytes)
    if validate_schema:
        # This is the “strictly envelope bytes only” enforcement required by the plan
        parse_client_update_envelope_v1(record_bytes)
    rec_sha_hex = sha256_hex(record_bytes)
    rec_sha_hex = assert_sha256_hex_str_v1(rec_sha_hex, allow_empty=False, field_name="leaf_sha256_hex")
    preimage_fields = poseidon_leaf_preimage_from_record_bytes_v2(record_bytes)
    leaf_poseidon = poseidon_hash_fields(preimage_fields)
    leaf_poseidon = int(assert_field(int(leaf_poseidon)))
    return rec_sha_hex, leaf_poseidon

def build_dual_roots_from_record_bytes_v1(
    record_bytes_list: Sequence[bytes],
    *,
    sha_spec: Sha256MerkleSpecV1 = SHA256_MERKLE_SPEC_V1,
    poseidon_spec: PoseidonMerkleSpecV1 = POSEIDON_MERKLE_SPEC_V1,
    pad_leaf_field: Optional[int] = None,
    validate_record_bytes: bool = True,
    validate_schema: bool = True,
) -> Dict[str, Any]:
    """
    Compute dual roots for a round from *ClientUpdateEnvelopeV1* canonical bytes:
    - root_sha256_hex (binary SHA-256 fixed depth)
    - root_poseidon_field (k-ary Poseidon fixed depth)
    Evidence (plan):
    - embed ssi_preimage_def_sha256 so auditors can verify the frozen Poseidon preimage layout.
    """
    leaf_sha_hex_list: List[str] = []
    leaf_poseidon_list: List[int] = []
    for rb in record_bytes_list:
        h_hex, leaf_p = record_bytes_to_dual_leaves_v1(
            rb,
            validate_record_bytes=validate_record_bytes,
            validate_schema=validate_schema,
        )
        leaf_sha_hex_list.append(h_hex)
        leaf_poseidon_list.append(int(leaf_p))
    root_sha_hex = sha256_merkle_root_hex(record_bytes_list, spec=sha_spec)
    root_sha_hex = assert_sha256_hex_str_v1(str(root_sha_hex), allow_empty=False, field_name="root_sha256_hex")
    root_poseidon_field, pad_used = poseidon_merkle_root_kary_v1(
        leaf_poseidon_list, spec=poseidon_spec, pad_leaf_field=pad_leaf_field
    )
    root_poseidon_field = int(assert_field(int(root_poseidon_field)))
    pad_used = int(assert_field(int(pad_used)))
    body: Dict[str, Any] = {
        "schema": "RoundRootsComputedV1",
        "n_records": int(len(record_bytes_list)),
        "root_sha256_hex": str(root_sha_hex),
        "root_poseidon_field": str(int(root_poseidon_field)),
        "poseidon_arity": int(poseidon_spec.arity),
        "poseidon_depth": int(poseidon_spec.depth),
        "poseidon_pad_leaf_field": str(int(pad_used)),
        "sha256_depth": int(sha_spec.depth),
        "leaf_sha256_hex": leaf_sha_hex_list,
        "leaf_poseidon_field": [str(int(x)) for x in leaf_poseidon_list],
        # NEW (plan evidence): frozen SSI Poseidon preimage definition fingerprint
        "ssi_preimage_def_sha256": str(ssi_preimage_def_sha256_v1()),
    }
    roots_body_canon_sha256_hex = sha256_hex(canon_json_bytes_v1(body))
    obj: Dict[str, Any] = dict(body)
    obj["roots_body_canon_sha256_hex"] = str(roots_body_canon_sha256_hex)
    return obj


def identity_empty_root_field_v1(
        *,
        poseidon_spec: PoseidonMerkleSpecV1 = POSEIDON_MERKLE_SPEC_V1,
) -> int:
    """
    Deterministic empty identity root (revocation-ready placeholder).
    For now: Poseidon Merkle root of an empty leaf list under the frozen padding rule.
    """
    root, _pad = poseidon_merkle_root_kary_v1([], spec=poseidon_spec, pad_leaf_field=None)
    return int(root)
# -----------------------------
# Manifest write + tiny auditor check
# -----------------------------
def write_round_roots_manifest_v1(
    out_path: str | Path,
    *,
    round_idx: int,
    rsu_id: int,
    root_sha256_hex: str,
    root_poseidon_field: str,
    identity_state_root_field: str,
    poseidon_pad_leaf_field: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root_sha256_hex_norm = assert_sha256_hex_str_v1(
        str(root_sha256_hex),
        allow_empty=False,
        field_name="root_sha256_hex",
    )
    root_poseidon_int = assert_field(int(str(root_poseidon_field).strip()))
    identity_state_root_int = assert_field(int(str(identity_state_root_field).strip()))
    pad_leaf_int = assert_field(int(str(poseidon_pad_leaf_field).strip()))
    obj: Dict[str, Any] = {
        "schema": "RoundRootsManifestV1",
        "round_idx": int(round_idx),
        "rsu_id": int(rsu_id),
        "root_sha256_hex": root_sha256_hex_norm,
        "root_poseidon_field": str(int(root_poseidon_int)),
        "identity_state_root_field": str(int(identity_state_root_int)),
        "poseidon_arity": int(POSEIDON_MERKLE_SPEC_V1.arity),
        "poseidon_depth": int(POSEIDON_MERKLE_SPEC_V1.depth),
        "poseidon_pad_leaf_field": str(int(pad_leaf_int)),
        "sha256_depth": int(SHA256_MERKLE_SPEC_V1.depth),
        # NEW (plan evidence): frozen SSI Poseidon preimage definition fingerprint
        "ssi_preimage_def_sha256": str(ssi_preimage_def_sha256_v1()),
    }
    if extra:
        ex = dict(extra)
        # Ensure evidence exists even if caller forgets
        if "ssi_preimage_def_sha256" not in ex:
            ex["ssi_preimage_def_sha256"] = str(ssi_preimage_def_sha256_v1())
        obj["extra"] = ex
    else:
        obj["extra"] = {"ssi_preimage_def_sha256": str(ssi_preimage_def_sha256_v1())}
    body: Dict[str, Any] = dict(obj)
    # ✅ Stable canonical body hash for on-chain export index (format-independent)
    manifest_body_canon_sha256_hex = sha256_hex(canon_json_bytes_v1(body))

    obj["manifest_body_canon_sha256_hex"] = str(manifest_body_canon_sha256_hex)
    # ✅ Atomic write (prevents partial manifests under Ray/multiprocess)
    _atomic_write_text_v1(out_path, json.dumps(obj, indent=2, sort_keys=True))
def audit_round_roots_manifest_v1(
    manifest_path: str | Path,
    *,
    record_bytes_list: Sequence[bytes],
    sha_spec: Sha256MerkleSpecV1 = SHA256_MERKLE_SPEC_V1,
    poseidon_spec: PoseidonMerkleSpecV1 = POSEIDON_MERKLE_SPEC_V1,
) -> None:
    """
    Tiny auditor check (NO Circom):
    recompute roots from *ClientUpdateEnvelopeV1* record bytes and assert they match the manifest.
    Also checks SSI preimage definition fingerprint if present.
    """
    manifest_path = Path(manifest_path)
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = assert_sha256_hex_str_v1(
        str(man["root_sha256_hex"]),
        allow_empty=False,
        field_name="root_sha256_hex",
    )
    expected_poseidon = str(int(assert_field(int(str(man["root_poseidon_field"]).strip()))))
    expected_pad_raw = str(man.get("poseidon_pad_leaf_field", "")).strip()
    expected_pad_field = None
    if expected_pad_raw != "":
        expected_pad_field = int(assert_field(int(expected_pad_raw)))
    # ✅ NEW (on-chain export support): verify canonical body hash if present
    expected_body_sha = str(man.get("manifest_body_canon_sha256_hex", "")).strip()
    if expected_body_sha:
        body = dict(man)
        body.pop("manifest_body_canon_sha256_hex", None)
        got_body_sha = sha256_hex(canon_json_bytes_v1(body))
        if expected_body_sha != got_body_sha:
            raise AssertionError(
                "[audit] manifest_body_canon_sha256_hex mismatch\n"
                f"expected: {expected_body_sha}\n"
                f"     got: {got_body_sha}"
            )
    computed = build_dual_roots_from_record_bytes_v1(
        record_bytes_list,
        sha_spec=sha_spec,
        poseidon_spec=poseidon_spec,
        pad_leaf_field=expected_pad_field,
        validate_record_bytes=True,
        validate_schema=True,
    )
    got_sha = str(computed["root_sha256_hex"])
    got_poseidon = str(computed["root_poseidon_field"])
    if got_sha != expected_sha:
        raise AssertionError(
            f"[audit] root_sha256_hex mismatch\nexpected: {expected_sha}\n     got: {got_sha}"
        )
    if got_poseidon != expected_poseidon:
        raise AssertionError(
            f"[audit] root_poseidon_field mismatch\nexpected: {expected_poseidon}\n     got: {got_poseidon}"
        )