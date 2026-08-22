#FL_IoV_CanonicalSpecV8c.py
# -*- coding: utf-8 -*-
"""
FL_IoV_CanonicalSpecV8b.py
Canonical, versioned "contract" module for your two-lane system:
- Training lane produces canonical bytes / hashes.
- Verification lane consumes only canonical bytes / hashes / field elements.
Key design rules:
- Never change fields/encodings without bumping schema/version names.
- Avoid floats in signed/hashed records; use ints/strings only.
- Vectors are NEVER JSON: they are packed int64-le bytes (then optionally zlib+base64 for transport).
- Poseidon *preimage layouts* and *domain separators* are frozen here so Python and Circom match exactly.
Primary references:
- RFC 8785 (JCS) JSON Canonicalization Scheme: https://www.rfc-editor.org/rfc/rfc8785.html
- circomlib Poseidon gadget: https://github.com/iden3/circomlib/blob/master/circuits/poseidon.circom
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import struct
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
# =============================================================================
# BN254 / alt_bn128 scalar field prime (Ethereum pairing-friendly curve used by Groth16 verifiers)
# =============================================================================
BN254_PRIME: int = 21888242871839275222246405745257275088548364400416034343698204186575808495617
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
# =============================================================================
# Anchor (P0) Spec
# =============================================================================
@dataclasses.dataclass(frozen=True)
class AnchorSpecV1:
    """
    The forever P0 object spec: quantized anchor vectors (predict_proba on a frozen anchor set).
    If you change ANY of these, bump `anchor_version` and treat it as a new protocol generation.
    """
    anchor_version: str
    M: int
    SCALE: int
    # Documentation-only invariants (do not "interpret" these at runtime)
    rounding_rule: str = (
        "clip to [0,1], multiply by SCALE, numpy.rint (ties-to-even), cast to int64"
    )
    vector_format: str = "int64 little-endian × M (raw bytes); transport may be zlib+base64"
    hash_alg: str = "sha256"
# Default (matches your current V8b shape)
ANCHOR_SPEC_V1 = AnchorSpecV1(anchor_version="v1", M=64, SCALE=100_000)
# =============================================================================
# Hash / Field helpers (the only sanctioned way to go bytes->field here)
# =============================================================================
def sha256_digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _reduce_to_field(x: int, *, prime: int = BN254_PRIME) -> int:
    return int(x) % int(prime)

def assert_field(x: int, *, prime: int = BN254_PRIME) -> int:
    xi = int(x)
    if not (0 <= xi < int(prime)):
        raise ValueError(f"Field element out of range: {xi}")
    return xi

def sha256_to_field(data: bytes, *, prime: int = BN254_PRIME) -> int:
    """
    Deterministic SHA-256 -> BN254 field element by reduction.
    Caller should domain-separate at the preimage/layout level (preferred),
    not by tweaking this function.
    """
    x = int.from_bytes(sha256_digest(data), "big")
    return _reduce_to_field(x, prime=prime)

def sha256_hex_to_field(hex_str: str, *, prime: int = BN254_PRIME) -> int:
    """
    Convert a SHA-256 hex string (64 hex chars) to a BN254 field element by reduction.
    Empty string maps to 0 for convenience (optional fields).
    """
    s = (hex_str or "").strip().lower()
    if s == "":
        return 0
    if len(s) != 64:
        raise ValueError(f"Expected 64-hex SHA-256 string, got len={len(s)}: {s[:16]}...")
    raw = bytes.fromhex(s)
    x = int.from_bytes(raw, "big")
    return _reduce_to_field(x, prime=prime)

def strint_to_field(s: str, *, prime: int = BN254_PRIME) -> int:
    """
    Parse a non-negative base-10 integer string to a field element.
    """
    ss = str(s).strip()
    if ss == "":
        raise ValueError("Empty int string")
    x = int(ss, 10)
    if x < 0:
        raise ValueError(f"Expected non-negative int string, got {x}")
    return _reduce_to_field(x, prime=prime)

def u64_to_field(x: int, *, prime: int = BN254_PRIME) -> int:
    xi = int(x)
    if not (0 <= xi <= (1 << 64) - 1):
        raise ValueError(f"Expected uint64, got {xi}")
    return _reduce_to_field(xi, prime=prime)

def assert_sha256_hex_str_v1(s: str, *, allow_empty: bool = False, field_name: str = "sha256_hex") -> str:
    """
    Enforce lowercase 64-hex SHA-256 strings at the boundary.
    Returns the normalized lowercase string.
    """
    ss = ("" if s is None else str(s)).strip().lower()
    if ss == "":
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} must be a 64-hex string, got empty")
    if len(ss) != 64:
        raise ValueError(f"{field_name} must be 64 hex chars, got len={len(ss)}: {ss}")
    # Validate hex chars (no try/except ambiguity)
    for ch in ss:
        if ch not in "0123456789abcdef":
            raise ValueError(f"{field_name} contains non-hex char {ch!r}: {ss}")
    return ss

def assert_nonempty_str_v1(s: str, *, field_name: str) -> str:
    ss = ("" if s is None else str(s)).strip()
    if ss == "":
        raise ValueError(f"{field_name} must be non-empty")
    return ss
# =============================================================================
# Canonical JSON bytes (restricted JCS-style)
# =============================================================================
# =============================================================================
# Canonical JSON bytes (restricted JCS-style; enforced subset)
# =============================================================================
def _assert_canon_json_subset_v1(x: Any, *, path: str = "$") -> None:
    """
    Enforce the exact subset we allow in any signed/hashed canonical JSON object:
      - dict with string keys only
      - values are: str, int, bool, None, list/tuple of allowed values, dict (recursive)
    Hard reject:
      - float (including 1.0), Decimal, NaN/Infinity
      - bytes/bytearray (must be encoded to string before reaching here)
      - any other JSON-unserializable custom objects
    """
    # Reject floats explicitly (Python json would serialize them; we forbid them).
    if isinstance(x, float):
        raise ValueError(f"Floats are forbidden in canonical JSON at {path}: {x!r}")
    # Common scalar types that are allowed
    if x is None or isinstance(x, (str, int, bool)):
        return
    # Reject bytes-like (must be string)
    if isinstance(x, (bytes, bytearray, memoryview)):
        raise ValueError(f"Bytes-like values are forbidden in canonical JSON at {path}")
    # Containers
    if isinstance(x, Mapping):
        for k, v in x.items():
            if not isinstance(k, str):
                raise ValueError(f"Non-string JSON key at {path}: {k!r}")
            _assert_canon_json_subset_v1(v, path=f"{path}.{k}")
        return
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _assert_canon_json_subset_v1(v, path=f"{path}[{i}]")
        return
    raise ValueError(f"Unsupported type in canonical JSON at {path}: {type(x).__name__}")

def canon_json_bytes_v1(obj: Mapping[str, Any]) -> bytes:
    """
    Restricted JCS-like canonicalization for hash/signature stability:
    - sort_keys=True (deterministic member ordering)
    - separators=(',', ':') (no whitespace)
    - ensure_ascii=False (preserve UTF-8)
    - allow_nan=False (reject NaN/Infinity)
    Additionally ENFORCED here:
    - no floats anywhere
    - dict keys must be strings
    - only {str,int,bool,None,list,dict} value subset is allowed
    """
    _assert_canon_json_subset_v1(obj, path="$")
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
# =============================================================================
# Canonical vector bytes (int64-le × M)
# =============================================================================
def pack_int64_le(vec: Sequence[int], *, expect_len: Optional[int] = None) -> bytes:
    if expect_len is not None and len(vec) != int(expect_len):
        raise ValueError(f"Expected vector length {expect_len}, got {len(vec)}")
    out = bytearray()
    for i, v in enumerate(vec):
        vi = int(v)
        if not (_I64_MIN <= vi <= _I64_MAX):
            raise ValueError(f"int64 out of range at idx={i}: {vi}")
        out += struct.pack("<q", vi)
    return bytes(out)

def unpack_int64_le(data: bytes, *, expect_len: Optional[int] = None) -> Tuple[int, ...]:
    if len(data) % 8 != 0:
        raise ValueError(f"int64-le bytes length must be multiple of 8, got {len(data)}")
    n = len(data) // 8
    if expect_len is not None and n != int(expect_len):
        raise ValueError(f"Expected {expect_len} int64s, got {n}")
    return tuple(struct.unpack("<" + "q" * n, data))

def vec_sha256_hex_from_ints(vec: Sequence[int], *, expect_len: Optional[int] = None) -> str:
    return sha256_hex(pack_int64_le(vec, expect_len=expect_len))

def vec_sha256_hex_from_bytes(vec_bytes: bytes) -> str:
    return sha256_hex(vec_bytes)
# =============================================================================
# Poseidon domain separators (frozen tags -> field elements)
# IMPORTANT: Circom constants MUST match these exact tag strings.
# =============================================================================
DOMAIN_TAGS_V1: Dict[str, str] = {
    # Leaves (Merkle)
    "leaf_client_update_v1": "SSI-MedAI:Leaf:ClientUpdate:v1",
    "leaf_rsu_aggregate_v1": "SSI-MedAI:Leaf:RSUAggregate:v1",
    # Commitments used by AnchorSum proofs (bind proofs to published roots)
    "commit_rsu_anchorsum_v1": "SSI-MedAI:Commit:RSU:AnchorSum:v1",
    "commit_global_anchorsum_v1": "SSI-MedAI:Commit:GLOBAL:AnchorSum:v1",
}
DOMAIN_FIELDS_V1: Dict[str, int] = {
    k: sha256_to_field(("DOMAIN|" + v).encode("utf-8"))
    for k, v in DOMAIN_TAGS_V1.items()
}
# -----------------------------------------------------------------------------
# SSI-capable leaf domains (new; does not alter existing V1 layouts)
# -----------------------------------------------------------------------------
DOMAIN_TAGS_SSI_V1: Dict[str, str] = {
    # RSU-accepted client update envelope (binds DID + update_commit + DP + sig/auth + replay fields)
    "leaf_client_update_envelope_v1": "SSI-MedAI:Leaf:ClientUpdateEnvelope:v1",
}
DOMAIN_FIELDS_SSI_V1: Dict[str, int] = {
    k: sha256_to_field(("DOMAIN|" + v).encode("utf-8"))
    for k, v in DOMAIN_TAGS_SSI_V1.items()
}

def did_to_field_v1(did: str, *, prime: int = BN254_PRIME) -> int:
    """
    Map DID string to BN254 field element using the single sanctioned rule (SHA256 -> mod prime),
    with a fixed, frozen prefix for domain separation.
    """
    d = assert_nonempty_str_v1(did, field_name="did")
    return sha256_to_field(("DID|" + d).encode("utf-8"), prime=prime)

def policy_id_to_field_v1(policy_id: str, *, prime: int = BN254_PRIME) -> int:
    """
    Map policy_id string to field with a frozen prefix (domain separation).
    Empty policy_id maps to 0.
    """
    p = ("" if policy_id is None else str(policy_id)).strip()
    if p == "":
        return 0
    return sha256_to_field(("POLICY|" + p).encode("utf-8"), prime=prime)
# =============================================================================
# Poseidon preimage layouts (frozen order)
# These functions DO NOT compute Poseidon; they only define the exact field list.
# =============================================================================
def poseidon_leaf_preimage_client_update_v1(
    *,
    round_idx: int,
    rsu_id: int,
    vehicle_id: int,
    anchor_id_field: int,
    q_anchor_sha256_hex: str,
    dp_round_json_sha256_hex: str = "",
) -> List[int]:
    """
    Poseidon leaf preimage for ClientUpdateRecordV1 (ZK-friendly leaf).
    Frozen layout (in-order):
      [ DOMAIN(leaf_client_update_v1),
        round_idx,
        rsu_id,
        vehicle_id,
        anchor_id_field,
        field(q_anchor_sha256),
        field(dp_round_json_sha256) ]
    """
    return [
        assert_field(DOMAIN_FIELDS_V1["leaf_client_update_v1"]),
        u64_to_field(round_idx),
        u64_to_field(rsu_id),
        u64_to_field(vehicle_id),
        assert_field(_reduce_to_field(int(anchor_id_field))),
        sha256_hex_to_field(q_anchor_sha256_hex),
        sha256_hex_to_field(dp_round_json_sha256_hex or ""),
    ]

def poseidon_leaf_preimage_rsu_aggregate_v1(
    *,
    round_idx: int,
    rsu_id: int,
    anchor_id_field: int,
    n_used: int,
    q_rsu_sha256_hex: str,
    zkp_artifact_sha256_hex: str = "",
    root_poseidon_field: int = 0,
    root_sha256_hex: str = "",
) -> List[int]:
    """
    Poseidon leaf preimage for RSUAggregateRecordV1 (used for GLOBAL round root).
    Frozen layout (in-order):
      [ DOMAIN(leaf_rsu_aggregate_v1),
        round_idx,
        rsu_id,
        anchor_id_field,
        n_used,
        field(q_rsu_sha256),
        field(zkp_artifact_sha256),
        root_poseidon_field,
        field(root_sha256_hex) ]
    """
    return [
        assert_field(DOMAIN_FIELDS_V1["leaf_rsu_aggregate_v1"]),
        u64_to_field(round_idx),
        u64_to_field(rsu_id),
        assert_field(_reduce_to_field(int(anchor_id_field))),
        u64_to_field(n_used),
        sha256_hex_to_field(q_rsu_sha256_hex),
        sha256_hex_to_field(zkp_artifact_sha256_hex or ""),
        assert_field(_reduce_to_field(int(root_poseidon_field))),
        sha256_hex_to_field(root_sha256_hex or ""),
    ]

def poseidon_commitment_preimage_v1(
    *,
    domain_key: str,
    round_idx: int,
    entity_id: int,
    anchor_id_field: int,
    q_sha256_hex: str,
    root_poseidon_field: int,
    extra_fields: Sequence[int] = (),
) -> List[int]:
    """
    Generic commitment preimage helper for AnchorSum proofs.
    You will typically use:
      - domain_key="commit_rsu_anchorsum_v1"
      - domain_key="commit_global_anchorsum_v1"
    Frozen base layout (in-order):
      [ DOMAIN(domain_key),
        round_idx,
        entity_id,          # rsu_id for RSU proof, or 0 for global proof
        anchor_id_field,
        field(q_sha256_hex),
        root_poseidon_field,
        ...extra_fields ]
    """
    if domain_key not in DOMAIN_FIELDS_V1:
        raise KeyError(f"Unknown domain_key: {domain_key}")
    pre = [
        assert_field(DOMAIN_FIELDS_V1[domain_key]),
        u64_to_field(round_idx),
        u64_to_field(entity_id),
        assert_field(_reduce_to_field(int(anchor_id_field))),
        sha256_hex_to_field(assert_sha256_hex_str_v1(q_sha256_hex, allow_empty=False, field_name="q_sha256_hex")),
        assert_field(_reduce_to_field(int(root_poseidon_field))),
    ]
    for x in extra_fields:
        pre.append(assert_field(_reduce_to_field(int(x))))
    return pre

def poseidon_leaf_preimage_client_update_envelope_v1(
    *,
    round_idx: int,
    rsu_id: int,
    did: str,
    update_commit_sha256_hex: str,
    dp_record_sha256_hex: str,
    sig_sha256_hex: str,
    auth_evidence_sha256_hex: str,
    nonce: int,
    timestamp: int,
    data_fingerprint_sha256_hex: str = "",
    policy_id: str = "",
    signed_payload_sha256_hex: str = "",
) -> List[int]:
    """
    Poseidon leaf preimage for ClientUpdateEnvelopeV1 (SSI-capable).
    Frozen layout (in-order):
      [ DOMAIN(leaf_client_update_envelope_v1),
        round_idx,
        rsu_id,
        did_field,
        field(update_commit_sha256),
        field(dp_record_sha256),
        field(sig_sha256),
        field(auth_evidence_sha256),
        nonce,
        timestamp,
        field(data_fingerprint_sha256),
        policy_id_field,
        field(signed_payload_sha256) ]
    NOTE: signed_payload_sha256 is REQUIRED for this SSI schema.
    """
    upd = assert_sha256_hex_str_v1(update_commit_sha256_hex, allow_empty=False, field_name="update_commit_sha256")
    dp = assert_sha256_hex_str_v1(dp_record_sha256_hex, allow_empty=False, field_name="dp_record_sha256")
    sig = assert_sha256_hex_str_v1(sig_sha256_hex, allow_empty=False, field_name="sig_sha256")
    auth = assert_sha256_hex_str_v1(auth_evidence_sha256_hex, allow_empty=False, field_name="auth_evidence_sha256")
    fp = assert_sha256_hex_str_v1(data_fingerprint_sha256_hex, allow_empty=True, field_name="data_fingerprint_sha256")
    msg = assert_sha256_hex_str_v1(signed_payload_sha256_hex, allow_empty=False, field_name="signed_payload_sha256")
    return [
        assert_field(DOMAIN_FIELDS_SSI_V1["leaf_client_update_envelope_v1"]),
        u64_to_field(round_idx),
        u64_to_field(rsu_id),
        assert_field(did_to_field_v1(did)),
        sha256_hex_to_field(upd),
        sha256_hex_to_field(dp),
        sha256_hex_to_field(sig),
        sha256_hex_to_field(auth),
        u64_to_field(nonce),
        u64_to_field(timestamp),
        sha256_hex_to_field(fp),
        assert_field(policy_id_to_field_v1(policy_id)),
        sha256_hex_to_field(msg),
    ]
# -----------------------------
# Bytes/Dict -> Poseidon leaf preimage (V1 dispatcher; legacy schemas)
# -----------------------------
def _parse_intish(v: Any, *, default: int = 0) -> int:
    if v is None:
        return default
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if not s:
        return default
    return int(s)

def poseidon_leaf_preimage_from_record_dict_v1(rec: Mapping[str, Any]) -> List[int]:
    """
    Legacy dispatcher used by MerkleSSI (V1 schemas only).
    """
    schema = str(rec.get("schema", "")).strip()
    if schema == "ClientUpdateRecordV1":
        return poseidon_leaf_preimage_client_update_v1(
            round_idx=_parse_intish(rec.get("round_idx")),
            rsu_id=_parse_intish(rec.get("rsu_id")),
            vehicle_id=_parse_intish(rec.get("vehicle_id")),
            anchor_id_field=_parse_intish(rec.get("anchor_id_field")),
            q_anchor_sha256_hex=str(rec.get("q_anchor_sha256", "")),
            dp_round_json_sha256_hex=str(rec.get("dp_round_json_sha256", "") or ""),
        )
    if schema == "RSUAggregateRecordV1":
        return poseidon_leaf_preimage_rsu_aggregate_v1(
            round_idx=_parse_intish(rec.get("round_idx")),
            rsu_id=_parse_intish(rec.get("rsu_id")),
            anchor_id_field=_parse_intish(rec.get("anchor_id_field")),
            n_used=_parse_intish(rec.get("n_used")),
            q_rsu_sha256_hex=str(rec.get("q_rsu_sha256", "")),
            zkp_artifact_sha256_hex=str(rec.get("zkp_artifact_sha256", "") or ""),
            root_poseidon_field=_parse_intish(rec.get("root_poseidon_field"), default=0),
            root_sha256_hex=str(rec.get("root_sha256_hex", "") or ""),
        )
    raise ValueError(f"Unsupported schema for Poseidon leaf preimage: {schema}")

def assert_is_canon_json_bytes_v1(record_bytes: bytes) -> Dict[str, Any]:
    try:
        obj = json.loads(record_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError("record_bytes must be UTF-8 JSON") from e
    if not isinstance(obj, dict):
        raise ValueError("Decoded record JSON must be an object/dict")
    canon = canon_json_bytes_v1(obj)
    if canon != record_bytes:
        raise ValueError("record_bytes are NOT canonical (bytes mismatch after canon_json_bytes_v1)")
    return obj

def poseidon_leaf_preimage_from_record_bytes_v1(record_bytes: bytes) -> List[int]:
    obj = assert_is_canon_json_bytes_v1(record_bytes)
    return poseidon_leaf_preimage_from_record_dict_v1(obj)

# -----------------------------
# Bytes/Dict -> Poseidon leaf preimage (V2 dispatcher; SSI schemas)
# -----------------------------
def poseidon_leaf_preimage_from_record_dict_v2(rec: Mapping[str, Any]) -> List[int]:
    """
    SSI-capable dispatcher (new schemas). Does not change V1 behavior.
    """
    schema = str(rec.get("schema", "")).strip()
    if schema == "ClientUpdateEnvelopeV1":
        return poseidon_leaf_preimage_client_update_envelope_v1(
            round_idx=_parse_intish(rec.get("round_idx")),
            rsu_id=_parse_intish(rec.get("rsu_id")),
            did=str(rec.get("did", "")),
            update_commit_sha256_hex=str(rec.get("update_commit_sha256", "")),
            dp_record_sha256_hex=str(rec.get("dp_record_sha256", "")),
            sig_sha256_hex=str(rec.get("sig_sha256", "")),
            auth_evidence_sha256_hex=str(rec.get("auth_evidence_sha256", "")),
            nonce=_parse_intish(rec.get("nonce")),
            timestamp=_parse_intish(rec.get("timestamp")),
            data_fingerprint_sha256_hex=str(rec.get("data_fingerprint_sha256", "") or ""),
            policy_id=str(rec.get("policy_id", "") or ""),
            signed_payload_sha256_hex=str(rec.get("signed_payload_sha256", "") or ""),
        )
    return poseidon_leaf_preimage_from_record_dict_v1(rec)

def poseidon_leaf_preimage_from_record_bytes_v2(record_bytes: bytes) -> List[int]:
    obj = assert_is_canon_json_bytes_v1(record_bytes)
    return poseidon_leaf_preimage_from_record_dict_v2(obj)
# =============================================================================
# Record schemas (minimal V1s) - canonical bytes for signing/hashing
# =============================================================================
def build_client_update_record_v1(
    *,
    round_idx: int,
    rsu_id: int,
    vehicle_id: int,
    anchor_version: str,
    anchor_id_field: int,
    anchor_M: int,
    anchor_SCALE: int,
    q_anchor_sha256: str,
    dp_round_json_sha256: str = "",
) -> Tuple[bytes, str]:
    """
    Minimal signed/hashed record for a vehicle contribution binding (legacy anchor-based client record).
    """
    qh = assert_sha256_hex_str_v1(q_anchor_sha256, allow_empty=False, field_name="q_anchor_sha256")
    dph = assert_sha256_hex_str_v1(dp_round_json_sha256, allow_empty=True, field_name="dp_round_json_sha256")
    rec: Dict[str, Any] = {
        "schema": "ClientUpdateRecordV1",
        "round_idx": int(round_idx),
        "rsu_id": int(rsu_id),
        "vehicle_id": int(vehicle_id),
        "anchor_version": str(anchor_version),
        "anchor_id_field": str(int(anchor_id_field)),
        "anchor_M": int(anchor_M),
        "anchor_SCALE": int(anchor_SCALE),
        "q_anchor_sha256": qh,
        "dp_round_json_sha256": dph,
    }
    b = canon_json_bytes_v1(rec)
    return b, sha256_hex(b)

def build_rsu_aggregate_record_v1(
    *,
    round_idx: int,
    rsu_id: int,
    anchor_version: str,
    anchor_id_field: int,
    anchor_M: int,
    anchor_SCALE: int,
    n_used: int,
    q_rsu_sha256: str,
    zkp_artifact_sha256: str = "",
    root_sha256_hex: str = "",
    root_poseidon_field: str = "",
) -> Tuple[bytes, str]:
    """
    Minimal RSU aggregate record (legacy).
    """
    qh = assert_sha256_hex_str_v1(q_rsu_sha256, allow_empty=False, field_name="q_rsu_sha256")
    zkh = assert_sha256_hex_str_v1(zkp_artifact_sha256, allow_empty=True, field_name="zkp_artifact_sha256")
    rsh = assert_sha256_hex_str_v1(root_sha256_hex, allow_empty=True, field_name="root_sha256_hex")

    root_poseidon_field_str = "" if root_poseidon_field is None else str(root_poseidon_field)
    rec: Dict[str, Any] = {
        "schema": "RSUAggregateRecordV1",
        "round_idx": int(round_idx),
        "rsu_id": int(rsu_id),
        "anchor_version": str(anchor_version),
        "anchor_id_field": str(int(anchor_id_field)),
        "anchor_M": int(anchor_M),
        "anchor_SCALE": int(anchor_SCALE),
        "n_used": int(n_used),
        "q_rsu_sha256": qh,
        "zkp_artifact_sha256": zkh,
        "root_sha256_hex": rsh,
        "root_poseidon_field": root_poseidon_field_str,
    }
    b = canon_json_bytes_v1(rec)
    return b, sha256_hex(b)

def build_dp_record_v1_minimal(
    *,
    round_idx: int,
    mechanism: str,
    clip_l1: int,
    epsilon: str,
    delta: str = "0",
    accountant: str = "RDP",
    extra: str = "",
) -> Tuple[bytes, str]:
    """
    Minimal DP record for auditing. Keep it small and bind it by hash.
    epsilon/delta MUST be strings (no floats).
    """
    rec: Dict[str, Any] = {
        "schema": "DPRecordV1",
        "round_idx": int(round_idx),
        "mechanism": str(mechanism),
        "clip_l1": int(clip_l1),
        "epsilon": str(epsilon),
        "delta": str(delta),
        "accountant": str(accountant),
        "extra": str(extra or ""),
    }
    b = canon_json_bytes_v1(rec)
    return b, sha256_hex(b)
# =============================================================================
# SSI schemas (V1) - canonical bytes for signing/hashing
# =============================================================================
def build_client_update_signed_v1(
    *,
    did: str,
    round_idx: int,
    rsu_id: int,
    update_commit_sha256: str,
    dp_record_sha256: str,
    nonce: int,
    timestamp: int,
    policy_id: str = "",
    data_fingerprint_sha256: str = "",
) -> Tuple[bytes, str]:
    """
    What the CLIENT signs (canonical bytes).
    """
    upd = assert_sha256_hex_str_v1(update_commit_sha256, allow_empty=False, field_name="update_commit_sha256")
    dp = assert_sha256_hex_str_v1(dp_record_sha256, allow_empty=False, field_name="dp_record_sha256")
    fp = assert_sha256_hex_str_v1(data_fingerprint_sha256, allow_empty=True, field_name="data_fingerprint_sha256")
    rec: Dict[str, Any] = {
        "schema": "ClientUpdateSignedV1",
        "did": assert_nonempty_str_v1(did, field_name="did"),
        "rsu_id": int(rsu_id),
        "round_idx": int(round_idx),
        "policy_id": str(policy_id or ""),
        "update_commit_sha256": upd,
        "dp_record_sha256": dp,
        "nonce": int(nonce),
        "timestamp": int(timestamp),
        "data_fingerprint_sha256": fp,
    }
    b = canon_json_bytes_v1(rec)
    return b, sha256_hex(b)

def build_client_update_envelope_v1(
    *,
    did: str,
    round_idx: int,
    rsu_id: int,
    signed_payload_sha256: str,
    update_commit_sha256: str,
    dp_record_sha256: str,
    sig_sha256: str,
    auth_evidence_sha256: str,
    nonce: int,
    timestamp: int,
    policy_id: str = "",
    data_fingerprint_sha256: str = "",
) -> Tuple[bytes, str]:
    """
    What the RSU logs/anchors after it verifies authorization + signature + replay gates.
    """
    msg = assert_sha256_hex_str_v1(signed_payload_sha256, allow_empty=False, field_name="signed_payload_sha256")
    upd = assert_sha256_hex_str_v1(update_commit_sha256, allow_empty=False, field_name="update_commit_sha256")
    dp = assert_sha256_hex_str_v1(dp_record_sha256, allow_empty=False, field_name="dp_record_sha256")
    sig = assert_sha256_hex_str_v1(sig_sha256, allow_empty=False, field_name="sig_sha256")
    auth = assert_sha256_hex_str_v1(auth_evidence_sha256, allow_empty=False, field_name="auth_evidence_sha256")
    fp = assert_sha256_hex_str_v1(data_fingerprint_sha256, allow_empty=True, field_name="data_fingerprint_sha256")
    rec: Dict[str, Any] = {
        "schema": "ClientUpdateEnvelopeV1",
        "did": assert_nonempty_str_v1(did, field_name="did"),
        "rsu_id": int(rsu_id),
        "round_idx": int(round_idx),
        "policy_id": str(policy_id or ""),
        "signed_payload_sha256": msg,
        "update_commit_sha256": upd,
        "dp_record_sha256": dp,
        "nonce": int(nonce),
        "timestamp": int(timestamp),
        "sig_sha256": sig,
        "auth_evidence_sha256": auth,
        "data_fingerprint_sha256": fp,
    }
    b = canon_json_bytes_v1(rec)
    return b, sha256_hex(b)
# =============================================================================
# Parsers / validators for canonical SSI records (evidence helpers)
# =============================================================================
def parse_client_update_signed_v1(record_bytes: bytes) -> Dict[str, Any]:
    obj = assert_is_canon_json_bytes_v1(record_bytes)
    if str(obj.get("schema", "")).strip() != "ClientUpdateSignedV1":
        raise ValueError("Expected schema ClientUpdateSignedV1")
    assert_nonempty_str_v1(obj.get("did", ""), field_name="did")
    assert_sha256_hex_str_v1(str(obj.get("update_commit_sha256", "")), allow_empty=False, field_name="update_commit_sha256")
    assert_sha256_hex_str_v1(str(obj.get("dp_record_sha256", "")), allow_empty=False, field_name="dp_record_sha256")
    assert_sha256_hex_str_v1(str(obj.get("data_fingerprint_sha256", "")), allow_empty=True, field_name="data_fingerprint_sha256")
    int(obj.get("rsu_id"))
    int(obj.get("round_idx"))
    int(obj.get("nonce"))
    int(obj.get("timestamp"))
    return obj

def parse_client_update_envelope_v1(record_bytes: bytes) -> Dict[str, Any]:
    obj = assert_is_canon_json_bytes_v1(record_bytes)
    if str(obj.get("schema", "")).strip() != "ClientUpdateEnvelopeV1":
        raise ValueError("Expected schema ClientUpdateEnvelopeV1")
    assert_nonempty_str_v1(obj.get("did", ""), field_name="did")
    assert_sha256_hex_str_v1(str(obj.get("signed_payload_sha256", "")), allow_empty=False, field_name="signed_payload_sha256")
    assert_sha256_hex_str_v1(str(obj.get("update_commit_sha256", "")), allow_empty=False, field_name="update_commit_sha256")
    assert_sha256_hex_str_v1(str(obj.get("dp_record_sha256", "")), allow_empty=False, field_name="dp_record_sha256")
    assert_sha256_hex_str_v1(str(obj.get("sig_sha256", "")), allow_empty=False, field_name="sig_sha256")
    assert_sha256_hex_str_v1(str(obj.get("auth_evidence_sha256", "")), allow_empty=False, field_name="auth_evidence_sha256")
    assert_sha256_hex_str_v1(str(obj.get("data_fingerprint_sha256", "")), allow_empty=True, field_name="data_fingerprint_sha256")
    int(obj.get("rsu_id"))
    int(obj.get("round_idx"))
    int(obj.get("nonce"))
    int(obj.get("timestamp"))
    return obj
# =============================================================================
# Spec bundle (handy for manifests / auditors)
# =============================================================================
# =============================================================================
# Cross-implementation conformance vectors (Stage 0.5)
# =============================================================================
CONFORMANCE_VECTORS_V1: Dict[str, Any] = {
    "dp_record_v1_example": {
        "obj": {
            "schema": "DPRecordV1",
            "round_idx": 1,
            "mechanism": "laplace",
            "clip_l1": 100000,
            "epsilon": "1.5",
            "delta": "0",
            "accountant": "RDP",
            "extra": "",
        },
        "expected_sha256_hex": "f71d93e2b3ced37f9baac540b0cda7a7b7fef44c91b4b63f66bd6d4464d27d6b",
        "expected_sha256_field_bn254": "17611597584384910490899413866065532974229296709647785863589674805863165821843",
    },
    "client_update_envelope_v1_example": {
        "signed_obj": {
            "schema": "ClientUpdateSignedV1",
            "did": "did:example:vehicle-1001",
            "rsu_id": 1,
            "round_idx": 1,
            "policy_id": "policy_v8b_rsu",
            "update_commit_sha256": "92439dd4e7652a0414d2cd9d220e8d2767ed71f9005327223fc5d12b7a5065c3",
            "dp_record_sha256": "f71d93e2b3ced37f9baac540b0cda7a7b7fef44c91b4b63f66bd6d4464d27d6b",
            "nonce": 42,
            "timestamp": 1700000000,
            "data_fingerprint_sha256": "",
        },
        "expected_signed_payload_sha256": "4a23dce08b1bc0c1a81632bb174977fc540b15644e5e3ea5a5f5a82b1d200c7a",
        "envelope_obj": {
            "schema": "ClientUpdateEnvelopeV1",
            "did": "did:example:vehicle-1001",
            "rsu_id": 1,
            "round_idx": 1,
            "policy_id": "policy_v8b_rsu",
            "signed_payload_sha256": "4a23dce08b1bc0c1a81632bb174977fc540b15644e5e3ea5a5f5a82b1d200c7a",
            "update_commit_sha256": "92439dd4e7652a0414d2cd9d220e8d2767ed71f9005327223fc5d12b7a5065c3",
            "dp_record_sha256": "f71d93e2b3ced37f9baac540b0cda7a7b7fef44c91b4b63f66bd6d4464d27d6b",
            "nonce": 42,
            "timestamp": 1700000000,
            "sig_sha256": "238905fcd43c169f77b598d48a482ce3968aedc8e8b651f5809256ddb1dde5a6",
            "auth_evidence_sha256": "9062ebbadd6dbefa9aeadb9aed724b39a8c096b8399e2286dbc19d05d190abe5",
            "data_fingerprint_sha256": "",
        },
        "expected_envelope_sha256": "546978ba90087e78dd6e75f96cff7f930bc9262202ad5645950f937ebc121fcf",
        "expected_leaf_preimage_fields": [
            "16099576858564589749369733319404514840088249097607380157969363409983256292263",
            "1",
            "1",
            "20941733557773143842625293060366780189329556845494768975230309609514646468672",
            "492415345033114222824380715851685453739394843223288030331891529237385995712",
            "2332318474345903509768284050688613138000819635091105803795635166591447104870",
            "16073049073278181076724779135178899810972865975060892429691770281990704457126",
            "21531342412920830367314627811382137142540562280113261408285902257524759899107",
            "42",
            "1700000000",
            "0",
            "9746524124148412409655605232884460067271203709297053651119989751244229907631",
            "11646272008463853780089521096472008287616817343893792209995661532594169711737",
        ],
    },
}

# =============================================================================
# Spec bundle (handy for manifests / auditors)
# =============================================================================
SPEC_BUNDLE_V1: Dict[str, Any] = {
    "anchor_spec": dataclasses.asdict(ANCHOR_SPEC_V1),
    "bn254_prime": str(BN254_PRIME),
    "canonical_json": "json.dumps(sort_keys=True,separators=(',',':'),allow_nan=False) over restricted schemas",
    "vector_format": "int64 little-endian × M (raw bytes)",
    "domain_tags_v1": DOMAIN_TAGS_V1,
    "domain_fields_v1": {k: str(v) for k, v in DOMAIN_FIELDS_V1.items()},
    "poseidon_leaf_layouts_v1": {
        "ClientUpdateRecordV1": [
            "DOMAIN(leaf_client_update_v1)",
            "round_idx", "rsu_id", "vehicle_id", "anchor_id_field",
            "field(q_anchor_sha256)", "field(dp_round_json_sha256)",
        ],
        "RSUAggregateRecordV1": [
            "DOMAIN(leaf_rsu_aggregate_v1)",
            "round_idx", "rsu_id", "anchor_id_field", "n_used",
            "field(q_rsu_sha256)", "field(zkp_artifact_sha256)",
            "root_poseidon_field", "field(root_sha256_hex)",
        ],
    },
    "poseidon_commit_layout_v1": [
        "DOMAIN(domain_key)",
        "round_idx", "entity_id", "anchor_id_field",
        "field(q_sha256_hex)", "root_poseidon_field", "...extra_fields",
    ],
    "schemas": ["ClientUpdateRecordV1", "RSUAggregateRecordV1", "DPRecordV1"],
}

SPEC_BUNDLE_V2: Dict[str, Any] = {
    "base": SPEC_BUNDLE_V1,
    "domain_tags_ssi_v1": DOMAIN_TAGS_SSI_V1,
    "domain_fields_ssi_v1": {k: str(v) for k, v in DOMAIN_FIELDS_SSI_V1.items()},
    "poseidon_leaf_layouts_ssi_v1": {
        "ClientUpdateEnvelopeV1": [
            "DOMAIN(leaf_client_update_envelope_v1)",
            "round_idx",
            "rsu_id",
            "did_field",
            "field(update_commit_sha256)",
            "field(dp_record_sha256)",
            "field(sig_sha256)",
            "field(auth_evidence_sha256)",
            "nonce",
            "timestamp",
            "field(data_fingerprint_sha256)",
            "policy_id_field",
            "field(signed_payload_sha256)",
        ],
    },
    "schemas_add": ["ClientUpdateSignedV1", "ClientUpdateEnvelopeV1"],
    "conformance_vectors_v1": CONFORMANCE_VECTORS_V1,
}
# =============================================================================
# SSI preimage definition fingerprint (V1)
# Used for run-self-auditing: all RSUs/Global must log the same value.
# =============================================================================
def ssi_preimage_def_obj_v1() -> Dict[str, Any]:
    """
    A minimal, frozen definition of the SSI Poseidon preimage rule-set that MUST
    match across Python and Circom implementations.
    Keep this object stable; if you change layouts/rules, bump the schema name.
    """
    return {
        "schema": "SSIPreimageDefV1",
        "bn254_prime": str(BN254_PRIME),
        "domain_tags_ssi_v1": DOMAIN_TAGS_SSI_V1,
        "poseidon_leaf_layouts_ssi_v1": SPEC_BUNDLE_V2["poseidon_leaf_layouts_ssi_v1"],
        "rules": {
            "domain_field_rule": "sha256('DOMAIN|' + tag_utf8) reduced mod BN254_PRIME",
            "sha256_hex_to_field_rule": "64-hex -> bytes -> big-endian int -> mod BN254_PRIME; empty -> 0",
            "did_to_field_rule": "sha256('DID|' + did_utf8) reduced mod BN254_PRIME",
            "policy_id_to_field_rule": "sha256('POLICY|' + policy_utf8) reduced mod BN254_PRIME; empty -> 0",
            "u64_to_field_rule": "uint64 checked then reduced mod BN254_PRIME",
        },
    }
def ssi_preimage_def_sha256_v1() -> str:
    """
    Canonical SHA256 hex of ssi_preimage_def_obj_v1().
    """
    b = canon_json_bytes_v1(ssi_preimage_def_obj_v1())
    return sha256_hex(b)
def ssi_preimage_def_field_bn254_v1() -> int:
    """
    BN254 field element of the SSI preimage definition fingerprint.
    """
    return sha256_hex_to_field(ssi_preimage_def_sha256_v1())