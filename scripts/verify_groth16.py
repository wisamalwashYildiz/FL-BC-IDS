#!/usr/bin/env python3
"""Structurally validate and optionally cryptographically verify retained Groth16 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Validate retained JSON shapes without invoking snarkjs.",
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    canonical = root / "verification" / "groth16" / "canonical"
    retained = root / "verification" / "groth16" / "retained_proofs"

    vkeys = sorted(canonical.glob("*verification_key.json"))
    proofs = sorted(retained.glob("*proof.json"))
    publics = sorted(retained.glob("*public_inputs*.json"))

    if len(vkeys) != 1 or len(proofs) != 1 or len(publics) != 1:
        print("GROTH16 EVIDENCE: FAIL")
        print("FAIL: expected exactly one retained verification triplet")
        print("vkeys:", [path.name for path in vkeys])
        print("proofs:", [path.name for path in proofs])
        print("public inputs:", [path.name for path in publics])
        return 2

    try:
        vkey = load_json(vkeys[0])
        proof = load_json(proofs[0])
        public_inputs = load_json(publics[0])
    except Exception as exc:
        print("GROTH16 EVIDENCE: FAIL")
        print("FAIL: invalid JSON:", exc)
        return 2

    failures: list[str] = []

    protocol = str(vkey.get("protocol", "")).lower()
    if protocol and protocol != "groth16":
        failures.append(f"verification key protocol is not groth16: {protocol!r}")

    curve = str(vkey.get("curve", "")).lower()
    if curve and curve not in {"bn128", "bn254", "altbn128"}:
        failures.append(f"unexpected verification-key curve: {curve!r}")

    expected_shapes = {"pi_a": 2, "pi_b": 2, "pi_c": 2}
    for field, minimum_length in expected_shapes.items():
        value = proof.get(field)
        if not isinstance(value, list) or len(value) < minimum_length:
            failures.append(
                f"proof field {field} missing or structurally incomplete"
            )

    if not isinstance(public_inputs, list) or not public_inputs:
        failures.append("public-input JSON is not a non-empty list")

    if failures:
        print("GROTH16 EVIDENCE: FAIL")
        for item in failures:
            print("FAIL:", item)
        return 1

    print(
        "GROTH16 STRUCTURAL CHECK: PASS "
        f"(vkey={vkeys[0].name}, proof={proofs[0].name}, "
        f"public_inputs={publics[0].name})"
    )

    if args.structural_only:
        return 0

    snarkjs = shutil.which("snarkjs")
    npx = shutil.which("npx")

    if snarkjs:
        cmd = [
            snarkjs,
            "groth16",
            "verify",
            str(vkeys[0]),
            str(publics[0]),
            str(proofs[0]),
        ]
    elif npx:
        cmd = [
            npx,
            "--no-install",
            "snarkjs",
            "groth16",
            "verify",
            str(vkeys[0]),
            str(publics[0]),
            str(proofs[0]),
        ]
    else:
        print("GROTH16 CRYPTOGRAPHIC CHECK: NOT RUN")
        print(
            "FAIL: snarkjs/npx not found. Install the archived Node dependencies "
            "or use --structural-only only for JSON-shape validation."
        )
        return 3

    print("Running:", " ".join(str(item) for item in cmd))
    completed = subprocess.run(cmd, cwd=root, check=False)
    if completed.returncode != 0:
        print(
            f"GROTH16 CRYPTOGRAPHIC CHECK: FAIL "
            f"(snarkjs exit code {completed.returncode})"
        )
        return completed.returncode or 1

    print("GROTH16 CRYPTOGRAPHIC CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
