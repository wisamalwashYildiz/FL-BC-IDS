#!/usr/bin/env python3
"""Validate the retained publication-facing Sepolia blockchain evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
HASH32_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{64}$")
SENSITIVE_KEY_RE = re.compile(
    r"^(private[_-]?key|mnemonic|seed[_-]?phrase|api[_-]?key|access[_-]?token)$",
    re.IGNORECASE,
)
MASKED = {"", "[masked]", "<masked>", "masked", "redacted", "<redacted>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/).",
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def scan_sensitive(value, path: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if SENSITIVE_KEY_RE.match(str(key)):
                rendered = str(child).strip().lower() if child is not None else ""
                if rendered not in MASKED:
                    failures.append(
                        f"sensitive field has a non-masked value: {child_path}"
                    )
            failures.extend(scan_sensitive(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(scan_sensitive(child, f"{path}[{index}]"))
    return failures


def collect_addresses(value) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            found |= collect_addresses(child)
    elif isinstance(value, list):
        for child in value:
            found |= collect_addresses(child)
    elif isinstance(value, str) and ADDRESS_RE.fullmatch(value):
        found.add(value.lower())
    return found


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    blockchain = root / "verification" / "blockchain"

    required = [
        "deployed_registry_sepolia.json",
        "deployed_verifiers_sepolia.json",
        "onchain_verification_report.json",
        "ProofRegistryV1.sol",
    ]
    missing = [name for name in required if not (blockchain / name).is_file()]
    if missing:
        print("BLOCKCHAIN EVIDENCE: FAIL")
        print("FAIL: missing blockchain evidence:", missing)
        return 2

    failures: list[str] = []
    report = load_json(blockchain / "onchain_verification_report.json")
    registry = load_json(blockchain / "deployed_registry_sepolia.json")
    verifiers = load_json(blockchain / "deployed_verifiers_sepolia.json")

    if report.get("chain_id") != 11155111:
        failures.append("chain_id is not Sepolia 11155111")

    network = report.get("network_evaluation", {}).get("network")
    if network is not None and str(network).lower() != "sepolia":
        failures.append(f"network_evaluation.network is not Sepolia: {network!r}")

    rpc_used = str(report.get("rpc_url_used", "")).strip().lower()
    if rpc_used not in MASKED:
        failures.append("rpc_url_used is not masked")

    failures.extend(scan_sensitive(report, "report"))
    failures.extend(scan_sensitive(registry, "registry"))
    failures.extend(scan_sensitive(verifiers, "verifiers"))

    submissions = report.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != 5:
        failures.append(
            f"expected exactly five proof submissions; got "
            f"{len(submissions) if isinstance(submissions, list) else 'non-list'}"
        )
        submissions = []

    for index, submission in enumerate(submissions, start=1):
        prefix = f"submission[{index}]"
        if submission.get("submitted") is not True:
            failures.append(f"{prefix}: submitted != true")
        if submission.get("mined") is not True:
            failures.append(f"{prefix}: mined != true")
        if submission.get("status") != 1:
            failures.append(f"{prefix}: receipt status != 1")
        if submission.get("verified_ok") is not True:
            failures.append(f"{prefix}: verified_ok != true")
        tx_hash = str(submission.get("tx_hash", ""))
        if not HASH32_RE.fullmatch(tx_hash):
            failures.append(f"{prefix}: invalid/missing transaction hash")

    finality = report.get("network_evaluation", {}).get(
        "finality_measurement", {}
    )
    if finality:
        if finality.get("timed_out_transactions") != 0:
            failures.append("one or more finality observations timed out")
        eligible = finality.get("eligible_transactions")
        finalized = finality.get("finalized_transactions")
        if eligible is not None and finalized != eligible:
            failures.append(
                f"finality mismatch: finalized={finalized}, eligible={eligible}"
            )

    security_scope = report.get("contracts", {}).get("security_scope", {})
    if security_scope:
        if security_scope.get("formal_security_audit_claimed") is not False:
            failures.append("formal_security_audit_claimed must remain false")
        if security_scope.get("formal_verification_claimed") is not False:
            failures.append("formal_verification_claimed must remain false")

    contract_addresses = {
        str(value).lower()
        for key, value in report.get("contracts", {}).items()
        if key in {"rsu_verifier", "global_verifier", "proof_registry"}
        and isinstance(value, str)
        and ADDRESS_RE.fullmatch(value)
    }
    if len(contract_addresses) != 3:
        failures.append("report does not contain three valid contract addresses")
    else:
        deployed_addresses = (
            collect_addresses(registry) | collect_addresses(verifiers)
        )
        missing_addresses = contract_addresses - deployed_addresses
        if missing_addresses:
            failures.append(
                "report contract addresses absent from deployment files: "
                + ", ".join(sorted(missing_addresses))
            )

    if failures:
        print("BLOCKCHAIN EVIDENCE: FAIL")
        for item in failures:
            print("FAIL:", item)
        return 1

    print(
        "BLOCKCHAIN EVIDENCE: PASS "
        "(Sepolia; 5/5 proof submissions valid; credentials masked; "
        "deployment addresses cross-linked)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
