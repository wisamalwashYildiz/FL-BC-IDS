#!/usr/bin/env python3
"""Publication-facing structural/security self-check for FL-BC-IDS."""

from __future__ import annotations

import argparse
import ast
import json
import hashlib
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


REQUIRED = [
    "README.md",
    "LICENSE",
    "MANIFEST.csv",
    "CHECKSUMS_SHA256.txt",
    "CITATION.cff",
    "docs/REPRODUCTION.md",
    "docs/VERIFICATION.md",
    "docs/DATASETS.md",
    "docs/ARTIFACT_MAP.md",
    "docs/SECURITY_AND_PRIVACY.md",
    "docs/HISTORICAL_PRIMARY_UTILITY_STATUS.md",
    "verification/groth16/circuit_sources",
    "verification/groth16/canonical",
    "verification/groth16/retained_proofs",
    "verification/blockchain/ProofRegistryV1.sol",
    "code/verification/On-Chain-RSU-GlobalServer-SSI-Verification_V10.py",
    "verification/public_keys",
    "experiments/02_multiseed/statistics/per_seed_metrics.csv",
    "experiments/02_multiseed/_publication_provenance/orchestration/multi_seed_master_manifest.json",
    "experiments/04_non_iid_stress/combined_stress",
    "experiments/07_model_consistency/results/comment4_validation_results.json",
    "experiments/08_compact_security_evidence",
]

FORBIDDEN_RELEASE_PATHS = [
    "verification/evidence_bundle/primary_compact",
    "docs/FINAL_PLACEHOLDER_AUDIT.csv",
    "docs/FINAL_PYTHON_SYNTAX_AUDIT.csv",
    "docs/FINAL_RUNNABLE_PATH_AUDIT.csv",
    "docs/FINAL_SECRET_AUDIT.csv",
    "docs/PORTABILITY_REPAIR_REPORT.csv",
    "docs/REPAIR_ACTION_LOG.txt",
]

STALE_PHRASES = [
    "PRIVATE STAGING PACKAGE",
    "DO NOT PUBLISH YET",
    "Add exact quick/full reproduction commands",
]

# Match actual ChatGPT content-reference annotations, not explanatory text such as
# the literal documentation example `:contentReference[...]`.
CONTENT_REFERENCE_RE = re.compile(r":contentReference\[[^\]\r\n]*\]\s*\{")
WINDOWS_ABS_RE = re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/]")
SECRET_ASSIGNMENT_RE = re.compile(
    r'''(?im)^\s*(PRIVATE_KEY|SEPOLIA_PRIVATE_KEY|WALLET_PRIVATE_KEY|MNEMONIC|SEED_PHRASE|API_KEY|ACCESS_TOKEN|RPC_URL|SEPOLIA_RPC_URL|WEB3_RPC_URL)\s*=\s*["']([^"']+)["']'''
)
SENSITIVE_JSON_KEY_RE = re.compile(
    r"^(private[_-]?key|ed25519[_-]?priv[_-]?b64|mnemonic|seed[_-]?phrase|api[_-]?key|access[_-]?token)$",
    re.IGNORECASE,
)
MASKED_VALUES = {"", "[masked]", "<masked>", "masked", "redacted", "<redacted>"}
TEXT_ARCHIVE_SUFFIXES = {".json", ".md", ".txt", ".csv", ".py", ".sol", ".circom", ".js", ".yml", ".yaml", ".cff"}
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
WARN_LARGE_FILE_BYTES = 50 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/).",
    )
    return parser.parse_args()


def is_retained_historical(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return "retained_v9_stack" in parts or "retained_v10_stack" in parts


def scan_json_secrets(value, rel: str, failures: list[str], prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}" if prefix else key_text
            if SENSITIVE_JSON_KEY_RE.match(key_text):
                rendered = str(child).strip().lower() if child is not None else ""
                if rendered not in MASKED_VALUES:
                    failures.append(
                        f"sensitive JSON field has a non-masked value: {rel}:{child_prefix}"
                    )
            scan_json_secrets(child, rel, failures, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_json_secrets(child, rel, failures, f"{prefix}[{index}]")


def string_literals(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value, getattr(node, "lineno", None)


def scan_zip(path: Path, root: Path, failures: list[str]) -> None:
    rel_zip = path.relative_to(root).as_posix()
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                member = info.filename
                pure = PurePosixPath(member)
                if pure.is_absolute() or ".." in pure.parts:
                    failures.append(f"unsafe ZIP member path: {rel_zip}:{member}")
                    continue
                mode = (info.external_attr >> 16) & 0xFFFF
                if mode and stat.S_ISLNK(mode):
                    failures.append(f"symlink ZIP member is not allowed: {rel_zip}:{member}")
                    continue
                if info.is_dir() or pure.suffix.lower() not in TEXT_ARCHIVE_SUFFIXES:
                    continue
                try:
                    data = zf.read(info)
                    text = data.decode("utf-8-sig")
                except (UnicodeDecodeError, RuntimeError, OSError, KeyError):
                    continue
                member_rel = f"{rel_zip}!/{member}"
                if CONTENT_REFERENCE_RE.search(text):
                    failures.append(f"stale contentReference annotation in ZIP member: {member_rel}")
                for match in SECRET_ASSIGNMENT_RE.finditer(text):
                    name, value = match.group(1), match.group(2).strip()
                    if value.lower() not in MASKED_VALUES:
                        failures.append(f"hard-coded credential candidate in ZIP member: {member_rel}:{name}")
                if pure.suffix.lower() == ".json":
                    try:
                        obj = json.loads(text)
                    except Exception:
                        continue
                    scan_json_secrets(obj, member_rel, failures)
    except zipfile.BadZipFile as exc:
        failures.append(f"invalid ZIP archive: {rel_zip}: {exc}")


def validate_nested_checksums(root: Path, failures: list[str]) -> None:
    """Verify every non-root checksum inventory using its declared relative paths.

    Historical subpackages use different path roots (for example, entries may
    start with ``statistics/`` or ``verification/``). We therefore resolve the
    nearest ancestor from which *all* declared paths exist, then verify hashes.
    """
    root_checksum = (root / "CHECKSUMS_SHA256.txt").resolve()
    for checksum_file in sorted(root.rglob("CHECKSUMS_SHA256.txt")):
        if checksum_file.resolve() == root_checksum:
            continue
        rel_checksum = checksum_file.relative_to(root).as_posix()
        entries: list[tuple[str, str]] = []
        try:
            for lineno, line in enumerate(checksum_file.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                if "  " not in line:
                    failures.append(f"malformed nested checksum line: {rel_checksum}:{lineno}")
                    entries = []
                    break
                digest, rel = line.split("  ", 1)
                if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                    failures.append(f"invalid nested SHA-256 digest: {rel_checksum}:{lineno}")
                    entries = []
                    break
                pure = PurePosixPath(rel)
                if pure.is_absolute() or ".." in pure.parts:
                    failures.append(f"unsafe nested checksum path: {rel_checksum}:{rel}")
                    entries = []
                    break
                entries.append((digest.lower(), rel))
        except OSError as exc:
            failures.append(f"cannot read nested checksum inventory: {rel_checksum}: {exc}")
            continue
        if not entries:
            continue

        candidates: list[Path] = []
        cur = checksum_file.parent
        while True:
            candidates.append(cur)
            if cur == root:
                break
            if root not in cur.parents:
                break
            cur = cur.parent
        base = next((cand for cand in candidates if all((cand / rel).is_file() for _, rel in entries)), None)
        if base is None:
            failures.append(f"cannot resolve nested checksum base: {rel_checksum}")
            continue
        for expected, rel in entries:
            target = base / rel
            got = hashlib.sha256(target.read_bytes()).hexdigest()
            if got != expected:
                failures.append(f"nested checksum mismatch: {rel_checksum} -> {rel}")


def validate_multiseed_completed(root: Path, failures: list[str]) -> None:
    path = root / "experiments/02_multiseed/_publication_provenance/orchestration/multi_seed_master_manifest.json"
    if not path.is_file():
        return
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        failures.append(f"invalid completed multi-seed orchestration manifest: {exc}")
        return
    if obj.get("state") != "completed":
        failures.append("completed multi-seed orchestration manifest does not have state=completed")
    stages = obj.get("stage_results")
    if not isinstance(stages, list) or len(stages) != 80:
        failures.append(f"completed multi-seed orchestration must contain 80 stage records; found {len(stages) if isinstance(stages, list) else 'non-list'}")
        return
    allowed = {"completed_validated", "skipped_valid_existing"}
    bad = [x.get("status") for x in stages if x.get("status") not in allowed]
    if bad:
        failures.append(f"completed multi-seed orchestration contains invalid stage status values: {sorted(set(map(str,bad)))}")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    failures: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED:
        if not (root / rel).exists():
            failures.append(f"missing required path: {rel}")
    for rel in FORBIDDEN_RELEASE_PATHS:
        if (root / rel).exists():
            failures.append(f"obsolete/stale release path must not be present: {rel}")

    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            failures.append(f"symbolic link is not allowed in the frozen release: {rel}")
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size >= MAX_GITHUB_FILE_BYTES:
            failures.append(f"GitHub-incompatible file >=100 MiB: {rel} ({size} bytes)")
        elif size >= WARN_LARGE_FILE_BYTES:
            warnings.append(f"large file >=50 MiB: {rel} ({size} bytes)")
        if len(rel) >= 240:
            warnings.append(f"long retained relative path; use a short clone root on Windows: {rel}")

    for path in root.rglob("*.md"):
        if is_retained_historical(path, root):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(root).as_posix()
        for phrase in STALE_PHRASES:
            if phrase in text:
                failures.append(f"stale placeholder phrase in {rel}: {phrase}")
        if CONTENT_REFERENCE_RE.search(text):
            failures.append(f"stale contentReference annotation in publication Markdown: {rel}")

    for path in root.rglob("*.py"):
        if is_retained_historical(path, root):
            continue
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            failures.append(f"Python syntax error: {rel}: {exc}")
            continue
        for literal, lineno in string_literals(tree):
            if WINDOWS_ABS_RE.search(literal):
                failures.append(f"absolute Windows path remains in runnable-code string: {rel}:{lineno}")
        if CONTENT_REFERENCE_RE.search(text):
            failures.append(f"stale contentReference annotation in runnable code: {rel}")
        for match in SECRET_ASSIGNMENT_RE.finditer(text):
            name, value = match.group(1), match.group(2).strip()
            if value.lower() not in MASKED_VALUES:
                failures.append(f"hard-coded credential/network secret candidate in {rel}: {name}")

    for path in root.rglob("*.json"):
        if is_retained_historical(path, root):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            failures.append(f"invalid JSON: {rel}: {exc}")
            continue
        scan_json_secrets(obj, rel, failures)
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if CONTENT_REFERENCE_RE.search(text):
            failures.append(f"stale contentReference annotation in publication JSON: {rel}")

    for path in root.rglob("*.zip"):
        scan_zip(path, root, failures)

    circuit_dir = root / "verification/groth16/circuit_sources"
    if len(list(circuit_dir.glob("*.circom"))) < 2 if circuit_dir.is_dir() else True:
        failures.append("fewer than two canonical AnchorSum .circom sources")

    public_key_dir = root / "verification/public_keys"
    if not public_key_dir.is_dir() or not list(public_key_dir.rglob("*.json")):
        failures.append("no public SSI verification JSON extracted")

    validate_multiseed_completed(root, failures)
    validate_nested_checksums(root, failures)
    warnings.append(
        "Historical main DP utility output bundle is disclosed as not retained; no substitute is fabricated."
    )

    if failures:
        print("PUBLICATION SELF-CHECK: FAIL")
        for item in failures:
            print("FAIL:", item)
        for item in warnings:
            print("WARN:", item)
        return 1

    print("PUBLICATION SELF-CHECK: PASS")
    for item in warnings:
        print("WARN:", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
