#!/usr/bin/env python3
"""Verify the frozen repository SHA-256 inventory exactly."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re


LINE_RE = re.compile(r"^([0-9a-fA-F]{64})  ([^\r\n]+)$")
IGNORED_PREFIXES = (
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    "environment/node_modules/",
    "data/raw/",
    "data/preprocessed/",
    "artifacts/",
    "logs/",
    ".pytest_cache/",
    ".mypy_cache/",
)
IGNORED_PARTS = {"__pycache__"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_ignored_rel(rel: str) -> bool:
    normalized = rel.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized in {".git", ".venv", "venv", "node_modules", "environment/node_modules", "data/raw", "data/preprocessed", "artifacts", "logs", ".pytest_cache", ".mypy_cache"}:
        return True
    if any(normalized.startswith(prefix) for prefix in IGNORED_PREFIXES):
        return True
    return bool(set(Path(normalized).parts) & IGNORED_PARTS)


def safe_repo_path(root: Path, rel: str) -> Path:
    if "\\" in rel:
        raise ValueError("checksum paths must use forward slashes")
    rel_path = Path(rel)
    if rel_path.is_absolute():
        raise ValueError("absolute path is not allowed")
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes repository root") from exc
    return candidate


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    checks = root / "CHECKSUMS_SHA256.txt"
    if not checks.is_file():
        print("CHECKSUM VERIFICATION: FAIL")
        print("FAIL: CHECKSUMS_SHA256.txt missing")
        return 2

    failures: list[str] = []
    entries: dict[str, str] = {}
    for line_number, line in enumerate(checks.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = LINE_RE.fullmatch(line)
        if not match:
            failures.append(f"malformed checksum line {line_number}: {line!r}")
            continue
        digest, rel = match.group(1).lower(), match.group(2)
        if rel == "CHECKSUMS_SHA256.txt":
            failures.append("checksum inventory must not self-reference CHECKSUMS_SHA256.txt")
            continue
        if is_ignored_rel(rel):
            failures.append(f"generated/local path must not be listed in checksum inventory: {rel}")
            continue
        if rel in entries:
            failures.append(f"duplicate checksum path: {rel}")
            continue
        try:
            safe_repo_path(root, rel)
        except ValueError as exc:
            failures.append(f"unsafe checksum path {rel!r}: {exc}")
            continue
        entries[rel] = digest

    for rel, digest in entries.items():
        path = safe_repo_path(root, rel)
        if not path.is_file():
            failures.append(f"missing: {rel}")
            continue
        actual = sha256_file(path)
        if actual != digest:
            failures.append(f"hash mismatch: {rel} expected={digest} actual={actual}")

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != checks
        and not is_ignored_rel(path.relative_to(root).as_posix())
    }
    listed_files = set(entries)
    for rel in sorted(actual_files - listed_files):
        failures.append(f"unlisted: {rel}")
    for rel in sorted(listed_files - actual_files):
        if not any(item.startswith(f"missing: {rel}") for item in failures):
            failures.append(f"listed but absent: {rel}")

    if failures:
        print("CHECKSUM VERIFICATION: FAIL")
        for item in failures:
            print("FAIL:", item)
        return 1
    print(f"CHECKSUM VERIFICATION: PASS ({len(entries)} files; exact inventory match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
