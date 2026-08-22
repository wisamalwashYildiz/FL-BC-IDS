#!/usr/bin/env python3
"""Regenerate MANIFEST.csv and CHECKSUMS_SHA256.txt from the final release tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path

from verify_checksums import is_ignored_rel


CONTROL_FILES = {"MANIFEST.csv", "CHECKSUMS_SHA256.txt"}
ROLE_BY_PREFIX = {
    "code/": "source_code",
    "data_preparation/": "data_preparation",
    "docs/": "documentation",
    "environment/": "environment_lock",
    "experiments/": "experiment_evidence",
    "scripts/": "verification_script",
    "verification/": "verification_evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="Check inventories without rewriting them.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def role_for(rel: str) -> str:
    for prefix, role in ROLE_BY_PREFIX.items():
        if rel.startswith(prefix):
            return role
    return "repository_metadata"


def release_files(root: Path, exclude_control: bool = False) -> list[Path]:
    out = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if is_ignored_rel(rel):
            continue
        if exclude_control and rel in CONTROL_FILES:
            continue
        out.append(path)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def render_manifest(root: Path) -> str:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["path", "size_bytes", "sha256", "role"])
    for path in release_files(root, exclude_control=True):
        rel = path.relative_to(root).as_posix()
        writer.writerow([rel, path.stat().st_size, sha256_file(path), role_for(rel)])
    return buf.getvalue()


def render_checksums(root: Path) -> str:
    lines = []
    for path in release_files(root):
        rel = path.relative_to(root).as_posix()
        if rel == "CHECKSUMS_SHA256.txt":
            continue
        lines.append(f"{sha256_file(path)}  {rel}\n")
    return "".join(lines)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest = root / "MANIFEST.csv"
    checksums = root / "CHECKSUMS_SHA256.txt"

    # MANIFEST deliberately excludes itself and the checksum file to avoid circular hashes.
    expected_manifest = render_manifest(root)
    if args.check:
        ok = True
        if not manifest.is_file() or manifest.read_text(encoding="utf-8") != expected_manifest:
            print("INVENTORY CHECK: FAIL: MANIFEST.csv is not current")
            ok = False
        # Only compute checksum expectation after using the on-disk manifest, because CHECKSUMS includes it.
        expected_checksums = render_checksums(root)
        if not checksums.is_file() or checksums.read_text(encoding="utf-8") != expected_checksums:
            print("INVENTORY CHECK: FAIL: CHECKSUMS_SHA256.txt is not current")
            ok = False
        if ok:
            print("INVENTORY CHECK: PASS")
            return 0
        return 1

    manifest.write_text(expected_manifest, encoding="utf-8", newline="")
    checksums.write_text(render_checksums(root), encoding="utf-8", newline="")
    print(f"WROTE {manifest}")
    print(f"WROTE {checksums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
