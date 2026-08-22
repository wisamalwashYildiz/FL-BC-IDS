# FL-BC-IDS Release Verification Scripts

Run these helpers from the repository root. Each accepts `--root <repository>`
for explicit testing where applicable.

- `release_self_check.py` checks required/forbidden release paths, Python syntax,
  actual stale content-reference annotations, machine-local Windows paths in
  runnable-code string literals, obvious hard-coded credential values, JSON
  private-key fields, nested ZIP path/secret safety, GitHub file-size limits,
  and the completed multi-seed orchestration contract. Historical retained V9/V10
  source snapshots are exempt from portability rewriting checks by design.
- `build_release_inventory.py` deterministically regenerates `MANIFEST.csv` and
  `CHECKSUMS_SHA256.txt`; `--check` proves that both inventories match the
  current release tree. Generated/local directories such as `.venv/`,
  `environment/node_modules/`, `data/raw/`, and `data/preprocessed/` are excluded.
- `verify_checksums.py` verifies the exact root SHA-256 inventory and rejects
  malformed lines, duplicates, self-reference, absolute paths, path traversal,
  hash mismatches, missing files, and unlisted publication files.
- `verify_blockchain_evidence.py` validates Sepolia chain/network identity, five
  successful proof submissions, finality consistency, masked RPC/secret fields,
  and deployment-address cross-links. It preserves the non-claims that no formal
  smart-contract security audit/formal verification is asserted by the retained report.
- `verify_groth16.py` validates the retained verification-key/proof/public-input
  triplet structurally and, by default, runs `snarkjs groth16 verify`.
  `--structural-only` is JSON/shape validation only and is not equivalent to
  cryptographic verification. `npx --no-install` is used to prevent an implicit
  package download.

The release inventory must be regenerated only after all publication files are
finalized:

```bash
python scripts/build_release_inventory.py
python scripts/build_release_inventory.py --check
python scripts/release_self_check.py
python scripts/verify_checksums.py
python scripts/verify_groth16.py --structural-only
python scripts/verify_blockchain_evidence.py
```

For full retained Groth16 cryptographic verification, first install the locked
Node dependencies with `npm ci --prefix environment`, then run:

```bash
python scripts/verify_groth16.py
```
