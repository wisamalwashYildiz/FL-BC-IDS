# Final Release Audit

Audit date: `2026-08-23`

## Status

**READY FOR PUBLIC GITHUB RELEASE, subject to preserving the frozen bytes in this package.**

This report describes checks executed against the final assembled release tree.
The executable root inventories are regenerated only after all content changes
are complete and are then verified again from the packaged ZIP.

## Release-safety checks

- Publication self-check: **PASS**
- Root exact-inventory/checksum verification: **PASS**
- Nested checksum inventories: **9 / 9 PASS**
- Runnable Python syntax/AST parse: **57 / 57 PASS**
- JSON parse audit: **586 / 586 PASS**
- CSV parse audit: **43 / 43 PASS**
- Symlinks in release tree: **0**
- Case-insensitive path collisions: **0**
- Windows-invalid/reserved path names: **0**
- Files at or above GitHub's 100 MiB single-file limit: **0**
- Public credential/private-material scan: **PASS**
- Nested `HISTORICAL_ORIGINAL_BYTES.zip` path-safety and targeted secret scan: **PASS**

## Verification evidence checks

- Groth16 retained-artifact structural verification: **PASS**
- Sepolia/blockchain retained-evidence verification: **PASS**
- Multi-seed completed orchestration manifest: **80 / 80 stages accounted for**
  (`58 completed_validated`, `22 skipped_valid_existing`)

Full Groth16 cryptographic verification is exposed by
`python scripts/verify_groth16.py` and intentionally requires the locked Node
dependencies under `environment/` to have been installed first. The release
helper does not silently download `snarkjs`. Structural proof/public-input/vkey
consistency was executed during release curation.

## Portability and provenance

The timestamped multi-seed orchestration snapshots are retained under their
historical names and are documented rather than renamed. Historical machine
paths embedded inside byte-preserved provenance/evidence are not executable
configuration. Current runnable code uses repository-relative paths or explicit
arguments/environment variables.

The longest retained relative artifact path is close to the legacy Windows path
limit. `PORTABILITY.md` therefore instructs Windows users to clone/extract to a
short root path or enable long-path support instead of renaming provenance-pinned
cryptographic artifacts.

## Known evidence boundary

The historical main DP predictive-utility output bundle was not retained in the
available source material. The repository does **not** manufacture or relabel a
substitute. This limitation is disclosed explicitly; retained Round-2 multi-seed,
generalization, stress, model-consistency, compact-security, and verification
evidence remains separately identified.

## Release rule

Any change to a tracked release file invalidates the frozen root inventory until
`python scripts/build_release_inventory.py` is rerun and all release checks pass
again. Do not edit files inside a published/tagged release without producing a new
versioned release.
