# Portability

Publication-facing runnable code uses repository-relative paths and/or explicit
environment-variable overrides. It must not depend on any original machine-local development path.

Run commands from the repository root. Configuration variables are documented
in `.env.example`.

### Windows path length

Some retained Groth16-generated filenames are intentionally long because their
names encode the circuit identity. On Windows, clone or extract the repository
to a short location (for example `C:\flbcids`) or enable long-path support in
the operating system/Git configuration. The retained cryptographic artifact
names are not shortened because renaming them would weaken provenance and break
cross-references.

## Python data and output paths

Raw datasets and generated/preprocessed data are local inputs and are excluded
from the release inventory:

- `data/raw/`
- `data/preprocessed/`
- `artifacts/`
- `logs/`

The checksum/self-check helpers intentionally ignore those local/generated
locations as well as `.venv/` and `node_modules/`.

## Node / snarkjs / Circom

Install the locked Node dependencies with:

```bash
npm ci --prefix environment
```

The verification helper resolves
`environment/node_modules/.bin/snarkjs` before falling back to a globally
available executable. Circom is a separate compiler; the retained version is
documented under `environment/` and can be selected with
`ANCHOR_ZKP_CIRCOM_CMD`.

## Retained source snapshots

Directories named `retained_v9_stack` and `retained_v10_stack` are historical
source-lineage evidence. They are preserved byte-for-byte and may contain
historical development paths. They are not the portable execution entry points.

Use:

- `experiments/05_long_round/code/reconstructed_20_round_v10_stack/` for the
  explicitly reconstructed 20-round reproduction;
- `experiments/06_expanded_topology/code/portable_v10_stack/` for the
  publication-portable expanded-topology runner.

## Verification evidence

Publication-facing verification JSON uses logical repository locators instead
of machine-local absolute paths. Public SSI files contain public keys only.
See `verification/SECURITY_REDACTION_NOTICE.md` and
`verification/PUBLICATION_REPAIR_REPORT.json`.
