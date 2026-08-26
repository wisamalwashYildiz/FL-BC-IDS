# FL-BC-IDS Reproducibility and Verification Package

This repository accompanies **FL-BC-IDS: Evidence-Native Privacy-Aware
Hierarchical Federated Intrusion Detection for the Internet of Vehicles**.

It contains implementation code, deterministic preprocessing material, retained
experiment results/configurations, Round-2 validation artifacts, Groth16 circuit
sources and compiled verification material, model-consistency checks, smart
contracts, Sepolia receipt evidence, **public-only** SSI verification material,
and executable verification helpers.

## Important provenance boundary

Historical retained outputs, later validation outputs, and reconstructed
reproduction artifacts are deliberately separated. Missing historical evidence
is not fabricated.

In particular:

- the exact historical main DP predictive-utility output bundle was not retained;
- `experiments/08_compact_security_evidence/` is a security/verification run,
  not a substitute for that missing main-utility bundle;
- the 20-round runner is explicitly reconstructed;
- expanded-topology retained source/configuration evidence is distinguished from
  missing raw historical output bundles.

See `docs/HISTORICAL_PRIMARY_UTILITY_STATUS.md` and
`docs/KNOWN_SCOPE_BOUNDARIES.md`.

## Environment setup

Python:

```bash
python -m venv .venv
# activate the environment, then:
python -m pip install --upgrade pip
pip install -r environment/requirements-repro.txt
```

Node/snarkjs:

```bash
npm ci --prefix environment
```

The Circom compiler is separate from npm; use the retained compiler version
documented under `environment/`.

## Repository verification

From the repository root:

```bash
python scripts/build_release_inventory.py --check
python scripts/release_self_check.py
python scripts/verify_checksums.py
python scripts/verify_groth16.py --structural-only
python scripts/verify_blockchain_evidence.py
```

For the actual retained Groth16 cryptographic proof verification, run:

```bash
python scripts/verify_groth16.py
```

after the locked Node dependencies have been installed. The helper searches
`environment/node_modules/.bin/snarkjs` first and does not silently download a
package.

## Structure

- `code/` — system, baselines, reviewer-validation code, verification utilities
- `data_preparation/` — deterministic preprocessing scripts/manifests
- `experiments/01_primary_compact/` — documented primary configuration,
  centralized references, explicitly labelled legacy non-DP evidence, and
  Round-2 seed-42 reproduction references
- `experiments/02_multiseed/` — paired seeds 42–51, results, provenance,
  statistical audits, timestamped orchestration snapshots, and a completed
  publication-safe orchestration manifest
- `experiments/03_generalization/` — temporal/generalization validation
- `experiments/04_non_iid_stress/` — controlled heterogeneity/participation stress
- `experiments/05_long_round/` — reconstructed 20-round reproduction scaffold
- `experiments/06_expanded_topology/` — 10-RSU / 200-vehicle source/configuration evidence
- `experiments/07_model_consistency/` — update→anchor and deterministic aggregation-replay evidence
- `experiments/08_compact_security_evidence/` — compact security/verification evidence
- `verification/` — Groth16, Sepolia, public SSI material, and selected evidence
- `environment/` — dependency/version snapshots
- `docs/` — reproduction, verification, dataset, artifact-map, privacy/scope documentation

## Privacy boundary

The numerical DP accounting is a learner-stage per-record-instance guarantee
conditional on the fixed preprocessing artifact/post-preprocessing training
multiset. It is **not** an unchanged end-to-end privacy guarantee for an
original raw record.

## Security

No SSI private signing keys, blockchain wallet private keys, seed phrases, or
hosted RPC credentials belong in the public release. See `SECURITY.md`.
Optional new Sepolia runs use environment variables documented in `.env.example`.

## Integrity inventory

`MANIFEST.csv` lists the release content, sizes, SHA-256 hashes, and artifact
roles; it excludes only the two root inventory-control files to avoid circular
self-hashing. `CHECKSUMS_SHA256.txt` includes `MANIFEST.csv` and all other
release content while excluding only itself. It is the executable integrity
inventory used by `scripts/verify_checksums.py`. The root `.gitattributes`
disables automatic line-ending conversion so these byte-level hashes remain
stable across platforms.

The stale pre-final audit CSV/Markdown files generated before the latest
security/path corrections are intentionally excluded. Regenerate any human
audit report only after the assembled repository passes the executable checks.

## License

Original FL-BC-IDS repository content is released under the MIT License; see
`LICENSE`. Third-party dependencies and generated artifacts remain subject to
their own applicable licenses and notices.
