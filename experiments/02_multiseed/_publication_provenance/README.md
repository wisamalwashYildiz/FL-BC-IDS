# FL-BC-IDS Multi-Seed Results — Publication-Corrected Bundle

This ZIP is a publication-safe derivative of the retained multi-seed result bundle.

## What was changed

- Machine-local Windows paths inside result JSON files were replaced by portable logical locators.
- Scientific values, metrics, hashes, seeds, timestamps, model/DP settings, proof-gating values, and experiment dimensions were not altered.
- JSON files with no machine-local paths were preserved byte-for-byte.
- Finalized per-seed provenance (`paired_input_contract.json` and `stage_status.json`) for both datasets and seeds 42–51 is included under `_publication_provenance/per_seed/`.
- The finalized completed orchestration master/preflight manifests are included under `_publication_provenance/orchestration/`.

## Completed orchestration semantics

The completed master covers all 80 planned dataset/seed/stage records:
- 58 `completed_validated`
- 22 `skipped_valid_existing`

`completed` means full plan coverage and successful validation. It does not mean that all 80 stages were freshly executed in that single orchestration session.

## Scope

This is a results/provenance bundle. Some JSON fields point to logical repository locations for larger or transient artifacts that are not necessarily included in this ZIP (for example preprocessed CSV/NPZ files, model files, anchor arrays, and intermediate proof artifacts).

The logical repository location for the results tree is `experiments/02_multiseed/results/` in the full reproducibility archive.
