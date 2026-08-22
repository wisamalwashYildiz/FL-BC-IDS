# Historical Orchestration Snapshots

This directory preserves timestamped orchestration snapshots exactly by their
historical run identity. The timestamp directory names are intentional and must
not be replaced by semantic aliases.

- `20260818T215520Z/` — non-resume mid-run snapshot (`state=running`; 12 validated stages recorded).
- `20260818T215920Z/` — failed non-resume restart attempt caused by already-populated validated output.
- `20260818T234015Z/` — resume mid-run snapshot (`resume=true`; 22 stage records).
- `20260818T234353Z/` — later resume mid-run snapshot (`resume=true`; 22 stage records).

These files are provenance snapshots, not the completed orchestration record.
The publication-safe completed manifest is:

`experiments/02_multiseed/_publication_provenance/orchestration/multi_seed_master_manifest.json`

That completed manifest covers all 80 planned dataset/seed/stage records: 58
`completed_validated` and 22 `skipped_valid_existing`.
