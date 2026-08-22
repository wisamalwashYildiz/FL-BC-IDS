# Supplementary 20-Round Reproduction Scaffold

## Status

This directory supports **reproduction of the documented configuration** for
the manuscript's historical 20-global-round CSE-CIC-IDS2018 supplementary run.

The project audit did **not** locate either:

1. a dedicated historical source snapshot with `NUM_ROUNDS = 20`, or
2. the original complete output bundle from that run.

Accordingly, this directory must not be presented as retained execution evidence
for the historical 20-round result.

## Documented configuration

- Dataset: CSE-CIC-IDS2018
- RSUs: 2
- Vehicles per RSU: 2
- Total vehicles: 4
- Global rounds: 20
- Local boosting rounds per participant per global round: 10

The learner-stage DP composition is:

- exact epsilon per round: `0.5525028433699525`
- exact 20-round epsilon: `11.05005686739905`
- manuscript four-decimal value: `11.0501`
- delta: `0`

This accounting is conditional on the fixed preprocessing artifact and the
post-preprocessing training multiset. It is not an end-to-end raw-record DP
guarantee.

## Retained source lineage

`code/retained_v9_stack/` and `code/retained_v10_stack/` are historical
snapshots and are preserved byte-for-byte. They are **not** the public runnable
copies and may contain machine-local paths or stale comments from development.

`SOURCE_LINEAGE_EVIDENCE.json` records an AST-level comparison of V9 and V10 and
the exact retained hashes. The comparison supports implementation-lineage
continuity; it does not establish which exact snapshot executed the historical
20-round run.

## Publication reproduction runner

`code/reconstructed_20_round_v10_stack/` contains V10 helper modules that remain
byte-identical to the retained V10 helpers plus the publication-hardened runner:

`FL_DP_SSI_DualMerklePoseidon_RSU+Global_ZKVerify_V10_REPRO_20ROUND.py`

The runner is explicitly a reconstruction, not a historical-executable claim.

See:

- `config/cse_20_round_reproduction_config.json`
- `SOURCE_LINEAGE_EVIDENCE.json`
- `REPRODUCTION_INSTRUCTIONS.md`
