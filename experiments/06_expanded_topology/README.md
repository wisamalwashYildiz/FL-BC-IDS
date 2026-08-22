# Expanded-Topology Experiment — Publication-Corrected

## Retained source evidence

The retained V10 main runner directly encodes:

- 10 RSUs
- 20 vehicles per RSU
- 200 vehicles total
- 2 global rounds
- 10 local boosting rounds per participant per global round

The retained V10 stack is preserved byte-for-byte under
`code/retained_v10_stack/`.

Its active dataset is CSE-CIC-IDS2018. CICIoV2024 appears only as the documented
alternate dataset selection, so the CIC configuration is not described as a
dedicated retained CIC executable.

## Publication-portable runner

`code/portable_v10_stack/` contains the same V10 helper stack plus a hardened
main runner. The helper files remain byte-identical to the retained V10 copies.
The main runner preserves the expanded-topology scientific constants but adds
portable repository discovery, explicit dataset selection through
`FLBCIDS_DATASET_NAME`, portable data/output paths, and removal of stale
contentReference comments.

## Historical result status

The original raw expanded-topology execution bundles were not located in the
project audit. `MANUSCRIPT_REPORTED_RESULTS.json` therefore records the values
reported by the manuscript as documentary metadata only; it is not represented
as independently retained execution evidence.

No missing raw output is fabricated.
