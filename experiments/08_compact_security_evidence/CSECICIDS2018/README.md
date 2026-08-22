# Compact Security-Evidence Run - CSE-CIC-IDS2018

This is the reclassified historical `rsu_outputs_dp` security-evidence run.
`run_outputs/` is a curated subset; the original bytes are identical to the
corresponding `verification_evidence/run_root/` files.

Important nuance: the historical in-run `ablation_summary.json` records the
public-input agreement helper as **fail** because the RSU artifact path was
empty and six global fields were unresolved. That failure is preserved.

A separate retained `PublicInputArtifactVerificationReportV1` reports **pass**:
all 8 RSU and all 7 global consistency checks succeed against retained
proof/summary/manifest/sidecar artifacts. Therefore do not say every in-run
ablation check passed.

The retained Sepolia report has 5/5 proof submissions with status=1 and
verified_ok=true. Six transactions including finalization are recorded as
finalized. Receipt cost for this security-evidence run is
0.008582244942364408 ETH. These values are not the manuscript's separately
reported main compact benchmark values.

No claim is made that this directory is the missing main utility run.
