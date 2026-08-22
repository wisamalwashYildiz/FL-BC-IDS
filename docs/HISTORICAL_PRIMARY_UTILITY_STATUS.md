# Historical Primary Predictive-Utility Output Status

The manuscript's documented main benchmark configuration is:

- 2 RSUs
- 2 vehicles per RSU (4 vehicles total)
- 2 global FL rounds
- 10 local boosting iterations per vehicle per global round
- `num_parallel_tree = 1`
- DP-enabled `approxDP`
- `dp_epsilon_per_tree = 0.25`
- `subsample = 0.2`
- `dp_delta_round = 0.0`

The original project tree contains a compact `rsu_outputs_dp` run, but its
dedicated ablation/tamper/verification artifacts show that it is the compact
**security-evidence** run. It is therefore not presented as the authoritative
historical main predictive-utility result bundle.

This repository does not fabricate or relabel another output as that historical
run. It instead provides the retained implementation/preprocessing artifacts,
complete 10-seed Round-2 validation outputs, the documented main configuration,
and a seed-42 reproduction-family reference.

See `PRIMARY_UTILITY_CANDIDATE_SEARCH.csv` for the final automated search for
possible historical metric fingerprints. A candidate is never automatically
promoted to authoritative historical evidence.
