# FL-BC-IDS Multi-Seed Statistical Outputs — Publication-Corrected

This bundle contains the retained seed-level metrics and the statistical
summaries used for the Reviewer 1 multi-seed analysis.

## Important correction

The original `summary_statistics.csv` used Student-t 95% confidence intervals.
That CI method is not the manuscript-facing method. The corrected file keeps the
same means and sample standard deviations but uses the retained submitted
percentile-bootstrap endpoints recorded in `bootstrap_exact_audit.csv`.

No seed-level metric value was changed.

## Bootstrap provenance

The original finite Monte-Carlo bootstrap RNG state and resample count were not
retained. `bootstrap_exact_audit.csv` therefore performs a deterministic exact
audit of the same nonparametric percentile-bootstrap estimand. It does not fit a
post-hoc RNG seed.

## Paired tests

`paired_permutation_tests_audit.csv` was independently reproduced from
`per_seed_metrics.csv` as exact two-sided paired sign-flip tests, paired by seed,
with Holm correction separately within each dataset across 15 tests.

## Paths

Machine-local path references were replaced by logical repository locators under
`experiments/02_multiseed/`.
