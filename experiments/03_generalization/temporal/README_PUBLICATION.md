# FL-BC-IDS Generalization Validation — Publication-Corrected Bundle

This bundle retains the complete temporal/generalization evidence, including
both strong and weak diagnostics.

## Primary temporal result

The primary adaptive temporal endpoint is a **supervised periodic-refresh**
scenario. Labels for the first 70% of the immediately preceding block are
assumed available before the next block is evaluated. Across the five
transitions containing seen attack families, the pooled seen-attack results are:

- recall: 0.9847876004592423
- precision: 0.4706124408476785
- F1: 0.636874100886352
- test FPR: 0.005699564502766714
- seen attack rows: 6,968

These values are not open-set/unseen-attack performance.

## Retained weak diagnostics

The strict non-adaptive rolling audit has zero eligible seen-attack evaluations.
The terminal timestamp-safe 70/15/15 split contains zero seen-attack rows and
7,560 novel-attack rows; its combined recall is 0.005555555555555556.

These weak results are intentionally retained and must not be relabeled as the
primary adaptive seen-attack endpoint.

## Threshold selection

The primary threshold is derived only from benign calibration scores at a
target <=1% empirical calibration FPR. Test rows are not used for fitting or
threshold selection.

## Publication corrections

Only machine-local path fields in three JSON files were regenerated. The Python
validation script and all other retained outputs were preserved byte-for-byte.
No scientific result was changed.
