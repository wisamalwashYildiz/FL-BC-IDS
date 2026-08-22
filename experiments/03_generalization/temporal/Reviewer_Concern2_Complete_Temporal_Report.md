# Reviewer Concern #2 — Complete Temporal Validation

## Source and chronology integrity

- CSE source SHA-256: `4335539845e880b1fb06703b5a68da0a03ed0682204bdda0863ddfc316782e3c`.
- Timestamp-valid deterministic sample rows: 3,000,000.
- Predictor count: 85.
- Ten chronological blocks were constructed without splitting identical timestamps.

## Why the strict rolling audit had no eligible seen-attack folds

The strict protocol reserves the immediately preceding block entirely for calibration. The dataset's attack campaigns are strongly scheduled in time, so attack families that recur in the next block commonly first appear in that calibration block rather than in the trained history. Therefore the strict protocol can legitimately contain zero attacks whose family has already been seen by the trained model. This is an eligibility property of the chronology, not a measured recall failure.

- Strict eligible seen-attack evaluations: 0.

## Supervised adaptive rolling protocol

- Test transitions: B1->B2 through B9->B10.
- For every transition, the immediately preceding block is split chronologically and timestamp-safely.
- First 70% of the preceding block is added to training as a supervised periodic refresh.
- Final 30% of the preceding block is used only for operating-point calibration.
- The primary threshold is fixed from benign calibration scores at <=1% empirical development FPR.
- The complete next block is untouched until final evaluation.
- Seen and novel attack families are reported separately.
- Operational assumption: labels for the adaptation segment become available before the subsequent block.

## Primary adaptive seen-attack temporal endpoint

- Eligible adaptive evaluations: 5.
- Pooled seen-attack rows: 6,968.
- Pooled recall: **0.984788**.
- Pooled precision: 0.470612.
- Pooled F1: 0.636874.
- Pooled test FPR: 0.005700.
- Evaluation recall mean ± SD: 0.904797 ± 0.138508.
- Worst eligible evaluation recall: 0.666667.

## Strict non-adaptive fold results

| evaluation_id   | train_blocks   |   calibration_block |   test_block |   seen_attack_rows |   novel_attack_rows |   combined_recall |   combined_test_fpr | status              |
|:----------------|:---------------|--------------------:|-------------:|-------------------:|--------------------:|------------------:|--------------------:|:--------------------|
| strict_test_B6  | B1-B4          |                   5 |            6 |                  0 |                   9 |        0.222222   |          0.010727   | no_seen_attack_rows |
| strict_test_B7  | B1-B5          |                   6 |            7 |                  0 |                  11 |        0.727273   |          0.0104437  | no_seen_attack_rows |
| strict_test_B8  | B1-B6          |                   7 |            8 |                  0 |                2359 |        0.0665536  |          0.0170104  | no_seen_attack_rows |
| strict_test_B9  | B1-B7          |                   8 |            9 |                  0 |                1835 |        0.0348774  |          0.00216323 | no_seen_attack_rows |
| strict_test_B10 | B1-B8          |                   9 |           10 |                  0 |                6813 |        0.00719213 |          0.00994928 | no_seen_attack_rows |

## Adaptive transition results

| evaluation_id     | transition   |   adaptation_rows |   calibration_rows_from_prior_block |   seen_test_attack_type_count |   seen_attack_rows |   novel_attack_rows |   seen_temporal_recall |   seen_temporal_precision |   seen_temporal_f1 |   seen_temporal_test_fpr |   seen_temporal_auc | status              |
|:------------------|:-------------|------------------:|------------------------------------:|------------------------------:|-------------------:|--------------------:|-----------------------:|--------------------------:|-------------------:|-------------------------:|--------------------:|:--------------------|
| adaptive_test_B2  | B1->B2       |            210000 |                               90000 |                             1 |               2360 |                1807 |               1        |                 0.597771  |         0.748256   |               0.00536789 |            1        | eligible            |
| adaptive_test_B3  | B2->B3       |            210000 |                               90000 |                             1 |               2676 |               85560 |               1        |                 0.627874  |         0.771404   |               0.00748947 |            0.999994 | eligible            |
| adaptive_test_B4  | B3->B4       |            210000 |                               90000 |                             0 |                  0 |               13777 |             nan        |               nan         |       nan          |             nan          |          nan        | no_seen_attack_rows |
| adaptive_test_B5  | B4->B5       |            210000 |                               90000 |                             2 |                 88 |               51349 |               0.909091 |                 0.0403023 |         0.0771828  |               0.00766405 |            0.993024 | eligible            |
| adaptive_test_B6  | B5->B6       |            210000 |                               90000 |                             0 |                  0 |                   9 |             nan        |               nan         |       nan          |             nan          |          nan        | no_seen_attack_rows |
| adaptive_test_B7  | B6->B7       |            210000 |                               90000 |                             3 |                  9 |                   2 |               0.666667 |                 0.0027137 |         0.00540541 |               0.00735027 |            0.992299 | eligible            |
| adaptive_test_B8  | B7->B8       |            210000 |                               90000 |                             0 |                  0 |                2359 |             nan        |               nan         |       nan          |             nan          |          nan        | no_seen_attack_rows |
| adaptive_test_B9  | B8->B9       |            210000 |                               90000 |                             3 |               1835 |                   0 |               0.948229 |                 0.8       |         0.86783    |               0.00145892 |            0.997044 | eligible            |
| adaptive_test_B10 | B9->B10      |            210000 |                               90000 |                             0 |                  0 |                6813 |             nan        |               nan         |       nan          |             nan          |          nan        | no_seen_attack_rows |

## Adaptive seen attack-family summary

| attack_type                                  |   evaluations_present |   total_seen_temporal_rows |   pooled_recall |   mean_evaluation_recall |   min_evaluation_recall |   max_evaluation_recall |
|:---------------------------------------------|----------------------:|---------------------------:|----------------:|-------------------------:|------------------------:|------------------------:|
| DDOS-LOIC-UDP                                |                     1 |                         80 |        1        |                 1        |                1        |                1        |
| DDOS-LOIC-UDP - ATTEMPTED                    |                     1 |                          8 |        0        |                 0        |                0        |                0        |
| FTP-BRUTEFORCE - ATTEMPTED                   |                     2 |                       5036 |        1        |                 1        |                1        |                1        |
| INFILTRATION - COMMUNICATION VICTIM ATTACKER |                     1 |                          3 |        0        |                 0        |                0        |                0        |
| INFILTRATION - DROPBOX DOWNLOAD              |                     1 |                          1 |        0        |                 0        |                0        |                0        |
| INFILTRATION - NMAP PORTSCAN                 |                     1 |                       1831 |        0.9503   |                 0.9503   |                0.9503   |                0.9503   |
| WEB ATTACK - BRUTE FORCE                     |                     1 |                          4 |        1        |                 1        |                1        |                1        |
| WEB ATTACK - BRUTE FORCE - ATTEMPTED         |                     1 |                          2 |        0        |                 0        |                0        |                0        |
| WEB ATTACK - XSS                             |                     1 |                          3 |        0.666667 |                 0.666667 |                0.666667 |                0.666667 |

## Terminal 70/15/15 stress diagnostic

| evaluation_id     |   seen_attack_rows |   novel_attack_rows | seen_temporal_recall   |   combined_recall |   combined_precision |   combined_f1 |   combined_test_fpr |   combined_auc | status              |
|:------------------|-------------------:|--------------------:|:-----------------------|------------------:|---------------------:|--------------:|--------------------:|---------------:|:--------------------|
| terminal_70_15_15 |                  0 |                7560 |                        |        0.00555556 |            0.0210632 |    0.00879213 |           0.0044119 |       0.873383 | no_seen_attack_rows |

## Interpretation constraints

1. The strict protocol is retained even if it has zero eligible seen-attack rows; it documents the dataset chronology and prevents selective omission.

2. The adaptive protocol is a supervised periodic-refresh scenario, not an unsupervised online detector. Its label-availability assumption must be stated.

3. Novel attacks remain an open-set problem and are never merged into the primary seen-attack temporal endpoint.

4. The terminal 70/15/15 result is a combined temporal plus attack-novelty stress diagnostic whenever its attack families are absent from training.

5. The 1% development-FPR threshold is primary regardless of which sensitivity target later produces the best test result.