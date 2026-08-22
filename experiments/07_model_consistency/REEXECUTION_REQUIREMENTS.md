# Re-execution Requirements

The retained result files in this ZIP are not sufficient by themselves to rerun
the Reviewer 1 Comment 4 harness.

A standalone re-execution requires, from the full FL-BC-IDS reproducibility
repository:

1. the Comment 4 verification harness / deterministic aggregation-replay code;
2. the exact serialized model-update bytes used by the positive and negative
   tests;
3. the retained quantized anchor vectors and fixed anchor input/evaluation
   contract;
4. the signed participation-report bytes and corresponding public verification
   key for the signature test;
5. the XGBoost/Flower environment matching the recorded diagnostic, including
   Flower 1.23.0 for the historical behavior comparison.

The expected validation interpretation is:

- signature-valid but unrelated artifacts are rejected by derivation checking;
- two 10-tree deltas expose an 11-vs-20 Flower contract mismatch;
- the corrected all-tree replay yields 20 trees;
- the second corrected transition yields 40 trees;
- all 16 validation assertions pass.

Do not use this ZIP alone to claim an independently rerunnable harness unless
the required implementation and retained binary/model evidence have been added
to the full archive.
