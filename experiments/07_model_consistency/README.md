# Model-Update / Anchor and Aggregation-Consistency Validation

This archive contains the retained **result/provenance records** for the
Reviewer 1 Comment 4 validation.

The retained harness records **16/16 validation assertions passed**. That result
must be interpreted precisely: the harness successfully checks the scoped
model-consistency relations and detects the historical Flower multi-tree
aggregation-contract mismatch. It does **not** mean that the unmodified Flower
1.23.0 aggregation behavior was itself correct.

## 1. Model-update -> anchor relation

The validation records the following verifier-side checks:

- exact received serialized model-update bytes must reproduce the
  `model_delta_sha256` bound by the signed report;
- the submitted quantized anchor vector must reproduce its signed anchor digest;
- the model update is deserialized and evaluated under the fixed anchor contract;
- a recomputed anchor must match the submitted/bound anchor relation.

The unrelated-artifact test deliberately uses a valid signature over an
internally unrelated model/anchor pairing. Signature verification succeeds, but
the derivation check returns `model_anchor_derivation_mismatch`.

This relation is checked by the deterministic verification layer. It is **not**
a claim that the Groth16 circuit proves XGBoost training or model-update
generation.

## 2. Deterministic RSU aggregation replay

Each retained client delta contains 10 trees.

The diagnostic records:

- two input deltas: 10 + 10 trees;
- Flower 1.23.0 later-update behavior: **11 trees**;
- intended all-tree aggregation contract: **20 trees**;
- corrected deterministic all-tree replay: **20 trees**;
- corrected second-round replay: **40 trees**.

Therefore, the 16/16 result includes successful **detection** of the Flower
multi-tree contract mismatch and successful validation of the corrected replay.
It must not be summarized as proof that the original Flower behavior already
consumed every tree.

The replay also checks deterministic ordering, byte stability, model-contract
compatibility, aggregate-artifact substitution, and second-round continuation.

## Hash semantics

`aggregation_replay_record_sha256 =
d4543b6ca653d588b30a813d3e718ed404e9d2289f6ee6cb423ff29fdfc02e18`
is the SHA-256 of the canonical record serialization:

- UTF-8 JSON;
- lexicographically sorted keys;
- separators `,` and `:` with no extra whitespace;
- no trailing newline.

The pretty-printed retained JSON file has a different file-byte hash. The exact
canonical byte stream is included as
`results/comment4_aggregation_replay_record.canonical.json`.

## Archive scope

This ZIP is a result/provenance subset. It does not contain the model-update
bytes, quantized-anchor arrays, signed-report bytes/public key, or the validation
implementation itself. Those are required for a standalone re-execution and
belong in the full reproducibility repository/evidence package.

Accordingly, the records in this ZIP are retained validation evidence, not a
self-contained rerunnable harness by themselves.
