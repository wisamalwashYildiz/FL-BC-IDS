# Verification Instructions

Run commands from the repository root after installing the Python and Node
dependencies described in `docs/REPRODUCTION.md`.

## 1. Groth16 retained-artifact verification

Circuit sources:

```text
verification/groth16/circuit_sources/
```

Compiled canonical artifacts:

```text
verification/groth16/canonical/
```

Retained proof/public-input artifacts:

```text
verification/groth16/retained_proofs/
```

Run the repository helper:

```bash
python scripts/verify_groth16.py
```

The large Powers-of-Tau file is not duplicated in Git. Its retained fingerprint
is:

```text
filename: powersOfTau28_hez_final_20.ptau
size:     1208042648 bytes
sha256:   159d3f938d941e06767d99f30b9fe59a245400a4aae138cf8e411732d7a2f6cd
```

## 2. Compact public-input / proof-bundle verification

Authoritative compact security-evidence run:

```text
experiments/08_compact_security_evidence/CSECICIDS2018/run_outputs/
```

The verifier defaults to that publication layout. To override it explicitly:

```text
FLBCIDS_COMPACT_EVIDENCE_DIR
```

Run the non-transactional proof-bundle audit:

```bash
python code/verification/DryRunRSU-GlobalSrver-SSI-Verification_V10.py
```

Run the artifact/public-input consistency verifier:

```bash
python code/verification/verify_public_inputs_from_artifacts.py
```

These checks verify retained proof/public-input/verifier/vkey/manifest bindings
and associated artifact consistency. They do not make a blockchain transaction.

## 3. Model-update -> anchor and aggregation replay

Implementation:

```text
code/verification/FL_IoV_VerifiableAggregation_UtilsV1.py
code/verification/Reviewer_Concern4_Model_Anchor_Aggregation_Verification.py
```

Run the controlled structural validation:

```bash
python code/verification/Reviewer_Concern4_Model_Anchor_Aggregation_Verification.py
```

Optional output override:

```text
FLBCIDS_COMMENT4_RESULTS_DIR
```

Retained results are under:

```text
experiments/07_model_consistency/results/
```

Scope: this validation checks the signed-delta-to-anchor derivation and
deterministic XGBoost aggregation-replay mechanisms. It does not claim that
Groth16 proves the complete XGBoost training program, and the controlled harness
does not by itself establish production-path integration for every historical
training run.

## 4. SSI/DID public verification material

Public-only verification material is under:

```text
verification/public_keys/
```

Private vehicle signing keys are excluded from the public archive.

## 5. Offline blockchain-evidence audit

Smart contract and retained Sepolia evidence are under:

```text
verification/blockchain/
```

Run:

```bash
python scripts/verify_blockchain_evidence.py
```

This is an offline structure/evidence check and does not spend gas.

## 6. Optional new Sepolia execution

Implementation:

```text
code/verification/On-Chain-RSU-GlobalServer-SSI-Verification_V10.py
```

Set credentials only through environment variables:

```text
SEPOLIA_RPC_URL
SEPOLIA_PRIVATE_KEY
```

An optional sender lock can be supplied using:

```text
SEPOLIA_EXPECTED_ADDRESS
```

Then inspect the executable options:

```bash
python code/verification/On-Chain-RSU-GlobalServer-SSI-Verification_V10.py --help
```

A new on-chain execution creates new transactions and therefore incurs Sepolia
gas. The retained historical receipts can be audited offline without performing
a new transaction.
