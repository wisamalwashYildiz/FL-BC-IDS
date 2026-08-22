# Security and Privacy Notes

## Differential privacy

The numerical values describe the DP-XGBoost learner stage conditional on the
fixed preprocessing artifact and post-preprocessing training multiset.

Compact two-round accounting:
- `epsilon_tree = 0.25`
- `delta = 0`
- `epsilon_round <= 0.5525`
- `epsilon_total <= 1.1050`

They are not presented as an unchanged end-to-end guarantee for an original raw
record before preprocessing.

## Groth16 scope

AnchorSum Groth16 checks the declared scoped arithmetic/public-input relation.
It does not prove the entire XGBoost training computation. Model-update/anchor
and deterministic aggregation consistency are checked separately.

## Identity and blockchain secrets

Only public SSI/DID verification material is published. No private signing key,
wallet private key, mnemonic, or hosted RPC secret is included.
