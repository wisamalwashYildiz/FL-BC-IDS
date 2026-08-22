# FL-BC-IDS Verification Package

This directory collects publication-facing verification artifacts for:

- Groth16 AnchorSum proof verification;
- Sepolia proof-registry evidence;
- public SSI/DID verification material;
- selected compact security-evidence records.

Important boundaries:

1. The historical main predictive-utility output bundle was not retained.
2. The selected `evidence_bundle/compact_security_evidence/` material is
   security/verification evidence, not a substitute main-utility bundle.
3. Public SSI files contain public keys only.
4. The retained Groth16 standalone triplet is a global round-2 proof. The
   deployment records use shared verifier/vkey hashes for RSU and GLOBAL logical
   scopes; see `groth16/VERIFIER_SCOPE_NOTE.json`.
5. Machine-local paths in publication-facing JSON have been replaced by logical
   repository locators into the full compact security-evidence experiment.
