# Security and Credential Handling

This public reproducibility repository is curated to contain **no private signing
key, wallet private key, mnemonic/seed phrase, access token, populated `.env`
file, or hosted RPC credential**.

Retained blockchain addresses, transaction hashes, receipts, DIDs, Ed25519
**public** keys, Groth16 verification keys, proofs, public inputs, and public
commitments are intentionally public verification material.

For optional new Sepolia executions, provide credentials only at runtime through
environment variables such as:

- `SEPOLIA_RPC_URL`
- `SEPOLIA_PRIVATE_KEY`
- optionally `SEPOLIA_EXPECTED_ADDRESS`

Never commit a populated `.env` file or any private key. `.env.example` contains
names/placeholders only. Offline verification of the retained publication
evidence does not require a wallet private key or hosted RPC credential.

The public SSI projection retains only verification material needed for
independent checking. Curation provenance for the removed historical test
signing-key fields is documented in
`verification/SECURITY_REDACTION_NOTICE.md`; the superseded private-material
source archive is not part of this release.

If you discover a credential, unsafe secret, or integrity problem in a public
release, do not post the secret in a public issue. Contact the corresponding
author or repository maintainer privately and rotate any affected credential.
