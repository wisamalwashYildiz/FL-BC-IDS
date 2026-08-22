# Public SSI / DID Verification Material

Files in this directory are publication-safe public projections of the retained
compact security-evidence identities.

Each vehicle file contains only:

- `did`
- `ed25519_pub_b64`

No private signing key, seed, secret, mnemonic, or password material is included.

The original uploaded `verification.zip` contained `ed25519_priv_b64` in all four
vehicle files. Those fields have been removed from this publication package.
Do not publish the original ZIP. If those test signing keys were ever reused
outside the historical experiment, retire/rotate them.
