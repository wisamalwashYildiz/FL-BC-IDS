# Security Redaction Notice

The source archive `verification.zip` contained four private Ed25519 key fields
(`ed25519_priv_b64`) under `verification/public_keys/`.

Those private values are **not present** in this corrected publication package.
The original archive is therefore sensitive and must not be published or nested
inside the public repository.

Original archive SHA-256 (identity only):

`84110572f51c89203f388beecbb27eb2a7a4cd53c22bcf3cfe6e906d14a8d6e0`

The corrected package preserves each affected file's original SHA-256 in
`ORIGINAL_TO_PUBLIC_SHA256.csv` without preserving the private bytes themselves.

If any of the exposed test signing keys were reused outside the historical
experiment, retire/rotate them.
