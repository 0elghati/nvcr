# Security policy

If the repository's Security page offers **Report a vulnerability**, use that
private form. Otherwise, open a minimal public issue asking a maintainer for a
private coordination contact. Do not disclose vulnerability or exploit details
in the issue.

Do not include credentials, access tokens, private artifact URLs, checkpoints,
TensorRT plans, or sensitive input streams in reports. Include the NVCR commit,
platform, build options, reproducible command, and sanitized logs when safe.

The `.nvcr`, `NVCR`, and `NVCS` formats are development/application wrappers;
malformed-input reports should include the smallest safe reproducer and whether
the bounded parser rejects it before backend execution.
