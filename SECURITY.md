# Security policy

Report suspected vulnerabilities privately to the repository maintainers using
the private security-reporting mechanism configured for the GitHub repository.
If no private channel is available, open a minimal issue requesting a private
contact without disclosing exploit details.

Do not include credentials, access tokens, private artifact URLs, checkpoints,
TensorRT plans, or sensitive input streams in reports. Include the NVCR commit,
platform, build options, reproducible command, and sanitized logs when safe.

The `.nvcr`, `NVCR`, and `NVCS` formats are development/application wrappers;
malformed-input reports should include the smallest safe reproducer and whether
the bounded parser rejects it before backend execution.
