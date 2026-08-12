# Support

Use the repository issue tracker for public support unless a release note
states a different contact.

- Usage questions: include NVCR version, platform/target profile, installation
  method, exact command, and the non-sensitive error output.
- Reproducibility questions: include commit, target profile, artifact manifest
  and bundle digests, input identity, and the relevant evidence-package row.
- Bug reports: include a minimal reproduction, build configuration, expected
  behavior, actual behavior, and whether CPU contract tests or target-local
  runtime tests reproduce it.
- Unsupported hardware requests: explain the device, CUDA/TensorRT versions,
  desired compatibility class, and the evidence you can collect. A matching
  configuration file does not establish support.
- Security reports: do not post credentials, private URLs, restricted model
  assets, or exploitable details in a public issue; follow [SECURITY.md](SECURITY.md).

Model checkpoints, exported graphs, TensorRT plans, and datasets have separate
redistribution terms. Do not attach them to public issues without confirming
their license and provenance.
