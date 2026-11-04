# Codex repository workflow

## Branch naming

- Never use the `codex/*` pattern for branch names. Use a descriptive,
  task-specific prefix such as `fix/`, `docs/`, or `test/`.

## Roadmap discipline

For every task affecting codec behavior, CUDA/TensorRT execution, entropy,
bitstreams, public APIs, packaging, performance, or FFmpeg integration:

1. Read `ROADMAP.md` before planning or editing.
2. Identify the active milestone and affected checklist items.
3. Stay within the active milestone unless the user reprioritizes the roadmap or
   a prerequisite must be repaired.
4. Do not mark work complete merely because code was written. Run the stated
   verification and record evidence in the appropriate result or evidence
   package.
5. Mark a milestone complete only when every exit criterion passes.
6. After material work, update the roadmap priorities and status. Keep detailed
   command output and measurements in the appropriate result or evidence
   package.
7. Keep one current publication result; do not commit raw runs or long historical
   benchmark logs. Retain an older result only when it explains a compatibility
   or safety decision that the current result cannot explain.

If requested work conflicts with the roadmap, explain the conflict and update the
roadmap only after the user chooses the new priority.

## Codec/provider separation and capability matrix

NVCR is a common neural-video-codec runtime. Codec integrations and execution
providers are independent axes; never add codec-specific provider branches or
provider-specific assumptions to codec semantics.

- The codec adapter owns codec sequencing, reference/state meaning, entropy
  decisions, payload syntax, and the normative stages it requires.
- The provider owns execution technology, device memory, kernels, artifacts,
  scheduling, synchronization, and provider-specific failures.
- A codec declares required versioned numerics profiles for normative stages.
  A provider advertises the profiles it implements and passes their conformance
  suites. CUTLASS, TensorRT, CUDA, CPU, and QNN are provider implementation
  choices, never codec behavior.
- Runtime compatibility is a codec-by-provider capability matrix. Track each
  cell as `exact`, `native`, `unsupported`, or `untested`; do not infer support
  from a GPU model or from another codec/provider pair.
- A provider that lacks a required normative profile must fail before decode
  with a structured incompatibility error. Never silently fall back to native
  arithmetic for a normative stage.
- GPU identity, SM version, TensorRT plan identity, and kernel selection are
  provider/artifact concerns. Do not special-case an RTX model in codec code.
- Do not claim cross-provider or cross-target support until the relevant matrix
  cell has a conformance result and retained evidence.

## Verification discipline

- Use release builds and the protocol in `docs/performance.md` for performance.
- Treat pinned Python DCVC-RT as the behavioral and performance reference until
  a documented replacement is accepted.
- Do not claim upstream compatibility without cross-runtime golden tests.
- Do not claim FFmpeg integration from a CLI-only demonstration. Encoder,
  decoder, drain/reset, timestamps, and a container path must pass their gates.
