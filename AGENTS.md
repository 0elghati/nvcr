# Codex repository workflow

## Roadmap discipline

For every task affecting codec behavior, CUDA/TensorRT execution, entropy,
bitstreams, public APIs, packaging, performance, or FFmpeg integration:

1. Read `ROADMAP.md` before planning or editing.
2. Identify the active milestone and affected checklist items.
3. Stay within the active milestone unless the user reprioritizes the roadmap or
   a prerequisite must be repaired.
4. Do not mark work complete merely because code was written. Run the stated
   verification and record evidence in `ROADMAP.md`.
5. Mark a milestone complete only when every exit criterion passes.
6. After material work, update roadmap status, current next action, evidence, and
   architectural decisions when applicable.
7. Preserve failed and superseded benchmark results as labeled history.

If requested work conflicts with the roadmap, explain the conflict and update the
roadmap only after the user chooses the new priority.

## Verification discipline

- Use release builds and the protocol in `docs/performance.md` for performance.
- Treat pinned Python DCVC-RT as the behavioral and performance reference until
  the roadmap records a replacement.
- Do not claim upstream compatibility without cross-runtime golden tests.
- Do not claim FFmpeg integration from a CLI-only demonstration. Encoder,
  decoder, drain/reset, timestamps, and a container path must pass their gates.
