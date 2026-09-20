# Measurement readiness assessment

Scope: reproducible evaluation roadmap gate; existing codec, provider, and formats.
The configured four QPs (0, 21, 42, 63), ten repetitions, 100 frames, and GOPs
1/30/100 are preserved. Historical records retain their original definitions.

| Finding and evidence | Consequence | Correction / verification | State |
|---|---|---|---|
| `cli/main.cpp:QualityAccumulator` pools integer-output SSE; pinned `assets/test_video.py:get_distortion` measures float output and `generate_log_json` averages frame PSNR | Reported PSNRs have different domains and temporal definitions | Common offline YUV420P8 evaluator, per-plane SSE/counts and both aggregation definitions; analytic fixtures | Fixed; NVCR validated, Python GPU validation pending |
| Legacy CLI times individual runtime calls; pinned evaluator excludes source conversion/transfer and bypasses its first ten measured frames | Timing and evaluated frame sets differ | Explicit host-to-host completed-frame boundary and in-process warm-up/reset; independent operation processes | Fixed; NVCR validated, Python GPU validation pending |
| Shell diagnostic warms another process | Measured model/context is cold | Same-session warm-up followed by reset and rewind | Fixed; NVCR validated, Python GPU validation pending |
| `profile_memory_command.py` samples VmHWM/RSS; historical Python exports allocator counters and process peaks | Different memory quantities can be compared as if identical | Same Linux wait4 process RSS high-water for both, separate memory passes, MiB, full-process window | Fixed; NVCR validated, Python GPU validation pending |
| Consolidation subtracts configurable fixed bytes and joins legacy PSNRs | Version-dependent byte accounting and incompatible quality | Parse actual nested streams, reconcile file sizes; new schema isolated from legacy | Fixed; NVCR validated, Python GPU validation pending |
| `assets` is clean pinned Microsoft commit 1feb52a592a9ff2c4e4ba2e5122e2da49a211466; `/home/oelghati/DCVC-RT` is modified commit 48ab0ac5 | Local historical wrapper is not the pinned evaluator | Preserve and fingerprint historical wrapper; new explicitly identified runner against clean pinned source | Fixed; NVCR validated, Python GPU validation pending |
| Historical raw wrapper version/command is not bound to each retained row | Historical measurements cannot be fully reconstructed | Mark historical reference provenance incomplete; never relabel as new runner data | Pending external historical evidence |
| Inputs have filenames but no complete acquisition/resize records | Source identity cannot be inferred from resolution | Hash whole input and evaluated prefix; explicit unknown provenance and scene grouping | Fixed; NVCR validated, Python GPU validation pending |
| Backend reset clears both DPBs and retains engines; decode returns synchronized host YUV; model profile pins FP16/checkpoints | Useful existing foundations | Keep semantics and verify reset on a bounded GPU smoke | Already correct; NVCR smoke/reset validated |
| NVCR uses two entropy coders at area >= 720p; upstream uses > 720p | Effective entropy setting differs at exactly 720p | Record and reject silent coding-state parity claims; match explicit reference switch to NVCR with documented wrapper deviation | Fixed; NVCR validated, Python GPU validation pending |

The focused software checks and completed NVCR campaign passed. See
[validation.md](validation.md) for test scope and the remaining reference and
TensorRT warning gates. Overall comparison qualification remains pending.

Historical acquisition/resize commands and historical wrapper-to-result binding
remain unavailable. Common GPU-allocation counters, historical floating metrics
for new runs, and causal runtime-architecture attribution are intentionally not
provided by this measurement change. RSS is not presented as GPU memory.

[The runbook](../measurement-runbook.md) provides a short launcher and the expanded
commands, including resume and independent analysis.
