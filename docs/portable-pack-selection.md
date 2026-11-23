# First portable codec pack: selection

Status: decision record for issue #280 (feature #268, epic #261),
2026-11-23. It changes no code, schema, bundle, or stream.
Per #280, reviewers can reject it before the pack schema or any export code is
written.

## Selection

| Choice | Selection |
|---|---|
| Baseline codec | MLVC-S, the libmlvc `dmc61sbr_mini_reglu-perceptual_v1-onnx-generic-v0.1` bundle (SHA-256 `c37ee2f5…`, libmlvc `c504aaa`, checkpoint `mlvc-s-perceptual-v1.ckpt`) |
| Portable representation | That ONNX bundle, unchanged: FP16 NCHW encoder and decoder graphs per canvas, one shared `weights.bin`, and the Gaussian and bit-estimator PMF tables |
| Broad provider | ONNX Runtime CPU on x86_64 |
| Declared capability range | Visible sizes up to 640x368, served by the 320x192 and 640x368 canvases. The proof resolutions are 176x144 and 640x360, which already have upstream-runtime parity vectors |
| Numerics | `native`. No cross-provider interoperability is claimed until #287 defines a bounded-quality profile |
| Derived accelerator | The C12 TensorRT plans built from the same ONNX parts, to show that a derived artifact keeps codec and model identity |

## Why the candidates split

The three candidates differ in what "portable" can mean for them today.

- **DCVC-RT and DCVC-UF can only be portable through exact numerics.** Their
  native streams fail with `malformed_bitstream` when decoded by a different
  provider or GPU ([compatibility](compatibility.md#dcvc-rt-capability-matrix)).
  Exact streams (`nvcr.fx.v1`) do decode across providers. However, the
  normative stages are NVCR's own `nvcr.fx-program.v1` programs, not ONNX. On
  the CPU they run in NVCR's reference executor, which is single-threaded
  scalar code (`src/deterministic/fx_program.cpp`). A 1080p decode takes about
  7 minutes per frame for DCVC-RT
  ([evidence](../evidence/fx-dcvcrt-rtx4070-20261112.md)) and about 2.8 minutes
  per frame for DCVC-UF
  ([evidence](../evidence/fx-dcvcuf-rtx4070-20261112.md)). That speed proves
  correctness, but it cannot serve as a product baseline.
- **MLVC can be portable through standard ONNX.** Stock ONNX Runtime CPU
  kernels run the graphs at 13–66 fps on the two proposed canvases. Its
  streams also decode across providers. The cost is bounded drift rather than
  exact output: decoders diverge to 57.5 dB Y at 176x144 and 44.4 dB at
  640x360 by frame 63, and recover to about 58.5 dB at the LTR recovery
  ([C12 evidence](../evidence/phase-c12-mlvc-tensorrt-20261104.md)). There is
  no exact profile, so cross-provider support depends on how #287 defines
  bounded-quality interoperability.

MLVC is therefore the only candidate that can complete a usable disconnected
first run with existing, already-redistributed artifacts. DCVC-RT's exact
numerics remain the strongest interoperability evidence in the repository.

## Comparison

All throughput figures are diagnostic single runs, none taken under
`docs/performance.md`. "Not recorded" means the repository has no figure.

| Field | DCVC-RT (`nvcr.fx.v1`) | DCVC-UF HT-S (`nvcr.fx.v1`) | MLVC-S (`native`) |
|---|---|---|---|
| Pinned source / checkpoint | microsoft/DCVC `1feb52a5`; `cvpr2025_image`, `cvpr2025_video` ([artifacts](dcvcrt-artifacts.md)) | microsoft/DCVC `cbdae87a`; `cvpr2026_image`, `cvpr2026_video_hts` ([licences](../MODEL_LICENSES.md)) | MLVC `1425880`, libmlvc `c504aaa`, bundle `c37ee2f5` ([C10](../evidence/phase-c10-mlvc-codec-20261102.md)) |
| Portable representation | 11 fx programs (`.fxprog.json` + `.fxw`) plus FP16 ONNX opset 18 for the encoder and display stages | 13 fx programs plus 5 FP16 ONNX stages. The ONNX stages are discarded unless `--onnx-dir` is passed, so published bundles contain no DCVC-UF ONNX | Standard ONNX, FP16 NCHW; graphs reference a shared external `weights.bin` |
| Internal graphs | 11 normative and 5 FP16 stages | 13 normative and 5 FP16 stages | 2 graphs per canvas (encoder, decoder), 4 canvases. LTR is codec state, not a graph |
| Shared weights | None. Each program carries its own weights, and the same normative set is copied into every per-resolution bundle | None. Each program has its own `.fxw` | One `weights.bin` shared by all 8 graphs (measured locally, below) |
| Non-model assets | CDF and quantisation binaries, `fx_entropy.json`, embedded sigmoid and WSiLU tables | `tables.bin` (per-QP quantisation rows, CDFs), `fx_entropy.json` | Gaussian and bit-estimator PMF JSON; the vendored `msrtc_rans` coder |
| Shapes | Dynamic, 1–65535, padded to 16 | Dynamic, 1–65535 | Fixed canvases with edge padding; `shape_profile: fixed` |
| Custom ops | fx ops need NVCR's executor or NVCR's TensorRT plugins; ONNX Runtime CUDA and QNN refuse them | Same, plus 10 Triton GEMM variants on TensorRT | None recorded; stock ONNX Runtime CPU runs both libmlvc bundles |
| Codec-owned host stages | rANS, NVI1/NVP1 payloads, QP schedule, references, colour conversion, padding | UF rANS, step masks, chunk schedule, tail handling | YUV conversion and padding, Gaussian indices, rANS, LTR and I-period management |
| Numerics required | `nvcr.fx.v1` for every normative stage; P display stays FP16 | `nvcr.fx.v1`; FP16 for display and encoder analysis | `native` only |
| Provider paths shown | TensorRT → ONNX Runtime CPU `exact`; ONNX Runtime CPU → TensorRT `untested` | TensorRT → ONNX Runtime CPU `exact`; reverse `untested` | ONNX Runtime CPU `native` (x86 parity); TensorRT (SM 12.0 only); QNN on S25 without parity |
| Redistribution | MIT basis. fx bundles are staged but not in the catalogs | MIT basis. fx bundles are staged but not in the catalogs | MIT. Already published: `engine-assets-mlvc` carries libmlvc's bundle unchanged with its licence |
| Portable size | Not recorded | Not recorded | 12.8 MB for all four canvases (measured locally); the RTX 5060 plans are 109 MB |
| CPU throughput | About 7 min per 1080p frame (3 frames, about 22 min) | About 2.8 min per 1080p frame (9 frames, about 25 min) | 66.6 / 52.0–65.6 fps encode/decode at 320x192; 20.4 / 12.8–15.3 fps at 640x368 |
| Parity | 11/11 stages bit-identical to the CPU reference; cross-provider I pictures byte-identical; BD-rate +0.16% against native | 13/13 stages bit-identical; one 9-frame 1080p cross-provider decode; BD-rate −1.5% to −1.8% against native; no upstream comparison | Byte-exact upstream-runtime parity on x86 ONNX Runtime 1.26.0 across 6 cases; the arm64 CPU does not reproduce it |
| Work to cover two resolutions | Pack layout and manifest, an ONNX Runtime CPU fx encode, runs at the new profiles. CPU speed is the blocker | As DCVC-RT, plus keeping the ONNX stages in bundles and removing the dependency on `engine_manifest.json` | A pack manifest that names two canvases. Both resolutions already have parity vectors |

MLVC-S's measured layout, from the installed RTX 5060 bundle
(`af32f837…`): 8 ONNX graphs total 1,561,226 bytes; `weights.bin` is
11,032,864 bytes; the PMF tables are 169,616 bytes; manifests and metadata are
about 34 KB. The 8 TensorRT plans total 109,089,256 bytes.

## Rejected alternatives

- **DCVC-RT under `nvcr.fx.v1`: deferred, not rejected.** It is the production
  codec and it already has exact cross-provider decode. Its portable path,
  however, depends on the CPU fx executor. That executor is a single-threaded
  reference implementation running at about 7 minutes per 1080p frame, and
  nothing has measured how much an optimised executor would gain. Its pack
  would also make NVCR's program format, not ONNX, the portable
  representation. That is consistent with the codec/provider split, but every
  new provider would then have to implement `nvcr.fx.v1` before running any
  pack. Proposed as the second pack, after a measured CPU executor task.
- **DCVC-UF HT-S under `nvcr.fx.v1`: rejected for the first proof.** It has the
  same executor dependency and less evidence: one 9-frame cross-provider
  decode, no ONNX Runtime CPU encode, no second resolution, and no ONNX stages
  in published bundles.
- **The standard MLVC `perceptual` and `psnr` models: rejected for the first
  proof.** They have no upstream reference vectors; their only check is a
  PyTorch-to-ONNX comparison within 0.07 dB. NVCR produces these exports
  itself and has no licence or notice record for them, although their 287 MB
  archives are published.

## Hidden dependencies

Each of these can reject the selection before implementation.

1. **Interoperability meaning (#287).** MLVC-S has no exact profile, and its
   parity holds only on x86. If #287 decides that a v2-supported path must be
   exact, MLVC-S can prove the offline product but not the cross-provider
   requirement. DCVC-RT would then become the first pack, with the CPU fx
   executor on the critical path.
2. **Fixed canvases (#283).** The proof uses two of libmlvc's four canvases as
   a measured choice. It does not adopt the four-canvas policy. The pack
   schema must express a canvas set as one kind of capability range, not as
   its only shape model.
3. **Not shaping the format around one codec.** Before #268 freezes the schema,
   the schema must also describe the DCVC-RT fx layout on paper: programs,
   FP16 ONNX stages, entropy assets, dynamic shapes, a numerics requirement,
   and no shared weight store.
4. **Packaging policy.** `ASSET_DISTRIBUTION_POLICY.md` excludes ONNX bundles
   from generic packages. The complete product (#269) needs an explicit policy
   entry that is separate from the core SDK package.
5. **Architecture scope.** The range is x86_64 only. NHWC and qualcomm bundles
   are rejected with `not_implemented`, and arm64 CPU parity is not
   established.

## Proof plan and exit measurements (for #268)

1. A schema and validator for the pack, reviewed against item 3 above.
2. An encode/decode run from the pack on ONNX Runtime CPU with no TensorRT plan
   present, at 176x144 and 640x360.
3. Correctness: byte-identical records and decoded YUV against MLVC's
   conversion runtime at both resolutions, plus reset, LTR recovery, and
   malformed-unit cases.
4. Startup: cold session-creation time.
5. Memory: peak RSS.
6. Throughput: process fps under `docs/performance.md`, encode and decode, at
   both resolutions, recording the CPU model and thread count.
7. Size: pack bytes and installed bytes.
8. Provenance: the libmlvc commit, the bundle digest, the checkpoint digest,
   and per-part digests.
9. Licence: the MIT licence and notice are inside the pack.
10. A derived accelerator: TensorRT plans built from the pack resolve to the
    same codec and model identity.
11. A fresh machine imports and validates the pack from documented inputs.
