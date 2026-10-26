# Cross-codec runtime requirements

## Status and scope

Status: Phase C0 complete on 2026-09-14.

This document executes the direction already established in
[`NVCR_VISION.md`](NVCR_VISION.md). It records requirements derived from the
current NVCR integration and three released codec implementations. It does not
design a new API or change codec behavior, provider behavior, artifacts, or the
stream format.

Evidence labels used below are intentional:

- **Verified fact** means the statement is supported by the pinned source,
  tests, official repository documentation, or paper listed in this document.
- **Inference** means it is an architectural conclusion drawn from those facts.
- **Open question** means source inspection does not settle the issue and a
  bounded prototype or compatibility test is required.

## Exact revisions audited

| Project | Repository | Branch | Commit |
|---|---|---|---|
| NVCR | `git@github.com:0elghati/nvcr.git` | `main` | `fe6cb8e9db0d90cf2ac33a6f1030a2600bd413e2` |
| DCVC-RT and DCVC-UF | <https://github.com/microsoft/DCVC.git> | `main` | `cbdae87a5445114cdc7f48816da63ea80bdeac40` |
| MLVC | <https://github.com/microsoft/mlvc.git> | `main` | `b881d799af62b8640c14c9ee6df3fb7fea679156` |

All three trees were clean when inspected. The Microsoft repositories were
fresh upstream clones; pre-existing local forks were not used as upstream
evidence.

## Method

The audit followed each implementation from its sequence loop through frame or
chunk policy, reference state, neural-model calls, entropy coding, payload
writing, decoding, and reconstructed output. It then compared those behaviors
with NVCR's public sessions, shared runtime, DCVC-RT backend, provider session,
artifact resolver, and NVAU implementations.

README and paper claims were used for design intent, availability, and declared
platform support. Executable behavior comes from source and tests. No model was
downloaded and no benchmark was run because C0 is a requirements audit, not a
performance or compatibility campaign.

## Current NVCR contract

### Session lifecycle

**Verified fact.** `IEncoderSession` and `IDecoderSession` expose separate
send/receive operations plus `flush()` and `reset()`. Their comments permit
delayed and flush-only output. A deterministic test session proves several
inputs before output and one final output during flush.

**Verified fact.** The production `Runtime` does not implement that general
behavior:

- `send_frame()` synchronously calls `codec::Runtime::encode()` and stores one
  result in `std::optional<Packet>`;
- `send_access_unit()` synchronously decodes and stores one
  `std::optional<Frame>`;
- submitting again before receiving replaces the pending item;
- neither direction can queue more than one output;
- `flush()` cannot collect codec output, marks both directions flushed, and
  calls a codec flush that resets both encoder and decoder state;
- decode cannot reorder output; and
- `try_again` from `receive_*()` means only that the single optional is empty
  and the shared runtime has not been flushed. It does not currently report
  genuine codec lookahead or input accumulation.

The public API comments define successful `send_frame()` and
`send_access_unit()` calls as consuming their input. Send methods never return
`try_again` in v1. Receive methods return `try_again` when no output is ready
and more input may be required. The current production implementation uses
successful sends followed by receive-side readiness because both codec
operations complete synchronously.

**Inference.** The public send/receive method shape is broader than its only
production implementation. Passing lifecycle fixture tests therefore does not
prove that the registered adapter path supports delayed or multi-output codecs.

### Current ownership

| Concern | Current owner | Evidence |
|---|---|---|
| Frame-type decision | Shared `codec::SequenceState` via `codec::Runtime` | `next_frame_type()` selects intra at sequence start and `frame_index % gop_size == 0`, otherwise predicted. |
| GOP policy | Shared `codec::SequenceState` and `RuntimeConfiguration::gop_size` | The policy is applied before the codec backend is called. |
| Reset semantics | Shared runtime, `codec::Runtime`, NVAU validation, and DCVC-RT backend | Public flush/reset affects both directions; NVAU reset is constrained to intra; DCVC-RT clears provider/reference state. |
| Reference selection | DCVC-RT `PredictedOrchestration` | It selects frame versus feature reference from DCVC-RT state and the 64-frame refresh rule. |
| Reference/feature state | Shared `SequenceState` plus DCVC-RT `ReferenceState` and provider buffers | Shared state carries one frame and one latent blob; the backend retains device DPBs and codec meaning. |
| Rate/quality controls | Shared configuration plus DCVC-RT orchestration | Base I QP is global; DCVC-RT applies its eight-position P-frame shift table. |
| Entropy | DCVC-RT orchestration | `IntraOrchestration` and `PredictedOrchestration` own CDF assets and native rANS sequencing. |
| Codec-private payload | DCVC-RT orchestration and payload parser | The codec creates and parses `NVI1` and `NVP1`. |
| Access-unit construction | Shared `codec::Runtime` | It constructs and serializes NVAU v1 after every backend encode. |
| Access-unit validation | Shared `AccessUnitIO` and `codec::Runtime` | Shared code validates framing, model ID, frame type, dimensions, QP, reset, and size before codec decode. |

**Inference.** Entropy, private payloads, stage order, and detailed reference
meaning are correctly codec-owned after Phase B. Frame/GOP selection and NVAU
construction still occur above the codec even though the other audited codecs
do not share those semantics.

### `RuntimeConfiguration` ownership

| Field | Current classification | Future owner |
|---|---|---|
| `intra_engine_path` | artifact-specific | Artifact selection; the codec maps selected components to roles. |
| `predicted_engine_path` | artifact-specific | Artifact selection; no generic runtime meaning. |
| `entropy_model_path` | artifact-specific | Codec artifact selection. |
| `codec_id` | runtime-generic | Runtime construction/dispatch. |
| `model_id` | artifact-specific | Selected codec model set. |
| `bitstream_model_id` | mixed/questionable | Derived from encoder artifact/codec identity; read from and checked against decoder stream metadata. |
| `provider_id` | runtime-generic | Runtime construction/dispatch. |
| `codec_api_version` | compatibility/versioning | Registry and artifact compatibility. |
| `provider_api_version` | compatibility/versioning | Registry and artifact compatibility. |
| `stream_format_version` | compatibility/versioning | Encoder stream choice and decoder stream dispatch, not codec rate policy. |
| `payload_syntax_version` | compatibility/versioning | Codec payload compatibility. |
| `model_set_version` | compatibility/versioning | Artifact/model compatibility. |
| `manifest_schema_version` | compatibility/versioning | Artifact catalog compatibility. |
| `device_id` | provider-specific | Provider configuration. |
| `intra_qp` | codec-specific | Codec rate/quality control. |
| `gop_size` | codec-specific | Codec sequence policy. |
| `memory_pool_bytes` | runtime-generic | Generic host-resource limit. |
| `device_arena_bytes` | provider-specific | Provider memory policy. |
| `max_packet_bytes` | runtime-generic | Common input/output resource bound. |
| `log_level` | runtime-generic | Common diagnostics policy. |
| `tensorrt_execution_mode` | provider-specific | TensorRT provider configuration. |
| `allow_legacy_access_units` | mixed/questionable | Decoder stream-compatibility policy; it should not share a bag with encoder codec settings. |
| `verify_encoder_reconstruction` | codec-specific | Codec validation/debug policy. |
| `enable_profiling` | provider-specific | Provider execution diagnostics, surfaced through runtime reporting. |

**Inference.** One struct currently crosses five ownership domains. C3 needs to
separate ownership, but it need not invent a configuration hierarchy: scoped
codec/provider option schemas and artifact requests already exist.

### Capability expression

**Verified fact.** `CodecCapabilities` represents intra, predicted,
bidirectional, hierarchical, delayed-output, and a single QP range. It cannot
state input grouping, output multiplicity, chunk size/tail behavior, arbitrary
dependencies, flush-produced output, or codec-defined modes such as target
bitrate, frame dropping, LTR recovery, and distinct model structures.

**Inference.** The flags are adequate for discovery of current DCVC-RT, not for
admission or negotiation of all three codecs. C0 does not establish a minimal
replacement capability vocabulary, so C1 should not add one.

## DCVC-RT requirements

### Paper/design intent

**Verified fact.** The DCVC-RT paper describes implicit temporal modeling, one
low-resolution latent representation, model integerization for cross-device
consistency, and module-bank rate control. These goals explain its low-latency,
single-frame implementation but do not establish lifecycle behavior by
themselves.

### Current implementation behavior

**Verified fact.** The pinned upstream implementation has the following
behavior:

- one input frame produces one I- or P-frame NAL and one reconstructed frame;
- frame zero and positive `frame_idx % intra_period == 0` positions are I
  frames; other positions are P frames;
- no lookahead, delayed output, output reordering, or multi-output path exists;
- an I frame clears the P-frame DPB and seeds it from the reconstructed I frame;
- P-frame state keeps a feature reference and reconstructs a frame reference
  when needed; a periodic `use_ada_i` signal resets the feature path;
- the P base QP is shifted through the eight-position frame-quality map, while
  I and P use separate checkpoints;
- neural analysis/hyperprior/prior/synthesis operations run through PyTorch and
  CUDA-oriented fused extensions, while native CPU rANS encodes `z` and the
  masked `y` partitions;
- the upstream stream is a private SPS plus I/P NAL syntax. SPS carries
  dimensions, entropy-partition mode, and feature-reset mode; each I/P record
  carries SPS ID, QP, length, and rANS payload;
- YUV420 input is converted to the model's three-channel representation,
  padded for model execution, and cropped after reconstruction; and
- the source test loop requires CUDA timing/execution, although individual
  fused operations can fall back to PyTorch implementations.

**Verified fact.** NVCR reproduces the I/P schedule, eight-position P QP shifts,
64-frame feature-reference refresh behavior, model sequencing, CPU rANS, and
reconstructed-reference update. It deliberately uses profiled TensorRT stage
bundles, `NVI1`/`NVP1` codec payloads, and NVAU rather than claiming byte
interchange with upstream Python SPS/NAL files. The current bundle contains
seven I stages, seven P stages, quantization tables, entropy CDFs, manifest
metadata, and target-local TensorRT plans; upstream distributes two PyTorch
checkpoints.

**Inference.** DCVC-RT explains the one-input/one-output, I/P-only, fixed-GOP,
and TensorRT-path assumptions that remain in shared NVCR code. Those assumptions
are not generic requirements.

## MLVC requirements

### Paper/design intent

**Verified fact.** The MLVC report targets cross-platform inference. It sends
entropy scales explicitly through the hyperprior to prevent platform numerical
variation from causing entropy-decoder failure, then uses gated memory,
long-term-reference recovery, and hardware-compatible models to regain coding
efficiency.

### Current implementation behavior

**Verified fact.** The pinned conversion/runtime path processes one frame at a
time and immediately returns one payload and one reconstructed frame. It has no
lookahead or display reordering, but its codec state is not DCVC-RT-shaped:

- I frames clear encoder and decoder reference managers;
- ordinary P frames reference the latest short-term frame;
- LTR-recovery frames reference the latest marked long-term frame;
- the manager retains one short-term reference and at most three long-term
  references;
- a feature reset can occur independently of I-frame reset;
- public standard-model defaults use an I period of 64 with feature reset
  disabled; the mini model uses I period 1024 and LTR start/period 8/64;
- fixed-Q-index and target-bitrate modes are supported, including sticky
  per-frame overrides;
- rate control can drop an ordinary P frame. The file syntax records that
  presentation position as Q index `-1` with an empty payload, and reconstruction
  repeats the previous frame without advancing codec reference state;
- the actual `.mlvc` record is only little-endian signed Q index plus unsigned
  payload length and payload. Dimensions, frame class, reference choice, reset,
  padding, and timing come from the same frame loop and model configuration;
- native `msrtc.rans` C++ code owns entropy. Encoding pushes `y1`, `y0`, then
  `z`; decoding reads `z`, `y0`, then `y1`, consistent with rANS stack order;
- the normal split has an encoder executable and decoder executable, with an
  optional scale decoder; other split definitions divide work differently;
- model bundles contain target-specific executable parts, conversion metadata,
  Gaussian and bit-estimator PMFs, model dimensions, hashes, and optional scale
  decoder files;
- input is raw YUV420, usually converted to YUV444 before inference. An I420
  split variant moves conversion into the graph;
- each executable model has a fixed width and height. A smaller frame is padded
  and may be rotated to fit; a larger frame is rejected; and
- CoreML, ONNX/ONNX Runtime, and OpenVINO paths use different target transforms
  and runtime settings. Apple, Intel, Qualcomm, DirectML, CUDA, and CPU targets
  therefore do not share one neutral executable artifact.

**Verified fact.** The repository is MIT licensed and publishes four current
checkpoint links with SHA-256 values. It notes that those OpenVid-trained
checkpoints differ from the Vimeo-trained models used in the paper. Conversion
tests export platform-selected models and check 128-frame median PSNR and BPP;
the platform cases are conditional, not one cross-platform bitstream golden run
inside a single CI job.

**Inference.** MLVC requires codec ownership of LTR/reset/drop/rate policy and
requires artifact selection to account for model split, fixed resolution,
target transforms, and provider settings. Multiple supported runtimes are
evidence for provider diversity, not evidence of one provider-neutral model
contract.

## DCVC-UF requirements

### Paper/design intent

**Verified fact.** The DCVC-UF paper defines a chunk as a non-overlapping group
of frames encoded into one compact latent and decoded together through
frame-specific reconstruction branches. It presents chunk size eight for the
high-throughput model and size one for low-delay use, plus one consolidated
entropy interaction per chunk.

### Current implementation behavior

**Verified fact.** The pinned source makes the algorithmic chunk semantics
explicit rather than using a tensor batch dimension:

- HT-S and HT-L set `g_frame_delay = 8`. Eight frames are concatenated along
  the channel dimension, processed by one chunk encoder, represented by one
  `y`/`z` entropy payload, and reconstructed by eight frame-specific decoder
  branches. LD sets `g_frame_delay = 1` and uses the frame-sequential model;
- an I position consumes one frame. Each following HT P chunk consumes up to
  eight real frames; a final short chunk repeats its last real frame until the
  eight-frame model input is full;
- one P compression call writes one P NAL. Per-frame accounting assigns all
  bits to the first real frame and zero to the remaining frames in the chunk;
- one P decompression call returns a list of eight reconstructed tensors. The
  sequence loop emits only the known number of real tail frames, in input and
  presentation order;
- the stream record does not carry chunk cardinality or valid tail count. The
  decoder infers eight versus one from the selected model structure and infers
  the final valid count from the externally configured total frame count;
- there is no B-frame reorder. The lifecycle delay is input accumulation for an
  eight-frame chunk, not presentation reordering;
- chunks are sequentially dependent through propagated feature/memory context.
  An I unit clears state and seeds the P model; periodic feature-memory reset is
  signaled in the P record and preserves a reconstructed feature for subsequent
  chunks;
- the sequence loop requires a positive multi-frame intra period to be aligned
  to the eight-frame delay, with the first P chunk beginning after the initial I
  frame;
- one selected QP, one entropy-parallelism value, and one reset flag apply to a
  whole P chunk. There are 64 QP levels and no released adaptive chunk-size
  selection;
- CPU rANS is driven by a worker thread alongside CUDA execution. HT-S uses a
  single masked-y entropy step, while HT-L retains four masked-y steps inside
  one chunk payload; both encode `y` and `z` into the private record;
- the released inference path is PyTorch plus custom C++/CUDA/CUTLASS kernels,
  CUDA graphs, preallocated device tensors, and CPU rANS. It does not publish a
  provider-neutral ONNX/CoreML/OpenVINO export path;
- four downloadable checkpoints cover the I model and HT-L, HT-S, and LD P
  models. The source is MIT licensed with retained CompressAI Apache notice;
  and
- YUV420 and PNG/RGB input are supported, original dimensions are padded and
  cropped, and performance is explicitly optimized only for listed resolutions
  and NVIDIA GPUs.

The repository's only forward-looking runtime statement is that more practical
functionality will be added; it does not specify adaptive chunks, random access,
or a portable artifact contract.

**Inference.** A faithful HT codec session must accept seven frames without an
access unit, emit one access unit after the eighth, accept one access unit that
creates eight frames, retain those frames for sequential `receive_frame()`
calls, and convert a short final input group into one padded codec chunk during
flush. The codec—not the generic runtime—must own padding, valid-tail count,
chunk QP/reset policy, and chunk reference state.

## Cross-codec matrix

Cells in the three codec columns and `Current NVCR` record verified current
behavior. `NVCR implication` is an inference from those facts.

| Concern | DCVC-RT | MLVC | DCVC-UF | Current NVCR | NVCR implication |
|---|---|---|---|---|---|
| Primary input unit | One frame | One frame | HT: one frame submitted into an eight-frame chunk; LD: one frame | One frame | Keep frame submission; let the codec group inputs. |
| Primary output unit | One I/P payload; one reconstructed frame | One frame record; one reconstructed or repeated presentation frame | HT: one P payload and eight reconstructions; LD: one and one | One NVAU v1 packet; one frame | Packet and frame cardinalities must be independent. |
| Inputs before output | One | One, except a dropped P skips model output | HT P: eight; I and LD: one | Exactly one synchronous input | Production sessions must permit accepted input with no output. |
| Outputs per input | One packet and one reconstruction | One record and one presentation result; a dropped P has an empty record and repeats prior output | Over a full HT chunk: one packet per eight submitted frames; decoder yields eight frames per packet | One packet or frame stored per send | Remove one-to-one backend assumptions. |
| Delayed output | None | None | HT P waits for eight submitted frames; final short chunk waits for drain | Advertised by interface/test fixture; absent in registered path | Preserve send/receive and implement codec-owned delay. |
| Encoder buffering | No codec lookahead | No lookahead; rate controller retains budget history | HT buffers up to eight frames; final short group is padded | One optional packet, not input buffering | Codec session owns bounded input buffering. |
| Decoder buffering | No | No model delay | HT decode returns eight tensors together | One optional frame | Codec session must queue all frames from a payload. |
| Reordering | None | None | None; chunk outputs remain presentation ordered | None | Do not add reorder machinery merely for these codecs. |
| Flush/drain semantics | No delayed codec output | No delayed codec output | HT flush must encode a short buffered group using repeated-tail padding | Shared flush emits nothing and resets both directions | Directional drain must reach codec session before reset. |
| Reset semantics | I clears DPB; periodic P feature refresh is codec syntax | I clears refs; feature reset and LTR policy are independent | I clears chunk context; P record can reset feature memory | NVAU reset iff intra; public reset/flush clears both sides | Distinguish sequence reset, codec feature refresh, and drain. |
| GOP/chunk semantics | Periodic I/P GOP plus eight-position QP map | I period, STR/LTR scheduling, optional drops | I followed by sequential eight-frame P chunks; aligned positive intra period | Shared fixed `gop_size` chooses I/P | All sequence policy belongs to codec. |
| Frame/chunk classes | I, P | I, P, LTR recovery, dropped presentation | I unit; P chunk; LD P frame | Intra, predicted | Common class is coarse metadata; codec-private syntax carries semantics. |
| Reference structure | One prior frame/feature path | One STR plus up to three LTR entries | One propagated prior-chunk feature/memory context | One shared frame/latent plus DCVC device state | Shared `SequenceState` cannot model all three faithfully. |
| Persistent feature state | Prior feature, periodically reconstructed/refreshed | Gated feature/memory; model-dependent reset; optional ref frame | Feature and hidden memory propagated between chunks | Opaque latent plus DCVC `ReferenceState` and device buffers | State representation and mutation stay codec-owned. |
| Rate/quality control | Base I/P QP; P shift map; 64 selectable rate points | Q index, target bitrate, sticky override, drop decision | One QP per I unit or P chunk; 64 levels | `intra_qp`, fixed P shifts, `gop_size` | Capability/configuration must not reduce rate control to one QP range. |
| Entropy coder | Native CPU rANS; `z` and masked `y` partitions | Native C++ `msrtc.rans`; PMF JSON; explicit scales | CPU rANS overlapped with CUDA; one chunk payload | Codec-owned native rANS and CDF assets | Keep entropy entirely behind codec session. |
| Payload ownership | Codec-private SPS/I/P NAL | Codec-private 8-byte record plus rANS payload | Codec-private SPS/I/P NAL; P payload represents a chunk | DCVC owns `NVI1`/`NVP1`; shared code owns NVAU | Codec returns complete private payload plus envelope metadata. |
| AU metadata required | Model/profile, dimensions, QP, I/P, sequence reset | Model/split, dimensions/padding policy, Q index, I/P/LTR/drop, reference/reset policy | Model structure, dimensions, QP, I/P chunk, valid frame count, feature reset | v1 has model, dimensions, QP, I/P, reset; v2 adds codec/profile/order/dependencies/sections | Known extra semantics can be codec-private; cardinality/timing needs a prototype contract check. |
| Model stages | Separate I/P networks with internal analysis, hyperprior, priors, synthesis | Usually encoder and decoder executables plus optional scale decoder; alternate splits exist | Separate I and P variants; HT encoder plus shared feature and eight frame-specific decoder branches | Seven I and seven P TensorRT stages | Stage topology must be artifact- and codec-defined. |
| Artifact layout | Two PyTorch checkpoints upstream | Target/model-type/fixed-resolution bundle with model parts, metadata, PMFs, hashes, optional scale decoder | I checkpoint plus one of HT-L/HT-S/LD checkpoints and compiled extensions | TensorRT bundle with plans, CDF/quant assets, manifest | Resolver concept survives; TensorRT-shaped catalog fields cannot be universal. |
| Pixel/color format | YUV420 input; three-channel model representation | YUV420 input; YUV444 conversion outside or in I420 graph | YUV420 or RGB/PNG; three-channel model input | YUV420P8 public path | Color conversion placement is codec/artifact-specific. |
| Resolution handling | Arbitrary original size padded and cropped | Fixed model canvas; smaller/rotated input padded, oversized input rejected | Arbitrary original size padded/cropped; optimized resolution/device list | Fixed validated TensorRT profiles, 64..1920 by 64..1080 | Artifact compatibility must express fixed canvases and profiled ranges. |
| Provider/runtime assumptions | PyTorch/CUDA-oriented inference plus CPU rANS | CoreML, ONNX Runtime EPs, OpenVINO, or Torch with target-specific settings | PyTorch, CUDA/CUTLASS extensions, CUDA graphs, CPU rANS | TensorRT FP16 and CUDA plus native CPU rANS | Provider execution and codec entropy can cross language/device boundaries. |
| Device-resident state | Feature/frame state retained for CUDA inference | Model outputs return through NumPy; provider-specific runtimes own execution state | Preallocated CUDA feature/memory/context tensors | TensorRT provider buffers retain DPBs; codec owns meaning | Opaque provider buffers are sufficient if codec controls lifetime/meaning. |
| Async execution need | CUDA streams overlap entropy/model work internally | Current frame loop invokes model parts synchronously | Worker CPU entropy overlaps CUDA; CUDA events/graphs coordinate work | Provider completion/dependency objects exist; runtime waits synchronously | Provider session already has the needed async primitive; public async API is not proven necessary. |
| Cross-provider concerns | Paper integerization goal; released path is CUDA-oriented | Explicit scale transmission, provider-specific graph passes, precision and EP settings | No released cross-provider path | Only TensorRT production evidence | ORT-CUDA is useful evidence, but cross-provider bitstream tests remain mandatory. |
| Reference compatibility testing | Upstream self-round-trip; NVCR has pinned Python/native behavioral golden and native byte parity across Phase B | Platform-conditional 128-frame conversion tests check PSNR/BPP; public weights differ from report weights | Test loop self-encodes/decodes; no released cross-provider golden | DCVC-RT golden plus deterministic CPU fixtures | Each new codec/provider pair needs pinned encode/decode and cross-provider reference gates. |

## NVCR gap classification

Each row receives one primary class:

- **A — Already generic enough**
- **B — Generic concept, implementation incomplete**
- **C — DCVC-RT-specific assumption in shared code**
- **D — Provider-specific assumption in shared code**
- **E — Stream contract may need extension**
- **F — Unknown; prototype/research required**

No audited abstraction is assigned F. The prototype questions below concern
future timing, portability, and capability decisions; they do not make the
current abstractions unclassifiable.

| Abstraction | Class | Evidence and consequence |
|---|---|---|
| `IEncoderSession` | **A — Already generic enough** | Repeated `send_frame`, repeated receive, flush, and reset can express an eight-frame UF encoder without a chunk-specific API. |
| `IDecoderSession` | **A — Already generic enough** | One `send_access_unit` followed by eight `receive_frame` calls can express UF; no new decoder verb is established. |
| `Runtime` | **C — DCVC-RT-specific assumption in shared code** | One optional per direction, synchronous one-to-one calls, and shared flush/reset contradict UF cardinality and directional drain. |
| `codec::Runtime` | **C — DCVC-RT-specific assumption in shared code** | It chooses fixed-GOP I/P classes, owns one frame/latent state, emits NVAU v1, and validates DCVC-shaped sequence state. |
| `ICodecAdapter` | **B — Generic concept, implementation incomplete** | Discovery and codec/provider composition are sound, but adapters create a one-to-one `CodecBackend` rather than the existing encoder/decoder session contract. |
| `CodecBackend` | **C — DCVC-RT-specific assumption in shared code** | `encode()` and `decode()` each return exactly one result and receive shared I/P plus `SequenceStateView`. |
| `CodecCapabilities` | **B — Generic concept, implementation incomplete** | It names delay and B/hierarchical support but cannot describe grouping, output multiplicity, drain output, dependencies, or codec rate modes. |
| `RuntimeConfiguration` | **D — Provider-specific assumption in shared code** | TensorRT mode and device arena are in the common struct; it also mixes DCVC GOP/QP, artifact paths, versions, and stream policy. |
| Registry/construction | **B — Generic concept, implementation incomplete** | Codec and provider factories are separate, but compatibility currently means only that both IDs are registered and the production adapter is TensorRT-build-gated. |
| Provider session | **A — Already generic enough** | Opaque buffers, typed bounded tensors, executable stages, dependencies, completions, and reset cover the observed graph-execution needs without exposing TensorRT types. |
| Artifact descriptors/resolution | **D — Provider-specific assumption in shared code** | Generic identity/version/digest fields coexist with engine-profile, CUDA-runtime, compute-capability, and TensorRT-oriented catalog assumptions. |
| NVAU v1 | **E — Stream contract may need extension** | DCVC-UF proves one AU may represent eight frames; MLVC proves I/P is not the complete private frame-class vocabulary. V1 cannot name codec/profile, chunk output count, order, or dependencies. No stream change is authorized by C0. |
| NVAU v2 | **B — Generic concept, implementation incomplete** | Codec/profile identity, order, dependencies, and typed/private sections can carry known codec data, but the production writer still emits v1 and common validation retains DCVC QP/I/P/reset constraints. |

## Safe conclusions now

### `codec::Runtime`

**Inference.** It should not survive as the shared semantic runtime. DCVC-RT,
MLVC, and DCVC-UF disagree on frame classes, input grouping, state, reset,
reference policy, output multiplicity, and rate control. The behavior currently
inside `codec::Runtime` should become codec-session behavior (or be absorbed by
the DCVC-RT adapter). Common dispatch, resource bounds, diagnostics, provider
construction, and bounded NVAU parsing/serialization remain shared. This
conclusion follows from real codec behavior, not layering preference.

### Session API and generic queues

**Inference.** The existing six session verbs survive: `send_frame`,
`receive_access_unit`, `send_access_unit`, `receive_frame`, `flush`, and `reset`
can express all known lifecycle shapes. The minimum missing contract is a real
registered-session path that:

1. permits successful input submission with zero immediate output;
2. permits repeated receives after one input;
3. lets directional flush produce output before end-of-stream; and
4. preserves every pending output instead of replacing a single optional.

The generic runtime should queue **nothing codec-semantic**. Encoder input
buffers and packet queues belong to the encoder session; decoded-frame queues
belong to the decoder session. A thin facade may enforce common packet-size and
state rules, but a second shared queue would duplicate session state and make
flush ownership ambiguous.

### GOP, chunk, and configuration ownership

**Inference.** GOP/chunk/LTR/drop policy belongs to the codec. Runtime owns
codec/provider selection, common resource limits, and logging. Codec owns
sequence policy, state, entropy, private payload, rate control, and model role
mapping. Provider owns device, execution mode, memory arena, profiling, and
executable mechanics. Artifact selection owns model/profile/component paths and
compatibility metadata. Stream metadata owns encoded identities and syntax
versions; decoder configuration supplies acceptance policy, not the encoded
truth.

### NVAU v2

**Inference.** For known requirements, NVAU v2 is **sufficient with existing
codec-private sections**. A DCVC-UF AU can be a chunk-level AU whose payload or
private/configuration section records model structure and valid tail count; its
outer order index identifies the chunk. MLVC LTR, feature reset, and drop
semantics can remain codec-private rather than expanding the common frame enum.

C0 did not prove that per-frame timestamps for a multi-frame AU were adequate.
Timestamp ownership normally belongs to a container/application layer, which is
outside C0, and the released UF file relies on external frame count and fixed
cadence. C4 tests the existing development packet boundary; a container-ready
mapping remains separate work.

### Second codec and second provider

**Inference.** MLVC is the stronger second **production codec** candidate after
C1-C3:

| Criterion | MLVC | DCVC-UF |
|---|---|---|
| Architectural value | Exercises LTR, independent feature reset, frame drop, bitrate control, fixed model canvases, target-specific bundles | Strongest lifecycle/cardinality stress through eight-frame chunks |
| Implementation cost | Existing ONNX/CoreML/OpenVINO conversion, two-part default split, published PMFs and hashes | Custom PyTorch/CUDA/CUTLASS proxy, eight decoder branches, no portable export path |
| Provider independence | Designed and tested across several runtimes, although artifacts differ by target | Released implementation is NVIDIA CUDA-specific |
| Artifact/license availability | MIT; four checkpoint URLs and hashes; automated bundler | MIT plus Apache notice; OneDrive checkpoint set without a repository hash table |
| Reproducibility | Public conversion tests and data paths; caveat that public weights differ from report weights | Self-round-trip test and public source; no cross-provider/export fixtures |
| Proof of codec genericity | Strongly disproves DCVC-only state/config/artifact assumptions | Strongly disproves one-to-one lifecycle assumptions but would also require a new execution path |

DCVC-UF remains the required C2 lifecycle acceptance case, first through a
deterministic fixture and later through a bounded feasibility prototype. That
separates lifecycle proof from the cost of porting its custom execution stack.

**Inference.** The audit does not displace ONNX Runtime CUDA as the strongest
first second-provider candidate. MLVC's released ONNX Runtime CUDA path makes it
more relevant, while its target-specific graph passes show why provider support
must be proven with a concrete artifact rather than registry claims. A second
codec and second provider should be validated as separate axes before their
combined path is treated as proof.

## Implementation follow-through — 2026-09-14

C1-C4 now implement and test the boundaries selected by this audit:

- registered adapters return codec-owned encoder and decoder sessions;
- DCVC-RT owns fixed-GOP I/P policy, sequence state, NVAU semantics, and output
  queues behind those sessions;
- a registered eight-frame grouped fixture proves delayed output, asymmetric
  cardinality, short-final-group drain, reset/reuse, repeated sessions, and
  independent directions without a chunk API;
- configuration is divided into runtime, codec, provider, artifact-selection,
  and stream-policy scopes without changing existing CLI or config-file keys; and
- the grouped fixture preserves distinct, non-uniform frame timestamps for a
  full eight-frame group and a short final group across `PacketIO`
  serialization without a chunk API or NVAU change.

The clean matched CUDA 12.8/TensorRT 10.9 Release suite passes 22/22 tests. The
65-frame QCIF GOP-8 stream and decoded output match the B7 hashes exactly. See
[the Phase C1-C3 evidence](../evidence/phase-c1-c3-runtime-ownership-20260914.md)
and [the C4 evidence](../evidence/phase-c4-grouped-au-timing-20260914.md).

C4 answers the development `PacketIO` timing question. It does not establish an
FFmpeg or standard-container mapping. No chunk interface, NVAU change, or
capability taxonomy is justified.

C5 tested `microsoft/mlvc` at the pinned revision with the public MLVC-S PSNR
checkpoint, the upstream default generic ONNX FP16 export, the in-memory
PyTorch CPU reference, and a proven ONNX Runtime 1.26.0 CUDA session. The
acceptance limits were fixed before the run at 0.05 dB absolute drift for every
reported PSNR aggregate and 0.0001 absolute mean-BPP drift at every rate point.
The two-frame QCIF vector failed: the largest PSNR drift was 0.1608 dB and
mean-BPP drift was 0.001736. The tested artifact/provider pair is not accepted
for that strict reference-consistency target. See
[the C5 evidence](../evidence/phase-c5-mlvc-reference-export-20260914.md).

C6 found that MLVC records `--torch-device cuda` without moving the in-memory
export reference off CPU. The named-state comparison remains a valid
CPU-reference/ORT-CUDA comparison. On the first frame, the paths have identical
`z_raw`, but 2 of 2,376 `y_raw_0` symbols and 7 of 2,376 `y_raw_1` symbols
differ by one before entropy coding. The entropy and decoder differences are
downstream, so the tested pair is not byte/payload interchangeable with the
reference path. MLVC's general suitability, an MLVC ORT self-conformant path,
and ONNX Runtime CUDA's suitability as an NVCR provider remain unresolved. See
[the C6 evidence](../evidence/phase-c6-mlvc-cuda-equivalence-20260914.md).

## Open questions requiring prototypes

| Question | Proposed experiment | Answer/decision rule |
|---|---|---|
| DCVC-RT second-provider feasibility | Hold DCVC-RT codec semantics constant and run a bounded ONNX Runtime CUDA provider experiment. | Continue only if the provider passes a separately declared self-conformance and reference-consistency gate; state cross-provider and byte-interchange claims independently. |
| DCVC-UF fused-operator export | Export one bounded fused operator and one frame-specific HT branch, then compare a pinned upstream encode/decode vector with the provider-stage prototype. | Continue only if the prototype preserves the selected bitstream/reconstruction contract and does not require generic runtime code to understand UF branch semantics. |
| Minimum capability vocabulary | Integrate a second real codec session behind the current descriptor and record each application admission decision that cannot be made. | Add only facts required by a failing admission test shared by at least two real codecs; do not create a speculative taxonomy. |

- **Open question:** Can DCVC-UF's custom fused operators and frame-specific HT
  branches be exported as provider stages without changing bitstream behavior?
- **Open question:** Which small capability facts are needed for application
  admission after two real codec sessions exist? C0 shows current omissions but
  does not justify a new capability taxonomy.

None of these questions was needed to complete C1-C6. The provider and codec
experiments remain separate axes.

## Recommended Phase C sequence

1. **C1 — Codec-runtime ownership cleanup.** Make registered adapters supply
   real encoder and decoder sessions through the existing session interfaces.
   Move fixed GOP/I/P selection, `SequenceState`, and codec-specific AU
   population/interpretation out of shared `codec::Runtime` into DCVC-RT-owned
   session behavior. Keep bounded envelope I/O common. Preserve bytes and
   reconstructed output.
2. **C2 — Buffered/delayed-output lifecycle.** Prove the registered path with a
   deterministic eight-input/one-packet and one-packet/eight-frame fixture,
   including a short final group emitted by encoder flush, full drain, reset,
   repeated sessions, and independent encoder/decoder state. Do not add a chunk
   interface.
3. **C3 — Configuration ownership split.** Separate runtime, codec, provider,
   artifact-selection, and decoder stream-policy inputs while preserving current
   CLI/config behavior and artifact validation.
4. **C4 — Grouped-AU timing prototype.** Preserve all frame timestamps and
   output order for full and short grouped access units across `PacketIO`.
   Keep the timing data codec-scoped and do not change NVAU.
5. **C5 — MLVC reference/export feasibility.** Pin one public model set and
   tolerance, then compare the upstream reference and ONNX Runtime CUDA paths
   before adding an NVCR provider or codec adapter.
6. **C6 — MLVC CUDA equivalence diagnosis.** Trace the first divergent state in
   the pinned C5 vector. Preserve its acceptance limits and do not start NVCR
   integration unless a concrete artifact/provider pair passes.
7. **C6.5 — Generic CLI session driving.** Drive codec sessions through send,
   receive-until-`try_again`, directional flush, and final drain using the same
   helper exercised by the grouped fixture.
8. **Provider-axis handoff.** Hold DCVC-RT codec semantics constant and test a
   separately defined provider path, with ONNX Runtime CUDA as the current
   candidate.
9. **C7 — DCVC-UF fused-operator feasibility.** Keep this as a pending,
   separate codec-axis experiment. Compare one fused operator and
   one frame-specific high-throughput branch against a pinned upstream vector
   without moving UF branch semantics into generic runtime code.

**Inference.** C5 and C6 reject the tested default FP16 MLVC/ORT-CUDA pair for
strict reference consistency and byte/payload interchange with the reference
path. They leave MLVC self-conformance and ONNX Runtime CUDA provider
suitability unresolved. After C6.5, the next bounded experiment holds DCVC-RT
constant while testing the provider axis. C7 remains pending until a concrete
DCVC-UF operator path preserves the selected bitstream and reconstruction
contract.

## Explicit non-goals

C0 does not implement a codec-runtime refactor, buffering, configuration split,
new capability enum, chunk API, dependency language, NVAU change, second codec,
second provider, FFmpeg/container integration, export pipeline, benchmark, or
training change. It does not claim upstream DCVC-RT byte compatibility, MLVC
cross-platform bit-exactness, or DCVC-UF provider portability.

## Source paths / references

### NVCR

- `AGENTS.md`, `.github/copilot-instructions.md`, `ROADMAP.md`
- `docs/NVCR_VISION.md`, `docs/provider-boundary-v2.md`,
  `docs/neural-bitstream-envelope.md`, `docs/bitstream.md`,
  `docs/extending-nvcr.md`, `docs/performance.md`
- `include/nvcr/codec/session.hpp`, `adapter.hpp`, `backend.hpp`,
  `descriptor.hpp`, `runtime.hpp`, and `sequence_state.hpp`
- `include/nvcr/runtime/runtime.hpp`, `packet.hpp`, and `registry.hpp`
- `include/nvcr/configuration/configuration.hpp`
- `include/nvcr/provider/provider_api.hpp`
- `include/nvcr/provider/experimental/session.hpp`
- `include/nvcr/artifacts/resolver.hpp`
- `include/nvcr/bitstream/access_unit.hpp`
- `src/codec/runtime.cpp` and `src/codec/sequence_state.cpp`
- `src/runtime/runtime.cpp` and `src/runtime/registry.cpp`
- `src/bitstream/access_unit.cpp`
- `src/dcvcrt/adapter.cpp`, `registration.cpp`, `orchestration.cpp`,
  `orchestration.hpp`, `payload.cpp`, `payload.hpp`, `rans_codec.cpp`, and
  `include/nvcr/dcvcrt/rans_codec.hpp`
- `src/dcvcrt/backend/tensorrt/backend.cpp`
- `tests/contract_tests.cpp`, `tests/support/test_codec/test_codec.cpp`, and
  DCVC-RT payload/orchestration/runtime/round-trip tests

### DCVC-RT at the pinned Microsoft/DCVC revision

- `DCVC-family/DCVC-RT/README.md` and
  `DCVC-family/DCVC-RT/test_video.py`
- `DCVC-family/DCVC-RT/src/models/image_model.py`,
  `video_model.py`, `common_model.py`, and `entropy_models.py`
- `DCVC-family/DCVC-RT/src/layers/cuda_inference.py`
- `DCVC-family/DCVC-RT/src/utils/stream_helper.py`, `transforms.py`,
  `video_reader.py`, and `video_writer.py`

Paper: Z. Jia et al., [*Towards Practical Real-Time Neural Video
Compression*](https://arxiv.org/abs/2502.20762), CVPR 2025.

### MLVC at the pinned Microsoft/mlvc revision

- `README.md`, `LICENSE`, `NOTICE`, `pyproject.toml`
- `video/conversion/_frame_loop.py`, `_coder.py`, `_model_wrapper.py`,
  `_rate_controller.py`, `_model_bundler.py`, `types.py`, and `utils.py`
- `video/conversion/_full_model/_base_model.py`,
  `_dmc61_model.py`, and `model_configs_example.yaml`
- `video/conversion/_split_model/_base_split_model.py`,
  `_split_model_factory.py`, `_dmc61sr_e1d1.py`, `_dmc61sbr_e1d1.py`,
  and `_dmc61sbr_e1d1_i420.py`
- `video/conversion/_exporter/_base_exporter.py`,
  `_coreml_exporter.py`, `_coreml_utils.py`, `_onnx_exporter.py`,
  `_onnx_utils.py`, and `_openvino_exporter.py`
- `video/conversion/_windowsml.py`
- `video/src/utils/rate_controller.py`
- `video/tests/test_conversion.py`
- `packages/msrtc_rans/`

Paper: T. Pärnamaa et al., [*MLVC: Multi-platform Learned Video Codec for
Real-World Deployment*](https://arxiv.org/abs/2606.28027), 2026.

### DCVC-UF at the pinned Microsoft/DCVC revision

- `README.md`, `training.md`, `test_video.py`, `test_compress_time.py`
- `src/models/image_model.py`, `video_model_ht.py`, `video_model_ld.py`,
  `common_model.py`, and `entropy_models.py`
- `src/utils/stream_helper.py`, `common.py`, `transforms.py`,
  `video_reader.py`, and `video_writer.py`
- `src/cpp/py_rans/`
- `src/layers/extensions/inference/bind.cpp`, `dmc_common.cpp`,
  `dmc_common.h`, `dmc_hts_proxy.cpp`, `dmc_hts_proxy.h`,
  `dmc_htl_proxy.cpp`, `dmc_htl_proxy.h`, `dmc_ld_proxy.cpp`,
  `dmc_ld_proxy.h`, `dmci_proxy.cpp`, and `dmci_proxy.h`
- `src/layers/extensions/inference/cutlass/` and `elementwise/`

Paper: J. Li et al., [*Ultra-Fast Neural Video
Compression*](https://arxiv.org/abs/2606.04410), CVPR 2026.
