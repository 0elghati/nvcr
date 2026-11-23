# Capability vocabulary v1

Status: design prototype for issue #279, 2026-11-23. Nothing in this document
changes production registration, artifact selection, or session creation.

NVCR currently answers “is this codec/provider pair compatible?” by checking
whether both IDs are registered. Real admission needs more information, but a
single capability bag would mix facts owned by codecs, providers, artifacts,
workloads, and retained evidence. This prototype keeps those five records
separate and evaluates them before a session executes.

The machine-readable files are:

- [JSON Schema](capability-vocabulary-v1.schema.json);
- [four evidence-backed examples](capability-vocabulary-v1.examples.json); and
- [`check_capability_admission.py`](../../scripts/check_capability_admission.py),
  the non-production admission prototype.

## The five owners

| Record | Owns | Does not own |
|---|---|---|
| Codec requirements | Stage direction, tensor contract, layout, shape behavior, state, required operations, usable memory domains, synchronization semantics, numerics | Provider names, GPU models, compiled plans |
| Provider capabilities | Facts known before a model loads: tensor types, layouts, shape modes, allocatable memory domains, completion semantics, implemented numerics profiles, loadable model formats | Whether one model's operators compile, exact shape coverage, conformance claims |
| Artifact capabilities | Codec/model/provider identity, stage set, format, validated operators, visible-size coverage, digest validation | General provider support or codec semantics |
| Workload requirements | Direction and visible dimensions for one requested job | Codec or provider capabilities |
| Conformance evidence | One tested codec/provider/artifact/workload cell and its `exact`, `native`, `unsupported`, or `untested` status | Support inferred from registration, GPU family, or another cell |

This split prevents two current mistakes. A registered provider is not assumed
to support every registered codec, and a provider's general ONNX or TensorRT
support is not treated as proof that a particular graph, plan, shape, or
numerics profile works.

## Representative pairs

The vocabulary comes from paths already implemented and measured in this
repository.

| Pair | Stage requirements represented | Artifact facts | Evidence status |
|---|---|---|---|
| DCVC-RT / TensorRT | Encode analysis and stateful reference stages; FP16 NCHW; bounded dynamic shapes; pinned-host or device buffers; completion objects; native numerics | Target-local plans, matching stage IDs, bounded 720p coverage, validated bindings/operators and digest | `native`: production path on its recorded target and bundle |
| DCVC-RT / ONNX Runtime CPU | Decoder prior/reference stages; INT32 NCHW portable programs; bounded shapes; host buffers; completion objects; `nvcr.fx.v1` | Portable normative programs and digests covering the selected profile | `exact`: CPU and TensorRT conformance evidence records bit-identical normative state |
| MLVC / ONNX Runtime CPU | Fixed-canvas encoder/decoder stages; FP16 and INT32 NCHW tensors; LTR sequence state; host buffers; completion objects; native numerics | Published bundle, fixed 640x368 canvas covering 640x360, model-part digests, validated ONNX operators | `native`: upstream-runtime parity on x86 for the published MLVC-S bundle |
| DCVC-RT / ONNX Runtime QNN with `nvcr.fx.v1` | Decoder prior stage requires the exact profile | QNN has neither an `nvcr.fx.v1` implementation nor a validated portable-program artifact | `unsupported`: reject before decode and name CPU/TensorRT exact alternatives |

The examples list representative stage classes rather than copying every
tensor binding from every manifest. Artifact validation remains responsible
for the complete immutable `TensorContract` and operator set before admission.

## Stage vocabulary

Each codec stage declares:

| Field | Meaning |
|---|---|
| `id` | Codec-owned semantic stage identity |
| `directions` | `encode`, `decode`, or both |
| `tensor_types` | Types that must cross the provider boundary |
| `layouts` | Logical tensor layouts required by this artifact family |
| `shape_mode` | `static` or `bounded_dynamic` |
| `state` | Human-readable codec state class, such as `stateless`, `sequence`, or `sequence-ltr` |
| `operators` | `artifact-validated` for provider graphs/plans or a named normative program contract |
| `memory_domains_any_of` | Domains the codec can use; the provider needs at least one |
| `synchronization` | Completion behavior required by the codec |
| `numerics_profile` | `native` or a named exact profile |

`state` and `operators` are recorded but not generically matched by the
prototype. State remains codec-owned, while operator support can only be
decided after validating the actual artifact. Turning either into provider
feature flags would repeat the speculative-schema problem this task avoids.

## Status and admission decisions

Conformance status and admission decision are separate:

| Status | Meaning | Decision when requirements also match |
|---|---|---|
| `exact` | The retained test proves the named normative profile for this cell | `admit` |
| `native` | The retained test proves the provider's native path for this exact cell; no cross-provider claim | `admit` |
| `unsupported` | A known requirement cannot be met or retained evidence records a clean incompatibility | `reject` |
| `untested` | Static requirements match, but no retained conformance result exists | `hold` |

`untested` must never collapse into `unsupported` or `native`. It says that the
project does not yet know. Likewise, `native` is scoped to its artifact and
target; it does not become `exact` because another provider supports the same
model.

The prototype checks, in order:

1. artifact codec and provider identity;
2. requested direction;
3. required stage presence;
4. tensor types, layouts, shape mode, synchronization, numerics, and one usable
   memory domain against pre-load provider capabilities;
5. artifact format, operator validation, digest validation, and visible-size
   coverage; then
6. retained conformance status.

It stops at the first failed requirement. Production ranking and fallback are
outside this prototype.

## Structured incompatibility

The known QNN mismatch produces this result before session execution:

```json
{
  "decision": "reject",
  "status": "unsupported",
  "incompatibility": {
    "owner": "provider",
    "requirement": "numerics_profile",
    "required": "nvcr.fx.v1",
    "available": ["native"],
    "stage_id": "i_prior"
  },
  "alternatives": [
    {"provider_id": "onnxruntime-cpu", "status": "exact"},
    {"provider_id": "tensorrt", "status": "exact", "requires": "matching fx bundle"}
  ]
}
```

The message names the owner and failed requirement instead of reporting only
that a provider is incompatible. Alternatives are data supplied by retained
evidence, not an automatic fallback.

Run all examples:

```bash
python3 scripts/check_capability_admission.py \
  docs/spec/capability-vocabulary-v1.examples.json
```

Run one case with `--case <id>`. Exit code 0 means every result matched the
example's expected decision, 1 means a decision changed, and 2 means the input
document was malformed.

## Deliberate omissions

- No provider-specific option vocabulary. QNN performance/finalization options
  stay in stage provider options until more than one codec needs a shared rule.
- No performance or preference score. Admission answers whether a path may run;
  artifact ranking belongs to issue #271.
- No automatic acquisition or fallback. Registry/cache work owns those actions.
- No generic operator list advertised by providers. Actual graph/plan loading
  and artifact validation own operator coverage.
- No production C++ types. The examples must settle before public API or
  manifest fields are added.

## Questions tied to missing evidence

- Add an `nhwc` layout rule only after a real MLVC NHWC bundle passes a provider
  gate; current bundles admitted here are NCHW.
- Add input-group/output-cardinality fields when DCVC-UF becomes a selected
  admission case; its eight-frame HT-S chunks are the concrete missing example.
- Add provider operator families only if the DCVC-UF QNN single-stage probe in
  issue #211 shows a useful fact that can be known before loading the graph.
- Add memory limits when one representative artifact is rejected or selected
  based on measured provider memory, not from a theoretical maximum.
- Expand conformance scope when retained evidence proves which target fields
  distinguish two otherwise identical cells.

These questions should remain open until their named example exists.
