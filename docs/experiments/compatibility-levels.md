# Engine hardware compatibility

Hardware compatibility describes the GPU boundary encoded in a TensorRT engine
bundle. It is independent of the container image and does not relax the
bundle's model, digest, CUDA, or TensorRT checks.

## Classes

| Manifest value | Hardware boundary | Intended use |
|---|---|---|
| `exact` | Named target profile, including GPU identity, numeric compute capability, and multiprocessor count | Default and controlled baseline |
| `same_compute_capability` | Desktop x86_64 GPUs with the same compute-capability major and minor | Evaluated same-SM fallback |
| `ampere_plus` | Desktop x86_64 GPUs with compute-capability major version 8 or newer | Broad evaluated fallback |

Compute capability is numeric. SM 8.9 means major 8, minor 9; it does not
match SM 8.6 or SM 12.0. The same-compute class is therefore narrower than a
GPU product family or marketing generation.

The Ampere-and-newer class is not an optimized universal plan. The source
builder supports this mode, but catalog installation can select it only when a
corresponding catalog entry exists. Do not infer download availability from
the build option or naming convention.

Jetson and other AArch64 targets are exact-only. Do not build or select
same-compute-capability or Ampere-and-newer bundles for Jetson.

## Runtime constraints common to every class

NVCR requires the TensorRT `major.minor.patch` version recorded in the engine
manifest. A broader hardware class cannot make an engine valid under another
TensorRT version.

On desktop, the active CUDA runtime must have the same major version and must
not be older than the engine's recorded runtime. Jetson requires the recorded
CUDA runtime exactly. Model, engine-profile, target-profile, manifest, and file
digests remain enforced for every class.

## Catalog selection

`nvcr-artifacts install` detects the target and chooses the strongest
compatible catalog candidate for each requested profile:

```text
exact -> same_compute_capability -> ampere_plus
```

This ordering is automatic. The installer does not expose a switch for forcing
a weaker catalog class, and it never starts a local build when no candidate
matches.

## Controlled evaluation

The evaluation driver records the intended class explicitly:

```text
--compatibility-class exact
--compatibility-class same_compute_capability
--compatibility-class ampere_plus
```

The requested value must match every selected engine manifest. A
same-compute-capability result requires matching numeric SM, and both broader
classes are rejected on AArch64/Jetson.

Exact results are the baseline. Each broader-class case requires a matching
exact row through `--exact-baseline-jsonl`, plus correctness and
performance-ratio validation on every target for which the result is retained.

## Naming

Exact bundles retain their registered target ID. Broader desktop bundles use
names that state the portability boundary, such as
`linux-amd64-sm89` or `linux-amd64-ampere-plus`. A name is metadata, not
evidence that the bundle passed runtime validation on a particular GPU.
