# Scope and support

NVCR separates an implemented runtime architecture from the narrower set of
codec, provider, and hardware combinations validated for use.

## Architecture and supported integration

| Layer | Current status |
|---|---|
| Runtime architecture | C++20 values, sessions, codec/provider contracts, registry, artifact resolution, bounded access units, CLI, and tests are implemented |
| Conformance fixtures | The deterministic codec and CPU provider validate contracts; they do not perform neural inference |
| Supported codec/provider integration | DCVC-RT with the TensorRT FP16 provider |
| Supported operating-system boundary | NVIDIA Linux, including Linux containers |
| Transitional work | Independent component-level TensorRT loading |
| Planned work | Additional codecs/providers, INT8, stable C ABI, FFmpeg, and standard-container mapping |

Checked-in target profiles describe requested hardware and artifact identities.
A profile or catalog entry alone is not evidence that every validation gate has
passed.

## Platform paths

| Environment | What NVCR provides | Important boundary |
|---|---|---|
| Linux x86_64 with NVIDIA GPU | Native package and CUDA/TensorRT container paths | GPU, driver, TensorRT, CUDA, and engine identity must be compatible |
| Windows with Docker Desktop/WSL2 | Linux container execution on the Windows host | This is container portability, not a native Windows implementation |
| Jetson Orin | Native AArch64 package and exact Jetson engine bundles | JetPack/L4T, CUDA, TensorRT, and target identity must match |
| CPU-only | Portable build and contract tests | No DCVC-RT/TensorRT inference |

Release availability can differ by platform. The
[installation guide](installation.md) identifies the recommended path for the
latest stable release.

## Artifact compatibility

The public artifact installer selects the best compatible published bundle in
this order:

1. exact target;
2. same compute capability on supported desktop systems;
3. Ampere-and-newer hardware compatibility on supported desktop systems.

Same compute capability means the same numeric SM version; SM 8.9 and SM 12.0
are different families. Broader hardware compatibility does not relax the
required TensorRT version or other catalog checks. Jetson uses exact bundles
only.

## What each validation level establishes

| Validation | What a pass demonstrates |
|---|---|
| Core contracts | Session lifecycle, registry, artifacts, access units, and malformed-input handling |
| Codec/provider integration | DCVC-RT/TensorRT construction, artifact checks, native entropy coding, and I/P encode/decode |
| Exact target | The recorded runtime and artifact bundle work on the named machine |
| Compatibility class | The declared broader engine class passed its correctness checks on the recorded targets |
| Container execution | The released Linux image runs in the recorded host environment |
| Controlled performance | Measurements follow the documented input, timing, repetition, and environment protocol |

A successful build, TensorRT deserialization, CPU test run, or matching target
profile proves only that narrower fact. Performance results are specific to
their recorded hardware and execution environment.

## Not currently claimed

NVCR does not currently claim:

- CPU neural-codec inference or native Windows/macOS support;
- multiple supported production codecs or providers;
- arbitrary TensorRT-plan portability or TensorRT version compatibility;
- byte compatibility with upstream Python DCVC-RT payloads;
- a frozen C++ ABI or stable C ABI;
- `NVAU` as an industry standard or `.nvcr` as a standard media container;
- FFmpeg integration; or
- redistribution rights for third-party checkpoints, model exports, engine
  plans, or datasets beyond their individual licences.

See [Identity and scope](identity-and-scope.md), [Compatibility](compatibility.md),
and the [Roadmap](../ROADMAP.md).
