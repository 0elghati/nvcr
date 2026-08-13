# Where NVCR runs today

NVCR currently runs the DCVC-RT codec integration through TensorRT FP16 on
NVIDIA Linux. The table below explains the available paths and what each one is
for.

| Environment | Recommended path | What to know |
|---|---|---|
| Linux x86_64 with an NVIDIA GPU | Native package or Docker | The GPU, CUDA, TensorRT, and selected engine must match. |
| Windows with Docker Desktop/WSL 2 | Linux Docker container | NVCR runs in the Linux container; this is not a native Windows build. |
| Jetson Orin | Native AArch64 package | JetPack/L4T, CUDA, TensorRT, and the Jetson engine must match. |
| CPU-only machine | Build and contract tests | This checks runtime behaviour; it does not run neural inference. |

## How engine selection works

An engine is the TensorRT-optimized model bundle used by the execution
provider. The installer detects the machine and chooses the best compatible
published bundle:

1. an exact target match;
2. the same GPU compute capability on supported desktop systems; or
3. an Ampere-and-newer compatibility bundle on supported desktop systems.

A broader engine still needs the matching TensorRT runtime. Jetson uses exact
engine bundles only. If the installer cannot find a match, see
[Troubleshooting](troubleshooting.md).

## What a successful check means

A check answers a specific question:

| Check | What it tells you |
|---|---|
| Core tests | The runtime, parser, and session contracts behave as expected. |
| Encode/decode round trip | NVCR, its engine, and the selected codec integration work together on that machine. |
| Target result | The recorded configuration ran on the named target. |
| Container run | The released Linux image ran on the recorded host setup. |
| Performance run | The recorded throughput and quality values came from the documented test setup. |

See [First run](first-run.md) for the installation procedures and
[Compatibility](compatibility.md) for target-specific requirements.
