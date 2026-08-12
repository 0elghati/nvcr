# Troubleshooting

Start with the first failing boundary. A successful registry check does not
prove that the GPU, runtime, or engine is compatible.

Container examples use `NVCR_IMAGE`. Set it to the current image selected in
[Installation](installation.md) before running them.

## Docker cannot see the GPU

**Symptom.** The host or container cannot report the expected NVIDIA GPU.

**What it means.** GPU access is failing below the NVCR runtime boundary.

**Check.** Inspect the host and container separately:

```bash
nvidia-smi
docker info
docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  "$NVCR_IMAGE"
```

**Fix.** Repair the driver, Docker Engine/Desktop WSL2 backend, and NVIDIA
Container Toolkit. On Windows, also try `wsl.exe nvidia-smi`.

**Do not.** Debug codec, provider, or artifact behavior until the container can
name the expected GPU.

## `nvidia-smi` works but CUDA reports an unknown error

**Symptom.** GPU management queries work, but artifact detection or backend
initialization fails with a CUDA unknown error.

**What it means.** `nvidia-smi` uses the management library and does not create
a CUDA context. The container may have stale UVM device mappings.

**Check.** Compare the UVM device identity:

```bash
stat -c '%t:%T %n' /dev/nvidia-uvm /dev/nvidia-uvm-tools
docker run --rm --gpus all --entrypoint sh "$NVCR_IMAGE" -lc \
  "stat -c '%t:%T %n' /dev/nvidia-uvm /dev/nvidia-uvm-tools"
```

**Fix.** If the host and container major/minor values differ, repair or restart
Docker and the NVIDIA Container Toolkit after the driver/UVM module change. If
CUDA fails only after adding `--user`, use the image's default user or configure
the device permissions deliberately.

**Do not.** Treat an explicit UVM mapping as the permanent repair, or add an
arbitrary user mapping without checking NVIDIA capability-node permissions.

## The container prints help instead of running a command

**Symptom.** A container invocation prints the NVCR help page instead of the
requested operation.

**What it means.** The image entrypoint is already `/opt/nvcr/bin/nvcr`.

**Check.** Pass only a subcommand:

```bash
docker run --rm --gpus all "$NVCR_IMAGE" codec list
```

**Fix.** Override the entrypoint when invoking the artifact client:

```bash
docker run --rm \
  --entrypoint /opt/nvcr/bin/nvcr-artifacts \
  "$NVCR_IMAGE" --help
```

**Do not.** Append another `nvcr` after the image name.

## No compatible engine is found

**Symptom.** Installation reports no compatible catalog entry, or encode cannot
resolve a bundle.

**What it means.** No catalog bundle satisfies the detected target and
requested profile under the implemented compatibility rules.

**Check.** The artifact client prints the detected target before catalog
selection:

```bash
nvcr-artifacts install --profile qcif
```

From a source checkout, `python3 scripts/nvcr_device.py` prints the detection
record without installing an engine. The script does not accept a `--json`
option; JSON is its default output.

**Fix.** In containers mount the engine collection at `/opt/nvcr/engines`.
`NVCR_ENGINE_ROOT` names a collection with `bundles/` and `profiles/`;
`NVCR_ENGINE_DIR` names one direct bundle with `engine_manifest.json`.

A target JSON file is not proof that an engine exists. If the public catalog
has no compatible entry, follow
[DCVC-RT artifact preparation](dcvcrt-artifacts.md). The current public client
needs no authentication and does not send `GH_TOKEN` or `GITHUB_TOKEN`.

The installer automatically ranks `exact`, then
`same_compute_capability`, then `ampere_plus`. Same-compute-capability
requires the same numeric major and minor: SM 8.9 does not match SM 8.6 or
SM 12.0. Ampere-and-newer is a desktop fallback for compute-capability major
version 8 or newer. Jetson is exact-only.

The source builder can create either broader desktop class, but this does not
mean an Ampere-and-newer catalog bundle is available. Local compatibility builds
are an advanced source-checkout workflow, and an explicit target JSON must
describe the machine performing the TensorRT build.

**Do not.** Treat a target-profile file as proof that an engine exists, or use
a token to mask a target mismatch.

## Digest, manifest, target, or license rejection

**Symptom.** Installation or validation rejects a digest, manifest, target, or
license decision.

**What it means.** Integrity metadata, target identity, or licensing policy does
not match the requested bundle. This is an enforcement boundary.

**Check.** Inspect and validate the exact bundle:

```bash
nvcr-artifacts inspect /path/to/bundle --json
nvcr-artifacts validate /path/to/bundle --json
```

**Fix.** For a catalog download, repeat installation only after confirming the
selected target and profile. For a locally built bundle, regenerate the
manifest from the bound model, engine, and target profiles, then verify
redistribution terms.

**Do not.** Edit a digest, rename a plan from another target, or override a
`license_restricted` decision.

## TensorRT or CUDA version mismatch

**Symptom.** A plan fails to deserialize or execute, or runtime identity differs
from its manifest.

**What it means.** Plans are bound to recorded GPU compatibility, CUDA,
TensorRT, model/profile, and digest identity. NVCR requires the TensorRT
`major.minor.patch` recorded in the manifest for every hardware class.

**Check.** Record the effective runtime and selected bundle:

```bash
nvidia-smi
dpkg-query -W 'libnvinfer*' 2>/dev/null
nvcr-artifacts inspect /path/to/bundle --json
nvcr-artifacts validate /path/to/bundle --json
```

**Fix.** Install a bundle whose manifest and detected runtime match, or rebuild
the plans with the active TensorRT runtime. Deserialization alone is not a
complete execution check.

**Do not.** Reuse a plan from a different TensorRT version merely because the
GPU name or compute capability looks similar. Same-compute-capability and
Ampere-and-newer modes broaden the hardware boundary only; they cannot repair a
TensorRT mismatch.

## Registry compatibility says yes, but encode fails

**Symptom.** Registry compatibility passes, but encode fails during artifact or
backend initialization.

**What it means.** `nvcr compatibility check --codec dcvc-rt --provider
tensorrt` confirms only that the IDs are registered as a pair.

**Check.** Run encode with `--verbose` to see the bundle selected during actual
backend initialization.

**Fix.** Diagnose the selected bundle and runtime using the artifact and
TensorRT sections above.

**Do not.** Interpret registry compatibility as GPU, runtime, or target
compatibility.

## Input is rejected or output has the wrong size

**Symptom.** Encode rejects the raw input, or decoded output has an unexpected
byte count.

**What it means.** The CLI accepts headerless planar 8-bit YUV420. Width and
height must be even. Raw bytes equal `width × height × 3 / 2 × frames`.

**Check.** From a source checkout, generate and verify the reference dimensions:

```bash
python3 scripts/generate_sample_yuv.py \
  --output nvcr-example/input.yuv \
  --width 176 --height 144 --frames 4
test "$(stat -c %s nvcr-example/input.yuv)" -eq 152064
```

Installed and container users can use the standard-library generator in
[First run](first-run.md), which does not require a source checkout.

**Fix.** Pass the same dimensions and frame count to encode. Decode obtains
dimensions from the access units.

**Do not.** Supply odd dimensions or assume a headered media-file layout.

## Cannot open a mounted input or output

**Symptom.** NVCR cannot open an input path or create or truncate an output path
inside the container.

**What it means.** The bind source is missing, uses the wrong host path syntax,
is not shared with Docker, or has incompatible permissions.

**Check.** Use absolute host paths and verify them before starting the container.
Keep input and output directories separate when applying read-only permissions.

```bash
export NVCR_INPUT_DIR="$PWD/nvcr-example/input"
export NVCR_OUTPUT_DIR="$PWD/nvcr-example/output"
test -r "$NVCR_INPUT_DIR/input.yuv"
test -w "$NVCR_OUTPUT_DIR"
docker run --rm \
  --mount "type=bind,source=$NVCR_INPUT_DIR,target=/input,readonly" \
  --mount "type=bind,source=$NVCR_OUTPUT_DIR,target=/output" \
  --entrypoint sh "$NVCR_IMAGE" \
  -lc "test -r /input/input.yuv && test -w /output"
```

**Fix.** In PowerShell, resolve the directory with `Resolve-Path` before using it in
`--mount`, and confirm that Docker Desktop can share the containing drive.
Under WSL, run the Linux commands with WSL paths or run the PowerShell commands
with Windows paths; do not mix path syntaxes within one command.

Mount input data read-only and the output directory writable. Confirm that each
path exists on the host before diagnosing the NVCR CLI.

**Do not.** Mix Windows and WSL path syntax in one invocation or mount the
output directory read-only.

## Bind-mounted output ownership

**Symptom.** Output files are owned by root, or CUDA fails after adding
`--user`.

**What it means.** NVIDIA device or capability nodes are not always available to
an arbitrary numeric user.

**Check.** Inspect host output ownership and NVIDIA device permissions.

**Fix.** Create the output files on the host before starting the container:

```bash
export NVCR_WORK_DIR="$PWD/nvcr-example"
mkdir -p "$NVCR_WORK_DIR"
touch "$NVCR_WORK_DIR/output.nvcr" "$NVCR_WORK_DIR/reconstructed.yuv"
```

Root can truncate those existing files without changing their host ownership.
If files were already created by root, change ownership only on the exact
output files. Docker Desktop handles Windows bind-mount ownership through its
Linux VM.

**Do not.** Recursively change ownership of a broad workspace or force a user
mapping that cannot access the GPU capability nodes.

## Windows/WSL2 performance is lower than expected

**Symptom.** The Windows/WSL2 container completes the workflow but reports lower
throughput than another machine or native Linux.

**What it means.** The result includes environment-specific host, WSL, Docker,
storage, power, and GPU effects.

**Check.** Complete the functional workflow before interpreting performance. Record
the Windows version, WSL2 kernel, Docker Desktop version, GPU, driver, image
tag and digest, CUDA/TensorRT versions, storage path, and power mode.

**Fix.** Keep Windows/WSL2 measurements labelled as environment-specific
portability evidence.

**Do not.** Pool them with native-Linux measurements without a controlled method,
or present them as a universal GPU ranking.

## Jetson-specific failures

**Symptom.** A Jetson install cannot select or execute an engine, or performance
is unstable.

**What it means.** Jetson engines are exact-target artifacts. Platform identity,
runtime versions, power mode, clocks, and thermals are part of the result.

**Check.** Record the module, JetPack/L4T, CUDA, TensorRT, and power state before
selecting a bundle:

```bash
uname -m
cat /etc/nv_tegra_release
/usr/local/cuda/bin/nvcc --version
dpkg-query -W 'libnvinfer*' 2>/dev/null
sudo nvpmodel -q
```

**Fix.** Install a catalog bundle only when its target identity matches the device. If
no accepted bundle exists, generate the engine from the validated model on the
target itself. Do not use a desktop plan or infer portability from compute
capability alone.

For performance investigation, stabilize the selected power mode and clocks,
monitor temperature and throttling, and record those settings with the result.

**Do not.** Use a desktop TensorRT plan or generalize one module's result to all
Jetson targets.

## CPU build has no NVCR CLI

**Symptom.** A CPU-only build completes without the `nvcr` CLI.

**What it means.** This is expected with
`-DNVCR_ENABLE_TENSORRT=OFF`; the configuration builds portable libraries and
contract tests but skips the TensorRT CLI.

**Check.** List and run the configured tests:

```bash
ctest --test-dir build-cpu -N
ctest --test-dir build-cpu --output-on-failure
```

**Fix.** Use a TensorRT-enabled build or GPU package for neural inference.

**Do not.** Treat passing CPU checks as evidence of neural inference, GPU support,
or performance.

## Version identity is unclear

**Symptom.** The installed build cannot be identified with a command-line
version flag.

**What it means.** There is no `nvcr --version` or
`nvcr-artifacts --version` yet.

**Check.** Use the identity source that matches the installation:

- Native install: record the release archive and
  `PACKAGE-MANIFEST.sha256`.
- Source build: run `git describe --tags --always --dirty`.
- Container: record immutable tag, digest, and OCI labels:

```bash
docker image inspect "$NVCR_IMAGE" \
  --format '{{json .RepoDigests}} {{json .Config.Labels}}'
```

**Fix.** Record the archive manifest, source revision, or immutable image digest
with the run.

**Do not.** Infer a release solely from the repository CMake version.

## A result is incomplete

**Symptom.** A package has missing rows, failed cases, conflicting dirty state,
or insufficient environment or artifact identity.

**What it means.** The package cannot support a controlled comparison.

**Check.** Compare the package with
[Performance and benchmarking](performance.md) and the
[retained-results inventory](../results/README.md).

**Fix.** Record command, source state, target, runtime, artifact hashes, input,
timing boundary, repetitions, and failure, then rerun missing required cases.

**Do not.** Average away failures or missing rows, or relabel diagnostic or
historical packages as controlled baselines.

## Getting help

Use [SUPPORT.md](../SUPPORT.md). Include installation method, package or source
revision, platform, GPU, CUDA/TensorRT versions, exact command, bundle
manifest/digests, and non-sensitive error output. Never attach credentials,
restricted model assets, checkpoints, or plans without confirming their terms.
