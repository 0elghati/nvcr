# Docker on Windows

NVCR does not provide a native Windows executable or native Windows TensorRT
path. The supported Windows-host procedure runs the Linux/amd64 image through
Docker Desktop's WSL 2 backend:

```text
Windows host -> Docker Desktop / WSL 2 -> Linux NVCR image -> NVIDIA GPU
```

The complete PowerShell workflow is in
[First run](first-run.md#windows-11-with-docker-desktop-and-an-nvidia-gpu).

## Requirements

- Windows 11 with WSL 2 enabled.
- A current NVIDIA Windows driver with CUDA-on-WSL support.
- Docker Desktop configured for the WSL 2 engine.
- GPU support enabled in Docker Desktop.
- Python 3 through the `py` launcher for the first-run input generator.

Check the environment from PowerShell:

```powershell
wsl --status
wsl --update
docker version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

The final command must list the intended GPU. The Windows driver provides the
WSL GPU interface; do not install a second Linux display driver inside WSL.

## Image identity

Select the published desktop release version and resolve its immutable digest:

```powershell
$NvcrVersion = "<version>"
$NvcrImage = "omarelghati/nvcr:${NvcrVersion}-amd64-cuda12.8-trt10.9"
docker pull $NvcrImage
$NvcrImageRef = docker image inspect $NvcrImage --format '{{ index .RepoDigests 0 }}'
docker image inspect $NvcrImage --format 'version={{ index .Config.Labels "org.opencontainers.image.version" }} revision={{ index .Config.Labels "org.opencontainers.image.revision" }} digest={{ index .RepoDigests 0 }}'
```

Record `$NvcrImageRef` with the run and use it for subsequent commands.
`latest-amd64-cuda12.8-trt10.9` is a rolling convenience alias. There is no
architecture-neutral NVCR `latest` image.

## Storage and paths

PowerShell can drive Docker Desktop directly. Use an absolute Windows
directory in a `--mount` specification and confirm Docker Desktop can access
the containing drive.

Alternatively, work entirely inside a WSL distribution and follow the Linux
commands with Linux paths. Do not combine Windows and WSL path syntax in one
command.

For large inputs or performance work, the WSL filesystem is generally more
appropriate than `/mnt/c`. The filesystem and storage path remain part of
the measured environment.

Use the same engine volume name as Linux:

```powershell
docker volume create nvcr-engines
```

The public artifact client automatically ranks exact, matching-SM, and
applicable Ampere-plus desktop bundles. If none matches the GPU and the image's
CUDA/TensorRT runtime, the failure is a compatibility result, not an
authentication prompt. The current public client does not consume a GitHub
token.

## Runtime command form

The image entrypoint is already `nvcr`:

```powershell
docker run --rm --gpus all $NvcrImageRef codec list
docker run --rm --gpus all $NvcrImageRef provider list
```

Override the entrypoint only for artifact operations:

```powershell
docker run --rm --gpus all --volume nvcr-engines:/opt/nvcr/engines --entrypoint /opt/nvcr/bin/nvcr-artifacts $NvcrImageRef install --profile qcif --engine-root /opt/nvcr/engines
```

## Interpretation of Windows results

A successful run establishes containerized Linux execution on the recorded
Windows/Docker Desktop/WSL 2 host. It does not establish a native Windows ABI
or make TensorRT plans portable across GPUs.

Performance reports should retain:

- Windows edition and build;
- WSL kernel and Docker Desktop versions;
- GPU and Windows driver;
- image version, revision, and digest;
- CUDA and TensorRT versions;
- storage path and filesystem;
- power and clock state; and
- timing boundary.

Do not pool Windows/WSL 2 and native-Linux measurements unless the comparison
uses an explicit controlled methodology.

## Next step

Run the canonical
[Windows functional validation](first-run.md#windows-11-with-docker-desktop-and-an-nvidia-gpu).
For path, GPU, or performance diagnosis, see
[Troubleshooting](troubleshooting.md).
