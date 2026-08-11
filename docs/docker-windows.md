# Docker benchmark from a Windows host

This document covers two ways to run the Linux/amd64 NVCR container from a
Windows host. It does not provide native Windows support for the NVCR binary
or TensorRT engine plans.

- **WSL 2 + Bash** (below): run `scripts/benchmark_docker.sh` from a WSL 2
  Linux distribution with Docker Desktop's WSL 2 backend. Recommended when
  the checkout lives in WSL.
- **Native PowerShell** ([jump to section](#native-powershell-launcher)):
  run `scripts/benchmark_docker.ps1` directly from Windows PowerShell when
  the checkout is not in WSL.

## Host prerequisites

Install or update the following on Windows:

1. Docker Desktop with the WSL 2 backend enabled and WSL integration enabled
   for the Linux distribution used for the checkout.
2. A current NVIDIA Windows driver that supports CUDA in WSL 2.
3. The current WSL kernel:

   ~~~powershell
   wsl --update
   wsl --status
   ~~~

GPU support in Docker Desktop for Windows requires the WSL 2 backend. The
Windows NVIDIA driver supplies the WSL GPU interface; do not install a Linux
NVIDIA display driver inside WSL.

Open the Ubuntu/WSL terminal for the remaining commands. Keep the NVCR
checkout, input YUV files, and result directory in the WSL Linux filesystem
when possible. Accessing large benchmark files through /mnt/c works, but is
usually slower than using paths below $HOME.

Confirm that Docker Desktop can see the GPU before running NVCR:

~~~bash
docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  nvidia/cuda:12.8.1-base-ubuntu24.04
~~~

If this fails, fix Docker Desktop/WSL GPU visibility first. Do not install a
second Linux NVIDIA driver in WSL as a workaround.

## Run the benchmark

From the NVCR checkout in WSL:

~~~bash
cd ~/src/NVCR
export NVCR_DATASET_DIR="$HOME/nvcr-data"
mkdir -p "$HOME/nvcr-results"

docker volume create nvcr-engines

scripts/benchmark_docker.sh \
  --image omarelghati/nvcr:0.19.1-amd64-cuda12.8-trt10.9 \
  --input-dir "$NVCR_DATASET_DIR" \
  --engine-volume nvcr-engines \
  --install-profiles "qcif 720p" \
  --results-dir "$HOME/nvcr-results/windows-wsl2-rtx4070" \
  --hardware windows-wsl2-rtx4070 \
  -- \
  --resolutions "qcif 720p" \
  --frames 300 \
  --qp 32 \
  --gops "1 299" \
  --repetitions 3 \
  --warmup-frames 10
~~~

The launcher pulls the immutable image, uses docker run --gpus all, and
records the resolved image identity. If nvcr-engines is already populated
with matching bundles, omit --install-profiles. Installing catalog assets
may require GH_TOKEN in the WSL environment.

The default input layout is the same as the Linux diagnostic runner, for
example:

~~~text
$HOME/nvcr-data/qcif/akiyo_qcif.yuv
$HOME/nvcr-data/720p/FourPeople_1280x720_60.yuv
~~~

Custom input paths passed after the -- separator are container paths, not
Windows paths. For example, use --qcif-input /input/my-qcif.yuv; the host
file must be under the directory supplied to --input-dir.

Results contain results.jsonl, results.csv, and summary.md. Compare the
Windows/WSL2 run with Linux using process_throughput_fps, and retain the host
label because Docker Desktop and WSL2 are part of the measured execution
environment. The launcher runs as container user 0:0 by default for NVIDIA
device compatibility; only use --user UID:GID after validating non-root GPU
access on that Windows/WSL2 installation.

## Native PowerShell launcher

Use `scripts/benchmark_docker.ps1` to drive Docker Desktop directly from
Windows PowerShell, without WSL or a dev container. This is the option for a
plain Windows checkout with Docker Desktop's default (non-WSL2-integrated)
engine, or any setup where the repository is not mounted inside WSL.

~~~powershell
& .\scripts\benchmark_docker.ps1 `
  -Image omarelghati/nvcr:0.19.1-amd64-cuda12.8-trt10.9 `
  -InputDir C:\nvcr-data `
  -EngineVolume nvcr-engines `
  -ResultsDir C:\nvcr-results\windows-rtx4070 `
  -Hardware windows-rtx4070 `
  -MatrixArgs @(
    '--resolutions', 'qcif 720p',
    '--frames', '300',
    '--qp', '32',
    '--gops', '1 299',
    '--repetitions', '3',
    '--warmup-frames', '10'
  )
~~~

### Parameters

| Parameter | Purpose |
|---|---|
| `-Image` | Runtime image to pull and run |
| `-InputDir` | Host folder with benchmark YUV inputs |
| `-EngineVolume` | Named Docker volume or a host engine folder |
| `-ResultsDir` | Host folder for `results.jsonl`, `results.csv`, `summary.md` |
| `-OutputDir` | Host folder for temporary encoded streams (default: `ResultsDir\streams`) |
| `-Hardware` | Label recorded in result rows |
| `-InstallProfiles` | Install engine profiles first, e.g. `"qcif 720p"` or `all` |
| `-HostRepoDir` | Override the checkout path if auto-detection fails |
| `-SkipPull` | Reuse a local image instead of pulling |
| `-MatrixArgs` | Options forwarded to `benchmark_resolution_matrix.sh` |

Notes:

- Paths for `-InputDir`, `-EngineVolume`, `-ResultsDir`, and `-OutputDir` are
  plain Windows paths; Docker translates them to daemon-visible mounts.
- `-EngineVolume` accepts a nested target-profile folder (for example
  `dcvcrt-cvpr2025\rtx5060-laptop-ubuntu2404\`) and detects it automatically.
- Set `GH_TOKEN` in the environment before running if `-InstallProfiles`
  needs to fetch catalog assets.
- Requires `--gpus all` support in Docker Desktop; confirm GPU visibility
  first with the `nvidia-smi` check above.

## Delegate to WSL from PowerShell

When the checkout lives inside WSL, start the Bash launcher from PowerShell
by delegating to WSL instead:

~~~powershell
wsl.exe bash -lc 'cd ~/src/NVCR && ./scripts/benchmark_docker.sh --image omarelghati/nvcr:0.19.1-amd64-cuda12.8-trt10.9 --input-dir "$HOME/nvcr-data" --engine-volume nvcr-engines --results-dir "$HOME/nvcr-results/windows-wsl2-rtx4070" --hardware windows-wsl2-rtx4070 -- --resolutions "qcif 720p" --frames 300 --qp 32 --gops "1 299" --repetitions 3'
~~~

For a Windows directory such as C:\nvcr-data, the corresponding WSL path
is usually /mnt/c/nvcr-data. Prefer copying the dataset into the WSL Linux
filesystem for long runs.

## References

- [Docker Desktop GPU support on Windows](https://docs.docker.com/desktop/features/gpu/)
- [Docker Desktop WSL 2 backend](https://docs.docker.com/desktop/features/wsl/)
- [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
