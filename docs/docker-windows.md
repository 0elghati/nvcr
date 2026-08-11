# Docker benchmark from a Windows host

This is a Windows-host procedure for running the Linux/amd64 NVCR container.
It does not provide native Windows support for the NVCR binary or TensorRT
engine plans. Run the commands from a WSL 2 Linux distribution with Docker
Desktop using the WSL 2 backend.

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

## PowerShell invocation

The benchmark launcher is a Bash script and should normally be run inside WSL.
If it must be started from PowerShell, delegate to WSL and use Linux paths:

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
