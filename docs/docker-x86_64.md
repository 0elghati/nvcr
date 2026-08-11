# Docker on x86_64 NVIDIA GPUs

For Linux x86_64 with Docker and NVIDIA Container Toolkit.

## Install

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

docker pull omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9

docker run --rm --gpus all \
  -v nvcr-engines:/opt/nvcr/engines \
  --entrypoint /opt/nvcr/bin/nvcr-artifacts \
  omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9 install \
    --engine-root /opt/nvcr/engines --profile qcif
```

Docker creates `nvcr-engines` automatically. The image already uses
`/opt/nvcr/engines` by default.

## Prepare Akiyo QCIF

NVCR expects headerless 8-bit YUV420. Convert Xiph's 176x144, 300-frame Akiyo
Y4M file:

```bash
export NVCR_DATASET_DIR="/absolute/path/to/yuv-dataset"
export NVCR_OUTPUT_DIR="/absolute/path/to/nvcr-output"
mkdir -p "$NVCR_DATASET_DIR" "$NVCR_OUTPUT_DIR"

curl -fL https://media.xiph.org/video/derf/y4m/akiyo_qcif.y4m \
  -o "$NVCR_DATASET_DIR/akiyo_qcif.y4m"
ffmpeg -i "$NVCR_DATASET_DIR/akiyo_qcif.y4m" \
  -pix_fmt yuv420p -f rawvideo \
  "$NVCR_DATASET_DIR/akiyo_qcif.yuv"
```

Other inputs: [Xiph Derf](https://media.xiph.org/video/derf/) and the
[UVG HD YUV420 dataset](https://tie-ultravideo.rd.tuni.fi/dataset.html). Check
each dataset's license; UVG is CC BY-NC.

## Encode and decode

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v nvcr-engines:/opt/nvcr/engines:ro \
  -v "$NVCR_DATASET_DIR:/input:ro" \
  -v "$NVCR_OUTPUT_DIR:/output" \
  omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9 encode \
    -i /input/akiyo_qcif.yuv -o /output/akiyo_qcif.nvcr \
    -s 176x144 -r 29.97 --frames 4 --gop-size 2 --qp 32

docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v nvcr-engines:/opt/nvcr/engines:ro \
  -v "$NVCR_OUTPUT_DIR:/output" \
  omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9 decode \
    -i /output/akiyo_qcif.nvcr \
    -o /output/akiyo_qcif_reconstructed.yuv
```

| Host storage | Container path | Mode |
|---|---|---|
| `nvcr-engines` volume | `/opt/nvcr/engines` | read-only when running |
| `$NVCR_DATASET_DIR` | `/input` | read-only |
| `$NVCR_OUTPUT_DIR` | `/output` | writable |

Use `/input/akiyo_qcif.yuv` inside the container. If CUDA fails after the
`nvidia-smi` check, add:

```bash
--device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools
```

## Compose or local build

From a source checkout:

```bash
export NVCR_X86_64_IMAGE="omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9"
export NVCR_INPUT_DIR="$NVCR_DATASET_DIR"
export NVCR_OUTPUT_DIR
docker compose -f docker/compose.x86_64.yaml run --rm engine-install \
  install --engine-root /opt/nvcr/engines --profile qcif

docker build --platform linux/amd64 --target runtime \
  -f docker/Dockerfile.x86_64 -t nvcr:desktop .
```

Build engines on the GPU and TensorRT runtime that will execute them; do not
reuse Jetson or another GPU's plans.

## Direct Docker benchmark

After the engine volume contains the matching target-local bundles, run the
same diagnostic matrix used by the native Linux workflow:

```bash
scripts/benchmark_docker.sh \
  --image omarelghati/nvcr:0.19.1-amd64-cuda12.8-trt10.9 \
  --input-dir "$NVCR_DATASET_DIR" \
  --engine-volume nvcr-engines \
  --results-dir "$NVCR_OUTPUT_DIR/benchmark-docker" \
  --hardware rtx4070-docker \
  -- \
  --resolutions "qcif 720p" --frames 300 --qp 32 \
  --gops "1 299" --repetitions 3 --warmup-frames 10
```

This command pulls the image before running `docker run`; it does not use
Docker Compose. Add `--install-profiles "qcif 720p"` when the named volume is
empty and the image can access the private engine-assets release through
`GH_TOKEN`. The result directory records the pulled image identity and process
wall-time throughput for comparison with native Linux results. The launcher
uses container user `0:0` by default because some NVIDIA Container Toolkit
setups reject CUDA device access for UID 1000; override with `--user UID:GID`
when non-root GPU access is validated on the target.
