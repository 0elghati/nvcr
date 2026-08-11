# Docker on Jetson Orin

For JetPack 6.1 / L4T 36.4. Run these commands on the Jetson.

## Install

```bash
docker pull omarelghati/nvcr:jetson-l4t36.4

docker run --rm --runtime=nvidia --gpus all --network=host \
  -v nvcr-engines:/opt/nvcr/engines \
  --entrypoint /opt/nvcr/bin/nvcr-artifacts \
  omarelghati/nvcr:jetson-l4t36.4 install \
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

Host directories map to `/input` and `/output` inside the container:

```bash
docker run --rm --runtime=nvidia --gpus all --network=host \
  --user "$(id -u):$(id -g)" \
  -v nvcr-engines:/opt/nvcr/engines:ro \
  -v "$NVCR_DATASET_DIR:/input:ro" \
  -v "$NVCR_OUTPUT_DIR:/output" \
  omarelghati/nvcr:jetson-l4t36.4 encode \
    -i /input/akiyo_qcif.yuv -o /output/akiyo_qcif.nvcr \
    -s 176x144 -r 29.97 --frames 4 --gop-size 2 --qp 32

docker run --rm --runtime=nvidia --gpus all --network=host \
  --user "$(id -u):$(id -g)" \
  -v nvcr-engines:/opt/nvcr/engines:ro \
  -v "$NVCR_OUTPUT_DIR:/output" \
  omarelghati/nvcr:jetson-l4t36.4 decode \
    -i /output/akiyo_qcif.nvcr \
    -o /output/akiyo_qcif_reconstructed.yuv
```

| Host storage | Container path | Mode |
|---|---|---|
| `nvcr-engines` volume | `/opt/nvcr/engines` | read-only when running |
| `$NVCR_DATASET_DIR` | `/input` | read-only |
| `$NVCR_OUTPUT_DIR` | `/output` | writable |

`cannot open input` means the container path is wrong; use
`/input/akiyo_qcif.yuv`, not its host path. If `--user` blocks GPU access, omit
it and correct output ownership afterward.

## Compose or local build

From a source checkout:

```bash
export NVCR_JETSON_IMAGE="omarelghati/nvcr:jetson-l4t36.4"
export NVCR_INPUT_DIR="$NVCR_DATASET_DIR"
export NVCR_OUTPUT_DIR
docker compose -f docker/compose.jetson.yaml run --rm engine-install \
  install --engine-root /opt/nvcr/engines --profile qcif

docker build --platform linux/arm64 --target runtime \
  -f docker/Dockerfile.jetson -t nvcr:orin-nano .
```

Jetson images and TensorRT engines must be built and run on the target Jetson;
do not reuse desktop engines.
