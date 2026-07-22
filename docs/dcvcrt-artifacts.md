# DCVC-RT Artifact Pipeline

NVCR does not load PyTorch checkpoints at runtime. The native backend needs a
directory containing TensorRT plans plus copied entropy, quantization, and
manifest assets.

Use this page when:

1. You are preparing release engine bundles.
2. Your architecture has no prebuilt engine bundle and you must build locally.

The artifact flow is:

```text
Microsoft DCVC-RT checkout + cvpr2025 .pth.tar checkpoints
        |
        | scripts/export_dcvcrt_onnx.py
        | scripts/export_dcvcrt_p_onnx.py
        v
build/models/dcvcrt/*.onnx + *.bin + *.json
        |
        | scripts/build_dcvcrt_tensorrt.sh
        v
build/engines/dcvcrt-1080p/*.plan + *.bin + *.json
```

ONNX files and runtime assets are portable. TensorRT `.plan` files are not:
build them on the final target machine with the final TensorRT runtime. If a
copied engine fails with `Serialization assertion plan->header.pad ==
expectedPlatformTag failed`, rebuild the engine directory on that machine.
NVCR also requires `engine_manifest.json`, generated beside `engine.sha256`, and
checks it against the selected `--device-id` before frame processing.

## Inputs

Local upstream checkout:

```bash
/home/oelghati/nvcr/assets
```

Remote upstream checkout:

```bash
https://github.com/microsoft/DCVC.git
```

Download the Microsoft DCVC-RT pretrained models from the upstream README and put
the two files here:

```text
/home/oelghati/nvcr/assets/checkpoints/cvpr2025_image.pth.tar
/home/oelghati/nvcr/assets/checkpoints/cvpr2025_video.pth.tar
```

Primary checkpoint page:

```text
https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8
```

Backup checkpoint page:

```text
https://1drv.ms/f/c/2866592d5c55df8c/EozfVVwtWWYggCitBAAAAAABbT4z2Z10fMXISnan72UtSA?e=BID7DA
```

Known checkpoint hashes used by the current NVCR manifests:

```text
555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9  cvpr2025_image.pth.tar
b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6  cvpr2025_video.pth.tar
```

## One Command

After the Python export environment contains PyTorch, ONNX, and ONNXScript, run:

```bash
python -c "import torch, onnx, onnxscript"
```

```bash
cd /path/to/nvcr
export PATH="$PWD/install-<platform>/bin:$PATH"  # from scripts/install.sh

./scripts/prepare_dcvcrt_artifacts.sh \
  --dcvcrt-root ./assets \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --skip-smoke
```

If `trtexec` is not on PATH, pass it explicitly (common location shown below):

```bash
./scripts/prepare_dcvcrt_artifacts.sh \
  --dcvcrt-root ./assets \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --skip-smoke
```

The script clones or updates the public DCVC repository, verifies that the two
checkpoints exist, exports ONNX/runtime assets, and builds target-local
TensorRT engines. During engine generation, it records the build GPU name,
compute capability, multiprocessor count, TensorRT version, precision,
optimization point, workspace, and builder optimization level in
`engine_manifest.json`.

`--device-id`, `--workspace-mib`, and `--builder-optimization-level` are
auto-tuned from [scripts/detect_platform.sh](../scripts/detect_platform.sh)
unless explicitly overridden (or auto-tuning is disabled with
`--no-auto-tune`). On an 8 GB Jetson Orin Nano this resolves to workspace
512 MiB / builder optimization level 1 for 1080p engine generation, matching
values previously found and pinned by hand; on hosts with more free memory
(e.g. discrete RTX/datacenter GPUs) it resolves to higher values
automatically.

If ONNX/runtime assets were exported on another machine and copied to
`build/models/dcvcrt`, skip the Python step and build only target-local engines:

```bash
./scripts/prepare_dcvcrt_artifacts.sh \
  --skip-clone \
  --skip-export \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --skip-smoke
```

## No prebuilt engine bundle for this architecture

If your release does not provide an engine bundle for this machine, build local
engines and install them under your NVCR prefix:

```bash
export NVCR_PREFIX="$HOME/.local/nvcr"
mkdir -p "$NVCR_PREFIX/engines"

./scripts/prepare_dcvcrt_artifacts.sh \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --engines "$NVCR_PREFIX/engines/dcvcrt" \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --python /path/to/DCVC-RT/src/venv/bin/python \
  --skip-clone --skip-smoke
```

Then run NVCR with:

```bash
nvcr encode -i input.yuv -o output.nvcr -s 176x144 --frames 1 --engine-dir "$NVCR_PREFIX/engines/dcvcrt"
```

## Direct NVCR Call

Use the generated engine directory directly. This example uses the Orin Nano
install verified during bring-up; substitute your own install prefix, input
file, and engine directory:

```bash
cd /path/to/nvcr
export PATH="$PWD/install-<platform>/bin:$PATH"

nvcr encode \
  -i /path/to/dataset/BasketballDrive_1920x1080_50.yuv \
  -o /tmp/basketball_hd.nvcr \
  -s 1920x1080 -r 50 --frames 97 --qp 32 \
  --engine-dir build/engines/dcvcrt
```

Decode with the same engine directory:

```bash
nvcr decode \
  -i /tmp/basketball_hd.nvcr \
  -o /tmp/basketball_hd.decoded.yuv \
  --engine-dir build/engines/dcvcrt
```
