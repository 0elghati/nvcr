# Hardware targets

Detect hardware identity on the machine that builds plans and on the machine
that runs the evaluation:

```bash
python3 scripts/nvcr_device.py > /tmp/nvcr-device.json
python3 -m json.tool /tmp/nvcr-device.json
```

Retain that JSON with the result. It records operating system, architecture,
GPU name, compute capability, multiprocessor count, CUDA runtime, and TensorRT
version. Also record driver, compiler, power mode, and clocks from the active
host or container.

## Registered exact targets

| Target profile | Expected family | Current requirement |
|---|---|---|
| `rtx3050-laptop-ubuntu2404` | Registered RTX 3050 Laptop identity | Confirm live detection and validate a complete exact engine set |
| `rtx4070-ubuntu2404` | SM 8.9 | Rebuild current exact engines from the current model profile |
| `rtx5060-laptop-ubuntu2404` | SM 12.0 Laptop identity | Keep the Laptop target distinct from other RTX 5060 devices |
| `orin-nano-l4t3647` | SM 8.7 Jetson Orin Nano identity | Validate exact target-local engines on the native device |

A target profile is part of the artifact identity, not evidence that a target
has passed. Its detected operating system, architecture, GPU identity, compute
capability, multiprocessor count, CUDA, and TensorRT constraints must agree with
the live machine.

When a source build receives `--target-profile`, the file must describe the
actual build host. The build command treats that file as an assertion and does
not emulate or independently prove the named hardware.

## Compatibility rules

- Exact artifacts are the controlled baseline.
- Same-compute-capability means the same numeric major and minor. SM 8.9 does
  not include SM 8.6 or SM 12.0.
- Ampere-and-newer is a desktop fallback for compute-capability major version
  8 or newer; builder support does not imply catalog availability.
- Every class requires the exact TensorRT `major.minor.patch` recorded in the
  manifest. A broader hardware class cannot resolve a TensorRT mismatch.
- Jetson/L4T uses exact target-local engines only.
- A container reproduces software userspace; it does not make a TensorRT plan
  portable.

## Required record

```text
target_id
gpu_name
compute_capability
multiprocessor_count
operating_system
cpu_architecture
driver_version
cuda_runtime_version
tensorrt_version
nvcr_commit
git_dirty
container_image and digest, or native build identifier
engine_bundle_path
engine_manifest_sha256
engine_bundle_sha256
model_profile_sha256
engine_profile_id
hardware_compatibility
```
