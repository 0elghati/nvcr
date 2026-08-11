# Hardware targets

Hardware identity must be detected on the machine that runs the experiment. Do not fill in an expected GPU or SM value by hand.

Run:

```bash
python3 scripts/nvcr_device.py > /tmp/nvcr-device.json
cat /tmp/nvcr-device.json
```

Record the JSON with the run metadata. It includes OS, architecture, GPU name, compute capability, multiprocessor count, CUDA runtime, and TensorRT version. Add the driver version, compiler, power mode, and clocks from the host/container environment.

## Exact targets

| Requested target | Actual target/profile today | Expected family | State |
|---|---|---|---|
| RTX 3050 exact | `rtx3050-laptop-ubuntu2404` | Detect from the registered Laptop target | Profile exists; a clean exact artifact/result set is still required |
| RTX 4070 exact | `rtx4070-ubuntu2404` | SM 8.9 | Current engines must be rebuilt from the current model profile |
| RTX 5060 exact | `rtx5060-laptop-ubuntu2404` | Detect exact Blackwell SM | Profile exists for the Laptop GPU; do not rename it to a generic RTX 5060 target |
| Jetson Orin Nano exact | `orin-nano-l4t3647` | SM 8.7 | Profile exists; a complete warning-free six-profile gate is required |

The target profile is part of the experiment identity. It binds the OS/platform, GPU family, CUDA, TensorRT, compiler expectations, and FP16 precision. The detected values and the profile must agree before an exact run is accepted.

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

## Practical rules

- Exact artifacts are the primary publication evidence.
- RTX 3050 cannot be claimed until the registered profile matches live
  detection and a complete engine set has passed validation.
- RTX 4070 exact cannot be claimed until the complete engine set is rebuilt
  from the current model profile and passes the publication protocol.
- The RTX 5060 profile currently names the Laptop GPU. Detect the real SM on the target and add a separate profile if a different RTX 5060 device is evaluated.
- Jetson/L4T uses exact target-local engines only.
- Do not copy desktop plans to Jetson or Jetson plans to desktop.
- A container reproduces software userspace; it does not make a TensorRT plan portable.
