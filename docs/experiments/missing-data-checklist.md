# Evaluation readiness checklist

Complete this list before treating a controlled evaluation as complete.

## Inputs and reference

- [ ] Every raw YUV input has width, height, frame rate, frame count, pixel
  format, SHA-256, and redistribution status.
- [ ] Local-only data remains outside Git.
- [ ] The bound DCVC-RT source revision and hash-verified image and video
  checkpoints are available.
- [ ] The Python reference environment contains PyTorch, ONNX, and ONNXScript.
- [ ] A reference wrapper can emit `nvcr.softwarex.python-reference.v1` rows,
  or equivalent precomputed rows are available.

## Build and artifacts

- [ ] TensorRT `trtexec` is identified on each target.
- [ ] Model and engine output directories are empty or deliberately versioned.
- [ ] The selected target JSON matches the live engine-build host.
- [ ] Every requested engine profile is present and validated.
- [ ] Engine, manifest, model-profile, target-profile, and engine-profile
  digests are recorded.
- [ ] The runtime TensorRT `major.minor.patch` exactly matches every selected
  engine manifest.
- [ ] The image tag and digest or native build identifier is recorded.

## Hardware

- [ ] RTX 3050 target access, live target confirmation, and exact engine set.
- [ ] RTX 4070 target access and exact engine set.
- [ ] RTX 5060 target access, with actual GPU name and numeric SM recorded.
- [ ] Jetson Orin Nano access and exact target-local engine set.
- [ ] Driver, CUDA, TensorRT, OS, architecture, compiler, clocks, and power mode
  are recorded.
- [ ] Compatibility class is recorded. Same-compute-capability matches both SM
  components; broader classes are desktop-only and do not relax TensorRT.

## Evaluation choices

- [ ] Selected profiles, QP values, GOP policy, and frame counts are fixed.
- [ ] RTX 4070 exact is included for the required Python comparison.
- [ ] An exact baseline row exists for every broader-class case.
- [ ] Warm-up and measured repetition counts are fixed.
- [ ] Timing and byte boundaries are fixed and documented.
- [ ] The complete command includes `--profile` with a fixed profile-repetition
  count; primary timing excludes verbose, quality, and memory instrumentation.
- [ ] External storage is selected for raw outputs when they must be retained.

## Remaining controlled-evaluation gates

- [ ] Confirm the RTX 3050 Laptop profile against live target detection.
- [ ] Rebuild every selected engine bundle from the current model-profile
  digest.
- [ ] Produce matching `nvcr.softwarex.python-reference.v1` rows for the
  required RTX 4070 cases.
- [ ] Provide a Jetson GPU-memory sampler when `nvidia-smi` cannot populate the
  required metric.
- [ ] Validate each same-compute-capability or Ampere-and-newer case against an
  exact baseline before retaining it as controlled evidence.
