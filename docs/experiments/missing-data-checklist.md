# Missing-data checklist

Complete this list before calling a SoftwareX run complete.

## Inputs and permissions

- [ ] Raw YUV input paths are available.
- [ ] Each sequence has width, height, frame rate, frame count, pixel format, and SHA-256.
- [ ] Sequence redistribution status is known; local-only data stays outside Git.
- [ ] Pinned DCVC-RT checkout is available or cloning is permitted.
- [ ] `cvpr2025_image.pth.tar` is available and hash-verified.
- [ ] `cvpr2025_video.pth.tar` is available and hash-verified.
- [ ] Python reference environment has PyTorch, ONNX, and ONNXScript.
- [ ] A pinned wrapper can emit `nvcr.softwarex.python-reference.v1` rows, or
  equivalent precomputed rows are available.

## Build and artifacts

- [ ] TensorRT `trtexec` path is known on each target.
- [ ] Model output directory is empty or explicitly versioned.
- [ ] Engine output directory is target-local and writable.
- [ ] All requested profiles are present and validated.
- [ ] Engine manifest and bundle digests are recorded.
- [ ] Model profile digest is recorded.
- [ ] Container image tag/digest or native build identifier is recorded.

## Hardware

- [ ] RTX 3050 target access, target profile, and exact engine set.
- [ ] RTX 4070 target access and exact engine set.
- [ ] RTX 5060 target access; actual GPU name and SM recorded.
- [ ] Jetson Orin Nano target access and exact engine set.
- [ ] Driver, CUDA, TensorRT, OS, architecture, compiler, clocks, and power mode recorded.
- [ ] Desired profiles selected: `qcif`, `cif`, `360p`, `540p`, `720p`, `1080p`.
- [ ] Desired QP set selected.
- [ ] Desired GOP policy selected.

## Evaluation choices

- [ ] Python comparison target selected; RTX 4070 exact is mandatory.
- [ ] Exact baseline JSONL is available for every compatibility-class case.
- [ ] Warm-up count and measured repetition count fixed.
- [ ] Timing boundary fixed and documented.
- [ ] Publication command includes `--profile` and a fixed profile repetition
  count; primary timing remains free of verbose, quality, and memory polling.
- [ ] External storage location chosen for raw outputs, if needed.
- [ ] Docker Hub namespace known only if images will be published.

## Current publication prerequisites

- [ ] Confirm the RTX 3050 Laptop profile against live target detection before
  running its exact row.
- [x] Reconcile `rtx4070-ubuntu2404` with the detected TensorRT 10.9 reference
  host.
- [ ] Rebuild all selected engine bundles from the current model-profile digest;
  the local 2026-08-05 RTX 4070 bundles predate it.
- [ ] Produce pinned `nvcr.softwarex.python-reference.v1` rows for the
  mandatory RTX 4070 comparison.
- [ ] Add a Jetson GPU-memory sampler if `nvidia-smi` cannot provide the
  required metric on the target.
- [ ] Validate any compatibility-policy assets against an exact baseline before
  publication.
