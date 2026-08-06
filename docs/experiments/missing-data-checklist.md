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
- [ ] Warm-up count and measured repetition count fixed.
- [ ] Timing boundary fixed and documented.
- [ ] External storage location chosen for raw outputs, if needed.
- [ ] Docker Hub namespace known only if images will be published.
- [ ] GitHub token available only when private engine catalog assets require it.

## Known blockers in this checkout

- [ ] Add an RTX 3050 target profile before running the RTX 3050 exact row.
- [ ] Add a SoftwareX result driver or adapter around the existing matrix script.
- [ ] Decide whether installer compatibility-policy flags are needed before publishing fallback assets.
