# DCVC-RT tests

GOP decisions, references, latents, reset/flush, and golden codec vectors belong here.

## 720p I-frame golden

`i_frame_golden.py` runs one FourPeople 1280x720 frame at QP 32 through the
pinned Python DCVC-RT runtime and native NVCR. It verifies the upstream commit,
checkpoint, source-frame, Python bitstream, and Python reconstruction hashes,
then gates native source quality and Python/native reconstruction PSNR.

Configure all three paths to register the opt-in CTest:

```bash
cmake -S . -B build-release \
  -DNVCR_DCVCRT_ROOT=/path/to/DCVC-RT \
  -DNVCR_TENSORRT_ENGINE_DIR=/path/to/dcvcrt-engines \
  -DNVCR_DCVCRT_720P_GOLDEN_INPUT=/path/to/FourPeople_1280x720_60.yuv
```

The test writes `tests/dcvcrt-i-frame-golden.json` under the build directory.
It is target evidence for the pinned RTX workflow, not a claim that native and
upstream Python payload formats are interchangeable.
