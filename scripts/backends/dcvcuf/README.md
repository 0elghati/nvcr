# DCVC-UF tools

DCVC-UF (pinned `microsoft/DCVC` `cbdae87a5445114cdc7f48816da63ea80bdeac40`)
runs in NVCR on two execution paths:

- **Native fused CUDA** (`-DNVCR_ENABLE_DCVC_UF=ON`): upstream's own fused
  inference sources, compiled with LibTorch and CUTLASS. Byte-identical to
  upstream `test_video.py`.
- **TensorRT** (`-DNVCR_ENABLE_TENSORRT=ON`): 20 TensorRT plans, the ported
  upstream glue kernels and NVCR's UF rANS coder. Not byte-identical to
  upstream (TensorRT FP16 kernels differ); see the
  [C11 evidence](../../../evidence/phase-c11-dcvcuf-tensorrt-20261102.md).

The CLI picks the path from `--engine-dir`: a directory with
`engine_manifest.json` runs on TensorRT, otherwise the native bundle
(`dcvcuf-cvpr2026-hts.pt`) is used.

## Native bundle

```text
python scripts/backends/dcvcuf/export_native_bundle.py --help
```

`materialize_native_sources.py` copies upstream's fused sources into the build
tree at configure time; CMake runs it.

## TensorRT bundle

1. Export the stages and build the plans (needs the pinned checkout, both
   checkpoints and `trtexec`):

   ```text
   python scripts/backends/dcvcuf/export_tensorrt_stages.py \
     --reference-root <dcvc-uf> --image-checkpoint cvpr2026_image.pth.tar \
     --image-checkpoint-sha256 <sha> --video-checkpoint cvpr2026_video_hts.pth.tar \
     --video-checkpoint-sha256 <sha> --builder-optimization-level 5 --output-dir <out>
   ```

2. Add the quantization and entropy tables from the native bundle:

   ```text
   python scripts/backends/dcvcuf/add_tensorrt_tables.py \
     --weights dcvcuf-cvpr2026-hts.pt --bundle <out>/bundle
   ```

3. Run: `nvcr encode --codec dcvc-uf --engine-dir <out>/bundle ...`. Each plan's
   execution context is sized for the 1080p profile; on a GPU shared with other
   work, `NVCR_TENSORRT_LOW_MEMORY_MODE=1` creates contexts per call instead.

`tensorrt_stages.py` defines the stages. `port_glue_kernels.py` regenerates
`src/dcvcuf/backend/tensorrt/glue_kernels.inc` from upstream (`--check`
verifies the committed file).

## References and checks

- `capture_rd_reference.py`: upstream `test_video.py` RD reference on the
  sequences in `c8_reference_sequences.json`.
- `capture_stage_vectors.py` and `check_tensorrt_stages.py`: per-stage tensors
  from upstream and the comparison of each TensorRT plan against them.
- `run_reconstruction_identity.py`: C13 G1, encoder-internal reconstruction
  against decoder output on both paths, via the test-only
  `nvcr_dcvcuf_reconstruction_identity` harness and the observer in
  `src/dcvcuf/diagnostics.hpp`; also checks that the harness streams equal the
  CLI's. The harness's `partial-group` mode checks reset and drain with a
  partly filled eight-frame group.
