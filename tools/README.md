# Developer reference tool

`dcvcrt-reference` is an opt-in development helper for invoking the pinned upstream
DCVC-RT checkout:

```bash
dcvcrt-reference doctor
dcvcrt-reference run --test_config config/dataset_config.akiyo.json \
  --cuda 1 --write_stream 1 --output_path output/results.json
```

The tool is not installed and is not part of libnvcr. It forwards codec options
to upstream `test_video.py` and supplies the
checkout's standard image and video checkpoints when they are not given explicitly.

