# Contributing

Work on a feature branch and submit a pull request. Do not push directly to
`main`, and do not commit checkpoints, exported ONNX/model assets, TensorRT
plans, datasets, credentials, or temporary asset URLs.

## Build and test

The CPU contract path is the fastest baseline:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

Use a Release TensorRT build, target-local artifacts, and the documented
registered GPU tests for production changes. Run `bash -n` and `shellcheck` for
changed shell scripts and `python3 -m py_compile` for changed Python files.

## Architecture and documentation

Keep codec semantics in a codec adapter and execution technology in a provider.
Use `RuntimeServices` for provider-mediated construction, preserve bounded
access-unit parsing, and add conformance tests for lifecycle, malformed input,
artifact compatibility, and registration behavior. A new codec/provider is not
production-supported until its target and evidence gates pass.

Update the relevant identity, scope, API, extension, stream, artifact, and
reproducibility and evidence documentation with the change. Keep implementation facts,
conformance fixtures, transitional paths, and roadmap work explicitly separate.
