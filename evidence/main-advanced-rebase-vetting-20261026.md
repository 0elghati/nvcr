# `main-advanced` rebase and minor-release vetting — 2026-10-26

## Scope and identity

- Source tip: `origin/main-advanced` at
  `6a564f191bcc76006df1d84378d4bf310b038b2c`
- Current-main base: `origin/main` at
  `9a09c696e8bf99b7022324c2a7ad3ec83ec630a2`
- Rebased branch: `feat/provider-runtime-v2-rebase`
- Rebased range: `5f7d74e9` through `e266cf8e`
- Intended release effect: a Conventional Commits `feat:` change. Release
  Please owns the eventual `1.1.0` version-file, citation, changelog, and tag
  update; this feature branch remains at the current released `1.0.8` identity.

The twelve `main-advanced` commits were rebased onto current `main`; the
original remote branch was not rewritten. A `git range-diff` confirmed that the
provider/session sequence remains present. Conflict resolution kept the current
release, licensing, citation, API-policy, and measurement work from `main` while
adopting the advanced provider/session architecture.

## Compatibility audit and repairs

Current `main` requires public C++ source compatibility in patch and minor
releases. The advanced branch had removed the original flat
`RuntimeConfiguration` fields, component construction seams, and low-level
`Runtime::create(configuration, Components)` overload. That would have made the
work a major release despite its intended `feat:` presentation.

The rebased branch therefore retains those 1.0 entry points as compatibility
shims while making scoped configuration and codec/provider sessions the
preferred path:

- flat configuration fields retain their original order for aggregate and
  designated initialization, then reconcile into scoped ownership groups;
- `codec::Components`, adapter `create_components()`, provider
  `component_factory`, and `RuntimeServices::create_components()` remain
  available;
- the original low-level runtime construction overload adapts components into
  sessions;
- new adapters may implement either construction seam, and production DCVC-RT
  continues to use provider and codec sessions;
- a public-API compatibility executable compiles and runs representative 1.0
  client code, including the original four-field `ProviderEntry` aggregate.

The audit also activated the previously inert version-identity test by adding
its missing `unittest.main()` entry point. Release Please now updates only
`version.txt` and `CITATION.cff`; CMake reads `version.txt`, so duplicate version
literals in CMake, README, and getting-started documentation are no longer
release targets.

## CPU, install, and safety gates

The clean CPU Release tree at `/tmp/nvcr-vetting-cpu-20261026` passed its build
and 23/23 CTest suite. The suite includes the active version-identity test and
the new public-API compatibility executable.

```text
cmake -S . -B /tmp/nvcr-vetting-cpu-20261026 \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF \
  -DNVCR_BUILD_TESTS=ON \
  -DNVCR_BUILD_CLI=ON
cmake --build /tmp/nvcr-vetting-cpu-20261026 -j2
ctest --test-dir /tmp/nvcr-vetting-cpu-20261026 --output-on-failure
cmake --install /tmp/nvcr-vetting-cpu-20261026 \
  --prefix /tmp/nvcr-vetting-install-20261026
```

The same compatibility source was then compiled as an external consumer using
only the installed headers and static library, and the resulting executable
passed.

The Clang 18 Debug ASan/UBSan tree at
`/tmp/nvcr-vetting-sanitized-20261026` passed its build and 20/20 CTest suite.
The access-unit libFuzzer target completed 1,000 bounded runs without a
sanitizer failure.

```text
ctest --test-dir /tmp/nvcr-vetting-sanitized-20261026 --output-on-failure
/tmp/nvcr-vetting-sanitized-20261026/tests/nvcr_access_unit_fuzz \
  -runs=1000 -timeout=5 -max_total_time=10
```

`shellcheck scripts/package_release.sh`, Python byte-compilation for changed
Python sources, and `git diff --check` passed. `actionlint` was not installed.

## TensorRT gates and parity

The clean TensorRT Release tree at `/tmp/nvcr-vetting-trt-20261026` used:

- NVIDIA GeForce RTX 4070, compute capability 8.9;
- driver 580.178.04;
- CUDA 12.8.93 and TensorRT 10.9.0;
- all six exact B3/B7 engine bundles from QCIF through 1080p;
- the pinned 720p Python/native golden input.

The warning-free build passed. The non-GPU selection passed 24/24 tests. The GPU
selection passed 11/13 tests, including the pinned golden and every profile
except the two 1080p tests. An unrelated training process, PID 636621, held
9,994 MiB and left only 1,820 MiB free; both 1080p attempts ended in CUDA
allocation failure while creating the P-analysis execution context. The user
explicitly accepted omitting tests that cannot coexist with that process, so
the job was not stopped and those two cases were not retried.

The Release CLI completed a 65-frame QCIF, QP-32, GOP-8 encode and decode
through the rebased production path:

- payload bytes: 14,669;
- complete `.nvcr` SHA-256:
  `9896684474818032506d32ac7d065a7710d9c26197a9d374f275d2b65f83f81a`;
- decoded YUV SHA-256:
  `10e02e460ec4477e9a7e41f72bbb44c5222e0f6773bf581cf874f74b97363008`.

All three values exactly match the retained B7 and C6.5/C6.6 evidence.

## Release decision

The rebase, source-compatibility, CPU, install-consumer, sanitizer, fuzz,
static, non-GPU TensorRT, available-profile GPU, pinned-golden, and
byte/reconstruction gates pass. The two 1080p tests are explicitly omitted
because they cannot coexist with the current GPU training workload. No new
performance claim is made; the change preserves the previously accepted
advanced-branch performance evidence.
