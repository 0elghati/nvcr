# RTX 4070 matched NVCR/Python measurement — 2026-09-21

The local campaign at `evidence/measurement-rtx4070-matched/` is complete.
This is the current matched RTX 4070 result. The committed
[operation measurements](operation-measurements.csv) retain one scalar record
for each of the 6,048 passed operations; raw JSONL, per-frame arrays,
commands, per-operation files, and logs remain local.

## Coverage and audit

- Six sequences/resolutions (QCIF, CIF, 360p, 540p, 720p, 1080p), QPs
  0/21/42/63, GOPs 1/30/100, 100 measured frames after 10 same-session
  warm-up frames, reset interval 64.
- Both implementations passed all 6,048 planned encode/decode operations:
  3,024 NVCR and 3,024 Python. Per implementation there are 720 encodes
  and 720 decodes in each of the separate throughput and memory modes,
  plus 72 quality encodes and 72 quality decodes.
- Re-running `measurement_campaign.py analyze` yielded `complete`, with
  no failed/skipped, missing, mismatched-attempt, or duplicate observations.
  All 6,048 execution IDs are unique, and one compatibility fingerprint is
  consistent throughout. The 576 throughput/memory aggregate records reproduce
  from individual process values (mean, sample SD, minimum, maximum); the 288
  quality aggregate records retain their separate evaluator outputs. All 144
  pooled quality values reproduce from per-frame plane SSE/sample counts.
- Each of the 144 implementation/sequence/QP/GOP conditions has the same
  recorded encoded-stream SHA256 in all 21 encode executions (10 throughput,
  10 memory, one quality). All 36 four-point entropy-rate/quality curves
  increase strictly in rate and pooled PSNR as QP increases.
- All observations use the `host-yuv420p8-completed-frame-v1` timing
  contract, 100 timed frames and 10 warm-up frames. The 504 non-empty
  operation stderr files are exactly the NVCR GOP-1 all-intra warning;
  no other stderr text was found.

## Timing and memory

FPS below is the synchronized completed-frame codec interval, excluding
initialization, warm-up, file I/O outside the interval and process teardown.
It is **not** process-level FPS. Each range spans the 12 separate condition
means for one resolution (four QPs × three GOPs), not a pooled estimate or
confidence interval.

| Resolution | NVCR encode FPS | Python encode FPS | NVCR decode FPS | Python decode FPS |
|---|---:|---:|---:|---:|
| QCIF | 520.42–959.62 | 193.18–280.27 | 578.01–1075.57 | 248.90–349.04 |
| CIF | 260.88–581.09 | 157.42–217.51 | 245.26–605.95 | 199.34–349.70 |
| 360p | 157.03–406.21 | 102.46–151.25 | 150.79–392.12 | 142.06–279.38 |
| 540p | 76.04–220.79 | 44.62–82.96 | 73.67–208.72 | 68.41–166.87 |
| 720p | 37.65–94.68 | 27.15–48.46 | 41.32–102.02 | 47.66–105.49 |
| 1080p | 16.15–44.06 | 11.71–20.68 | 17.60–47.05 | 21.23–45.32 |

For each of 72 matched sequence/QP/GOP conditions per operation, the
ratio of the NVCR mean FPS to the Python mean FPS has median **2.35× encode**
(range 1.37–3.43; NVCR higher in 72/72) and **1.24× decode** (range
0.82–3.08; NVCR higher in 56/72). These are medians of condition-level
ratios, not a pooled equal-work FPS or a claim that one software layer alone
caused the difference. Repetitions are independent processes, not paired
cross-implementation samples.

| Completed-frame FPS variability | Median within-condition CV | Maximum CV |
|---|---:|---:|
| NVCR encode | 0.13% | 4.10% |
| NVCR decode | 0.16% | 3.89% |
| Python encode | 0.60% | 2.74% |
| Python decode | 0.60% | 1.75% |

Memory is Linux `wait4` whole-process RSS high-water, including imports,
model/engine loading and warm-up—not CUDA allocation. Across the same 72
conditions per operation, the median ratio of NVCR mean RSS to Python mean RSS
is **0.40 encode** (range 0.36–0.43) and **0.41 decode** (range 0.36–0.44).
All 1,152 per-condition timing/RSS statistics are in
[condition-statistics.csv](condition-statistics.csv). Individual values for
all operation modes are in [operation-measurements.csv](operation-measurements.csv);
only `mode=throughput` FPS should be used for the throughput comparison above.
The local `analysis.json` and `observations.jsonl` retain their source records.
No repetition was excluded.

## Rate–distortion

The common quality metric is decoded YUV420P8 pooled plane PSNR (6:1:1).
Every 100-frame decoded output was evaluated by the same offline metric.
The table gives NVCR relative to Python BD-rate, using the repository's
SciPy 1.16.3 PCHIP log-rate integration on each curve's *quality overlap
only*. Positive values mean NVCR used more bits at equal interpolated PSNR.
All 18 paired sequence/GOP curves had positive overlap. Entropy BPP is the
more comparable codec payload boundary; complete-file BPP includes different
NVCS/NVAU versus Python SPS/IP framing and must not be called an entropy
coding difference.

| Sequence | GOP | Entropy BD-rate | Complete-file BD-rate |
|---|---:|---:|---:|
| Akiyo QCIF | 1 | +0.38% | +19.93% |
| Akiyo QCIF | 30 | +1.29% | +153.65% |
| Akiyo QCIF | 100 | −0.03% | +253.10% |
| Waterfall CIF | 1 | +0.24% | +3.83% |
| Waterfall CIF | 30 | +0.41% | +30.57% |
| Waterfall CIF | 100 | +0.90% | +60.99% |
| Basketball 360p | 1 | +0.42% | +3.45% |
| Basketball 360p | 30 | +0.40% | +17.31% |
| Basketball 360p | 100 | +0.48% | +21.90% |
| Basketball 540p | 1 | +0.54% | +2.31% |
| Basketball 540p | 30 | +0.45% | +10.56% |
| Basketball 540p | 100 | +0.31% | +13.20% |
| FourPeople 720p | 1 | +1.18% | +2.31% |
| FourPeople 720p | 30 | +2.03% | +13.16% |
| FourPeople 720p | 100 | +3.70% | +26.37% |
| Basketball 1080p | 1 | +0.80% | +1.57% |
| Basketball 1080p | 30 | +0.70% | +5.04% |
| Basketball 1080p | 100 | +0.70% | +6.20% |

The median across these 18 separate BD integrations is +0.51% for entropy
BPP and +13.18% for complete-file BPP. The medians are descriptive, not
a pooled BD-rate. All 144 measured points are in
[rd-points.csv](rd-points.csv), with unrounded source JSON retained locally.

## Provenance and limits

The measured NVCR source was
`caa58507b2d42dffb58902b8719b7859b53c5a3b` **plus its captured working
diff**; the Release CLI SHA256 is
`d22754daa957bc9dfc7aa65dab5732cd698c5bba369014400588c108e7b27596`.
The GPU is NVIDIA GeForce RTX 4070 (SM 8.9, 46 SMs), with CUDA runtime
12.8 and TensorRT 10.9.0. All 20 preflight checks passed, including six
exact-target engine bundles, inputs, checkpoints, and required Python CUDA/
rANS extensions. Python used PyTorch 2.9.1+cu126 with its custom CUDA path.

The Python checkout was fork commit
`1aa39a0ec756669af2ca3aedf9cf934aef55b1a4`, **not** model-profile
commit `1feb52a592a9ff2c4e4ba2e5122e2da49a211466`. The user-authorized
`explicit-reference-source-override` recorded 73 tracked changes in runner,
experiment and input files; none were under `src/` or `test_video.py`.
The exact diff and extension hashes are retained in `run.json`. This is a
matched *fork* result, not an official pinned-source result.

The full run was explicitly authorized without a matched smoke. Stable hashes
across repeated warmed executions do not establish cold-versus-warmed reset
equivalence. The run is internally complete, but the pinned-source and
matched-smoke publication gates remain open. Environment snapshots and CPU
`performance` governors do not prove fixed GPU clocks or absence of thermal
drift throughout the overnight campaign. Successful streams/reconstructions
were removed by the runner after hashing and evaluation, so this audit cannot
re-decode them.

## Local evidence identity

| File under `evidence/measurement-rtx4070-matched/` | SHA256 |
|---|---|
| `run.json` | `cfbb1fbb029b64dc7d5969f240694057ebc91d9646d89c27f184d21601e2f1ff` |
| `observations.jsonl` | `a9c4b84091c769b6d735fd7f61d92ecb003c9bdf17f85c91eb8c95f475f347ff` |
| `analysis.json` | `b4013563558caf4181fdedd85abf730f62d9ba5e519dc3b2188920f04833afff` |
| `rd-points.json` | `1bb769af1ada04974154abc875693549f22863fa870ce0dbe2fc01090c55a41c` |

The compact exports have SHA256
`92859f738b0c477afd51f94c17d030f6c5f7b42d97a307d116ac7c94795a4f6f`
(`operation-measurements.csv`),
`604445284a80f09591fb659dba73f582b66e8ba9be2fe42e36b408c8c4f3b936`
(`condition-statistics.csv`), and
`34a91b5b23d78d9de8e5adfb5ec256954da33c0d24cdee9c9c51312bd6125316`
(`rd-points.csv`). They were generated with:

```bash
python3 scripts/export_measurement_summary.py \
  evidence/measurement-rtx4070-matched results/rtx4070/measurement
```

The retained local raw package is the authority for every unrounded value,
quality-overlap bound, source diff, command, and artifact digest. This summary
does not import historical QP32 results or the separate Jetson campaign.
