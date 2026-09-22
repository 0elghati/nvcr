# RTX 4070 matched NVCR/Python measurement — 2026-09-21

The full matched campaign at `evidence/measurement-rtx4070-matched/` is
complete. This current RTX 4070 result also retains two targeted throughput
repeats in the compact [CSV](operation-measurements.csv) and
[JSONL](operation-measurements.jsonl): 6,048 full-campaign operations plus 640
repeat operations, each labelled by `campaign` and `campaign_fingerprint`.
Use `campaign=full` for the six-resolution primary comparison; the extra
repeats independently check 16 selected decode conditions. Raw JSONLs,
per-frame arrays, commands, per-operation files, and logs remain local.

## Coverage and audit

- Six sequences/resolutions (QCIF, CIF, 360p, 540p, 720p, 1080p), QPs
  0/21/42/63, GOPs 1/30/100, 100 measured frames after 10 same-session
  warm-up frames, reset interval 64.
- In `campaign=full`, both implementations passed all 6,048 planned
  operations: 3,024 NVCR and 3,024 Python. Per implementation there are
  720 encodes and 720 decodes in each of the separate throughput and memory
  modes, plus 72 quality encodes and 72 quality decodes. The two targeted
  throughput campaigns add 480 operations at 720p and 160 at 1080p; all passed.
- Re-running `measurement_campaign.py analyze` yielded `complete`, with
  no failed/skipped, missing, mismatched-attempt, or duplicate observations.
  All 6,688 merged execution IDs are unique. Each source campaign has its
  own fingerprint because the targeted manifests select fewer conditions;
  each fingerprint is consistent within its campaign. The 576 full-campaign
  throughput/memory aggregate records reproduce from individual process
  values (mean, sample SD, minimum, maximum); the 288
  quality aggregate records retain their separate evaluator outputs. All 144
  pooled quality values reproduce from per-frame plane SSE/sample counts.
- Each of the 144 full-campaign implementation/sequence/QP/GOP conditions
  has the same recorded encoded-stream SHA256 in all 21 encode executions
  (10 throughput, 10 memory, one quality). The targeted encode repeats match
  those hashes for their selected conditions. All 36 four-point
  entropy-rate/quality curves increase strictly in rate and pooled PSNR
  as QP increases.
- All observations use the `host-yuv420p8-completed-frame-v1` timing
  contract, 100 timed frames and 10 warm-up frames. In the full campaign,
  the 504 non-empty operation stderr files are exactly the NVCR GOP-1
  all-intra warning;
  no other stderr text was found.

## Timing and memory

### Process-level FPS: 100-frame job throughput

For each clean throughput run, `process_fps = frames / process_seconds`.
Each process completes 100 measured frames after 10 warm-up frames; the
warm-up time is included in the duration but its frames are excluded from
the numerator. The parent-observed process duration includes startup,
imports, engine/model loading, warm-up/reset, source and bitstream file I/O,
codec work with per-frame CUDA synchronization, and teardown. Offline quality
evaluation and hashing are outside the codec process.

The [condition statistics](condition-statistics.csv) contain a separate
`campaign`-labelled `process_fps` row for every
sequence/QP/GOP/implementation/operation: 288 in `campaign=full` and 64 in
the two targeted cohorts. Each row reports `n=10`, the arithmetic mean of
ten per-run process FPS values, sample SD (`ddof=1`), and a two-sided 95%
Student-t confidence interval for that mean:
`mean ± t(0.975, 9) × sample_SD / sqrt(10)`, with
`t(0.975, 9) = 2.2621571628540993`. The interval assumes independent
fresh-process repetitions and describes the condition mean; it is not an
interval for a pooled resolution result or an NVCR/Python ratio. Other
metric rows leave the CI columns blank.

The table pools the 12 QP/GOP conditions and ten fresh-process repetitions
per condition from `campaign=full`: 12,000 frames divided by the sum of
process durations for each implementation, resolution, and operation.
The speedup is the ratio of those pooled FPS values. Pooled throughput
uses total frames divided by total process seconds; the per-condition mean
above instead gives every repetition equal weight in FPS space. Neither
the condition-level mean nor its confidence interval is substituted for
the pooled table.

| Resolution | NVCR encode FPS | Python encode FPS | NVCR/Python | NVCR decode FPS | Python decode FPS | NVCR/Python |
|---|---:|---:|---:|---:|---:|---:|
| QCIF | 82.63 | 27.13 | 3.05× | 84.07 | 27.96 | 3.01× |
| CIF | 74.18 | 26.26 | 2.82× | 74.54 | 27.79 | 2.68× |
| 360p | 65.11 | 24.72 | 2.63× | 64.82 | 27.09 | 2.39× |
| 540p | 51.92 | 20.30 | 2.56× | 51.36 | 24.05 | 2.14× |
| 720p | 34.82 | 16.34 | 2.13× | 36.61 | 21.35 | 1.71× |
| 1080p | 19.67 | 10.03 | 1.96× | 21.00 | 15.28 | 1.37× |

NVCR has higher process FPS in all 72 encode and all 72 decode matched
conditions. The median condition-level NVCR/Python ratio is 2.69× for encode
(range 1.59–3.14×) and 2.35× for decode (range 1.14–3.11×).
These are 100-frame job results, not steady-state codec FPS; a different
job length changes the weight of startup and warm-up. For one concrete
condition, full-campaign QCIF QP 0/GOP 1 NVCR decode has a per-run FPS mean
of 79.88 (95% CI 79.07–80.70), while its pooled condition throughput is
79.87 FPS.

### Synchronized completed-frame codec FPS

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

### Targeted repeat of the 16 Python-leading codec-decode conditions

The extra `targeted-720p-r10` cohort covers all 12 FourPeople QP/GOP
conditions; `targeted-1080p-r10` covers the four BasketballDrive GOP-1 QPs.
Each selected condition adds ten fresh-process throughput repetitions for
both implementations and both operations. All 640 operations passed.
Python again leads in synchronized completed-frame decode FPS in 16/16;
NVCR leads in process-level decode FPS in 16/16 (1.14–1.95×), with
nonoverlapping observed per-run ranges. NVCR also leads in process-level
encode FPS in 16/16. Their 64 process-FPS condition rows have separate
`campaign` labels and confidence intervals; these selected repeats corroborate
the reversal and are not pooled into the balanced six-resolution table above.

Memory is Linux `wait4` whole-process RSS high-water, including imports,
model/engine loading and warm-up—not CUDA allocation. Across the same 72
conditions per operation, the median ratio of NVCR mean RSS to Python mean RSS
is **0.40 encode** (range 0.36–0.43) and **0.41 decode** (range 0.36–0.44).
The [condition statistics](condition-statistics.csv) retain all 1,152
original full-campaign timing/RSS rows, add 288 full-campaign process-FPS
rows, and keep 192 targeted timing rows plus 64 targeted process-FPS rows
separate by `campaign` (1,696 rows total). Individual values for all
operation modes and the targeted repeats remain in the merged
[CSV](operation-measurements.csv) and [JSONL](operation-measurements.jsonl);
only `mode=throughput` rows should be used for FPS comparisons.
`throughput_fps` is the completed-frame codec metric; `process_fps` is
the 100-frame process metric. RD points are from `campaign=full` only.
Each local campaign retains its own `analysis.json` and
`observations.jsonl` source records.
No repetition was excluded.

## Rate–distortion

The common quality metric is decoded YUV420P8 pooled plane PSNR (6:1:1).
Every 100-frame decoded output was evaluated by the same offline metric.
Quality was measured once per condition and implementation in `campaign=full`;
the targeted repeats added throughput measurements only. At the same QP,
the median NVCR-minus-Python pooled PSNR difference across 72 matched points
is −0.0150 dB; NVCR is lower at 63 points and higher at nine. The largest
deficit is −0.1186 dB (FourPeople 720p, QP 21, GOP 100). Same-QP PSNR
does not hold bitrate fixed, so the BD-rate results below answer a different
question.

| Resolution | Median NVCR − Python PSNR | Range across 12 QP/GOP points | NVCR lower |
|---|---:|---:|---:|
| QCIF | -0.0099 dB | -0.1019 to +0.0844 dB | 8/12 |
| CIF | -0.0125 dB | -0.0459 to +0.0449 dB | 10/12 |
| 360p | -0.0116 dB | -0.0425 to +0.0025 dB | 10/12 |
| 540p | -0.0138 dB | -0.0469 to +0.0110 dB | 11/12 |
| 720p | -0.0849 dB | -0.1186 to -0.0247 dB | 12/12 |
| 1080p | -0.0118 dB | -0.0374 to -0.0001 dB | 12/12 |

The next table gives NVCR relative to Python BD-rate, using the repository's
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

The full campaign measured NVCR source
`caa58507b2d42dffb58902b8719b7859b53c5a3b` **plus its captured working
diff**. The targeted repeats used source head
`d81fc793f5adb527676e7718e85942765e6aaf2a` after the compact
result commit and a campaign scheduling edit; the Release CLI and
completed-frame timer source hashes match the full campaign. The Release
CLI SHA256 is
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

The complete source observations remain under local ignored `evidence/`.
The committed CSV and JSONL retain scalar fields and source labels;
they do not contain raw commands, per-frame arrays, or logs.

| Local campaign/file | SHA256 |
|---|---|
| `measurement-rtx4070-matched/run.json` | `cfbb1fbb029b64dc7d5969f240694057ebc91d9646d89c27f184d21601e2f1ff` |
| `measurement-rtx4070-matched/observations.jsonl` | `a9c4b84091c769b6d735fd7f61d92ecb003c9bdf17f85c91eb8c95f475f347ff` |
| `measurement-rtx4070-matched/analysis.json` | `b4013563558caf4181fdedd85abf730f62d9ba5e519dc3b2188920f04833afff` |
| `measurement-rtx4070-matched/rd-points.json` | `1bb769af1ada04974154abc875693549f22863fa870ce0dbe2fc01090c55a41c` |
| `measurement-rtx4070-python-leading-720p-r10/run.json` | `e310e31a0c1e07f1f39d30d776647b62cf3a6868434be3044df8ae196b4b2bc1` |
| `measurement-rtx4070-python-leading-720p-r10/observations.jsonl` | `4676ace17ca27e335fb859103f485111647d2acdd7cbb8e5bfe3838d762137cc` |
| `measurement-rtx4070-python-leading-720p-r10/analysis.json` | `7a58b19cd18e9d297bc1e5851f0515e82a036f8ae974764df45bc176ec6a7282` |
| `measurement-rtx4070-python-leading-1080p-r10/run.json` | `310304c31a131508e0e1b4d1e03faeb081e34222c7f3a1237f8143d33fe37bf8` |
| `measurement-rtx4070-python-leading-1080p-r10/observations.jsonl` | `43f2ce6fae2f9010fbe9b23403cb60207822aaf3c879e1c6a41d69cd373569fb` |
| `measurement-rtx4070-python-leading-1080p-r10/analysis.json` | `1baeda1d611ca4244caf9ef26b4b00de586efa856143ab93e277e2f967cf4bc0` |

| Committed compact export | SHA256 |
|---|---|
| `operation-measurements.csv` | `69e939eb9f29f3820e5dd5dc200a62bffdd13a6da53e951ab0c1dc3523f2ef56` |
| `operation-measurements.jsonl` | `576346627a4bed69c463b93a43e19ab7e068c6e8a27fdb359e4f7fd0a0b2e077` |
| `condition-statistics.csv` | `bdcfd4e5f8994dca6a08dd330f0dfe3d3029c95b6861313c3cc9a782395dd548` |
| `rd-points.csv` | `34a91b5b23d78d9de8e5adfb5ec256954da33c0d24cdee9c9c51312bd6125316` |

Reproduce the compact exports from the three local source packages with:

```bash
python3 scripts/export_measurement_summary.py \
  evidence/measurement-rtx4070-matched results/rtx4070/measurement \
  --repeat-campaign targeted-720p-r10 \
    evidence/measurement-rtx4070-python-leading-720p-r10 \
  --repeat-campaign targeted-1080p-r10 \
    evidence/measurement-rtx4070-python-leading-1080p-r10
```

The local raw packages remain the authority for unrounded quality-overlap
bounds, source diffs, commands, and artifact digests. This result does not
import historical QP32 measurements or the separate Jetson campaign.
