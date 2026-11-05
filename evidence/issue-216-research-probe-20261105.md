# Issue #216 research probe: faster exact decoding — 2026-11-05

## Scope

- Branch: `fix/cross-gpu-entropy-determinism` on `ef4dcfd8` (`main` `163441e9`),
  uncommitted. Roadmap: C12 (research input; no roadmap change proposed here
  without the user's priority decision).
- Question: the exact profile (`dcvc-rt.exact.v1`) decodes 1080p at 18.4 fps
  against 48.7 fps native on an RTX 4070 (see the
  [#216 evidence](issue-216-cross-hardware-fix-20261104.md)). Engineering bounds
  put the remaining kernel headroom near 10-20%. Can a change to the codec, without
  quality loss, bring exact decoding near native speed?
- Inference only. No training was run.

## Why exact decoding is expensive

DCVC-RT's P-frame entropy parameters depend on a temporal prior computed from
the previous decoded feature, so the whole decoder-side path must be exact:
346 GMAC per 1080p P-frame, every multiply at 16 bits (four INT8 products).
The entropy-parameter networks alone (hyper decoder 2.6, temporal prior 6.4,
prior fusion 30.2, spatial prior 21.7 GMAC) total about 61 GMAC, 5.7 times
less. Two codec changes would shrink or cheapen the exact work:

1. Condition the entropy model only on values both sides know exactly
   (hyperprior, the current frame's decoded latents, and the previous frame's
   decoded latents) so only the entropy networks must be exact and synthesis
   runs native FP16.
2. Retrain for 8-bit integer arithmetic (one INT8 product per multiply).

## Method

`scripts/backends/dcvcrt/integer_gop_study.py`, pinned DCVC-RT `1feb52a5`,
pure-PyTorch path (fused kernels disabled; the script checks every emulated
layer runs), QPs 0/21/42/63.

- Temporal-prior ablations (`fp32@temporal=zero`, `fp32@temporal=mean`) on the
  seven-sequence set (BasketballDrive, Kimono, ReadySteadyGo, Jockey, HoneyBee
  1080p; FourPeople, Johnny 720p), 96 frames, reference reset every 32: the
  temporal-prior output feeding the prior fusion network is replaced by zeros,
  or by its per-channel spatial mean.
- Per-module W8A8 screen: Kimono and BasketballDrive 1080p, FourPeople and
  Johnny 720p, 40 frames; one module group at 8-bit weights and activations,
  the shipped map elsewhere.

## Results

### Temporal prior

Both ablations shift the trained fusion network out of distribution, so they
bound what the existing model can do without its temporal prior, not what a
retrained model could do. BD-rate is not meaningful here (+1010% and +389%,
curves barely overlap); raw points:

| Sequence, QP | Reference | Temporal prior zeroed | Temporal prior channel mean |
|---|---|---|---|
| BasketballDrive, 21 | 0.0072 bpp, 33.05 dB | +832% bits, +1.74 dB | +186% bits, -1.84 dB |
| BasketballDrive, 42 | 0.0236 bpp, 36.32 dB | +269%, -0.09 dB | +181%, -1.02 dB |
| Johnny, 42 | 0.0069 bpp, 41.04 dB | +289%, -1.40 dB | +269%, -2.65 dB |
| Kimono, 42 | 0.0233 bpp, 38.37 dB | +199%, +0.81 dB | +97%, -0.62 dB |

Zeroing also changes the quantization step the fusion network emits, moving
the codec to a different operating point; the channel-mean variant is the
interpretable one: +71% to +270% bits at 0.5-3.7 dB lower PSNR. DCVC-RT relies
heavily on a spatially detailed temporal prior; it cannot be removed or
replaced at inference time.

### 8-bit arithmetic

BD-rate change against the shipped map's own result on the screen set
(-0.332%):

| Module at W8A8 | GMAC (1080p) | Change |
|---|---:|---:|
| `P.hyper_decoder` | 2.6 | +0.02% |
| `P.temporal_prior_encoder` | 6.4 | +0.10% |
| `P.y_spatial_prior` | 21.7 | +2.29% |
| `I.dec` | 476 (I-frames) | +3.09% |
| `P.feature_adaptor_i`, `_p` | 18.8 / 2.1 | +5.12% |
| `P.feature_extractor` | 103 | +5.89% |
| `P.recon_generation_net` | 112 | +7.25% |
| `P.decoder` | 68 | +7.73% |
| `P.y_prior_fusion` | 30.2 | +33.0% |

Post-training 8-bit is far from lossless in every heavy module, and worst in
the prior fusion network, which a latent-conditioned design would also need
exact.

### Literature

- DCVC-HEM (Li, Li, Lu, ACM MM 2022) conditions its entropy model on the
  previous frame's decoded latent (latent prior) alongside a temporal context
  prior; removing the latent prior costs 10.1% bitrate, and alone it saves
  19.3%. Latent conditioning is an established DCVC-family design, but HEM did
  not drop the feature-domain temporal context prior.
- DCVC-RT's own int16 integerization costs about 3% BD-rate and runs 4-5 times
  slower than its FP16 model (28.3/20.9 against 125.2/112.8 fps at 1080p on an
  A100). NVCR's exact profile is 2.6 times slower with no BD-rate cost.
- MobileNVC reaches real-time 1080p on a phone with an int8 model, but its int8
  model trails its own float model by a wide margin.
- DCVC-RT training code appears not to be public (GitHub issue #137 requests
  it); OpenDCVCs publishes training implementations of earlier DCVC models.

## Accepting a small quality loss

Measured: the cheapest no-training compromise, 8-bit weights in the modules
that tolerated them best (reconstruction, I-frame synthesis and spatial prior at
W8A16; hyper decoder and temporal prior at W8A8), cuts the 1080p P-frame exact
stages from 47.9 to 44.8 ms (-6.5%) and I-frame synthesis from 59.3 to 50.7 ms.
Halving the integer products barely helps because memory traffic and per-tile
overhead dominate; its BD-rate was not measured. Width compromises are not a
speed lever.

Approaches from the literature that trade a small loss for near-native speed,
all needing training:

| Approach | What must be exact | Expected decode speed (1080p, RTX 4070) | Quality risk | Source |
|---|---|---|---|---|
| Entropy model conditioned on the previous frame's decoded latent instead of the decoded feature; synthesis native, reconstruction drift bounded by I-frames | Hyper decoder, prior fusion, spatial prior, a new small latent-temporal net (~65 GMAC) | Estimated ~35 fps (exact entropy ~10 ms replaces ~2 ms of TensorRT; other stages native) | Unknown; DCVC-RT relies heavily on its feature-domain temporal prior. Cross-target drift measured earlier at PSNR-Y 52-57 dB against the encoder's decode over 33 frames | DCVC-HEM latent prior |
| Scale indices transmitted through the hyperprior and looked up in a fixed codebook (MLVC) | The hyper-latent's factorized model only | Near native | Scales no longer use spatial or temporal context; needs a redesigned, retrained entropy model | MLVC (ECCV 2026); a review notes the factorized model must still be bit-exact |
| Mixed-precision quantization-aware training, whole decoder integer | Whole decoder, mostly 8-bit | Estimated 1.5-2 times the current exact speed | JOMP reports DCVC-FM-level rate-distortion with 87.6% fewer bit operations and bit-exact cross-platform decoding | JOMP (arXiv 2606.13110) |
| Integer codec trained from scratch | Everything | Depends on design | New codec | Integer-Centric Neural Video Compression (ICLR 2026) |

NVCR already integrates MLVC (roadmap C10), a codec designed for
multi-platform decoding; its C10 evidence shows streams decoding across NPU,
phone CPU and x86 CPU with bounded drift that recovers at long-term reference
frames.

## Conclusion

- No inference-time change speeds up exact decoding without losing quality:
  the temporal prior is essential to the trained model, and 8-bit arithmetic
  costs 2-33% per heavy module.
- Both codec changes need retraining, and neither has evidence that it can be
  quality-neutral against DCVC-RT; upstream's own integerized model already
  loses about 3%. Under the no-quality-compromise rule they are research bets,
  not plans.
- A retrained model would be a new model profile with its own pinned reference
  and golden tests (`AGENTS.md`), and a new roadmap item by user decision. Its
  first gate would be a training pipeline that reproduces DCVC-RT's FP32
  results within noise, since none is public.
- The two small modules that tolerate 8-bit (hyper decoder, temporal prior
  encoder, about 9 GMAC) are too small to matter for speed.

The consolidated research report, with the option space and the next
experiments, is [docs/research/issue-216-cross-target-research.md](../docs/research/issue-216-cross-target-research.md).

## Not verified

- Any retrained variant: the latent-prior and QAT outcomes are unknown.
- The screen uses four sequences and 40 frames; its noise level is about 0.1%.
