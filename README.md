# striate

<p align="center"><img src="docs/4e198-cellwarp.gif" width="512" alt="Green, pink and black cells with bright membranes drifting and breathing in a seamless loop"></p>
<p align="center"><sub>GoogLeNet <code>inception4e</code>, channel 198: the cells this one filter wants to see, drifting and breathing as far as the filter allows. <a href="experiments/loops/cellwarp.md">How it was made</a>.</sub></p>

An image classifier is a stack of filters, and feature visualization asks each
filter what it wants to see: start from noise and nudge the pixels until that one
filter fires as hard as it can. striate reproduces the method of the Distill
article *["Feature Visualization"](https://distill.pub/2017/feature-visualization/)*
(Olah et al., 2017) as a small clean-room implementation, at high resolution, on
any torchvision ImageNet backbone, and then tries to make the results move.

Moving is the hard part. Optimize each frame on its own and the pictures flicker;
tie the frames together pixel by pixel and they dissolve into one another instead
of moving. What works is building the motion into the parameterization, so that a
frame can only be a moved copy of a picture, and then letting the filter decide
how much motion it tolerates. The name is the aim: smooth along time, striated
along x and y, with the texture crisp while the motion stays continuous.

There is no single recipe yet. Each loop below gets its own parameterization and
its own loss, chosen for what that filter draws, and the write-up for each gives
the loss in full alongside a plain-language account.

## Gallery

| loop | backbone, layer, channel | method | cost to the filter |
|---|---|---|---|
| [Cell warp](experiments/loops/cellwarp.md) | GoogLeNet, `inception4e`, 198 | segment the still into its own cells, then learn periodic per-cell drift and breathing; Real-ESRGAN round trip | 7.4 % (warp) + 1.5 % (round trip) |

More loops will join this table as they mature.

## Setup (uv)

```bash
uv sync                 # create the env and install torch/torchvision/etc.
```

On Apple Silicon, FFT/grid_sample backward fall back to CPU; the package sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` automatically.

## Stills — `striate.cli`

```bash
# list layer names to target (ConvNeXt-tiny by default; --model googlenet, resnet50_v1, ...)
uv run python -m striate.cli --list-layers

# a channel at 512px (faithful paper reproduction)
uv run python -m striate.cli --layer features.5 --channel 42 --size 512

# climb toward hi-res with octave seeding
uv run python -m striate.cli --layer features.7 --channel 100 --progressive --width 1920 --height 1080
```

ConvNeXt stage layers for `--layer` (shallow → deep): `features.1`, `features.3`,
`features.5`, `features.7`. Channel counts grow with depth (tiny: 96 / 192 /
384 / 768). `striate.model.NETS` lists the other backbones; GoogLeNet gives the
crispest renders.

## Motion

- **`sweep.py`** — lo-res channel sweep + contact sheet, for browsing before
  committing compute.
- **`morph.py`** — within-layer channel morph via warm-start continuation, with
  optional adaptive pacing, a soft **low-frequency freeze** (kills large-scale
  brightness drift), and a temporal-coherence term.
- **`interp.py`** — joint *aligned* interpolation (the paper's method: a batch
  optimized together with shared init), plus a tunable **high-frequency
  temporal-smoothness** term that tames "feature shimmer" without braking the
  morph or saturating (it lives in unbounded spectrum space).
- **`upscale.py`** — off-the-shelf Real-ESRGAN super-res of rendered frames
  (via spandrel; weights go in `models/`, not versioned). `--down N` brings the
  frames back to N px, a round trip that crispens outlines.

Rendered output lands in `outputs/` (gitignored).

## Findings

Measurements from building the morph pipeline. Strobe metric: per spatial-frequency
band, R = E[d2²]/E[d1²], where d1 and d2 are the first and second temporal
differences of the frames. R = 0 is smooth motion, 2 a random walk, 3 i.i.d.
flicker. `strobe_bench.py` reproduces these at low res (texture scale is fixed
in pixels by the network, so a small render behaves like a crop of the same
phenomenon).

- **The morph strobe is optimizer noise, not the morph.** Warm-start continuation
  (`morph.py`) at 1080p gives R ≈ 2.0–2.4 across 4–64px bands, with a ~0.5
  frame-difference sign-flip rate: frame-to-frame change is a random walk riding
  on top of the intended motion.
- **Penalizing temporal curvature directly is a trap.** A constant-velocity
  coherence loss in pre-sigmoid space drives R to ~0.1, but it gets there by
  saturating ~40% of pixels: linear drift of preactivations toward ±∞ satisfies it
  perfectly. Any smoothness metric needs a saturation guard next to it.
- **Iterate averaging fixes it.** Rendering frames from an EMA of the spectrum
  instead of the raw Adam iterate (`morph.py --ema 0.97`) cuts R to 0.36–0.48 and
  the flip rate to 0.07–0.11, for ~4% fine-detail loss and ~5 frames of lag
  (0.99: R ≈ 0.1, −11% detail, ~17 frames of lag). Raising Adam's eps did nothing,
  which rules out sign-like steps on unconstrained coefficients as the cause.
- **Two facets of one channel are connected by a high-activation path, but not a
  straight one.** Between two seeds of `features.5` ch42, linear interpolation loses
  65% of the activation midway (67% in pixel space, 85% with polar Fourier interp).
  A string-method minimum-energy path loses ~10%, by detouring around the
  crossfade valley on a route ~4× longer than the straight line. Respacing the
  string by band-passed image distance rather than spectrum distance evens out its
  pacing (speed CV 0.65 → 0.08) and halves the activation drop.

- **A loop coupled pixel by pixel dissolves instead of moving.** Optical flow
  (RAFT) explains about 0% of the frame-to-frame change in ridge-walk loops, and a
  brightness-constancy loss between frames is satisfied most cheaply by a slow
  crossfade. A shared picture warped by a learned periodic field is explained
  about 95% by flow at no cost in activation.
- **Motion has a price, and the channel sets it.** Warping a still of
  `inception4e` channel 198 along its own cell walls costs about 2% of the
  activation at 4 px of motion and about 7% at 9 px. Left free, the optimizer
  shrinks the motion back toward the still, so the amplitude has to be held by a
  target.

## Experiments

`experiments/` holds the scripts behind the work above and the dead ends around
it: the blend-alpha de-smear critic and critic morphs (`critic.py`,
`morph_critic.py`, `morph_fb.py`, `morph_hires.py`, …), facet corpora and
clustering (`make_corpus.py`, `facets.py`, `cluster_facets.py`), within-channel
swims and optical-flow morphing (`swim.py`, `flowmorph.py`), keyframe
interpolation (`keyframe.py`), post-process stabilization (`stabilize.py`,
`flatten.py`), a ViT/SigLIP backbone (`vit_still.py`, `vit_sweep.py`), and `loops/`: every
approach to seamless loops, from ridge walks and transport to multiplane layers and the cell
warp loop in the gallery, with an index of what each found (`loops/README.md`). They are
research code: run them from the repo root (`uv run python experiments/swim.py`),
and expect defaults that point at artifacts in `outputs/` from earlier runs.

## Layout

```
striate/
  param.py        Fourier / decorrelated parameterization (+ pre-sigmoid access,
                  exact image->spectrum inversion for seeding/octaves)
  transforms.py   transformation robustness (pad / jitter / scale / rotate)
  objectives.py   channel / layer / neuron objectives + clip penalty
  model.py        torchvision backbones (ConvNeXt, GoogLeNet, ResNet, ...)
                  with activation hooks; the forward pass stops at the
                  deepest hooked layer
  render.py       optimization loop + progressive octave render
  util.py         device selection + image I/O
  cli.py          stills entry point
```

`param.py` is written to double as the reference spec for an eventual port to
another backend (e.g. hasktorch).

## License

MIT (see `LICENSE`). That covers this code only; model weights carry their own
terms (below).

## Model weights

ConvNeXt code is MIT, torchvision is BSD-3. The ImageNet-1k weights carry the
unsettled ImageNet-dataset commercial caveat; feature-viz *output* is the
lowest-risk use of them. For airtight commercial cleanliness, swap to a
non-ImageNet ConvNeXt variant (timm CLIP/LAION) or DINOv2 — only `model.py`
changes.
