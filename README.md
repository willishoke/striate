# striate

Feature visualization (Distill 2017, *["Feature Visualization"](https://distill.pub/2017/feature-visualization/)*),
reproduced as a small clean-room implementation, pointed at a **modern CNN
(ConvNeXt)**, at **high resolution**, and extended into **temporal morphing** —
smooth video that interpolates between what different neurons "want to see."

The animating idea: smoothness is a function of *spatial frequency*. A morph
should be **smooth along time** but **striated along x/y** — the feature texture
stays crisp while the motion stays continuous. Most of the temporal machinery
here is about enforcing that decomposition cleanly (and discovering which
artifacts come from violating it).

## Setup (uv)

```bash
uv sync                 # create the env and install torch/torchvision/etc.
```

On Apple Silicon, FFT/grid_sample backward fall back to CPU; the package sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` automatically.

## Stills — `striate.cli`

```bash
# list ConvNeXt layer names to target
uv run python -m striate.cli --list-layers

# a channel at 512px (faithful paper reproduction)
uv run python -m striate.cli --layer features.5 --channel 42 --size 512

# climb toward hi-res with octave seeding
uv run python -m striate.cli --layer features.7 --channel 100 --progressive --width 1920 --height 1080
```

Stage layers for `--layer` (shallow → deep): `features.1`, `features.3`,
`features.5`, `features.7`. Channel counts grow with depth (tiny: 96 / 192 /
384 / 768).

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
- **`stabilize.py` / `flatten.py`** — post-process experiments (temporal
  brightness LPF; spatial illumination flatten).

Rendered output lands in `outputs/` (gitignored).

## Layout

```
striate/
  param.py        Fourier / decorrelated parameterization (+ pre-sigmoid access,
                  exact image->spectrum inversion for seeding/octaves)
  transforms.py   transformation robustness (pad / jitter / scale / rotate)
  objectives.py   channel / layer / neuron objectives + clip penalty
  model.py        ConvNeXt backbone with activation hooks (only file that
                  changes to swap backbones)
  render.py       optimization loop + progressive octave render
  util.py         device selection + image I/O
  cli.py          stills entry point
```

`param.py` is written to double as the reference spec for an eventual port to
another backend (e.g. hasktorch).

## Licensing note

ConvNeXt code is MIT, torchvision is BSD-3. The ImageNet-1k weights carry the
unsettled ImageNet-dataset commercial caveat; feature-viz *output* is the
lowest-risk use of them. For airtight commercial cleanliness, swap to a
non-ImageNet ConvNeXt variant (timm CLIP/LAION) or DINOv2 — only `model.py`
changes.
