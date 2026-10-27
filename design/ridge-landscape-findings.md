# Ridge landscape: findings

Results of the five steps proposed in `ridge-landscape-handoff.md`, run 2026-10-27 at 256px.
Scripts: `ridge-landscape-scratch/landscape.py` (corpus, describe, coverage, width, graph),
`walk.py` (Langevin and noisy-ascent walks), `ring.py --anchors-from` (string tours through
chosen attractors), `viz.py` (strips, MP4), `run_landscape.sh` (steps 1–4 in sequence, one
job at a time). Videos and full tables: https://claude.ai/artifact/RCBYHqScfPsXTfMSBNsfqS (private page). Strips and sheets are in `figures/`.

Two channels: ResNet-50 V1 `layer3` ch 530 (the "cells" channel) in full, GoogLeNet
`inception4c` ch 64 (leopard-spotted domes) for steps 1–4. Activation numbers are "robust":
the channel mean over four fixed view draws (`MultiScaleCrop`, whole canvas at 224 to 1:1,
plus jitter, scale, rotation) shared by every image in a comparison. Ratios are to the best
attractor of the 64-attractor corpus unless stated. Nothing in any run clipped pixels except
the one collapsed Langevin run noted below.

## Short version

- **There are no clusters.** Among 64 attractors, every string between kNN neighbours
  bottoms out between 0.87 and 0.91 of the maximum (180 strings, median 0.893 for
  layer3 · 530; 191 strings, median 0.885 for inception4c · 64). The disconnectivity tree is a
  flat palm: one cluster at θ = 0.85, all singletons at θ = 0.92. The intra/inter-cluster
  dial has nothing to act on, for either channel.
- **The ridge is isotropic at texture scale.** From an attractor, a random spectrum
  perturbation drops activation to 0.9 after the same RMS step whether it points at random,
  at the nearest other attractor, or at a random other attractor (0.36–0.39 logit RMS). The
  straight line to the nearest attractor leaves the 0.9 set after 16–20% of its length.
  Perturbations coarser than 32px are tolerated about 2.5× further than texture-scale ones.
- **A loop is a thread.** The pinned-anchor ring comes within one nearest-neighbour distance
  of 5 of 64 attractors, three of them its own anchors. The smoothed loop (anchors free)
  comes within 1.6 NN of none while holding 0.97 of the maximum: the near-maximal set is far
  larger than the attractor set. In descriptor space the loop moves along the corpus's own
  directions (its span holds 84% of corpus variance; a random subspace of the same rank
  holds 0.2%) but is lower-dimensional (participation ratio 3.2 vs 5.3).
- **Langevin does not walk.** With the renderer's Adam-style preconditioner, temperature
  boils fine texture and never moves the arrangement, at every temperature up to collapse.
  Noise that is isotropic in image terms does the same. String tours do walk; tours through
  the tightest and the loosest attractor triplet are indistinguishable.
- **Translation is not free.** Rolling an attractor by a random offset costs 8% (ResNet,
  ratio 0.917, min 0.906) to 13% (GoogLeNet, 0.867, min 0.837) of activation. The renders
  are tuned to the views' borders (gray padding, the network's own zero padding).

## Step 1: corpus and descriptor

64 attractors per channel: four batches of 16 seeds, joint octaves 224 → 256, 384 steps per
octave, with `ring.py`'s clip hinge and chroma cap (weights 4, 4). Render time 5.3 min
(ResNet) and 2.8 min (GoogLeNet).

| channel | robust activation (min, mean, max) | NN distance CV (Gram) |
|---|---|---|
| layer3 · 530 | 2.648, 2.671, 2.693 | 0.085 |
| inception4c · 64 | 11.03, 11.18, 11.32 | 0.065 |

Sheets: `figures/530_corpus_sheet.jpg`, `figures/4c64_corpus_sheet.jpg`. Visually each is 64
arrangements of one texture; nothing suggests groups.

Descriptor: the Gram matrix of an early layer (ResNet `layer1`, GoogLeNet `inception3a`;
256 channels, 65,536 numbers), unit normalised, computed on the circularly padded image with
the padded border of the feature map dropped, averaged over the sub-stride sampling phases
(4 × 4 for stride 4). Shift check: rolling 16 attractors by random offsets moves this
descriptor by 1e-5 of the corpus's median nearest-neighbour distance. Without the phase
average the same check gave 0.47 (a roll by a multiple of the stride gave exactly 0, so the
residual was the sampling grid alone); without circular padding, 0.67. The radial log power
spectrum (3 × 64 bins) moves 0.05 and is kept as a second opinion; the 16 × 16 squint (1.19)
is the non-invariant control.

Why Gram: it is what `facets.py` and `cluster_facets.py` already use, it keeps arrangement
statistics the power spectrum discards, and once periodic it is exactly shift-invariant.
The nearest-neighbour CV says every attractor is about as far from every other as from its
nearest: a high-dimensional cloud with no near-duplicates, which is what random arrangements
of one texture look like.

## Step 2: coverage

Loops re-rendered at 256 with the same scripts (`ring.py --size 256`, then `smooth.py
--v 1.0 --wmax 0.125`) so loop and corpus share a resolution and a recipe. "Within 1 NN"
counts attractors whose distance to the nearest frame is below the corpus's median
nearest-neighbour distance. PR is the participation ratio of the PCA spectrum.

| loop | activation mean, min | of max | corpus variance in loop span | within 1 NN, 2 NN | median attractor-to-loop (NN) | PR loop, corpus |
|---|---|---|---|---|---|---|
| 530 ring, 3 pinned anchors | 2.625, 2.577 | 0.975 | 0.84 (random 0.002) | 5, 63 | 1.25 | 3.2, 5.3 |
| 530 smoothed, v = 1 | 2.615, 2.579 | 0.971 | 0.77 | 0, 37 | 1.95 | 3.4, 5.3 |
| 530 tour, tightest triplet (9, 16, 20) | 2.623, 2.576 | 0.974 | 0.84 | 3, 62 | 1.36 | 3.1 |
| 530 tour, loosest triplet (6, 36, 57) | 2.634, 2.586 | 0.978 | 0.85 | 4, 64 | 1.41 | 2.8 |
| 530 Langevin T = 1, raw | 2.781, 2.694 | 1.033 | 0.84 | 1, 58 | 1.53 | 1.1 |
| 4c64 ring | 10.96, 10.73 | 0.968 | 0.76 (random 0.002) | 7, 64 | 1.19 | 4.2, 7.5 |
| 4c64 smoothed | 10.95, 10.77 | 0.967 | 0.73 | 0, 62 | 1.76 | 4.0, 7.5 |

The radial-power descriptor agrees (shares 0.86–0.89; PR 4–6 vs corpus 12–16). The
attractors the 530 ring visits are its three anchors (corpus items 0–2, which share their
seeds) plus two more. Frames sit a median 2.4 NN distances from the nearest attractor.

Reading: coverage of the *attractor set* by one loop is close to zero and would not improve
with longer loops, because the attractors sample a set whose effective dimension (5–8 in
Gram space, 12–16 in radial-power space, both lower bounds for a nonlinear set) a
one-dimensional path cannot fill. Coverage of the *directions* of variation is high. The
useful question is not "how much of the ridge does a loop visit" but "does the loop move
along the ridge's own directions", and it does.

## Step 3: width

Eight attractors spread over the activation ranking, 16 random Gaussian directions per
frequency band, each scaled so a step of 1 changes the pre-sigmoid image by RMS 1. The
radius is the step at which robust activation falls to 0.9 of the attractor's (doubling
expansion, then 6 bisections; no direction was censored).

| direction | 530 radius, logit RMS (q10–q90) | pixel RMS there | 4c64 radius | pixel RMS there |
|---|---|---|---|---|
| > 32 px | 0.78 (0.73–0.83) | 0.134 | 0.68 (0.64–0.74) | 0.122 |
| 8–32 px | 0.33 (0.33–0.34) | 0.059 | 0.32 (0.31–0.33) | 0.058 |
| < 8 px | 0.29 (0.28–0.29) | 0.051 | 0.36 | 0.065 |
| all frequencies | 0.37 (0.36–0.39) | 0.067 | 0.41 | 0.075 |
| straight toward nearest attractor | 0.36 (line length 2.24) | | 0.39 (line length 1.97) | |
| straight toward a random attractor | 0.36 | | 0.39 | |

Reading: the spread across directions and across attractors is within ±5%. In a
200,000-dimensional space a random direction is a typical direction, and typical is the
same everywhere: the superlevel set is a round tube at texture scale. The direction to
another attractor is no different from random, which is the pre-image of the flat tree.
Low spatial frequencies are the exception in both channels: the channel tolerates 2.5×
larger coarse perturbations, which is where the smoothing speed limit (v · |k|) lets the
least motion through. The "carrier plus texture" hypothesis for inception4c · 64 predicted
free spot-moving (fine) directions and stiff dome-reshaping (coarse) ones; the band split
shows coarse directions freest in both channels, with only a modest extra freedom at fine
scale for the leopard channel (0.36 vs 0.29). A Hessian/Lanczos pass was not needed.

## Step 4: disconnectivity graph

kNN graph with k = 4 on the Gram descriptor. One string per edge (open string, 4 → 9 → 19
interior nodes, 100 + 60 + 40 steps, perpendicular ascent, equal band-passed image spacing,
8 strings batched), 12 s per string at 256px: 37 min for 180 edges (ResNet), 21 min for 191
(GoogLeNet). Barrier = lowest robust activation on the converged chain.

| channel | barrier min, q10, median, q90, max | vs lower endpoint | straight-line midpoint | clusters at θ = 0.85 / 0.90 / 0.92 |
|---|---|---|---|---|
| layer3 · 530 | 0.874, 0.886, 0.893, 0.898, 0.906 | 0.902 | 0.594 | 1 / groups of 4, 3, 3, 3, 2, 2, 2 and 45 singletons / 64 singletons |
| inception4c · 64 | 0.863, 0.876, 0.885, 0.893, 0.904 | 0.899 | 0.533 | 1 / one pair, 62 singletons / 64 singletons |

Mean activation profile along the 21 nodes (530): 0.99, 0.99, 0.96, 0.94, 0.93, 0.92, 0.91,
0.90, 0.90, 0.89, 0.89, 0.89, then symmetric back up. Every string has this shape. The string
recovers 0.30–0.35 of the maximum over the straight line, as the earlier ConvNeXt
experiment did.

Trees: `figures/530_tree.png`, `figures/4c64_tree.png` (leaves at their own activation
ratio, bars at the barrier where two groups connect). Both are one basin complex with
uniform barriers: there is nothing to cut.

Caveats on the absolute level. The 19-node strings are coarser and shorter-trained than the
39-node ring, whose minimum reached 0.958 of its anchors, so 0.89 is a lower bound and the
true barriers are nearer 0.95. The corpus maximum is itself under-converged: a 3000-step
continued ascent from attractor 24 reached 2.82, 4.6% above it, so every ratio's denominator
is low by that much. Neither changes the shape of the tree.

## Step 5: walks by design

### Langevin

Preconditioned Langevin in spectrum space: `x += η M ∇a + sqrt(2ηT) sqrt(M) ξ`, with M a
fixed diagonal preconditioner (1 / per-coefficient gradient RMS at the start, so the drift
is Adam-like and the stationary measure is still ∝ exp(a/T)). T in units of
T_ref = η · median(gradient RMS) / 2, the temperature at which the per-coefficient noise
step equals the typical gradient step (T_ref = 2e-8 at η = 0.01). 3000 steps from attractor
24, 120 frames saved at equal intervals.

| T / T_ref | activation of loop frames (of max) | descriptor moved (NN) | nearest attractor (NN) | clipped |
|---|---|---|---|---|
| 0 (pure stochastic ascent) | 1.033 | 4.4 | 2.2 | 0 |
| 0.3, 1, 3 | 1.033 (identical to T = 0 to three digits) | 4.4–4.5 | 2.2 | 0 |
| 30 | 1.019 | 5.8 | 3.2 | 0 |
| 300 | 0.937 | 12.7 | 10.8 | 0 |
| 3000 | 0.46 | 23 | 22 | 11% |

In every run the composition is the same in frame 1 and frame 120 (strips in `figures/`,
videos on the page). The preconditioner puts the noise where the gradient is small, at high
spatial frequencies, where a coefficient barely changes the image; the drift restores the
texture each step. Raw frames strobe (R ≈ 2.4 at 4–16px, a random walk), and smoothing the
T = 300 trajectory with `smooth.py` gives a loop at 0.982 of the maximum that is nearly
still (R 0.13, sharpest turn 0.92).

Replacing the Langevin term with white spectrum noise (isotropic in image terms, the width
probes' unit) at σ = 0.03 and 0.1 logit RMS per step gives loops at 1.01 and 0.82 of the
maximum that drift 6 and 13 NN distances in descriptor space and keep every cell where it
was. The ascent pulls the arrangement back faster than noise can move it: the attractor is
sticky at the arrangement level even though the straight-line barrier to the next
attractor is only 10%, because in 200,000 dimensions almost no noise points along that
line. Annealing between T = 3 and 0.3 (period 1000 steps) changed nothing.

Conclusion: temperature is not a walk generator here. The "intra vs inter" dial does not
exist at the landscape level (step 4) and cannot be driven by noise at the dynamics level.

### String tours

`ring.py --anchors-from corpus.pt --anchor-ids a,b,c` builds the usual pinned ring through
chosen corpus attractors; `smooth.py` then frees the anchors. The tightest triplet in
descriptor space (9, 16, 20; summed pairwise distance 0.088) and the loosest (6, 36, 57;
0.261):

| tour | ring activation (of max, min) | after smoothing | R 4–16px after | sharpest turn |
|---|---|---|---|---|
| tightest | 0.974, 0.957 | 0.971, 0.957 | 0.04 | 0.95 |
| loosest | 0.978, 0.960 | 0.975, 0.962 | 0.04 | 0.95 |
| fresh seeds (step 2 ring) | 0.975, 0.957 | 0.971, 0.958 | 0.04 | 0.94 |

Indistinguishable in activation, smoothness, coverage and look. This is the walk-level
statement of the flat tree. Both are on the page at 1 s and at 1/8 speed (8 s).

## What this means for the project

- For these channels the ridge is one round, high-dimensional tube with uniform shallow
  dips between attractors. Attractors are not landmarks of its structure; they are random
  samples of its surface. Choosing waypoints by cluster is moot; choosing them by descriptor
  distance changes nothing measurable.
- What a walk generator can control is pace and frequency budget, not which basin it is in.
  Strings with free nodes plus the space-time cone (`smooth.py`) are the working generator.
  Low frequencies are where the ridge is widest (step 3) and where the cone lets the least
  motion through; a looser cone at low |k| is the lever for larger-scale change in a
  1-second loop, and the "longer loop" question from the handoff is really a question about
  that budget.
- Coverage of the attractor set is the wrong measure of a loop. Share of corpus variance in
  the loop's span (0.77–0.85) and effective dimension (loop 3 vs corpus 5) are the measures
  that move, and they say a loop is a thread with the right orientation.
- Two side findings worth keeping. The render is tuned to the view's borders (8–13% lost
  under a roll); if translation invariance matters for tiling or parallax, render with
  circular padding in the view. And the attractor renders are about 4% under-converged at
  384 + 384 steps.
- Not tried: other descriptor layers, k > 4, the Hessian/Lanczos check (unneeded), and the
  README GIF channels. A different channel family (object parts rather than textures) might
  have real clusters; the pipeline runs end to end in about an hour per channel.

## Practical notes

- Runs seeded from high-temperature Langevin frames ran 10–30× slower than identical runs
  from clean frames (T = 300: 17 min instead of 100 s; its `smooth.py` pass 63 min instead of
  2). Cause not found; possibly subnormal spectrum coefficients hitting CPU fallback ops.
- `run_landscape.sh` takes the interpreter from `STRIATE_PY` (worktrees have no `.venv`).
- Memory stayed calm throughout (one job at a time, caps 0.45/0.35, no pageouts beyond
  idle paging).
- Videos: `viz.py mp4` uses ffmpeg from `$FFMPEG` or an `imageio_ffmpeg` on the path; a
  scratch venv with `imageio-ffmpeg` is enough. 256px 120-frame MP4s are about 1MB.
