# Handoff: measuring the ridge, and walking it on purpose

Status: exploration, nothing built yet. Written 2026-10-27 at the end of a session that
produced 120-frame "ridge-walk" loops of single channels. This document hands the next
step to a new agent: quantify how much of a channel's high-activation set a loop actually
visits, find the structure of that set (clusters of attractors and the barriers between
them), and use that structure to choose walks deliberately, within a cluster or between
clusters.

Read the whole document before running anything. The constraints section is not optional.

## The question

The user's words: "Is it possible to quantify how big the 'ridge loop' is relative to the
entire state space? I wonder if some kind of simulated annealing approach would be useful
here: find attractors in the state space, cluster them, and then constrain walks to either
be inter- or intra-cluster."

Restated as three deliverables:

1. **Coverage.** A number, or a small set of numbers, for how much of a channel's set of
   near-maximal images one loop visits.
2. **Structure.** A map of that set: which attractors (local maxima) group together, and
   how deep the activation valleys between groups are.
3. **Walks by design.** A way to generate a walk constrained to one cluster
   (intra-cluster) or one that deliberately crosses between clusters (inter-cluster), with
   temperature-style control over which.

## Vocabulary

- **Channel objective**: the mean activation of one channel at one layer, measured on
  randomly transformed views of the image (the views come from `MultiScaleCrop` in
  `zoo.py`: random scale from the whole canvas shrunk to 224 down to a 1:1 crop, then
  jitter, scale and rotation).
- **Attractor / facet**: a local maximum of that objective, reached by gradient ascent from
  a random start. Different seeds land on different attractors.
- **Ridge, or superlevel set at θ**: all images whose activation is at least θ times the
  best attractor's. "A ridge walk" is a path that stays inside it.
- **Image**: always the Fourier-parameterized image of `striate/param.py` (`FourierImage`):
  an rfft2 spectrum, scaled by 1/f, decorrelated color, sigmoid. Distances and
  interpolation happen on that spectrum unless stated otherwise.

## What exists now

### Prior findings (repo README and author's memory notes)

- **Two attractors of one channel are connected by a high-activation path, but not a
  straight one.** Between two seeds of ConvNeXt `features.5` ch 42, linear interpolation
  loses 65% of activation midway. A string-method minimum-energy path loses about 10%,
  detouring on a route about 4x longer than the straight line. Respacing the string by
  band-passed image distance evens its pacing (speed CV 0.65 to 0.08).
- **Facet corpora and clustering already exist for ConvNeXt** in `experiments/`:
  - `facets.py`: renders N distinct maxima of one channel as a batch, pushed apart by
    Gram-matrix (texture-statistics) diversity. Exports `gram_vecs`.
  - `cluster_facets.py`: clusters facets with HDBSCAN on a combined feature of a
    downsampled "squint" image (color and coarse form) plus Gram statistics, each
    PCA-reduced and z-scored. It notes that k-means failed because the facets form a dense
    core with a diffuse halo. Visualized with t-SNE.
  - `swim.py`: walks within one facet mode (a tight nearest-neighbor tour of facets,
    projected back onto the mode with a discriminator, the channel objective and
    frequency-banded temporal coherence). This is the "intra-cluster walk" idea, built
    before. Read what it found before rebuilding anything.
  - These scripts use `striate.model.Backbone`, which is ConvNeXt only. The ridge-walk
    scripts below use a separate wrapper (`zoo.py: Net`) that also loads GoogLeNet and
    ResNet-50.

### This session's loops

Scripts are in `design/ridge-landscape-scratch/` (copied from a session temp folder, not
committed). Results were published as a private comparison page:
https://claude.ai/artifact/GQmRiv1AGA3WfgxJAFXCAg

Pipeline for one channel (`run_walk.sh NET LAYER:CH TAG`):

1. `ring.py`: three anchor attractors, each rendered with "joint" octaves (224 to 1024 in
   √2 steps, all frequencies free, multi-scale views) plus two color penalties scaled by
   the channel's activation: a hinge on pre-sigmoid values past the 0.05/0.95 points, and a
   cap on pixel-scale chroma relative to luma. The anchors are joined in a ring by three
   open strings (perpendicular-only ascent, the same random view for every node at each
   step, equal band-passed image spacing), coarse to fine (4, 9, 19, 39 nodes per
   segment), giving 120 frames.
2. `smooth.py`: all 120 frames become free nodes of one closed string (no pinned anchors),
   and after every step the video's space-time spectrum is projected onto
   |ω| ≤ min(v·|k|, ω_max): a feature at spatial frequency k may change only as fast as
   motion at v pixels per frame would change it.

Measurements (activation is the channel mean over fixed view draws, averaged over frames;
"turn" is the lowest cosine between successive frame-to-frame changes in the 2–32px band;
R is the strobe ratio E[d2²]/E[d1²], 0 smooth, 2 random walk):

| channel | loop | activation | sharpest turn | R 4–16px |
|---|---|---|---|---|
| ResNet-50 V1 layer3 · 530 (anchors 2.27) | original ring | 2.14 | −0.82 | 0.53 |
| | v = 1 px/frame | 2.10 | 0.96 | 0.04 |
| | v = 0.5 px/frame | 2.02 | 0.99 | 0.01 |
| GoogLeNet inception4c · 64 (anchors 9.5) | original ring | 8.04 | −0.80 | 0.60 |
| | v = 1 px/frame | 7.84 | 0.95 | 0.05 |
| | v = 0.5 px/frame | 7.45 | 0.99 | 0.02 |

Observations that bear on this handoff:

- **Pinned maxima cause U-turns.** The path into an attractor and the path out of it both
  follow its flattest exit, so a ring through pinned attractors reverses direction at each
  one. This is a fact about the landscape near a maximum, and it will matter for any walk
  that visits attractors in sequence.
- **The ridge is wide.** Every loop stays within 5–15% of the anchors' activation while
  changing the image completely (band-passed correlation between frames 60 apart is under
  0.1). That suggests the near-maximal set is high-dimensional, with any one loop a thin
  thread through it.
- **Channels differ in what moves.** In layer3 · 530 the cells themselves rearrange. In
  inception4c · 64 (leopard-spotted fur domes) the user observed that the domes keep their
  topology and only the spots slide over them. A hypothesis: the domes are a carrier and
  the channel encodes the spot texture on curved surfaces. The ridge measurements below
  could test this.
- **The speed limit freezes large structure.** Structures above roughly 32px barely change
  in a 1-second loop at v = 1. Any walk generator that wants to travel between clusters
  needs either a longer loop or a different motion budget at low spatial frequency.

## Proposed work

Do these in order. Each step produces something the user can look at and decide on.
Start every experiment at 256px: texture scale is fixed in pixels by the network, so a
small render behaves like a crop of the same phenomenon (the strobe bench in the README
relies on the same fact). Move to 1024 only for final animations.

### Step 1: an attractor corpus with translation-invariant descriptors

- Render N = 64 attractors of one channel at 256px in batches (ordinary seeds, the joint
  view, the two color penalties). Start with ResNet-50 V1 layer3 ch 530, which the user
  has seen the most of, then GoogLeNet inception4c ch 64.
- Record for each: the spectrum, the robust activation (fixed view draws, as in
  `ring.py`'s final evaluation), and descriptors that ignore translation. The image is
  periodic, so two attractors that are the same pattern shifted are far apart in pixels
  but are the same facet. Candidates: Gram matrices of an early layer (`facets.gram_vecs`),
  the radially averaged power spectrum per color channel, and the cluster features used
  by `cluster_facets.py`. Pick one and say why.
- Check that the descriptor really is shift-invariant: roll an attractor by random
  offsets and confirm its descriptor distance to the original is far below the typical
  distance between attractors.

### Step 2: coverage of a loop

- Project the 120 frames of an existing loop and the 64 corpus attractors into the same
  descriptor space.
- Report: (a) the share of the corpus's descriptor variance lying in the subspace the
  loop's frames span; (b) for each attractor, the distance to the nearest loop frame
  divided by the median nearest-neighbor distance within the corpus; (c) the loop's
  effective dimension (participation ratio of its PCA spectrum) beside the corpus's.
- The loops exist only at 1024. Either downsample frames to 256 for descriptors, or
  re-render a 256px loop with the same scripts (`--size 256`). Use one resolution
  throughout a comparison.

### Step 3: the width of the ridge at an attractor

- From each of several attractors, probe random directions in spectrum space (with the
  1/f scaling, so a direction means a natural-looking perturbation). For each direction,
  find by bisection the step at which activation falls to 0.9 of the attractor's.
- Report the distribution of those radii. Many large radii mean many nearly free
  directions (a wide, high-dimensional plateau); a few large ones and many small ones mean
  a thin ridge. Compare channels: the "carrier plus texture" hypothesis for inception4c ·
  64 predicts large radii in directions that move spots and small ones in directions that
  reshape domes.
- A stronger tool if the probes are ambiguous: Hessian-vector products by double
  backpropagation and a Lanczos estimate of the flattest curvature directions. Translation
  should show up as two exactly flat directions, which makes a good correctness check.

### Step 4: structure, as a disconnectivity graph

This is the principled version of "find attractors and cluster them", from the
energy-landscape literature (disconnectivity graphs, as used for protein folding).

- Build a k-nearest-neighbor graph on the corpus in descriptor space (k around 4).
- For each edge, run a string between the two attractors at 256px (the open-string code in
  `ring.py`, or `strobe_bench.string_path` in the repo) and record the lowest robust
  activation along the converged path: the barrier.
- Two attractors are in the same cluster at threshold θ if some path of edges between
  them has all barriers above θ. Sweeping θ from 1 down gives a tree: clusters merge as
  the threshold drops. Draw it (attractor thumbnails at the leaves, θ on the vertical
  axis).
- The earlier 10% barrier suggests a whole channel may be one cluster at θ = 0.9 and split
  at 0.95. The tree will say.
- Cost: about 3N strings for k = 4, which is around 200 at N = 64. Measure one string's
  time at 256px before committing, and report the estimate to the user.

### Step 5: walks by design

- **Langevin dynamics as the walker.** Gradient ascent plus Gaussian noise at temperature
  T samples images with probability proportional to exp(activation / T), which is the
  natural measure on the ridge. Its trajectory is itself a ridge walk.
  - Low T wanders inside one cluster; high T hops barriers; an annealing schedule
    alternates them. That maps directly onto the inter/intra-cluster dial. Calibrate T
    against the barrier heights from step 4.
  - Raw Langevin trajectories are noisy frame to frame. Smooth them with the space-time
    speed limit from `smooth.py` (project the trajectory, not just the final frames), or
    render frames from an exponential moving average of the iterate (the README reports
    that EMA 0.97 cut strobe R from about 2 to 0.4).
  - Langevin does not return to its start. For a seamless loop, close it with a string
    from the last frame back to the first, or run a closed string seeded from the
    trajectory.
- **String tours as the alternative.** Choose waypoints from the step 4 clusters (inside
  one cluster for intra, cluster representatives for inter), connect them with strings,
  then smooth with `smooth.py`. Avoid pinning the waypoints, or the walk U-turns at each
  one (see observations above).
- Deliver a side-by-side of an intra-cluster and an inter-cluster loop of the same
  channel. The user prefers a slow, not-quite-periodic feel: at 1/8 speed the original
  loop has what they called a "quiet relabi vibe". Offer 8-second loops alongside 1-second
  ones.

## Constraints (these come from the user and from things that went wrong)

- **Memory.** The Mac has 36GB of unified memory, and these renders have pushed it into
  swap repeatedly. The user noticed and was unhappy about it.
  - Run anything that holds a whole video or chain one job at a time, never two in
    parallel.
  - Keep the MPS caps in the scripts (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45`, low 0.35):
    an overshoot then raises an out-of-memory error instead of swapping.
  - Avoid full-size temporaries: no `torch.cat`/`torch.stack` copies of whole chains,
    chunk FFTs over spatial rows, and use `ring.ChunkedAdam` instead of `torch.optim.Adam`
    for chain-sized parameters (torch's Adam allocates full-size temporaries every step).
  - MPS allocations do not show in `ps` RSS. Check `sysctl vm.swapusage` and
    `memory_pressure` early in a long run.
- **Sleep.** Wrap long runs in `caffeinate -i` so the machine doesn't idle-sleep.
- **Where things go.** Write experiment outputs to a scratch folder, never into the repo's
  `outputs/` (it holds the user's renders, which are not backed up). Don't commit or push
  unless the user asks; branch first if they do. No AI attribution lines in commits or PRs.
- **Fidelity.** The user cares whether a constraint is faithful to the network. Their line:
  path-smoothness constraints are fine ("a path smoothness constraint, not an accuracy
  constraint; we're still maximizing activations"), terms that pull images away from
  maximal activation need justification. Report robust activation beside every result,
  relative to the attractors.
- **Visual judgement.** Numbers alone have misled before (a "fix" that drove the strobe
  metric to 0.1 by saturating 40% of pixels). Look at frames, report clipped-pixel share,
  and show the user video, not just tables.

## Practical notes

- Environment: run from the repo root with `PYTHONPATH=.:design/ridge-landscape-scratch`
  and the repo's `.venv` (`uv run` works too). `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the
  scripts because some backward ops (grid sampling) are missing on MPS.
- MPS FFTs run only along trailing axes; `smooth.project` moves time last for that reason.
- Timings on the M5 Max at 1024: anchors about 2 minutes, a 120-frame ring about 7
  minutes, one `smooth.py` pass about 6–9 minutes. At 256px everything is roughly 10x
  faster.
- Video: GIF delays below 2/100 s are slowed to 10/100 s by browsers, so a 120-frame GIF
  plays at most 50fps (2.4 s). The 512px GIFs come out at 17–23MB because every pixel
  changes every frame. MP4 (H.264, 120fps) is the better format for review and for a
  GitHub README. `encode.sh` needs an ffmpeg binary; `imageio-ffmpeg` in a scratch venv
  provides one without touching the project environment.
- Artifact pages have a 15MB per-file and 256MB per-version limit. Make a fresh page per
  study rather than growing one.

## Open questions for the user

- Which channels matter most for the structure study: the ones already looped (layer3 ·
  530, inception4c · 64, inception4e · 93), or the README GIF candidates?
- Is a 1-second loop the target, or are longer loops acceptable? Inter-cluster walks need
  either more time or a looser speed limit at low spatial frequencies.
- Should a measured landscape (the disconnectivity tree) become a page of its own, as part
  of the atlas side of the project?
