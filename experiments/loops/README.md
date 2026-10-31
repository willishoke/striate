# Loops

Research code for making feature visualizations move: seamless loops of one channel, 2026-10-27 to 2026-10-30. Every approach here keeps the channel firing while the picture moves; they differ in what the motion is built from. The finished loop is the cell warp, written up in [`cellwarp.md`](cellwarp.md). The rest are the studies that led to it, kept because their results are cited in that write-up or in the main README.

Run any script from the repo root, e.g. `uv run python experiments/loops/warploop.py --help`. They import each other from this folder. Activation numbers below are channel scores (spatial mean of the channel over four fixed multi-scale crop draws); "motion explained" is the share of frame-to-frame change that optical flow accounts for (`flowfade.py`: 1 is pure transport, 0 a crossfade).

## The studies, in order

| script | idea | outcome |
|---|---|---|
| `ring.py`, `smooth.py` | ridge walk: a closed string of near-maximal images, then smoothed under a space-time speed limit | holds activation but is a crossfade: about 0 motion explained, ghosting on large-cell channels |
| `warploop.py` | transport: one still carried by a looping divergence-free swirl, re-optimized over all frames | 0.99 motion explained at 1 px/frame and no activation loss; reads as a funhouse mirror. Also makes the cell warp still |
| `warploop2.py`, `contourflow.py` | transport with motion blur, steady drift, and flows along the image's own level sets | contour currents shear thin edge bands and lose 14 to 34 % activation |
| `volslice.py` | slice a seamless 3D volume with a moving plane, so features split and merge like grain | topology events are physical and contrast swing stays low, but it scores about 7 % below the carried loops after long refinement and reads as a fade to optical flow |
| `regrow.py` | every frame its own image, tied to the previous frame carried along a steady flow | works only on gentle flows; mid-loop frames stay ghosted |
| `layers.py`, `sam_cells.py` | rigid periodic layers sliding at their own speeds, composited with learned alpha | identity kept by construction (0.9 to 0.98 motion explained) at 2 to 9 % activation; every occlusion edge costs activation, and cutting along the texture's own outlines (SAM) costs least |
| `cells.py`, `tissue_wfit.py` | an imposed cell tissue (power diagram) that divides and extrudes on a schedule | failed: the channel regrew its own smaller cells inside the imposed walls, and per-frame area solves made walls jitter |
| `seg.py`, `cellwarp.py` | segment the still into its own cells, then learn periodic per-cell drift and breathing | the gallery loop: 7.4 % activation cost at 5.7 px rms motion. See [`cellwarp.md`](cellwarp.md) |
| `lucid.py`, `hires.py` | a faithful port of the Distill paper's interpolation and diversity recipe, extended with a shared carrier image and displacement field; `hires.py` renders one of these at 4K through octaves | the paper's jointly optimized frames share only coarse layout, so a brightness-constancy loss between frames buys a dissolve; a shared carrier warped by a shared field moves (about 0.95 motion explained) at no activation cost |

## Tools

| script | what it does |
|---|---|
| `measure.py` | loop scores (activation, contrast swing, motion explained) and before/after comparisons |
| `flowfade.py` | the motion-explained measure, with a self-test (a pure shift scores about 1, a crossfade 0) |
| `content.py` | hue-swing and interior-change metrics for a cell warp loop |
| `cellwarp_viz.py`, `lucid_viz.py` | displacement maps and frame strips for `cellwarp.py` and `lucid.py` runs |
| `sheet.py` | contact sheet of stills across nine backbones; the survey that picked GoogLeNet |
| `octave.py`, `zoo.py` | octave renders across backbones, and the multi-scale crop every score uses |

`sam_cells.py` needs `transformers`, which is not a project dependency. `tissue_wfit.py` reads a `cells.py` run. Trained outputs live in `assets/`, which is not in git.
