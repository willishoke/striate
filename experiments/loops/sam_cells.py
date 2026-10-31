"""SAM instances on a periodic canvas -> a 0/1 alpha of whole cells covering ~cover of the canvas.
Usage: sam_cells.py IN.png OUT_alpha.png cover
Needs `transformers`, which is not a project dependency (uv pip install transformers)."""

import sys
import numpy as np, torch
from PIL import Image
from transformers import pipeline

gen = pipeline("mask-generation", model="facebook/sam-vit-base", device="cpu")
im = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
S = im.shape[0]
P = S // 4
pad = np.pad(im, ((P, P), (P, P), (0, 0)), mode="wrap")  # 1.5 S, periodic context at the borders
W = 1024
out = gen(
    Image.fromarray(pad).resize((W, W), Image.LANCZOS),
    points_per_batch=64,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.9,
)
masks = [
    np.asarray(
        Image.fromarray(np.asarray(m).astype(np.uint8) * 255).resize(
            (S + 2 * P,) * 2, Image.NEAREST
        )
    )
    > 127
    for m in out["masks"]
]
alpha = np.zeros((S, S), bool)
taken = np.zeros((S, S), bool)
rng = np.random.default_rng(0)
cand = []
for m in masks:
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        continue
    cy, cx = ys.mean(), xs.mean()
    if not (
        P <= cy < P + S and P <= cx < P + S
    ):  # each instance once: its centroid in the central tile
        continue
    a = len(ys) / S / S
    if a > 0.03 or a < 0.0008:  # cells, not clusters or crumbs
        continue
    cand.append((ys, xs, a))
print(f"{len(masks)} masks, {len(cand)} cell candidates", flush=True)
cover = float(sys.argv[3])
for i in rng.permutation(len(cand)):
    ys, xs, a = cand[i]
    yy, xx = (ys - P) % S, (xs - P) % S
    if taken[yy, xx].mean() > 0.1:
        continue
    alpha[yy, xx] = True
    taken[yy, xx] = True
    if alpha.mean() >= cover:
        break
print(f"alpha coverage {alpha.mean():.2f}")
Image.fromarray((alpha * 255).astype(np.uint8)).save(sys.argv[2])
