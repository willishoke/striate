"""Same-object affinity between adjacent segments of a segobj.py partition.

Object identity is a continuum (a coil and its shell, a tip that changes colour), so instead of
merging or splitting, each pair of touching segments gets a score in [0, 1]:

  cooc  share of SAM2 masks, at every granularity, that cover most of either segment and
        most of both (a whole-shell mask covers coil and body; a coil-only mask covers one)
  edge  how weak the render's own edge is along the shared wall, relative to all walls
        (a colour change with no shading edge is weak; an occluding contour is strong)

affinity = cooc * (0.5 + 0.5 * edge). Writes graph.json and a graph overlay.

usage: objgraph.py image.png segdir [--labels labels_display.npy]
"""

import argparse, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from seg import boundaries
from segobj import sam_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("segdir")
    ap.add_argument("--labels", default="labels_display.npy")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--cover", type=float, default=0.6, help="share of a segment a mask must cover")
    a = ap.parse_args()

    import torch
    from transformers import pipeline

    L = np.load(f"{a.segdir}/{a.labels}")
    pri = np.load(f"{a.segdir}/labels.npz")["edge"]
    img = np.asarray(Image.open(a.image).convert("RGB"))
    S = img.shape[0]
    P = S // 4
    gen = pipeline(
        "mask-generation", model=a.model, device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    masks, _ = sam_masks(gen, np.pad(img, ((P, P), (P, P), (0, 0)), mode="wrap"), a.grid)

    # every mask onto the torus (duplicates across the padding are harmless for ratios)
    n = int(L.max())
    size = np.bincount(L.ravel(), minlength=n + 1).astype(np.float64)
    covers = []  # per mask: set of segments it covers
    for m, s in masks:
        ys, xs = np.nonzero(m)
        t = np.zeros((S, S), bool)
        t[(ys - P) % S, (xs - P) % S] = True
        if t.mean() > 0.12:
            continue
        hit = np.bincount(L[t], minlength=n + 1) / np.maximum(size, 1)
        covers.append((set(np.nonzero(hit >= a.cover)[0].tolist()) - {0}, s))

    # adjacency and wall strength
    walls = {}
    for ax in (0, 1):
        A, B = L, np.roll(L, -1, ax)
        e = 0.5 * (pri + np.roll(pri, -1, ax))
        d = A != B
        for i, j, v in zip(A[d], B[d], e[d]):
            k = (int(min(i, j)), int(max(i, j)))
            walls.setdefault(k, []).append(float(v))
    allw = np.concatenate([np.array(v) for v in walls.values()])
    lo, hi = np.percentile(allw, 10), np.percentile(allw, 90)

    pairs = []
    for (i, j), v in walls.items():
        if len(v) < 6:
            continue
        both = sum(s for c, s in covers if i in c and j in c)
        either = sum(s for c, s in covers if i in c or j in c)
        cooc = both / either if either else 0.0
        edge = float(np.clip((hi - np.mean(v)) / (hi - lo), 0, 1))
        pairs.append(
            {"i": i, "j": j, "wall_px": len(v), "cooc": round(cooc, 3), "edge": round(edge, 3),
             "affinity": round(cooc * (0.5 + 0.5 * edge), 3)}
        )
    pairs.sort(key=lambda p: -p["affinity"])
    json.dump({"masks": len(covers), "pairs": pairs}, open(f"{a.segdir}/graph.json", "w"), indent=1)
    for p in pairs[:25]:
        print(p)

    # overlay: walls coloured by affinity, drawn at 2x
    k = 2
    big = np.asarray(Image.fromarray(img).resize((S * k, S * k), Image.LANCZOS)).copy()
    big = (0.55 * big).astype(np.uint8)
    Lb = np.kron(L, np.ones((k, k), np.int32))
    aff = {(p["i"], p["j"]): p["affinity"] for p in pairs}
    for ax in (0, 1):
        A, B = Lb, np.roll(Lb, -1, ax)
        ys, xs = np.nonzero(A != B)
        for y, x in zip(ys, xs):
            i, j = int(A[y, x]), int(B[y, x])
            w = aff.get((min(i, j), max(i, j)), 0.0)
            c = np.array([255, 255, 255]) * (1 - w) + np.array([255, 40, 160]) * w
            big[y, x] = c if w > 0.05 else (90, 90, 90)
    pil = Image.fromarray(big)
    d = ImageDraw.Draw(pil)
    cm = ndi.center_of_mass(np.ones_like(L), L, range(1, n + 1))
    for i, (y, x) in enumerate(cm, 1):
        d.text((x * k - 6, y * k - 6), str(i), fill=(255, 255, 255))
    pil.save(f"{a.segdir}/graph.png")


if __name__ == "__main__":
    main()
