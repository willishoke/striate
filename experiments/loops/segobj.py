"""Object segmentation of a periodic render: SAM2 instances, snapped to the render's own edges.

1. SAM2 automatic masks on the wrap-padded canvas and on a grid of zoomed crops of it.
2. Instances are mapped onto the torus (each kept once: centroid in the central tile),
   filtered by area, and accepted greedily by score so they don't overlap.
3. Each accepted instance is eroded to a core; regions no instance reached become extra
   cores. A periodic watershed on the edge strength grows the cores into a full partition,
   so every boundary is redrawn along the render's own edges.

Writes labels.npz (labels, source of each label: 1 SAM, 2 gap fill), an overlay at 2x,
a colour-fill view, 3x crops and stats.json.

usage: segobj.py image.png outdir [--grid 3] [--amin 0.0008] [--amax 0.03] [--erode 3]
Needs `transformers` (SAM2), which is not a project dependency.
"""

import argparse, json, os, time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from seg import gb, watershed, boundaries


def sam_masks(gen, img, grid, size=1024):
    """Masks over img (H, W, 3 uint8) from one full pass plus grid x grid overlapping crops.
    Returns a list of (bool mask at img resolution, score)."""
    H, W = img.shape[:2]
    views = [(0, 0, H, W)]
    if grid > 1:
        c = int(round(2 * H / (grid + 1)))  # crops overlap by half
        step = (H - c) / (grid - 1)
        views += [
            (round(i * step), round(j * step), c, c) for i in range(grid) for j in range(grid)
        ]
    out = []
    for y0, x0, h, w in views:
        crop = Image.fromarray(img[y0 : y0 + h, x0 : x0 + w]).resize((size, size), Image.LANCZOS)
        res = gen(crop, points_per_batch=64, pred_iou_thresh=0.8, stability_score_thresh=0.85)
        for m, s in zip(res["masks"], res["scores"]):
            m = np.asarray(m, dtype=np.float32)
            m = np.asarray(Image.fromarray(m).resize((w, h), Image.BILINEAR)) > 0.5
            if not m.any():
                continue
            full = np.zeros((H, W), bool)
            full[y0 : y0 + h, x0 : x0 + w] = m
            # a mask cut by a crop edge is a fragment; the full pass or another crop has it whole
            if (y0 > 0 and m[0].any()) or (x0 > 0 and m[:, 0].any()):
                continue
            if (y0 + h < H and m[-1].any()) or (x0 + w < W and m[:, -1].any()):
                continue
            out.append((full, float(s)))
    return out, len(views)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("outdir")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--amin", type=float, default=0.0008, help="min instance area, fraction of canvas")
    ap.add_argument("--amax", type=float, default=0.03, help="max instance area, fraction of canvas")
    ap.add_argument("--overlap", type=float, default=0.15, help="max share of an instance already taken")
    ap.add_argument("--erode", type=int, default=3, help="core erosion in px; boundaries are redrawn within it")
    ap.add_argument("--edge-sigma", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    import torch
    from transformers import pipeline

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    gen = pipeline("mask-generation", model=a.model, device=dev)

    img = np.asarray(Image.open(a.image).convert("RGB"))
    S = img.shape[0]
    P = S // 4
    pad = np.pad(img, ((P, P), (P, P), (0, 0)), mode="wrap")
    t0 = time.time()
    masks, nviews = sam_masks(gen, pad, a.grid)
    print(f"{len(masks)} masks from {nviews} views in {time.time() - t0:.0f}s", flush=True)

    # onto the torus, each instance once
    cand = []
    for m, s in masks:
        ys, xs = np.nonzero(m)
        cy, cx = ys.mean(), xs.mean()
        if not (P <= cy < P + S and P <= cx < P + S):
            continue
        t = np.zeros((S, S), bool)
        t[(ys - P) % S, (xs - P) % S] = True
        area = t.mean()
        if a.amin <= area <= a.amax:
            cand.append((s, area, t))
    cand.sort(key=lambda c: -c[0])
    print(f"{len(cand)} candidates in the area range", flush=True)

    # greedy acceptance by score
    lab = np.zeros((S, S), np.int32)
    n = 0
    for s, area, t in cand:
        if (lab[t] > 0).mean() > a.overlap:
            continue
        n += 1
        lab[t & (lab == 0)] = n
    sam_cover = (lab > 0).mean()

    # cores: eroded instances, plus gap regions no instance reached
    core = np.zeros_like(lab)
    st = ndi.generate_binary_structure(2, 1)
    for i in range(1, n + 1):
        m = lab == i
        e = ndi.binary_erosion(m, st, iterations=a.erode)
        core[e if e.sum() >= 12 else m] = i
    reach = ndi.binary_dilation(lab > 0, st, iterations=2 * a.erode + 2)
    gaps, ng = ndi.label(~reach)
    src = {i: 1 for i in range(1, n + 1)}
    for g in range(1, ng + 1):
        m = gaps == g
        if m.mean() >= a.amin / 2:
            n += 1
            core[m] = n
            src[n] = 2

    # watershed on edge strength, periodic
    lum = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255
    col = img.astype(np.float32) / 255
    edge = np.zeros_like(lum)
    for c in range(3):
        gy = ndi.sobel(gb(col[..., c], a.edge_sigma), 0, mode="wrap")
        gx = ndi.sobel(gb(col[..., c], a.edge_sigma), 1, mode="wrap")
        edge += gx**2 + gy**2
    edge = np.sqrt(edge)
    dark = gb(lum, 6) - gb(lum, 1)  # crevices: darker than their surroundings
    pri = edge / edge.max() + 0.5 * np.maximum(dark, 0) / max(dark.max(), 1e-6)
    labels = watershed(pri.astype(np.float64), core)

    # report
    ids, counts = np.unique(labels, return_counts=True)
    areas = counts / S / S
    srcarr = np.array([src.get(int(i), 0) for i in ids])
    stats = {
        "objects": int(len(ids)),
        "from_sam": int((srcarr == 1).sum()),
        "gap_fill": int((srcarr == 2).sum()),
        "sam_cover_before_fill": round(float(sam_cover), 3),
        "area_px_median": int(np.median(counts)),
        "area_px_p10_p90": [int(np.percentile(counts, 10)), int(np.percentile(counts, 90))],
        "diameter_px_median": round(float(2 * np.sqrt(np.median(counts) / np.pi)), 1),
        "masks": len(masks),
        "candidates": len(cand),
        "args": vars(a),
    }
    print(json.dumps(stats, indent=1))
    json.dump(stats, open(f"{a.outdir}/stats.json", "w"), indent=1)
    np.savez_compressed(f"{a.outdir}/labels.npz", labels=labels, ids=ids, source=srcarr, edge=pri)

    # views
    rng = np.random.default_rng(1)
    pal = rng.integers(40, 255, (labels.max() + 1, 3)).astype(np.float32)
    for scale in (2,):
        big = np.asarray(Image.fromarray(img).resize((S * scale, S * scale), Image.LANCZOS)).copy()
        lb = np.kron(labels, np.ones((scale, scale), np.int32))
        gap = np.isin(lb, ids[srcarr == 2])
        line = np.where(gap[..., None], [255, 60, 200], [255, 240, 0]).astype(np.uint8)
        thin = boundaries(lb)
        big[thin] = line[thin]
        Image.fromarray(big).save(f"{a.outdir}/overlay.png")
        fill = (0.5 * big + 0.5 * pal[lb]).astype(np.uint8)
        fill[thin] = 0
        Image.fromarray(fill).save(f"{a.outdir}/fill.png")
    for k, (y, x) in enumerate([(S // 8, S // 8), (S // 2, S // 2)]):
        w = S // 3
        ov = np.asarray(Image.open(f"{a.outdir}/overlay.png"))
        Image.fromarray(ov[2 * y : 2 * (y + w), 2 * x : 2 * (x + w)]).resize(
            (3 * w, 3 * w), Image.LANCZOS
        ).save(f"{a.outdir}/crop{k}.png")


if __name__ == "__main__":
    main()
