"""Depth order of a segobj.py partition, read off the walls between touching objects.

For every pair of touching segments, two cues vote on which one is in front:

  mono  Depth Anything V2, run twice on the wrap-padded render (once rolled by half a tile,
        so every wall is far from a crop seam in one run) with its ground-plane tilt removed:
        mean disparity in a band 2-8 px inside each object along the shared wall
  rim   a rounded object darkens toward its own silhouette, so the side that darkens more
        at the wall is the front one: luma 1-5 px from the wall against luma 6-14 px in

Each cue is z-scored over all pairs and the vote is their mean. There is no ground truth, so
cues are scored against a small set of walls judged by eye (--judged, a JSON list of
[i, j, front]). On ch 7's 14 judged walls: mono 11, rim 13, outline convexity 7 (dropped);
the expected contact-shadow sign (the side behind darkens) was wrong on 13 of 14. A per-object depth is then fitted to the votes by
weighted least squares (pairs weighted by vote strength times wall length, and by
1 - same-object affinity), and every confident vote it contradicts is listed: those are the
cyclic overlaps a single depth per object cannot hold.

Writes depth.json (per-object depth, per-pair votes), mono.npy and views.

usage: objdepth.py image.png segdir
"""

import argparse, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from seg import gb

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def disk_conv(m, r):
    """Periodic convolution of a mask with a normalized disk of radius r."""
    S = m.shape[0]
    yy, xx = np.mgrid[:S, :S]
    yy, xx = np.minimum(yy, S - yy), np.minimum(xx, S - xx)
    k = (yy**2 + xx**2 <= r * r).astype(np.float64)
    k /= k.sum()
    return np.real(np.fft.ifft2(np.fft.fft2(m.astype(np.float64)) * np.fft.fft2(k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("segdir")
    ap.add_argument("--labels", default="labels_display.npy")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Base-hf")
    ap.add_argument("--judged", default="", help="JSON list of [i, j, front] judged by eye")
    a = ap.parse_args()

    import torch
    from transformers import pipeline

    L = np.load(f"{a.segdir}/{a.labels}")
    n = int(L.max())
    img = np.asarray(Image.open(a.image).convert("RGB"))
    S = img.shape[0]
    lum = img.astype(np.float32) @ LUMA / 255

    # monocular depth on the wrapped canvas, twice: as is and rolled by half a tile
    P = S // 2
    pipe = pipeline(
        "depth-estimation", model=a.model, device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    monos = []
    for sh in (0, S // 2):
        pad = np.pad(np.roll(img, (sh, sh), (0, 1)), ((P, P), (P, P), (0, 0)), mode="wrap")
        d = pipe(Image.fromarray(pad))["predicted_depth"].squeeze().float().cpu().numpy()
        d = np.asarray(Image.fromarray(d).resize((2 * P + S,) * 2, Image.BILINEAR))
        d = d - ndi.gaussian_filter(d, S / 8)  # the ground-plane tilt a tiling has no use for
        d = np.roll(d[P : P + S, P : P + S], (-sh, -sh), (0, 1))
        monos.append((d - d.mean()) / d.std())
    # distance of every pixel to each run's crop seam
    yy, xx = np.mgrid[:S, :S]
    seam = [np.minimum(np.minimum(yy, S - 1 - yy), np.minimum(xx, S - 1 - xx))]
    seam.append(np.roll(seam[0], (-(S // 2), -(S // 2)), (0, 1)))
    mono = np.where(seam[0] >= seam[1], monos[0], monos[1])
    np.save(f"{a.segdir}/mono.npy", mono)

    # per-object masks, their disk shares, and distance bands (periodic, via FFT)
    masks = [None] + [L == i for i in range(1, n + 1)]
    near = {r: [None] + [disk_conv(masks[i], r) > 1e-3 for i in range(1, n + 1)] for r in (1, 5, 8, 14)}

    aff = {}
    try:
        g = json.load(open(f"{a.segdir}/graph.json"))
        aff = {(p["i"], p["j"]): p["affinity"] for p in g["pairs"]}
    except FileNotFoundError:
        pass

    # touching pairs
    pairs = set()
    for ax in (0, 1):
        A, B = L, np.roll(L, -1, ax)
        dd = A != B
        for i, j in zip(A[dd], B[dd]):
            pairs.add((int(min(i, j)), int(max(i, j))))

    raw = []
    for i, j in sorted(pairs):
        wall = (masks[i] & near[1][j]) | (masks[j] & near[1][i])
        if wall.sum() < 12:
            continue
        bi = masks[i] & near[8][j] & ~near[1][j]
        bj = masks[j] & near[8][i] & ~near[1][i]
        if bi.sum() < 8 or bj.sum() < 8:
            continue
        r = 0 if seam[0][wall].min() >= seam[1][wall].min() else 1  # the run whose seam is farther
        mono_v = monos[r][bi].mean() - monos[r][bj].mean()  # > 0: i nearer

        def shade(k, o):
            inner = masks[k] & near[5][o]
            deep = masks[k] & near[14][o] & ~near[5][o]
            return lum[deep].mean() - lum[inner].mean() if inner.any() and deep.any() else 0.0

        rim_v = shade(i, j) - shade(j, i)  # > 0: i darkens more at the wall, its own rim: i front
        raw.append([i, j, int(wall.sum() // 2), mono_v, rim_v])

    R = np.array([r[3:] for r in raw])
    Z = R / R.std(0)
    vote = Z.mean(1)
    sg = np.sign(Z)
    agree = float((sg[:, 0] == sg[:, 1]).mean())
    judged = {}
    if a.judged:
        for i, j, f in json.load(open(a.judged)):
            judged[(min(i, j), max(i, j))] = 1 if f == min(i, j) else -1
    score = {}
    for name, col in (("mono", Z[:, 0]), ("rim", Z[:, 1]), ("vote", vote)):
        hits = [np.sign(c) == judged[(r[0], r[1])] for r, c in zip(raw, col) if (r[0], r[1]) in judged]
        score[name] = f"{sum(hits)}/{len(hits)}"

    # fit one depth per object: z_i - z_j ~ vote, weighted
    rows, rhs, w = [], [], []
    for (i, j, wl, *_), v in zip(raw, vote):
        a_ij = aff.get((i, j), 0.0)
        rows.append((i, j))
        rhs.append(np.clip(v, -2, 2))
        w.append(np.sqrt(wl) * (1 - min(1.0, a_ij / 0.25)) + 1e-3)
    M = np.zeros((len(rows) + 1, n + 1))
    for k, (i, j) in enumerate(rows):
        M[k, i], M[k, j] = w[k], -w[k]
    M[-1, 1:] = 1  # mean depth 0
    b = np.concatenate([np.array(rhs) * np.array(w), [0]])
    z = np.linalg.lstsq(M[:, 1:], b, rcond=None)[0]
    z = np.concatenate([[0], z])

    conf = np.abs(vote) > 0.5
    contra = [(int(i), int(j), round(float(v), 2)) for (i, j, *_), v, c in zip(raw, vote, conf)
              if c and np.sign(z[i] - z[j]) != np.sign(v)]

    out = {
        "objects": n,
        "pairs": len(raw),
        "cue_agreement": round(agree, 3),
        "judged_correct": score,
        "confident_pairs": int(conf.sum()),
        "contradicted_by_fit": contra,
        "z": [round(float(v), 3) for v in z],
        "votes": [
            {"i": i, "j": j, "wall_px": wl, "mono": round(float(zz[0]), 2), "rim": round(float(zz[1]), 2),
             "vote": round(float(v), 2)}
            for (i, j, wl, *_), zz, v in zip(raw, Z, vote)
        ],
    }
    json.dump(out, open(f"{a.segdir}/depth.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("z", "votes")}, indent=1))

    # views: depth anything map, fitted object depth
    def ramp(x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        c0, c1, c2 = np.array([20, 24, 60]), np.array([200, 80, 120]), np.array([255, 236, 170])
        x = x[..., None]
        return np.where(x < 0.5, c0 + (c1 - c0) * 2 * x, c1 + (c2 - c1) * (2 * x - 1)).astype(np.uint8)

    k = 2
    Image.fromarray(ramp(gb(mono, 0.5))).resize((S * k,) * 2, Image.BILINEAR).save(f"{a.segdir}/mono.png")
    zi = z[L]
    shade_img = (ramp(zi) * (0.35 + 0.65 * lum[..., None])).astype(np.uint8)  # depth colour, render shading
    edge = (L != np.roll(L, 1, 0)) | (L != np.roll(L, 1, 1))
    shade_img[edge] = 0
    Image.fromarray(shade_img).resize((S * k,) * 2, Image.NEAREST).save(f"{a.segdir}/order.png")


if __name__ == "__main__":
    main()
