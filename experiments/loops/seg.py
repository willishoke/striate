"""Two-level periodic watershed segmentation of a cells-like render.

Level 1: domains, bounded by the bright, wide membranes.
Level 2: cells inside each domain, bounded by the thinner walls.
Writes labels (npz), an overlay at full size and 3x crops.

Also writes the membrane band (the top 15% of the domain ridge) that cellwarp.py reads.

usage: seg.py base.png outdir [--r1 R] [--b1 B] [--r2 R] [--mb2 B] [--band-q Q]
"""

import argparse, heapq, json, os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

LUMA = np.array([0.299, 0.587, 0.114])


def gb(a, s):
    return ndi.gaussian_filter(a, s, mode="wrap")


def ridge(lum, s):
    return np.maximum(lum - gb(lum, s), 0)


def local_min_markers(R, r, mask=None):
    """Minima of R within radius r (periodic), as a label image."""
    mn = ndi.minimum_filter(R, size=2 * r + 1, mode="wrap")
    m = R <= mn + 1e-9
    if mask is not None:
        m &= mask
    lab, n = ndi.label(m)  # minima plateaus -> one marker each
    # collapse each plateau to its centroid pixel
    out = np.zeros_like(lab)
    cs = ndi.center_of_mass(m, lab, range(1, n + 1))
    for i, (y, x) in enumerate(cs, 1):
        out[int(round(y)) % R.shape[0], int(round(x)) % R.shape[1]] = i
    return out, n


def watershed(pri, markers, region=None):
    """Priority-flood watershed on a torus. If region is given, floods never cross region labels."""
    H, W = pri.shape
    lab = markers.copy()
    P = pri.tolist()
    L = lab.tolist()
    Rg = region.tolist() if region is not None else None
    heap = []
    ys, xs = np.nonzero(markers)
    for y, x in zip(ys.tolist(), xs.tolist()):
        heap.append((P[y][x], y, x))
    heapq.heapify(heap)
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        p, y, x = pop(heap)
        l = L[y][x]
        for ny, nx in ((y - 1) % H, x), ((y + 1) % H, x), (y, (x - 1) % W), (y, (x + 1) % W):
            if L[ny][nx] == 0 and (Rg is None or Rg[ny][nx] == Rg[y][x]):
                L[ny][nx] = l
                q = P[ny][nx]
                push(heap, (q if q > p else p, ny, nx))
    return np.array(L, dtype=np.int32)


def boundaries(lab):
    b = (lab != np.roll(lab, 1, 0)) | (lab != np.roll(lab, 1, 1))
    return b


def overlay(img, b1, b2, scale, box=None, c1=(255, 230, 0), c2=(0, 230, 255)):
    im = img
    B1, B2 = b1, b2
    if box is not None:
        x0, y0, w = box
        im = im[y0 : y0 + w, x0 : x0 + w]
        B1, B2 = b1[y0 : y0 + w, x0 : x0 + w], b2[y0 : y0 + w, x0 : x0 + w]
    pil = Image.fromarray(im).resize((im.shape[1] * scale, im.shape[0] * scale), Image.LANCZOS)
    arr = np.array(pil)
    if B2 is not None:
        m2 = np.kron(B2, np.ones((scale, scale), bool))
        arr[m2] = c2
    if B1 is not None:
        m1 = np.kron(B1, np.ones((scale, scale), bool))
        if scale > 1:  # 2 px line at crop scale
            m1 = ndi.binary_dilation(m1)
        arr[m1] = c1
    return Image.fromarray(arr)


def band_views(img, cells, band, outdir):
    """Random-colour label map and membrane band over the render, full size and 3x crops."""
    rng = np.random.default_rng(0)
    cols = rng.integers(60, 255, (cells.max() + 1, 3))
    lab = cols[cells].astype(np.float64)
    mix = (0.45 * lab + 0.55 * img).astype(np.uint8)
    mix[band] = (0.5 * mix[band] + 0.5 * np.array([255, 255, 255])).astype(np.uint8)
    Image.fromarray(mix).save(os.path.join(outdir, "labels_full.png"))
    bimg = img.copy()
    bimg[band] = (0.35 * bimg[band] + 0.65 * np.array([255, 240, 120])).astype(np.uint8)
    Image.fromarray(bimg).save(os.path.join(outdir, "band_full.png"))
    for i, (x0, y0, w) in enumerate(
        [(0, 0, 256), (384, 384, 256), (640, 128, 256), (128, 640, 256)]
    ):
        Image.fromarray(mix[y0 : y0 + w, x0 : x0 + w]).resize((768, 768), Image.NEAREST).save(
            os.path.join(outdir, f"crop{i}_labels.png")
        )
        Image.fromarray(bimg[y0 : y0 + w, x0 : x0 + w]).resize((768, 768), Image.LANCZOS).save(
            os.path.join(outdir, f"crop{i}_band.png")
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--s1", type=float, default=10, help="background blur for the domain ridge")
    ap.add_argument("--r1", type=int, default=24, help="marker radius for domains (px)")
    ap.add_argument(
        "--b1", type=float, default=4, help="smoothing of the domain ridge before flooding"
    )
    ap.add_argument("--s2", type=float, default=4, help="background blur for the cell ridge")
    ap.add_argument("--r2", type=int, default=8, help="marker radius for cells (px)")
    ap.add_argument("--b2", type=float, default=1.5)
    ap.add_argument(
        "--mb2", type=float, default=3.0, help="blur of the cell ridge used to find markers"
    )
    ap.add_argument(
        "--min2",
        type=int,
        default=120,
        help="drop cells smaller than this (px), merge into neighbour",
    )
    ap.add_argument(
        "--band-q",
        type=float,
        default=0.85,
        help="membrane band: pixels above this quantile of the blurred domain ridge",
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    img = np.array(Image.open(a.base).convert("RGB"))
    lum = (img.astype(np.float64) / 255 * LUMA).sum(-1)

    # level 1: domains
    rd1 = ridge(lum, a.s1)
    R1 = gb(rd1, a.b1)
    mk1, n1 = local_min_markers(R1, a.r1)
    dom = watershed(R1, mk1)

    # level 2: cells within domains
    rd2 = ridge(lum, a.s2)
    R2 = gb(rd2, a.b2)
    Rm = gb(rd2, a.mb2)
    mk2, n2 = local_min_markers(Rm, a.r2)
    cells = watershed(R2, mk2, region=dom)
    # cells never cross domain walls, but a domain with no marker stays 0: give it its own label
    z = cells == 0
    if z.any():
        extra, ne = ndi.label(z)
        cells = np.where(z, extra + cells.max(), cells)
    # merge tiny cells into the neighbour they share the most boundary with
    sizes = np.bincount(cells.ravel())
    small = np.nonzero(sizes < a.min2)[0]
    for s in small:
        if s == 0:
            continue
        m = cells == s
        ring = ndi.binary_dilation(m, structure=np.ones((3, 3)), border_value=0) & ~m
        nb = cells[ring]
        nb = nb[(nb != s)]
        if len(nb):
            cells[m] = np.bincount(nb).argmax()
    # relabel
    _, cells = np.unique(cells, return_inverse=True)
    cells = cells.reshape(dom.shape) + 1
    _, dom = np.unique(dom, return_inverse=True)
    dom = dom.reshape(cells.shape) + 1

    # membrane band: the strongest domain ridge, where cellwarp.py blends memberships more widely
    rb = gb(rd1, 2)
    band = rb > np.quantile(rb, a.band_q)

    b1, b2 = boundaries(dom), boundaries(cells)
    a1 = np.bincount(dom.ravel())[1:]
    a2 = np.bincount(cells.ravel())[1:]
    stats = dict(
        domains=int(dom.max()),
        cells=int(cells.max()),
        domain_px_median=float(np.median(a1)),
        domain_diam_median=float(2 * np.sqrt(np.median(a1) / np.pi)),
        cell_px_median=float(np.median(a2)),
        cell_diam_median=float(2 * np.sqrt(np.median(a2) / np.pi)),
        cell_diam_q10=float(2 * np.sqrt(np.quantile(a2, 0.1) / np.pi)),
        cell_diam_q90=float(2 * np.sqrt(np.quantile(a2, 0.9) / np.pi)),
        band_fraction=float(band.mean()),
        args=vars(a),
    )
    print(json.dumps(stats, indent=1))
    np.savez_compressed(os.path.join(a.out, "labels.npz"), dom=dom, cells=cells, band=band)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    overlay(img, b1, b2, 1).save(os.path.join(a.out, "overlay_full.png"))
    overlay(img, b1, None, 1).save(os.path.join(a.out, "overlay_domains.png"))
    for i, box in enumerate([(0, 0, 256), (384, 384, 256), (640, 128, 256), (128, 640, 256)]):
        overlay(img, b1, b2, 3, box).save(os.path.join(a.out, f"crop{i}.png"))
        Image.fromarray(img[box[1] : box[1] + 256, box[0] : box[0] + 256]).resize(
            (768, 768), Image.LANCZOS
        ).save(os.path.join(a.out, f"crop{i}_plain.png"))
    band_views(img, cells, band, a.out)


if __name__ == "__main__":
    main()
