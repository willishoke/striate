"""Rigorous facet clustering for the SWIM: isolate one tight homotopy class.

k-means failed because it carves convex Voronoi cells by variance -- it can't
separate a dense core from a diffuse halo (a *density* structure) or follow a
curved manifold. So:
  - features: combined squint (downsampled RGB = color/coarse form) + Gram texture
    statistics, each PCA-reduced and z-scored so they contribute equally.
  - clustering: HDBSCAN (density-based) -> dense core = a cluster, halo = noise(-1),
    no convexity assumption. The core IS the homotopy class we swim within.
  - viz: t-SNE nonlinear embedding (reveals manifold/core-halo that linear PCA hides).
The largest dense cluster is saved as the core corpus for the discriminator + swim.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE

from striate.model import Backbone
from striate.param import FourierImage
from striate.util import pick_device
from facets import gram_vecs

PALETTE = [(255, 90, 90), (90, 170, 255), (110, 230, 110), (255, 210, 80), (210, 120, 255), (120, 230, 230)]
NOISE = (110, 110, 110)


def pca(X, q):
    Xc = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vh[:min(q, Vh.shape[0])].t()


def zscore(X):
    return (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facets", default="outputs/facets_l3_84.pt")
    ap.add_argument("--squint", type=int, default=16)
    ap.add_argument("--pca-dims", type=int, default=8)
    ap.add_argument("--min-cluster", type=int, default=6)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--core-out", default="outputs/facets_core.pt")
    ap.add_argument("--out", default="outputs/facet_clusters.pt")
    args = ap.parse_args()

    dev = pick_device()
    blob = torch.load(args.facets, weights_only=False)
    spectra, size, layer, chans = blob["spectra"].to(dev), blob["size"], blob["layer"], blob["channels"]
    N = spectra.shape[0]
    bb = Backbone("tiny", device=dev)
    sub = bb.model.features[:int(layer.split(".")[1]) + 1]

    fp = FourierImage(size, size, batch=N, device=dev)
    with torch.no_grad():
        fp.spectrum.copy_(spectra)
        imgs = fp.to_image()
        squint = F.interpolate(imgs, size=(args.squint, args.squint), mode="area").reshape(N, -1).cpu()
        gram = gram_vecs(sub(bb.normalize(imgs))).cpu()
    imgs = imgs.cpu()

    feat = torch.cat([zscore(pca(squint, args.pca_dims)), zscore(pca(gram, args.pca_dims))], dim=1).numpy()
    lab = HDBSCAN(min_cluster_size=args.min_cluster, min_samples=args.min_samples).fit_predict(feat)
    emb = TSNE(n_components=2, perplexity=min(12, (N - 1) // 3), init="pca", random_state=0).fit_transform(feat)

    ids = sorted(set(lab))
    sizes = {c: int((lab == c).sum()) for c in ids}
    print(f"  HDBSCAN: " + "  ".join(f"{'noise' if c == -1 else f'c{c}'}={sizes[c]}" for c in ids))

    core_ids = [c for c in ids if c != -1]
    if core_ids:
        core = max(core_ids, key=lambda c: sizes[c])
        order = [i for i in range(N) if lab[i] == core]
        torch.save({"spectra": spectra[order].cpu(), "channels": [chans[0]] * len(order),
                    "size": size, "layer": layer}, args.core_out)
        print(f"  core = cluster {core} ({len(order)} facets) -> {args.core_out}  idx={order}")

    torch.save({"labels": torch.tensor(lab), "emb": torch.tensor(emb)}, args.out)

    # t-SNE scatter
    W = 560
    e = emb - emb.min(0)
    e = e / (e.max(0) + 1e-6) * (W - 60) + 30
    scat = Image.new("RGB", (W, W), (18, 18, 18))
    sd = ImageDraw.Draw(scat)
    for i in range(N):
        col = NOISE if lab[i] == -1 else PALETTE[int(lab[i]) % len(PALETTE)]
        sd.ellipse([e[i, 0] - 7, e[i, 1] - 7, e[i, 0] + 7, e[i, 1] + 7], fill=col)
        sd.text((e[i, 0] + 8, e[i, 1] - 6), str(i), fill=(230, 230, 230))
    scat.save(args.out.replace(".pt", "_scatter.png"))

    # grouped sheet (clusters first, noise last)
    groups = [(c, [i for i in range(N) if lab[i] == c]) for c in core_ids] + \
             ([(-1, [i for i in range(N) if lab[i] == -1])] if -1 in ids else [])
    cols = max(len(m) for _, m in groups)
    T, band = 120, 18
    sheet = Image.new("RGB", (cols * T, len(groups) * (T + band)), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for r, (c, members) in enumerate(groups):
        top = r * (T + band)
        col = NOISE if c == -1 else PALETTE[c % len(PALETTE)]
        d.text((6, top + 3), f"{'halo/noise' if c == -1 else f'cluster {c}'} (n={len(members)})", fill=col)
        for cc, i in enumerate(members):
            a = (imgs[i].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            sheet.paste(Image.fromarray(a).resize((T, T)), (cc * T, top + band))
            d.text((cc * T + 3, top + band + 1), f"{i}", fill=(255, 255, 0))
    sheet.save(args.out.replace(".pt", ".png"))
    print(f"{N} facets  scatter -> {args.out.replace('.pt', '_scatter.png')}  sheet -> {args.out.replace('.pt', '.png')}")


if __name__ == "__main__":
    main()
