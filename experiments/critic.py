"""Spike: the blend-alpha de-smear critic.

A small CNN regresses how *superimposed* an image is -- 0 for a clean single
feature-viz still, 1 for a 50/50 superposition -- trained on controlled
spectrum-space blends of the corpus stills (the same space the morph interpolates
in, so the synthetic smear matches the real failure). Target is 2*min(a, 1-a):
it's a "distance from either clean endpoint", peaking at the 50/50 blend. Two
tests:

  (a) does it learn?  -> MAE / correlation on held-out channels.
  (b) does its gradient de-smear a real superimposed midpoint without being
      adversarially cheated?  -> re-run the t=0.5 morph objective with the critic
      added as a loss, at a few weights, and compare to the baseline. The critic
      is run on the same transform-robustness crops it trained on, which is the
      guard against the pixel-optimizer finding a high-freq cheat.

If (b) shows the midpoint committing to a clean texture instead of ghosting --
and the critic's own score actually drops on the result -- the approach is real
and worth lifting into the full keyframe morph.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageDraw
from tqdm import trange

from striate.model import Backbone
from striate.param import FourierImage
from striate.transforms import TransformRobustness
from striate.util import pick_device


class Critic(nn.Module):
    def __init__(self, ch=32):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.GELU())

        self.body = nn.Sequential(blk(3, ch), blk(ch, ch * 2), blk(ch * 2, ch * 4),
                                  blk(ch * 4, ch * 4), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(ch * 4, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)           # logit


class OrbitAugment:
    """Euclidean-isometry augmentation: rotation + mirroring + translation only.

    These preserve the texture's intrinsic structure, so a facet's orbit under them is
    genuinely 'clean' -- and the disc must see it, or it learns a pose-rigid manifold
    and fights the structure-preserving deformation that IS the swim. Deliberately NO
    shear/scale: those are non-isometric (distort / rescale) and would corrupt the manifold.

    Translation = honest shift-and-trim: a `crop`-px window roams over the LARGER source
    canvas (real different content, no re-phasing, no seam), and across samples the roam
    covers the whole facet -- nothing optimized is discarded. Rotation then crop is
    scale-exact (1:1, no resize-blur) and corner-free given the octave margin: the window
    center may stray up to R/2 - crop*sqrt(2)/2 and still stay inside any rotation."""

    def __init__(self, crop=160, degrees=180.0, translate=60):
        self.crop, self.degrees, self.translate = crop, degrees, translate

    def __call__(self, img):
        H = img.shape[-1]
        if random.random() < 0.5:
            img = torch.flip(img, [3])          # mirror
        if random.random() < 0.5:
            img = torch.flip(img, [2])
        mid = min(H, int(self.crop * 1.5))                    # rotation-safe intermediate (>= crop*sqrt2)
        off = max(0, min(self.translate, (H - mid) // 2))
        cy = H // 2 + random.randint(-off, off)               # roaming-crop translation over the canvas
        cx = H // 2 + random.randint(-off, off)
        img = img[:, :, cy - mid // 2:cy - mid // 2 + mid, cx - mid // 2:cx - mid // 2 + mid]
        if self.degrees:                                      # rotate the small patch (cheap CPU fallback)
            img = TF.affine(img, angle=random.uniform(-self.degrees, self.degrees),
                            translate=[0, 0], scale=1.0, shear=[0.0, 0.0],
                            interpolation=TF.InterpolationMode.BILINEAR)
        o = (img.shape[-1] - self.crop) // 2
        return img[:, :, o:o + self.crop, o:o + self.crop].contiguous()   # scale-exact 1:1 final crop


def render_blend(spectra, i, j, alpha, size, device):
    """Spectrum-space blend (1-a)*spec_i + a*spec_j, rendered to [0,1] images."""
    a = alpha.view(-1, 1, 1, 1, 1)
    blended = (1 - a) * spectra[i] + a * spectra[j]
    fp = FourierImage(size, size, batch=blended.shape[0], device=device)
    fp.spectrum.copy_(blended)
    return fp.to_image()


def prep(img, crit_size, transforms=None):
    if transforms is not None:
        img = transforms(img)
    img = F.interpolate(img, size=(crit_size, crit_size), mode="bilinear", align_corners=False)
    return (img - 0.5) / 0.5


def sample_batch(spectra, pool, B, size, crit_size, transforms, device):
    ii = torch.randint(len(pool), (B,), device=pool.device)
    jj = (ii + torch.randint(1, len(pool), (B,), device=pool.device)) % len(pool)   # j != i
    i, j = pool[ii], pool[jj]
    alpha = torch.rand(B, device=device)
    with torch.no_grad():
        img = render_blend(spectra, i, j, alpha, size, device)
        x = prep(img, crit_size, transforms)
    target = 2 * torch.minimum(alpha, 1 - alpha)
    return x, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=["outputs/corpus.pt"])
    ap.add_argument("--critic-out", default="outputs/critic.pt")
    ap.add_argument("--crit-size", type=int, default=160)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--from-ch", type=int, default=42)
    ap.add_argument("--to-ch", type=int, default=90)
    ap.add_argument("--layer", default="features.5")
    ap.add_argument("--morph-size", type=int, default=192)
    ap.add_argument("--morph-steps", type=int, default=160)
    ap.add_argument("--lams", default="0,2,8")
    ap.add_argument("--out", default="outputs/desmear_sheet.png")
    args = ap.parse_args()

    dev = pick_device()
    blobs = [torch.load(c, weights_only=False) for c in args.corpus]
    spectra = torch.cat([b["spectra"] for b in blobs], dim=0).to(dev)   # (N,3,H,Wf,2)
    size, N = blobs[0]["size"], spectra.shape[0]
    nval = max(8, N // 5)
    perm = torch.randperm(N)
    val_pool, train_pool = perm[:nval].to(dev), perm[nval:].to(dev)
    transforms = OrbitAugment(crop=args.crit_size)   # isometries: teach the disc the facets' structure-preserving orbit is clean
    layers = "+".join(b.get("layer", "?") for b in blobs)
    print(f"union [{layers}]  {N} stills @ {size}px  (train {len(train_pool)} / val {len(val_pool)})  "
          f"crit {args.crit_size}px")

    critic = Critic().to(dev)
    opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    critic.train()
    for _ in trange(args.steps, desc="train critic"):
        x, target = sample_batch(spectra, train_pool, args.batch, size, args.crit_size, transforms, dev)
        loss = F.mse_loss(torch.sigmoid(critic(x)), target)
        opt.zero_grad(); loss.backward(); opt.step()
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)

    # (a) does it learn? -- held-out channels
    with torch.no_grad():
        xv, tv = sample_batch(spectra, val_pool, 256, size, args.crit_size, transforms, dev)
        pv = torch.sigmoid(critic(xv))
        mae = (pv - tv).abs().mean().item()
        corr = torch.corrcoef(torch.stack([pv, tv]))[0, 1].item()
    print(f"(a) held-out: MAE={mae:.3f}  corr={corr:.3f}  (target range [0,1])")

    torch.save(critic.state_dict(), args.critic_out)

    def critic_score(img, draws=8):
        with torch.no_grad():
            s = torch.stack([torch.sigmoid(critic(prep(img, args.crit_size, transforms)))
                             for _ in range(draws)])
        return s.mean().item()

    # (b) de-smear the REAL mush: a spectrum-interpolated frame -- which is exactly
    # what the morph's Catmull-Rom densify produces -- projected back toward the
    # clean manifold by minimizing the critic. (The single jointly-optimized midpoint
    # was the wrong object: it's coherent, and the critic rightly rates it clean. The
    # pipeline's mush is *interpolation*, and that is what we test here.)
    A, B = int(train_pool[0]), int(train_pool[1])
    sA, sB = spectra[A:A + 1], spectra[B:B + 1]
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]

    def render_spec(spec):
        fp = FourierImage(size, size, device=dev)
        with torch.no_grad():
            fp.spectrum.copy_(spec)
        return fp.to_image().detach()

    def desmear(interp_spec, lam, steps=160):
        param = FourierImage(size, size, device=dev)
        with torch.no_grad():
            param.spectrum.copy_(interp_spec)
        o = torch.optim.Adam([param.spectrum], lr=0.03)
        for _ in trange(steps, leave=False, desc=f"desmear lam={lam:g}"):
            o.zero_grad()
            (lam * torch.sigmoid(critic(prep(param.to_image(), args.crit_size, transforms)))).backward()
            o.step()
        return param.to_image().detach()

    rows = [("raw interp (mush)", []), ("de-smeared (crit only)", [])]
    for t in ts:
        interp = (1 - t) * sA + t * sB
        raw = render_spec(interp)
        des = desmear(interp, lam=5.0)
        rows[0][1].append((raw, critic_score(raw)))
        rows[1][1].append((des, critic_score(des)))
    for label, cells in rows:
        print(f"  {label:24s} " + "  ".join(f"t{ts[k]:.2f}:{cells[k][1]:.2f}" for k in range(len(ts))))

    tw, band = 224, 22
    sheet = Image.new("RGB", (len(ts) * tw, len(rows) * (tw + band)), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for r, (label, cells) in enumerate(rows):
        top = r * (tw + band)
        d.text((6, top + 4), label, fill=(255, 255, 0))
        for c, (im, sc) in enumerate(cells):
            arr = (im.cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            sheet.paste(Image.fromarray(arr).resize((tw, tw)), (c * tw, top + band))
            d.text((c * tw + 6, top + band + 4), f"t{ts[c]:.2f} s{sc:.2f}", fill=(255, 255, 0))
    sheet.save(args.out)
    print(f"sheet -> {args.out}  (top: raw spectrum-interp mush; bottom: critic-projected)")


if __name__ == "__main__":
    main()
