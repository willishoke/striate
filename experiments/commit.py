"""POC: a commitment prior vs. superposition mush at the morph midpoint.

Background: the keyframe morph loss maximizes (1-t)*A + t*B over the layer
activation. That objective is linear, so at t=0.5 it is *maximized* by an image
that faintly shows channel A overlaid with faintly-shown channel B -- the ghosty
double-exposure that reads as "mush". Smoothness/align terms can't fix it; they
only smear that superposition evenly across time.

Hypothesis: superposition is off the feature-viz manifold, and the net is its own
manifold critic -- a committed, on-manifold image dumps its activation *energy*
into a few channels, while a superposition smears energy across many junk
channels. So add a commitment term: reward the fraction of total layer energy
sitting in the two target channels. No new network; reuses the forward pass.

This isolates the worst case (a t-sweep, midpoint = money shot) and renders two
rows -- baseline vs. +commit -- so the de-mushing is visible at a glance. If the
midpoint commits to a coherent structure instead of ghosting, the hypothesis
holds and we lift it into the keyframe pipeline (and only then consider the
heavier trained-discriminator version).
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import torch
from PIL import Image, ImageDraw
from tqdm import trange

from striate.model import Backbone
from striate.param import FourierImage
from striate.transforms import TransformRobustness
from striate.util import pick_device, save_image


def optimize_frame(bb, sub, ch_a, ch_b, *, t, commit_weight, size, steps, lr, transforms, device):
    param = FourierImage(size, size, device=device)
    opt = torch.optim.Adam([param.spectrum], lr=lr)
    for _ in trange(steps, leave=False, desc=f"t={t:.2f} c={commit_weight:g}"):
        opt.zero_grad()
        img = param.to_image()
        act = sub(bb.normalize(transforms(img)))  # (1, C, H, W)
        v = act.mean(dim=(2, 3))[0]  # signed channel means
        drive = (1 - t) * v[ch_a] + t * v[ch_b]  # existing blended objective
        loss = -drive
        if commit_weight > 0:
            e = (act**2).mean(dim=(2, 3))[0]  # per-channel energy (>=0)
            concentration = (e[ch_a] + e[ch_b]) / (e.sum() + 1e-6)
            loss = loss - commit_weight * concentration  # reward commitment
        loss.backward()
        opt.step()
    with torch.no_grad():
        img = param.to_image()
        act = sub(bb.normalize(img))
        v = act.mean(dim=(2, 3))[0]
        e = (act**2).mean(dim=(2, 3))[0]
        drive = ((1 - t) * v[ch_a] + t * v[ch_b]).item()
        conc = ((e[ch_a] + e[ch_b]) / (e.sum() + 1e-6)).item()
    return img.detach(), drive, conc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="features.5")
    ap.add_argument("--from-ch", type=int, default=42)
    ap.add_argument("--to-ch", type=int, default=90)
    ap.add_argument("--commit-weight", type=float, default=10.0)
    ap.add_argument("--tpoints", type=int, default=5)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument(
        "--no-rotate",
        action="store_true",
        help="drop rotation (the lone CPU-fallback grid_sample on MPS) for speed",
    )
    ap.add_argument("--out", default="outputs/commit_sheet.png")
    args = ap.parse_args()

    dev = pick_device()
    bb = Backbone("tiny", device=dev)
    idx = int(args.layer.split(".")[1])
    sub = bb.model.features[: idx + 1]
    transforms = TransformRobustness(degrees=[] if args.no_rotate else None)
    ts = [i / (args.tpoints - 1) for i in range(args.tpoints)]
    print(
        f"commit POC  {args.layer}  ch {args.from_ch}->{args.to_ch}  @ {args.size}px  "
        f"commit_weight={args.commit_weight}  steps={args.steps}  rotate={not args.no_rotate}"
    )

    os.makedirs("outputs/commit", exist_ok=True)
    rows = []  # (label, [(path, drive, conc), ...])
    for cw in (0.0, args.commit_weight):
        tag = "base" if cw == 0 else "commit"
        frames = []
        for t in ts:
            img, drive, conc = optimize_frame(
                bb,
                sub,
                args.from_ch,
                args.to_ch,
                t=t,
                commit_weight=cw,
                size=args.size,
                steps=args.steps,
                lr=args.lr,
                transforms=transforms,
                device=dev,
            )
            p = f"outputs/commit/{tag}_t{t:.2f}.png"
            save_image(img, p)
            frames.append((p, drive, conc))
            print(f"  {tag:6s} t={t:.2f}  drive={drive:7.3f}  conc={conc:.3f}")
        rows.append(("baseline" if cw == 0 else f"+commit({args.commit_weight:g})", frames))

    tw, band = 256, 24
    sheet = Image.new("RGB", (len(ts) * tw, len(rows) * (tw + band)), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for r, (label, frames) in enumerate(rows):
        top = r * (tw + band)
        d.text((8, top + 6), label, fill=(255, 255, 0))
        for c, (p, _drive, conc) in enumerate(frames):
            sheet.paste(Image.open(p).resize((tw, tw)), (c * tw, top + band))
            d.rectangle([c * tw + 4, top + band + 4, c * tw + 150, top + band + 22], fill=(0, 0, 0))
            d.text((c * tw + 8, top + band + 8), f"t{ts[c]:.2f} c{conc:.2f}", fill=(255, 255, 0))
    sheet.save(args.out)
    print(f"sheet -> {args.out}  (top: baseline mush, bottom: commitment prior)")


if __name__ == "__main__":
    main()
