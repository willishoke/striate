"""Joint aligned interpolation -- the Distill paper's method, plus a tunable
high-frequency temporal-smoothness term.

A batch of N frames optimized together, each with objective (1-t)*A + t*B.
Alignment from shared init + a light adjacent-frame penalty. On top, an optional
*frequency-selective* temporal smoothness: penalize the adjacent-frame change of
HIGH-frequency spectrum coefficients, which tames the jagged "feature shimmer"
without braking the mid-frequency morph -- and lives in unbounded spectrum
space, so it can't saturate like the old image-space coherence loss.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import trange

from striate.model import Backbone
from striate.param import FourierImage, _rfft2d_freqs
from striate.transforms import TransformRobustness
from striate.util import pick_device, save_image


def run_interp(bb, layer, ch_a, ch_b, *, N, size, steps, align_weight, lr,
               hf_smooth, hf_cutoff, hf_width, device):
    bb.clear_hooks()
    bb.hook(layer)
    param = FourierImage(size, size, batch=N, device=device)
    with torch.no_grad():  # shared init -> natural alignment
        param.spectrum.copy_(param.spectrum[:1].expand_as(param.spectrum).clone())

    # high-pass mask over radial frequency ( ~0 at low/mid freq, ~1 at high freq)
    f = _rfft2d_freqs(size, size)
    hp = 1.0 / (1.0 + np.exp(-(f - hf_cutoff) / hf_width))
    hp_mask = torch.tensor(hp, dtype=torch.float32, device=device)[None, None, ..., None]

    t = torch.linspace(0, 1, N, device=device)
    transforms = TransformRobustness()
    opt = torch.optim.Adam([param.spectrum], lr=lr)

    for _ in trange(steps, leave=False):
        opt.zero_grad()
        images = param.to_image()
        act = bb(transforms(images))[layer]
        a = act[:, ch_a].mean(dim=(1, 2))
        b = act[:, ch_b].mean(dim=(1, 2))
        loss = -((1 - t) * a + t * b).mean()
        loss = loss + align_weight * ((images[1:] - images[:-1]) ** 2).mean()
        if hf_smooth > 0.0:
            dspec = param.spectrum[1:] - param.spectrum[:-1]      # adjacent-frame spectrum change
            loss = loss + hf_smooth * (hp_mask * dspec ** 2).mean()  # penalize HIGH-freq change only
        loss.backward()
        opt.step()

    bb.clear_hooks()
    return param.to_image().detach()


def metric(images):
    N = images.shape[0]
    arr = [images[i].permute(1, 2, 0).cpu().numpy() for i in range(N)]
    L = np.stack([0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2] for a in arr])
    return np.abs(L[2:] - 2 * L[1:-1] + L[:-2]).mean() / np.abs(L[1:] - L[:-1]).mean()


def export(images, tag, fps):
    N = images.shape[0]
    d = f"outputs/interp_{tag}"
    os.makedirs(d, exist_ok=True)
    for i in range(N):
        save_image(images[i:i + 1], f"{d}/frame_{i:04d}.png")
    pil = [Image.open(f"{d}/frame_{i:04d}.png") for i in range(N)]
    loop = pil + pil[-2:0:-1]
    loop[0].save(f"outputs/interp_{tag}.gif", save_all=True, append_images=loop[1:],
                 duration=1000 // fps, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="features.5")
    ap.add_argument("--from-ch", type=int, default=42)
    ap.add_argument("--to-ch", type=int, default=90)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--align-weight", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--hf-smooth", default="0,10,100,1000", help="comma list to sweep")
    ap.add_argument("--hf-cutoff", type=float, default=0.25)
    ap.add_argument("--hf-width", type=float, default=0.05)
    args = ap.parse_args()

    dev = pick_device()
    bb = Backbone("tiny", device=dev)
    print(f"sweep {args.layer} ch{args.from_ch}->ch{args.to_ch}  N={args.frames} @ {args.size}px, hf-smooth sweep:")
    for hs in [float(x) for x in args.hf_smooth.split(",")]:
        imgs = run_interp(bb, args.layer, args.from_ch, args.to_ch, N=args.frames,
                          size=args.size, steps=args.steps, align_weight=args.align_weight,
                          lr=args.lr, hf_smooth=hs, hf_cutoff=args.hf_cutoff,
                          hf_width=args.hf_width, device=dev)
        tag = f"hf{hs:g}"
        export(imgs, tag, args.fps)
        print(f"  hf_smooth={hs:7g}: jitter/motion={metric(imgs):.2f}  -> outputs/interp_{tag}.gif")


if __name__ == "__main__":
    main()
