"""Render a corpus of clean single-channel ConvNeXt feature-viz stills, saving
their *spectra* (not just PNGs).

These stills are the "clean / on-manifold" examples (smear = 0) the blend-alpha
critic trains against. Blending two of their spectra at alpha synthesizes a
controlled superposition -- and because the blend happens in the same Fourier
space the morph's Catmull-Rom interpolation uses, the synthetic smear matches the
real failure mode rather than approximating it with a pixel average. Doubles as
the first slice of the atlas.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import trange

from striate.model import Backbone
from striate.param import FourierImage
from striate.transforms import TransformRobustness
from striate.util import pick_device


def render_group(bb, sub, channels, *, size, steps, lr, transforms, device):
    B = len(channels)
    param = FourierImage(size, size, batch=B, device=device)
    chs = torch.tensor(channels, device=device)
    opt = torch.optim.Adam([param.spectrum], lr=lr)
    for _ in trange(steps, leave=False, desc=f"ch {channels[0]}..{channels[-1]}"):
        opt.zero_grad()
        img = param.to_image()
        act = sub(bb.normalize(transforms(img)))             # (B, C, h, w)
        sel = act[torch.arange(B, device=device), chs]       # (B, h, w): each elem its own channel
        (-sel.mean()).backward()
        opt.step()
    return param.spectrum.detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="features.5")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--stride", type=int, default=6, help="spacing between sampled channels")
    ap.add_argument("--group", type=int, default=16, help="render batch size")
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="outputs/corpus.pt")
    args = ap.parse_args()

    dev = pick_device()
    bb = Backbone("tiny", device=dev)
    idx = int(args.layer.split(".")[1])
    sub = bb.model.features[:idx + 1]
    transforms = TransformRobustness(degrees=[])             # no rotation (CPU fallback)
    channels = list(range(0, args.n * args.stride, args.stride))
    print(f"corpus: {len(channels)} stills  {args.layer}  @ {args.size}px  groups of {args.group}")

    spectra = []
    for s in range(0, len(channels), args.group):
        grp = channels[s:s + args.group]
        spectra.append(render_group(bb, sub, grp, size=args.size, steps=args.steps,
                                     lr=args.lr, transforms=transforms, device=dev))
        print(f"  rendered {s + len(grp)}/{len(channels)}")
    spectra = torch.cat(spectra, dim=0)                      # (N, 3, H, Wf, 2)

    os.makedirs("outputs", exist_ok=True)
    torch.save({"spectra": spectra, "channels": channels, "size": args.size,
                "layer": args.layer}, args.out)
    print(f"saved spectra {tuple(spectra.shape)} -> {args.out}")

    cols, tw = 8, 128
    rows = (len(channels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * tw), (12, 12, 12))
    fp = FourierImage(args.size, args.size, batch=len(channels), device=dev)
    with torch.no_grad():
        fp.spectrum.copy_(spectra.to(dev))
        imgs = fp.to_image().cpu()
    for i in range(len(channels)):
        arr = (imgs[i].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        sheet.paste(Image.fromarray(arr).resize((tw, tw)), ((i % cols) * tw, (i // cols) * tw))
    sheet.save("outputs/corpus_preview.png")
    print("preview -> outputs/corpus_preview.png")


if __name__ == "__main__":
    main()
