"""Diversity-batch facet generator: N distinct local maxima of ONE channel.

Maximize the channel's activation across a batch while pushing the batch apart
(Gram-matrix / texture-statistics diversity), so the facets are different
*arrangements* of the same feature. This is the axis-2 material the per-channel
discriminator trains on and the facet-walk swims through. Output format matches
the critic's corpus schema, so critic.py trains the discriminator directly on it.
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


def gram_vecs(act):
    """(B,C,h,w) -> unit Gram vectors (B, C*C): texture statistics per facet."""
    B, C, h, w = act.shape
    A = act.reshape(B, C, h * w)
    G = torch.bmm(A, A.transpose(1, 2)) / (h * w)
    g = G.reshape(B, -1)
    return g / (g.norm(dim=1, keepdim=True) + 1e-6)


def render_facets(bb, sub, channel, count, *, size, steps, lr, div, transforms, device):
    param = FourierImage(size, size, batch=count, device=device)
    opt = torch.optim.Adam([param.spectrum], lr=lr)
    for _ in trange(steps, leave=False, desc=f"x{count}"):
        opt.zero_grad()
        act = sub(bb.normalize(transforms(param.to_image())))
        drive = act[:, channel].mean()  # all facets fire the channel
        if count > 1 and div > 0:
            g = gram_vecs(act)
            sim = g @ g.t()
            diversity = (sim - torch.eye(count, device=device)).clamp(min=0).sum() / (
                count * (count - 1)
            )
        else:
            diversity = 0.0
        (-drive + div * diversity).backward()
        opt.step()
    return param.spectrum.detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="features.3")
    ap.add_argument("--channel", type=int, default=84)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--group", type=int, default=16, help="render batch size (groups concatenated)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--div", type=float, default=1.0, help="within-group diversity weight")
    ap.add_argument("--out", default="outputs/facets.pt")
    args = ap.parse_args()

    dev = pick_device()
    bb = Backbone("tiny", device=dev)
    sub = bb.model.features[: int(args.layer.split(".")[1]) + 1]
    transforms = TransformRobustness(degrees=[])

    spectra = []
    for s in range(0, args.n, args.group):
        cnt = min(args.group, args.n - s)
        spectra.append(
            render_facets(
                bb,
                sub,
                args.channel,
                cnt,
                size=args.size,
                steps=args.steps,
                lr=args.lr,
                div=args.div,
                transforms=transforms,
                device=dev,
            )
        )
        print(f"  group {s // args.group + 1}: {cnt} facets")
    spectra = torch.cat(spectra, dim=0)
    torch.save(
        {
            "spectra": spectra,
            "channels": [args.channel] * args.n,
            "size": args.size,
            "layer": args.layer,
        },
        args.out,
    )

    fp = FourierImage(args.size, args.size, batch=args.n, device=dev)
    with torch.no_grad():
        fp.spectrum.copy_(spectra.to(dev))
        imgs = fp.to_image().cpu()
    print(f"{args.n} facets ch{args.channel} {args.layer} @ {args.size}px -> {args.out}")

    cols, T = 4, 160
    rows = (args.n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * T, rows * T), (12, 12, 12))
    for i in range(args.n):
        a = (imgs[i].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        sheet.paste(Image.fromarray(a).resize((T, T)), ((i % cols) * T, (i // cols) * T))
    sheet.save(args.out.replace(".pt", "_preview.png"))
    print(f"preview -> {args.out.replace('.pt', '_preview.png')}")


if __name__ == "__main__":
    main()
