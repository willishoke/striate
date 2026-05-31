"""Critic-free morph for abstract / early layers.

Abstract patterns (gratings, ripples) blend into valid patterns rather than
ghosted mush, so no de-smear is needed -- the raw spectrum interpolation is
already smooth and clean. Just optimize two endpoints and interpolate.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time

import torch
from PIL import Image

from striate.model import Backbone
from striate.transforms import TransformRobustness
from striate.util import pick_device, save_image
from keyframe import render_spectra
from make_corpus import render_group
from morph_critic import save_gif, to_pil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="features.1")
    ap.add_argument("--from-ch", type=int, default=74)
    ap.add_argument("--to-ch", type=int, default=40)
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--endpoint-steps", type=int, default=130)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--tag", default="naive")
    args = ap.parse_args()

    dev = pick_device()
    transforms = TransformRobustness(degrees=[])
    bb = Backbone("tiny", device=dev)
    sub = bb.model.features[:int(args.layer.split(".")[1]) + 1]
    print(f"naive {args.size}px  {args.layer}:{args.from_ch} -> {args.to_ch}  "
          f"frames={args.frames}  (critic-free interpolation)")

    t0 = time.perf_counter()
    ep = render_group(bb, sub, [args.from_ch, args.to_ch], size=args.size, steps=args.endpoint_steps,
                      lr=args.lr, transforms=transforms, device=dev).to(dev)
    print(f"  endpoints: {time.perf_counter() - t0:.1f}s")

    sA, sB = ep[0:1], ep[1:2]
    ts = torch.linspace(0, 1, args.frames)
    dense = torch.cat([((1 - t) * sA + t * sB).cpu() for t in ts], dim=0)
    imgs = render_spectra(dense, args.size, args.size, dev, chunk=4)

    save_gif(imgs, f"outputs/morph_{args.tag}.gif", args.fps)
    save_image(imgs[0][None], f"outputs/morph_{args.tag}_A.png")
    save_image(imgs[args.frames // 2][None], f"outputs/morph_{args.tag}_mid.png")
    save_image(imgs[-1][None], f"outputs/morph_{args.tag}_B.png")

    n, tw = 6, 240
    idxs = [round(k * (args.frames - 1) / (n - 1)) for k in range(n)]
    strip = Image.new("RGB", (n * tw, tw), (12, 12, 12))
    for c, i in enumerate(idxs):
        strip.paste(to_pil(imgs[i]).resize((tw, tw)), (c * tw, 0))
    strip.save(f"outputs/morph_{args.tag}_strip.png")
    print(f"-> outputs/morph_{args.tag}.gif, _strip.png, _A/_mid/_B.png")


if __name__ == "__main__":
    main()
