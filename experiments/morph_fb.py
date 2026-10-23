"""Frequency-banded temporally-coherent critic morph: sweep band configs, pick the
best (low jitter, motion preserved, mush gone), render strip + gifs + consecutive
transition frames to judge crispness and coherence by eye."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import numpy as np
import torch
from PIL import Image

from striate.model import Backbone
from striate.transforms import TransformRobustness
from striate.util import pick_device
from keyframe import render_spectra
from make_corpus import render_group
from morph_critic import (Critic, desmear_sequence_fb, comparison_strip, save_gif, to_pil,
                          prep, CRIT_SIZE)
from morph_sweep import jitter_motion

CONFIGS = [
    dict(name="k8-t40",  k=8, lam=3, anchor_lo=40, temporal_hi=40,  anchor_hi=0),
    dict(name="k8-t120", k=8, lam=3, anchor_lo=40, temporal_hi=120, anchor_hi=0),
    dict(name="k4-t80",  k=4, lam=3, anchor_lo=40, temporal_hi=80,  anchor_hi=0),
    dict(name="k3-t120", k=3, lam=2, anchor_lo=40, temporal_hi=120, anchor_hi=0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-ch", type=int, default=42)
    ap.add_argument("--to-ch", type=int, default=90)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--k", type=int, default=8, help="band split: low-pass downsample factor")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--proj-lr", type=float, default=0.03)
    ap.add_argument("--layer", default="features.5")
    ap.add_argument("--critic", default="outputs/critic.pt")
    args = ap.parse_args()

    dev = pick_device()
    critic = Critic().to(dev)
    critic.load_state_dict(torch.load(args.critic))
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    transforms = TransformRobustness(degrees=[])
    bb = Backbone("tiny", device=dev)
    sub = bb.model.features[:int(args.layer.split(".")[1]) + 1]

    ep = render_group(bb, sub, [args.from_ch, args.to_ch], size=args.size, steps=160,
                      lr=0.05, transforms=transforms, device=dev).to(dev)
    sA, sB = ep[0:1], ep[1:2]
    ts = torch.linspace(0, 1, args.frames)
    dense = torch.cat([((1 - t) * sA + t * sB).cpu() for t in ts], dim=0)
    base_imgs = render_spectra(dense, args.size, args.size, dev)

    def mid_critic(imgs):
        mid = range(args.frames // 4, 3 * args.frames // 4)
        with torch.no_grad():
            return float(np.mean([torch.sigmoid(critic(prep(imgs[i][None].to(dev), CRIT_SIZE, transforms))).item()
                                  for i in mid]))

    bj, bd = jitter_motion(base_imgs)
    print(f"baseline (raw interp):  jitter={bj:.3f}  motion={bd:.4f}  mid-critic={mid_critic(base_imgs):.2f}")
    print(f"{'config':8s} {'jitter':>8s} {'motion':>8s} {'mid-crit':>9s}")

    results = []
    for cfg in CONFIGS:
        spec = desmear_sequence_fb(dense, critic, size=args.size, lam=cfg["lam"], anchor_lo=cfg["anchor_lo"],
                                   temporal_hi=cfg["temporal_hi"], anchor_hi=cfg["anchor_hi"], k=cfg.get("k", args.k),
                                   steps=args.steps, lr=args.proj_lr, transforms=transforms, device=dev)
        imgs = render_spectra(spec, args.size, args.size, dev)
        j, d = jitter_motion(imgs)
        mc = mid_critic(imgs)
        results.append((cfg, imgs, j, d, mc))
        print(f"{cfg['name']:8s} {j:8.3f} {d:8.4f} {mc:9.2f}")

    ok = [r for r in results if r[3] >= 0.4 * bd and r[4] < 0.3]
    pick = min(ok, key=lambda r: r[2]) if ok else min(results, key=lambda r: r[2])
    cfg, imgs = pick[0], pick[1]
    print(f"chosen: {cfg['name']}  (jitter={pick[2]:.3f} motion={pick[3]:.4f} mid-crit={pick[4]:.2f})  vs baseline jitter={bj:.3f}")

    comparison_strip(base_imgs, imgs, 6, "outputs/morph_fb_strip.png")
    save_gif(base_imgs, "outputs/morph_fb_base.gif", 20)
    save_gif(imgs, "outputs/morph_fb_best.gif", 20)
    sheet = Image.new("RGB", (2 * args.size, 2 * args.size), (12, 12, 12))
    for n, i in enumerate([9, 10, 11, 12]):
        sheet.paste(to_pil(imgs[i]), ((n % 2) * args.size, (n // 2) * args.size))
    sheet.save("outputs/morph_fb_consec.png")
    print("-> outputs/morph_fb_strip.png, morph_fb_consec.png, morph_fb_best.gif, morph_fb_base.gif")


if __name__ == "__main__":
    main()
