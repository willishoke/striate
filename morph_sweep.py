"""Tune the temporal-coherence balance for the two-endpoint critic morph.

Optimize the endpoints once, then try several (lam, anchor, temporal, steps)
projection configs. Report for each: jitter = high-freq flicker / motion (lower
smoother), motion = mean frame-to-frame change (if it collapses ~0 the morph
froze), and mid-critic = residual mush. We want low jitter + motion near the
baseline's + low mid-critic. Render the chosen config's strip + gif.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import numpy as np
import torch

from striate.model import Backbone
from striate.transforms import TransformRobustness
from striate.util import pick_device
from keyframe import render_spectra
from make_corpus import render_group
from morph_critic import Critic, desmear_sequence, comparison_strip, save_gif, prep, CRIT_SIZE


def jitter_motion(imgs):
    arr = imgs.permute(0, 2, 3, 1).numpy()
    L = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    num = float(np.abs(L[2:] - 2 * L[1:-1] + L[:-2]).mean())
    den = float(np.abs(L[1:] - L[:-1]).mean())
    return num / (den + 1e-9), den


CONFIGS = [
    dict(name="continuation", lam=3, anchor=25, temporal=0, steps=20),
    dict(name="balanced",     lam=3, anchor=20, temporal=60, steps=30),
    dict(name="strong-temp",  lam=2, anchor=18, temporal=140, steps=40),
    dict(name="gentle-warm",  lam=2, anchor=22, temporal=0, steps=12),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-ch", type=int, default=42)
    ap.add_argument("--to-ch", type=int, default=90)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--layer", default="features.5")
    ap.add_argument("--critic", default="outputs/critic.pt")
    ap.add_argument("--proj-lr", type=float, default=0.03)
    args = ap.parse_args()

    dev = pick_device()
    critic = Critic().to(dev)
    critic.load_state_dict(torch.load(args.critic))
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    transforms = TransformRobustness(degrees=[])
    bb = Backbone("tiny", device=dev)
    idx = int(args.layer.split(".")[1])
    sub = bb.model.features[:idx + 1]

    ep = render_group(bb, sub, [args.from_ch, args.to_ch], size=args.size, steps=160,
                      lr=0.05, transforms=transforms, device=dev).to(dev)
    sA, sB = ep[0:1], ep[1:2]
    ts = torch.linspace(0, 1, args.frames)
    dense = torch.cat([((1 - t) * sA + t * sB).cpu() for t in ts], dim=0)
    base_imgs = render_spectra(dense, args.size, args.size, dev)

    def mid_critic(imgs):
        mid = range(args.frames // 4, 3 * args.frames // 4)
        with torch.no_grad():
            s = [torch.sigmoid(critic(prep(imgs[i][None].to(dev), CRIT_SIZE, transforms))).item() for i in mid]
        return float(np.mean(s))

    bj, bd = jitter_motion(base_imgs)
    print(f"baseline (raw interp):  jitter={bj:.3f}  motion={bd:.4f}  mid-critic={mid_critic(base_imgs):.2f}")
    print(f"{'config':14s} {'jitter':>8s} {'motion':>8s} {'mid-crit':>9s}")

    results = []
    for cfg in CONFIGS:
        des_spec = desmear_sequence(dense, critic, size=args.size, lam=cfg["lam"], anchor=cfg["anchor"],
                                    temporal=cfg["temporal"], steps=cfg["steps"], lr=args.proj_lr,
                                    warmstart=True, transforms=transforms, device=dev)
        des = render_spectra(des_spec, args.size, args.size, dev)
        j, d = jitter_motion(des)
        mc = mid_critic(des)
        results.append((cfg, des, j, d, mc))
        print(f"{cfg['name']:14s} {j:8.3f} {d:8.4f} {mc:9.2f}")

    # pick: lowest jitter among configs that kept motion (>= 40% of baseline) and de-smeared (mid-crit < 0.3)
    ok = [r for r in results if r[3] >= 0.4 * bd and r[4] < 0.3]
    pick = min(ok, key=lambda r: r[2]) if ok else min(results, key=lambda r: r[2])
    cfg, des = pick[0], pick[1]
    print(f"chosen: {cfg['name']}  (jitter={pick[2]:.3f} motion={pick[3]:.4f} mid-crit={pick[4]:.2f})")
    comparison_strip(base_imgs, des, 6, "outputs/morph_sweep_strip.png")
    save_gif(base_imgs, "outputs/morph_sweep_base.gif", 20)
    save_gif(des, "outputs/morph_sweep_best.gif", 20)
    print("strip -> outputs/morph_sweep_strip.png   gifs -> morph_sweep_base.gif, morph_sweep_best.gif")


if __name__ == "__main__":
    main()
