"""Hi-res two-endpoint frequency-banded critic morph (single config = fb-a winner),
with per-phase timing so we can size the run to a compute budget.

Run a small --frames/--proj-steps first as a probe to read sec/step, then scale.
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
from morph_critic import Critic, desmear_sequence_fb, comparison_strip, save_gif, to_pil, prep, CRIT_SIZE
from morph_sweep import jitter_motion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-ch", type=int, default=42)
    ap.add_argument("--to-ch", type=int, default=90)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--frames", type=int, default=14)
    ap.add_argument("--proj-steps", type=int, default=30)
    ap.add_argument("--endpoint-steps", type=int, default=120)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--anchor-lo", type=float, default=40.0)
    ap.add_argument("--temporal-hi", type=float, default=40.0)
    ap.add_argument("--proj-lr", type=float, default=0.03)
    ap.add_argument("--layer", default="features.5")
    ap.add_argument("--from-layer", default=None, help="layer for endpoint A (default --layer)")
    ap.add_argument("--to-layer", default=None, help="layer for endpoint B (default --layer)")
    ap.add_argument("--critic", default="outputs/critic.pt")
    ap.add_argument("--tag", default="hires")
    args = ap.parse_args()

    dev = pick_device()
    critic = Critic().to(dev)
    critic.load_state_dict(torch.load(args.critic))
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    transforms = TransformRobustness(degrees=[])
    bb = Backbone("tiny", device=dev)
    la, lb = args.from_layer or args.layer, args.to_layer or args.layer
    sub_a = bb.model.features[:int(la.split(".")[1]) + 1]
    sub_b = bb.model.features[:int(lb.split(".")[1]) + 1]
    print(f"hires {args.size}px  {la}:{args.from_ch} -> {lb}:{args.to_ch}  frames={args.frames} "
          f"proj-steps={args.proj_steps} endpoint-steps={args.endpoint_steps}")

    t0 = time.perf_counter()
    sA = render_group(bb, sub_a, [args.from_ch], size=args.size, steps=args.endpoint_steps,
                      lr=0.05, transforms=transforms, device=dev)
    sB = render_group(bb, sub_b, [args.to_ch], size=args.size, steps=args.endpoint_steps,
                      lr=0.05, transforms=transforms, device=dev)
    ep = torch.cat([sA, sB], dim=0).to(dev)
    t_ep = time.perf_counter() - t0
    print(f"  endpoints: {t_ep:5.1f}s  ({t_ep / (2 * args.endpoint_steps) * 1000:.0f} ms/step, 2x batch-1)")

    sA, sB = ep[0:1], ep[1:2]
    ts = torch.linspace(0, 1, args.frames)
    dense = torch.cat([((1 - t) * sA + t * sB).cpu() for t in ts], dim=0)

    t1 = time.perf_counter()
    des_spec = desmear_sequence_fb(dense, critic, size=args.size, lam=args.lam, anchor_lo=args.anchor_lo,
                                   temporal_hi=args.temporal_hi, anchor_hi=0.0, k=args.k,
                                   steps=args.proj_steps, lr=args.proj_lr, transforms=transforms, device=dev)
    t_des = time.perf_counter() - t1
    nstep = args.frames * args.proj_steps
    print(f"  de-smear:  {t_des:5.1f}s  ({t_des / nstep * 1000:.0f} ms/frame-step over {nstep} steps)")

    t2 = time.perf_counter()
    base_imgs = render_spectra(dense, args.size, args.size, dev, chunk=4)
    des_imgs = render_spectra(des_spec, args.size, args.size, dev, chunk=4)
    t_rend = time.perf_counter() - t2

    def mid_critic(imgs):
        mid = range(args.frames // 4, 3 * args.frames // 4)
        with torch.no_grad():
            return sum(torch.sigmoid(critic(prep(imgs[i][None].to(dev), CRIT_SIZE, transforms))).item()
                       for i in mid) / max(1, len(list(mid)))

    bj, bd = jitter_motion(base_imgs)
    dj, dd = jitter_motion(des_imgs)
    print(f"  render:    {t_rend:5.1f}s")
    print(f"  baseline jitter={bj:.3f} motion={bd:.4f} mid-crit={mid_critic(base_imgs):.2f}")
    print(f"  +critic  jitter={dj:.3f} motion={dd:.4f} mid-crit={mid_critic(des_imgs):.2f}")
    print(f"  TOTAL {t_ep + t_des + t_rend:.1f}s")

    comparison_strip(base_imgs, des_imgs, 6, f"outputs/morph_{args.tag}_strip.png")
    save_gif(des_imgs, f"outputs/morph_{args.tag}.gif", 20)
    save_image(des_imgs[0][None], f"outputs/morph_{args.tag}_A.png")
    save_image(des_imgs[args.frames // 2][None], f"outputs/morph_{args.tag}_mid.png")
    save_image(des_imgs[-1][None], f"outputs/morph_{args.tag}_B.png")
    print(f"-> outputs/morph_{args.tag}_strip.png, _A/_mid/_B.png, morph_{args.tag}.gif")


if __name__ == "__main__":
    main()
