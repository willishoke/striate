"""Two-endpoint morph with per-frame critic de-smear -- no keyframe density.

The bet: we no longer need closely-spaced keyframes (and the memory wall / windowing
that forced). We optimize *two* endpoints, linearly interpolate their spectra to M
frames (deliberately mushy), and de-smear each frame independently with the critic +
a temporal-coherence term. Memory is constant in M and in resolution -- one frame on
the GPU at a time -- so this scales to 1080p/4K without batching keyframes.

Baseline (raw interp) vs +critic projection: strips, gifs, jitter, critic profile.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import torch

from striate.model import Backbone
from striate.transforms import TransformRobustness
from striate.util import pick_device
from keyframe import render_spectra
from make_corpus import render_group
from morph_critic import (Critic, desmear_sequence, jitter, comparison_strip, save_gif, prep, CRIT_SIZE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="features.5")
    ap.add_argument("--from-ch", type=int, default=42)
    ap.add_argument("--to-ch", type=int, default=90)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--endpoint-steps", type=int, default=160)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--critic", default="outputs/critic.pt")
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--anchor", type=float, default=20.0, help="pull to interp trajectory (motion)")
    ap.add_argument("--temporal", type=float, default=30.0, help="pull to previous frame (smoothness)")
    ap.add_argument("--proj-steps", type=int, default=80)
    ap.add_argument("--proj-lr", type=float, default=0.03)
    ap.add_argument("--no-warmstart", action="store_true")
    ap.add_argument("--fps", type=int, default=24)
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
    print(f"endpoints ch {args.from_ch},{args.to_ch} @ {args.size}px  -> interp {args.frames} frames"
          f"  -> critic projection (lam={args.lam} anchor={args.anchor} temporal={args.temporal})")

    ep = render_group(bb, sub, [args.from_ch, args.to_ch], size=args.size, steps=args.endpoint_steps,
                      lr=args.lr, transforms=transforms, device=dev).to(dev)
    sA, sB = ep[0:1], ep[1:2]
    ts = torch.linspace(0, 1, args.frames)
    dense = torch.cat([((1 - t) * sA + t * sB).cpu() for t in ts], dim=0)   # the mush

    base_imgs = render_spectra(dense, args.size, args.size, dev)
    des_spec = desmear_sequence(dense, critic, size=args.size, lam=args.lam, anchor=args.anchor,
                                temporal=args.temporal, steps=args.proj_steps, lr=args.proj_lr,
                                warmstart=not args.no_warmstart, transforms=transforms, device=dev)
    des_imgs = render_spectra(des_spec, args.size, args.size, dev)

    def score(img):
        with torch.no_grad():
            return torch.stack([torch.sigmoid(critic(prep(img[None].to(dev), CRIT_SIZE, transforms)))
                                for _ in range(6)]).mean().item()

    sample = [round(k * (args.frames - 1) / 5) for k in range(6)]
    sb = [score(base_imgs[i]) for i in sample]
    sd = [score(des_imgs[i]) for i in sample]
    print(f"  jitter/motion   baseline={jitter(base_imgs):.3f}   +critic={jitter(des_imgs):.3f}   (lower=smoother)")
    print("  critic-s base   " + "  ".join(f"{x:.2f}" for x in sb))
    print("  critic-s +crit  " + "  ".join(f"{x:.2f}" for x in sd))

    os.makedirs("outputs", exist_ok=True)
    comparison_strip(base_imgs, des_imgs, 6, "outputs/morph_endpoints_strip.png")
    save_gif(base_imgs, "outputs/morph_ep_base.gif", args.fps)
    save_gif(des_imgs, "outputs/morph_ep_critic.gif", args.fps)
    print("strip -> outputs/morph_endpoints_strip.png   gifs -> outputs/morph_ep_base.gif, morph_ep_critic.gif")


if __name__ == "__main__":
    main()
