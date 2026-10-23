"""End-to-end test: does the blend-alpha critic de-smear an actual morph -- without
breaking temporal smoothness?

  1. optimize K joint-aligned keyframes (reuse keyframe.optimize_keyframes)
  2. Catmull-Rom densify to M frame spectra  (this is where the mush lives)
  3a. baseline: render the densified spectra directly (current best pipeline)
  3b. +critic: a *temporally-coherent* projection -- de-smear each frame by minimizing
      the critic, warm-started from the previous de-smeared frame and anchored (image
      space) to its interpolated trajectory position. The warm-start + anchor are what
      stop the per-frame de-fog from reintroducing flicker.

Reports both strips + gifs, the jitter/motion ratio for each (lower = smoother), and
the critic-score profile, so we see whether de-smearing fixes the video and what it
costs in time-smoothness.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from striate.model import Backbone
from striate.param import FourierImage
from striate.transforms import TransformRobustness
from striate.util import pick_device
from keyframe import optimize_keyframes, catmull_rom, render_spectra
from critic import Critic, prep

CRIT_SIZE = 160


def render_one(spec, size, device):
    fp = FourierImage(size, size, device=device)
    with torch.no_grad():
        fp.spectrum.copy_(spec)
    return fp.to_image().detach()


def desmear_sequence(dense, critic, *, size, lam, anchor, steps, lr, warmstart, transforms, device,
                     temporal=0.0):
    """Temporally-coherent critic projection over a sequence of (mushy) interp frames.

    Per frame, three forces: (lam) the critic de-fogs toward the clean manifold;
    (anchor) an image-space pull to this frame's interpolated trajectory position
    keeps it from drifting/collapsing to an endpoint; (temporal) a pull to the
    previous *de-smeared* frame keeps the sequence smooth. Warm-starting from the
    previous frame gives a coherent starting point. anchor vs temporal is the
    motion<->smoothness balance."""
    out, prev_spec, prev_img = [], None, None
    for k in range(dense.shape[0]):
        interp = dense[k:k + 1].to(device)
        interp_img = render_one(interp, size, device)
        param = FourierImage(size, size, device=device)
        with torch.no_grad():
            param.spectrum.copy_(prev_spec if (warmstart and prev_spec is not None) else interp)
        opt = torch.optim.Adam([param.spectrum], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            img = param.to_image()
            loss = (lam * torch.sigmoid(critic(prep(img, CRIT_SIZE, transforms)))
                    + anchor * ((img - interp_img) ** 2).mean())
            if temporal > 0 and prev_img is not None:
                loss = loss + temporal * ((img - prev_img) ** 2).mean()
            loss.backward()
            opt.step()
        prev_spec = param.spectrum.detach().clone()
        prev_img = param.to_image().detach()
        out.append(prev_spec.cpu())
    return torch.cat(out, dim=0)


def _lowpass(img, k):
    """Crude low-pass: downsample by k then upsample back. k sets the band split."""
    h, w = img.shape[-2:]
    small = F.interpolate(img, size=(max(1, h // k), max(1, w // k)), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def desmear_sequence_fb(dense, critic, *, size, lam, anchor_lo, temporal_hi, anchor_hi,
                        k, steps, lr, transforms, device):
    """Frequency-banded temporally-coherent projection.

    Low band -> the smooth interpolation (stable A->B trajectory, temporally smooth
    for free). High band -> the previous de-smeared frame (crisp detail, coherent,
    NOT pulled toward the grey interp -- this is the de-muting). Optional anchor_hi
    lets the high band also track the interp a little, for a more directed morph.
    Critic de-fogs throughout. Warm-start from the previous frame's spectrum."""
    out, prev_spec, prev_hi = [], None, None
    for fi in range(dense.shape[0]):
        interp = dense[fi:fi + 1].to(device)
        interp_img = render_one(interp, size, device)
        interp_lo = _lowpass(interp_img, k)
        interp_hi = interp_img - interp_lo
        param = FourierImage(size, size, device=device)
        with torch.no_grad():
            param.spectrum.copy_(prev_spec if prev_spec is not None else interp)
        opt = torch.optim.Adam([param.spectrum], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            img = param.to_image()
            lo = _lowpass(img, k)
            hi = img - lo
            loss = (lam * torch.sigmoid(critic(prep(img, CRIT_SIZE, transforms)))
                    + anchor_lo * ((lo - interp_lo) ** 2).mean())
            if anchor_hi > 0:
                loss = loss + anchor_hi * ((hi - interp_hi) ** 2).mean()
            if temporal_hi > 0 and prev_hi is not None:
                loss = loss + temporal_hi * ((hi - prev_hi) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = param.to_image()
            prev_hi = (final - _lowpass(final, k)).detach()
        prev_spec = param.spectrum.detach().clone()
        out.append(prev_spec.cpu())
    return torch.cat(out, dim=0)


def jitter(imgs):
    arr = imgs.permute(0, 2, 3, 1).numpy()
    L = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(np.abs(L[2:] - 2 * L[1:-1] + L[:-2]).mean() / (np.abs(L[1:] - L[:-1]).mean() + 1e-9))


def to_pil(t):
    return Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def comparison_strip(base, des, n, path):
    M = base.shape[0]
    idxs = [round(k * (M - 1) / (n - 1)) for k in range(n)]
    tw, band = 224, 22
    sheet = Image.new("RGB", (n * tw, 2 * (tw + band)), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for r, (label, imgs) in enumerate([("baseline (raw densify)", base), ("+critic projection", des)]):
        top = r * (tw + band)
        d.text((6, top + 4), label, fill=(255, 255, 0))
        for c, i in enumerate(idxs):
            sheet.paste(to_pil(imgs[i]).resize((tw, tw)), (c * tw, top + band))
    sheet.save(path)


def save_gif(imgs, path, fps=24):
    pil = [to_pil(imgs[i]) for i in range(imgs.shape[0])]
    loop = pil + pil[-2:0:-1]
    loop[0].save(path, save_all=True, append_images=loop[1:], duration=1000 // fps, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="features.5")
    ap.add_argument("--from-ch", type=int, default=42)
    ap.add_argument("--to-ch", type=int, default=90)
    ap.add_argument("--keyframes", type=int, default=6)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--kf-steps", type=int, default=140)
    ap.add_argument("--align-weight", type=float, default=10.0)
    ap.add_argument("--hf-smooth", type=float, default=100.0)
    ap.add_argument("--hf-cutoff", type=float, default=0.25)
    ap.add_argument("--hf-width", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.05)
    # critic projection
    ap.add_argument("--critic", default="outputs/critic.pt")
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--anchor", type=float, default=60.0)
    ap.add_argument("--proj-steps", type=int, default=60)
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
    print(f"keyframes K={args.keyframes} @ {args.size}px  {args.layer} ch {args.from_ch}->{args.to_ch}"
          f"  -> densify {args.frames}  -> critic projection (lam={args.lam} anchor={args.anchor})")
    kf = optimize_keyframes(bb, args.layer, args.from_ch, args.to_ch, K=args.keyframes,
                            height=args.size, width=args.size, steps=args.kf_steps,
                            align_weight=args.align_weight, hf_smooth=args.hf_smooth,
                            hf_cutoff=args.hf_cutoff, hf_width=args.hf_width, lr=args.lr,
                            device=dev, segments=0, rotate=False)
    dense = catmull_rom(kf.cpu(), args.frames)

    base_imgs = render_spectra(dense, args.size, args.size, dev)
    des_spec = desmear_sequence(dense, critic, size=args.size, lam=args.lam, anchor=args.anchor,
                                steps=args.proj_steps, lr=args.proj_lr,
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
    comparison_strip(base_imgs, des_imgs, 6, "outputs/morph_critic_strip.png")
    save_gif(base_imgs, "outputs/morph_base.gif", args.fps)
    save_gif(des_imgs, "outputs/morph_critic.gif", args.fps)
    print("strip -> outputs/morph_critic_strip.png   gifs -> outputs/morph_base.gif, morph_critic.gif")


if __name__ == "__main__":
    main()
