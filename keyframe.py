"""Keyframe interpolation for smooth high-res video without holding every frame.

  1. Optimize K joint-aligned keyframes (shared init + adjacent alignment + a
     high-freq temporal-smoothness term). Memory scales with K, so at 1080p we
     use gradient checkpointing (--segments) to fit a useful keyframe batch.
  2. Densify to M frames via centripetal Catmull-Rom through the keyframe
     *spectra* -- C1-continuous (no boundary kinks), and the spline's smoothness
     is supplied by the *interpolation*: fewer keyframes = smoother (the spline
     low-passes the inherently-jaggy optimized trajectory), more keyframes =
     more "alive" feature character. Keyframe count is the smooth<->alive dial.

Densification is just FFTs, so M can be arbitrary; only K touches the GPU-heavy path.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.checkpoint import checkpoint_sequential
from tqdm import trange

from striate.model import Backbone
from striate.param import FourierImage, _rfft2d_freqs
from striate.transforms import TransformRobustness
from striate.util import pick_device, save_image


def optimize_keyframes(bb, layer, ch_a, ch_b, *, K, height, width, steps, align_weight,
                       hf_smooth, hf_cutoff, hf_width, lr, device, segments=0, rotate=True):
    idx = int(layer.split(".")[1])
    # Flatten to per-block granularity: each stage is a Sequential of CNBlocks;
    # checkpointing at the block level (not the top level) is what actually cuts
    # the peak, since stage 3 alone is 9 blocks. CNBlocks keep their internal
    # residual intact -- we only flatten the Sequential grouping, not the blocks.
    blocks = []
    for m in bb.model.features[:idx + 1]:
        blocks.extend(list(m.children()))
    sub = torch.nn.Sequential(*blocks)
    seg = min(segments, len(blocks)) if segments > 0 else 0
    param = FourierImage(height, width, batch=K, device=device)
    with torch.no_grad():  # shared init -> alignment
        param.spectrum.copy_(param.spectrum[:1].expand_as(param.spectrum).clone())
    f = _rfft2d_freqs(height, width)
    hp = 1.0 / (1.0 + np.exp(-(f - hf_cutoff) / hf_width))
    hp_mask = torch.tensor(hp, dtype=torch.float32, device=device)[None, None, ..., None]
    t = torch.linspace(0, 1, K, device=device)
    transforms = TransformRobustness(degrees=None if rotate else [])  # rotation = CPU-fallback grid_sample
    opt = torch.optim.Adam([param.spectrum], lr=lr)
    for _ in trange(steps, desc="keyframes", leave=False):
        opt.zero_grad()
        images = param.to_image()
        x = bb.normalize(transforms(images))
        act = checkpoint_sequential(sub, seg, x, use_reentrant=False) if seg > 0 else sub(x)
        a = act[:, ch_a].mean(dim=(1, 2))
        b = act[:, ch_b].mean(dim=(1, 2))
        loss = -((1 - t) * a + t * b).mean()
        loss = loss + align_weight * ((images[1:] - images[:-1]) ** 2).mean()
        if hf_smooth > 0:
            dspec = param.spectrum[1:] - param.spectrum[:-1]
            loss = loss + hf_smooth * (hp_mask * dspec ** 2).mean()
        loss.backward()
        opt.step()
    return param.spectrum.detach()


def _cr_point(p0, p1, p2, p3, local, alpha=0.5, eps=1e-5):
    def dt(a, b):
        return max((((b - a) ** 2).sum().sqrt().item()) ** alpha, eps)
    t0 = 0.0
    t1 = t0 + dt(p0, p1)
    t2 = t1 + dt(p1, p2)
    t3 = t2 + dt(p2, p3)
    t = t1 + local * (t2 - t1)
    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


def catmull_rom(keyframes, M, alpha=0.5):
    K = keyframes.shape[0]
    out = []
    for u in np.linspace(0, K - 1, M):
        i = min(int(np.floor(u)), K - 2)
        local = float(u - i)
        p0 = keyframes[max(i - 1, 0)]
        p1 = keyframes[i]
        p2 = keyframes[i + 1]
        p3 = keyframes[min(i + 2, K - 1)]
        out.append(_cr_point(p0, p1, p2, p3, local, alpha))
    return torch.stack(out)


def render_spectra(spectra, height, width, device, chunk=8):
    out = []
    for s in range(0, spectra.shape[0], chunk):
        fb = spectra[s:s + chunk].to(device)
        param = FourierImage(height, width, batch=fb.shape[0], device=device)
        with torch.no_grad():
            param.spectrum.copy_(fb)
        out.append(param.to_image().detach().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="features.5")
    ap.add_argument("--from-ch", type=int, default=42)
    ap.add_argument("--to-ch", type=int, default=90)
    ap.add_argument("--keyframes", type=int, default=15)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--segments", type=int, default=4, help="gradient-checkpoint segments (0=off)")
    ap.add_argument("--no-rotate", action="store_true", help="drop rotation (the lone CPU-fallback grid_sample)")
    ap.add_argument("--align-weight", type=float, default=10.0)
    ap.add_argument("--hf-smooth", type=float, default=100.0)
    ap.add_argument("--hf-cutoff", type=float, default=0.25)
    ap.add_argument("--hf-width", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--outdir", default="outputs/keyframe_frames")
    args = ap.parse_args()

    dev = pick_device()
    bb = Backbone("tiny", device=dev)
    print(f"{args.keyframes} keyframes @ {args.width}x{args.height} (ckpt segments={args.segments}) "
          f"-> Catmull-Rom densify to {args.frames}")
    kf = optimize_keyframes(bb, args.layer, args.from_ch, args.to_ch, K=args.keyframes,
                            height=args.height, width=args.width, steps=args.steps,
                            align_weight=args.align_weight, hf_smooth=args.hf_smooth,
                            hf_cutoff=args.hf_cutoff, hf_width=args.hf_width, lr=args.lr,
                            device=dev, segments=args.segments, rotate=not args.no_rotate)
    dense = catmull_rom(kf.cpu(), args.frames)
    imgs = render_spectra(dense, args.height, args.width, dev)

    os.makedirs(args.outdir, exist_ok=True)
    for i in range(imgs.shape[0]):
        save_image(imgs[i:i + 1], f"{args.outdir}/frame_{i:04d}.png")

    arr = [imgs[i].permute(1, 2, 0).numpy() for i in range(imgs.shape[0])]
    L = np.stack([0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2] for a in arr])
    jm = np.abs(L[2:] - 2 * L[1:-1] + L[:-2]).mean() / np.abs(L[1:] - L[:-1]).mean()

    pil = [Image.open(f"{args.outdir}/frame_{i:04d}.png") for i in range(imgs.shape[0])]
    loop = pil + pil[-2:0:-1]
    loop[0].save("outputs/keyframe.gif", save_all=True, append_images=loop[1:],
                 duration=1000 // args.fps, loop=0)

    tw = 300
    th = round(tw * args.height / args.width)
    idxs = [round(k * (len(pil) - 1) / 5) for k in range(6)]
    sheet = Image.new("RGB", (6 * tw, th), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for c, idx in enumerate(idxs):
        sheet.paste(pil[idx].resize((tw, th)), (c * tw, 0))
        d.rectangle([c * tw + 4, 4, c * tw + 60, 22], fill=(0, 0, 0))
        d.text((c * tw + 8, 8), f"f{idx}", fill=(255, 255, 0))
    sheet.save("outputs/keyframe_strip.png")
    print(f"done: {len(pil)} frames, jitter/motion={jm:.2f} -> outputs/keyframe.gif + keyframe_strip.png")


if __name__ == "__main__":
    main()
