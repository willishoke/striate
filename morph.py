"""Within-layer channel morph via warm-started continuation.

One persistent image + one persistent optimizer. Warm up on channel A, then
slide the objective A->B. Two anti-flicker mechanisms:

  * temporal coherence loss -- each frame is pulled toward the previous frame
    (lambda * MSE), so underdetermined high-frequency detail can't reshuffle
    unless the objective genuinely demands it. lambda is the smoothness dial.
  * adaptive pacing -- a cheap low-res pre-pass measures per-frame change vs t,
    then reparameterizes the t-schedule to equal perceptual arc-length, so
    frames cluster where the morph actually moves.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from striate.model import Backbone
from striate.objectives import clip_penalty
from striate.param import FourierImage, _rfft2d_freqs
from striate.transforms import TransformRobustness
from striate.util import pick_device, save_image


class InterpChannel:
    """Maximize (1-t)*chan_a + t*chan_b at a layer. ``t`` is updated per frame."""

    def __init__(self, layer, ch_a, ch_b):
        self.layer = layer
        self.a = ch_a
        self.b = ch_b
        self.t = 0.0

    def __call__(self, acts):
        act = acts[self.layer]
        return -((1.0 - self.t) * act[:, self.a].mean() + self.t * act[:, self.b].mean())


def run_morph(
    backbone,
    layer,
    ch_a,
    ch_b,
    t_schedule,
    *,
    width=1280,
    height=720,
    steps_per_frame=8,
    warmup=150,
    lr=0.05,
    device="cpu",
    coherent=True,
    batch=1,
    coherence_loss=0.0,
    clip_weight=0.0,
    freeze_cutoff=0.0,
    freeze_width=0.03,
    seed=0,
    outdir=None,
    collect=False,
    ema=0.0,
    adam_eps=1e-8,
):
    backbone.clear_hooks()
    backbone.hook(layer)
    transforms = TransformRobustness()
    param = FourierImage(height, width, device=device)
    obj = InterpChannel(layer, ch_a, ch_b)
    optimizer = torch.optim.Adam([param.spectrum], lr=lr, eps=adam_eps)

    # Iterate averaging. Frames are rendered from an EMA of the spectrum rather
    # than the raw Adam iterate, so per-step optimizer noise is low-passed in
    # time instead of landing in every frame. Optimization itself is unchanged.
    ema_spec = param.spectrum.detach().clone() if ema > 0.0 else None

    # Soft low-frequency freeze. A per-frequency gain (sigmoid in radial
    # frequency) that throttles *updates* to the lowest frequencies during the
    # morph -- so the large-scale brightness field set during warmup stays put
    # (no drift) while the feature (mid/high freq) keeps morphing. Throttles
    # updates, never drives anything, so it can't saturate. Soft, not a gate.
    freeze_mask = None
    if freeze_cutoff > 0.0:
        f = _rfft2d_freqs(height, width)
        gain = 1.0 / (1.0 + np.exp(-(f - freeze_cutoff) / freeze_width))
        freeze_mask = torch.tensor(gain, dtype=torch.float32, device=device)[None, None, ..., None]
    freezing = False

    def step(target):
        optimizer.zero_grad()
        preact = param.preact()
        image = torch.sigmoid(preact)
        loss = 0.0
        for _ in range(batch):  # average objective over a batch of transforms
            loss = loss + obj(backbone(transforms(image)))
        loss = loss / batch
        if coherence_loss > 0.0 and target is not None:
            loss = loss + coherence_loss * F.mse_loss(preact, target)
        if clip_weight > 0.0:
            loss = loss + clip_weight * clip_penalty(image)
        loss.backward()
        if freezing and freeze_mask is not None:
            param.spectrum.grad.mul_(freeze_mask)  # throttle low-freq updates
        optimizer.step()
        if ema_spec is not None:
            ema_spec.mul_(ema).add_(param.spectrum.detach(), alpha=1.0 - ema)

    obj.t = float(t_schedule[0])
    for _ in range(warmup):  # warmup optimizes ALL frequencies (establishes the field)
        step(None)
    freezing = True  # from here on, the lowest frequencies are soft-frozen
    if ema_spec is not None:
        ema_spec.copy_(param.spectrum.detach())  # start the average at the warmed-up image

    if outdir:
        os.makedirs(outdir, exist_ok=True)
    prev = prev2 = None  # pre-sigmoid history
    images = []
    n = len(t_schedule)
    for i, t in enumerate(t_schedule):
        obj.t = float(t)
        if prev2 is not None:
            target = 2.0 * prev - prev2  # constant-velocity prediction (unbounded space)
        else:
            target = prev  # first morph frame: hold position
        if coherent:
            random.seed(seed)  # replay identical augmentation sequence each frame
        for _ in range(steps_per_frame):
            step(target)
        cur = param.preact().detach()
        img = torch.sigmoid(cur if ema_spec is None else param.preact(ema_spec).detach())
        prev2, prev = prev, cur  # roll the two-frame pre-sigmoid history (raw iterate)
        if outdir:
            save_image(img, f"{outdir}/frame_{i:04d}.png")
        if collect:
            images.append(img.cpu())
        if i % 30 == 0 or i == n - 1:
            print(f"  frame {i:>3}/{n}  t={t:.3f}")

    backbone.clear_hooks()
    return images


def measure_velocity(
    backbone,
    layer,
    ch_a,
    ch_b,
    *,
    n=80,
    width=426,
    height=240,
    steps_per_frame=4,
    warmup=60,
    lr=0.05,
    device="cpu",
    coherence_loss=0.0,
):
    """Low-res linear-t pass; return per-frame consecutive-image MSE (the velocity)."""
    imgs = run_morph(
        backbone,
        layer,
        ch_a,
        ch_b,
        np.linspace(0, 1, n),
        width=width,
        height=height,
        steps_per_frame=steps_per_frame,
        warmup=warmup,
        lr=lr,
        device=device,
        coherence_loss=coherence_loss,
        collect=True,
    )
    return np.array([float(((imgs[i] - imgs[i - 1]) ** 2).mean()) for i in range(1, len(imgs))])


def pacing_schedule(velocity, n_out, smooth=5):
    """Reparameterize t to equal perceptual arc-length (constant visual velocity)."""
    v = np.convolve(velocity, np.ones(smooth) / smooth, mode="same")
    v = np.maximum(v, v.max() * 0.05)  # floor so flat stretches still advance
    t_pre = np.linspace(0, 1, len(v) + 1)
    s = np.concatenate([[0], np.cumsum(v)])
    s /= s[-1]
    return np.interp(np.linspace(0, 1, n_out), s, t_pre)


def strip(outdir, frames, out, n=5, thumb_w=384):
    th = int(thumb_w * 9 / 16)
    idxs = [round(k * (frames - 1) / (n - 1)) for k in range(n)]
    sheet = Image.new("RGB", (n * thumb_w, th), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    for col, idx in enumerate(idxs):
        thumb = Image.open(f"{outdir}/frame_{idx:04d}.png").resize((thumb_w, th))
        x = col * thumb_w
        sheet.paste(thumb, (x, 0))
        draw.rectangle([x + 4, 4, x + 78, 22], fill=(0, 0, 0))
        draw.text((x + 8, 8), f"f{idx}", fill=(255, 255, 0))
    sheet.save(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tiny")
    p.add_argument("--layer", default="features.7")
    p.add_argument("--from-ch", type=int, default=100)
    p.add_argument("--to-ch", type=int, default=104)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--steps-per-frame", type=int, default=6)
    p.add_argument("--warmup", type=int, default=120)
    p.add_argument("--coherence-loss", type=float, default=0.0)
    p.add_argument("--transform-batch", type=int, default=1)
    p.add_argument("--no-coherent", action="store_true")
    p.add_argument("--adaptive-pacing", action="store_true")
    p.add_argument(
        "--freeze-cutoff",
        type=float,
        default=0.0,
        help="soft low-freq freeze cutoff (radial freq); 0=off",
    )
    p.add_argument("--freeze-width", type=float, default=0.03)
    p.add_argument(
        "--ema",
        type=float,
        default=0.0,
        help="render frames from an EMA of the iterate (decay per step); 0=off",
    )
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--outdir", default="outputs/morph_frames")
    p.add_argument("--strip", default="outputs/morph_strip.png")
    args = p.parse_args()

    device = pick_device()
    print(
        f"device: {device}  morph {args.layer} ch{args.from_ch} -> ch{args.to_ch} "
        f"@ {args.width}x{args.height} x{args.frames}  lambda={args.coherence_loss}"
    )
    backbone = Backbone(args.model, device=device)

    t0 = time.time()
    if args.adaptive_pacing:
        print("  measuring velocity (low-res pre-pass)...")
        vel = measure_velocity(
            backbone,
            args.layer,
            args.from_ch,
            args.to_ch,
            device=device,
            coherence_loss=args.coherence_loss,
        )
        t_sched = pacing_schedule(vel, args.frames)
        print(f"  pacing: velocity ratio max/min = {vel.max() / max(vel.min(), 1e-9):.1f}")
    else:
        t_sched = np.linspace(0, 1, args.frames)

    run_morph(
        backbone,
        args.layer,
        args.from_ch,
        args.to_ch,
        t_sched,
        width=args.width,
        height=args.height,
        steps_per_frame=args.steps_per_frame,
        warmup=args.warmup,
        lr=0.05,
        device=device,
        coherent=not args.no_coherent,
        batch=args.transform_batch,
        coherence_loss=args.coherence_loss,
        freeze_cutoff=args.freeze_cutoff,
        freeze_width=args.freeze_width,
        outdir=args.outdir,
        ema=args.ema,
        adam_eps=args.adam_eps,
    )
    strip(args.outdir, args.frames, args.strip)
    print(f"done in {time.time() - t0:.0f}s  frames -> {args.outdir}  strip -> {args.strip}")


if __name__ == "__main__":
    main()
