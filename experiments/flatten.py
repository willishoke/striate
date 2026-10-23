"""Spatial high-pass / illumination-flatten: remove the large-scale brightness
drift (the dodge/burn field) while keeping the feature texture.

The low spatial frequencies are the objective's nullspace -- they wander freely
and overlay a slow dodge/burn pattern on the morph. Subtracting each frame's
own heavy blur removes that field; the mid/high-frequency feature survives.

Post-process only; no re-render.
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def strip(outdir, n, out, k=5, tw=320):
    th = int(tw * 9 / 16)
    idxs = [round(j * (n - 1) / (k - 1)) for j in range(k)]
    sheet = Image.new("RGB", (k * tw, th), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for c, idx in enumerate(idxs):
        t = Image.open(f"{outdir}/frame_{idx:04d}.png").resize((tw, th))
        sheet.paste(t, (c * tw, 0))
        d.rectangle([c * tw + 4, 4, c * tw + 72, 22], fill=(0, 0, 0))
        d.text((c * tw + 8, 8), f"f{idx}", fill=(255, 255, 0))
    sheet.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="outputs/clean_frames")
    ap.add_argument("--outdir", default="outputs/flat_frames")
    ap.add_argument("--radius", type=float, default=100.0, help="blur radius (px); larger = removes only the biggest-scale field")
    ap.add_argument("--strip", default="outputs/flat_strip.png")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    frames = sorted(glob.glob(f"{args.indir}/frame_*.png"))
    for i, f in enumerate(frames):
        im = Image.open(f).convert("RGB")
        a = np.asarray(im).astype(np.float64)
        blur = np.asarray(im.filter(ImageFilter.GaussianBlur(args.radius))).astype(np.float64)
        # subtract large-scale field, add back its global mean per channel
        out = a - blur + blur.reshape(-1, 3).mean(0)
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(f"{args.outdir}/frame_{i:04d}.png")

    strip(args.outdir, len(frames), args.strip)
    print(f"flattened {len(frames)} frames (radius={args.radius}) -> {args.outdir}  strip -> {args.strip}")


if __name__ == "__main__":
    main()
