"""Temporal brightness stabilization: low-pass the per-frame mean luminance.

The morph's flicker lives almost entirely in global brightness (white temporal
noise from the per-frame random transforms). Form evolves slowly because
warm-start holds it continuous; brightness doesn't, so it jumps at frame rate.
This filters ONLY the luminance trajectory -- multiplicative rescale to a
smoothed mean -- leaving spatial form untouched and color ratios intact.

Post-process only; no re-render needed.
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image


def mean_luminance(path):
    a = np.asarray(Image.open(path)).astype(np.float64)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).mean()


def lowpass(x, win):
    pad = win // 2
    xp = np.pad(x, pad, mode="reflect")
    k = np.ones(win) / win
    return np.convolve(xp, k, mode="same")[pad : pad + len(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="outputs/morph_frames")
    ap.add_argument("--outdir", default="outputs/morph_stab")
    ap.add_argument("--win", type=int, default=11, help="LPF window in frames")
    ap.add_argument("--gif", default="outputs/morph_ch60_ch64_stab.gif")
    args = ap.parse_args()

    frames = sorted(glob.glob(f"{args.indir}/frame_*.png"))
    L = np.array([mean_luminance(f) for f in frames])
    Ls = lowpass(L, args.win)

    os.makedirs(args.outdir, exist_ok=True)
    thumbs = []
    for i, f in enumerate(frames):
        a = np.asarray(Image.open(f)).astype(np.float64)
        b = np.clip(a * (Ls[i] / max(L[i], 1e-6)), 0, 255).astype(np.uint8)
        Image.fromarray(b).save(f"{args.outdir}/frame_{i:04d}.png")
        thumbs.append(Image.fromarray(b).resize((800, 450)))

    g = thumbs[::2]
    g[0].save(args.gif, save_all=True, append_images=g[1:], duration=66, loop=0, optimize=True)

    flicker_before = float(np.std(np.diff(L)))
    flicker_after = float(np.std(np.diff(Ls)))
    print(f"frame-to-frame brightness jitter: {flicker_before:.3f} -> {flicker_after:.3f} "
          f"({100 * (1 - flicker_after / flicker_before):.0f}% reduction)")
    print(f"stabilized frames -> {args.outdir}   gif -> {args.gif}")


if __name__ == "__main__":
    main()
