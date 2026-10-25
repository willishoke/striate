"""Test the lambda theory on features.3 (ch50, the liked one): does lambda=800
clip and lambda=200 stay clean, while both stay smooth? 480p, 1s."""

import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)  # morph.py lives at the repo root

import numpy as np

from morph import measure_velocity, pacing_schedule, run_morph, strip
from striate.model import Backbone
from striate.util import pick_device


def metrics(imgs):
    arr = [im[0].permute(1, 2, 0).numpy() for im in imgs]
    L = np.stack([0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2] for a in arr])
    sec = np.abs(L[2:] - 2 * L[1:-1] + L[:-2]).mean()
    vel = np.abs(L[1:] - L[:-1]).mean()
    sh = np.mean([(a < 0.02).mean() for a in arr]) * 100
    hi = np.mean([(a > 0.98).mean() for a in arr]) * 100
    mid = np.mean([((a > 0.2) & (a < 0.8)).mean() for a in arr]) * 100
    return sec / vel, sh, hi, mid


dev = pick_device()
bb = Backbone("tiny", device=dev)
LAYER, A, B, W, H, FR = "features.3", 50, 54, 854, 480, 60

vel = measure_velocity(bb, LAYER, A, B, device=dev)
tsched = pacing_schedule(vel, FR)
print(f"theory test {LAYER} ch{A}->ch{B} @ {W}x{H}, {FR} frames:")
for name, lam in [("lambda200", 200.0), ("lambda800", 800.0)]:
    outdir = f"outputs/theory_{name}"
    imgs = run_morph(
        bb,
        LAYER,
        A,
        B,
        tsched,
        width=W,
        height=H,
        steps_per_frame=8,
        warmup=80,
        lr=0.05,
        device=dev,
        coherence_loss=lam,
        collect=True,
        outdir=outdir,
    )
    r = metrics(imgs)
    print(
        f"  {name}: jitter/motion={r[0]:.2f}  shadows={r[1]:.1f}%  highlights={r[2]:.1f}%  midtones={r[3]:.1f}%"
    )
    strip(outdir, FR, f"outputs/theory_{name}_strip.png")
print("done")
