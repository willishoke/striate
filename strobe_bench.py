"""Strobe bench: a cheap low-res proxy for temporal artifacts in morph paths.

Texture scale is fixed in pixels by the network, not relative to the frame, so a
small render is effectively a crop of the same phenomenon -- band metrics in px
transfer directly to 1080p. Each config reports:

  R      E[d2^2]/E[d1^2] per band: 0 = smooth motion, 2 = random walk, 3 = i.i.d. flicker
  flip   rate of frame-difference sign reversals (above-median moves)
  CV     std/mean of per-frame speed (uneven pacing / surges); pk = max/median speed
  drop   worst activation along the path vs the endpoints
  sat    fraction of saturated pixels

Methods: ``morph`` (morph.py warm-start continuation) and ``string`` (coarse-to-fine
string-method minimum-energy path between two same-channel attractors), with the
string's arc length measured either in spectrum space or in band-passed image space.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import random
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

import morph
from striate.model import Backbone
from striate.param import FourierImage
from striate.transforms import TransformRobustness
from striate.util import pick_device

warnings.filterwarnings("ignore")
LUMA = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
BANDS = [("0-4px", 0.5, 2), ("4-16px", 2, 8), ("16-64px", 8, 32)]


def blur(x, s):
    k = int(3 * s) * 2 + 1
    g = torch.exp(-((torch.arange(k, device=x.device) - k // 2) ** 2) / (2 * s * s))
    g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


def bandpass(frames, lo, hi):
    lum = (frames * LUMA.to(frames.device)).sum(1, keepdim=True)
    return blur(lum, lo) - blur(lum, hi)


def temporal_report(frames):
    """frames: (T, 3, H, W) in [0, 1] on CPU."""
    out = {
        "sat": float(((frames < 0.02) | (frames > 0.98)).float().mean()),
        "detail": float(bandpass(frames, 0.5, 2).std()),
    }  # fine-detail contrast: catches averaging blur
    for name, lo, hi in BANDS:
        b = bandpass(frames, lo, hi)
        d1 = b[1:] - b[:-1]
        d2 = d1[1:] - d1[:-1]
        big = d1[1:].abs() > d1.abs().median()
        v = d1.pow(2).mean((1, 2, 3)).sqrt()
        out[name] = dict(
            R=float(d2.pow(2).mean() / d1.pow(2).mean()),
            flip=float(((torch.sign(d1[1:]) != torch.sign(d1[:-1])) & big).sum() / big.sum()),
            CV=float(v.std() / v.mean()),
            pk=float(v.max() / v.median()),
        )
    return out


class Scorer:
    def __init__(self, bb, layer, ch):
        self.bb, self.layer, self.ch = bb, layer, ch
        self.tf = TransformRobustness()

    def act(self, img):
        return self.bb(self.tf(img))[self.layer][:, self.ch].mean((1, 2))

    @torch.no_grad()
    def robust(self, frames, mb=16, draws=4):
        """Activation averaged over fixed transform draws (comparable across methods)."""
        tot = 0.0
        for k in range(draws):
            random.seed(1000 + k)
            tot = tot + torch.cat([self.act(frames[i : i + mb]) for i in range(0, len(frames), mb)])
        return (tot / draws).cpu().numpy()


def string_path(
    sc,
    *,
    width,
    height,
    seeds=(1, 2),
    endpoint_steps=512,
    schedule="6:150,13:50,27:25,55:15",
    lr=0.02,
    mb=16,
    arc="spectrum",
    arc_band=(2, 32),
    device="cpu",
):
    """Coarse-to-fine string method. ``arc`` = 'spectrum' (L2 of spectrum params) or
    'image' (L2 of band-passed luminance) decides what 'equal spacing' means."""
    helper = FourierImage(height, width, device=device)

    def image(spec):
        z = torch.view_as_complex((spec * helper.spectrum_scale).contiguous())
        pre = torch.fft.irfftn(z, s=(height, width), dim=(2, 3)) / 4.0
        return torch.sigmoid(helper._decorrelate_color(pre))

    ends = []
    for sd in seeds:
        torch.manual_seed(sd)
        p = FourierImage(height, width, device=device)
        opt = torch.optim.Adam([p.spectrum], lr=0.05)
        for _ in range(endpoint_steps):
            opt.zero_grad()
            (-sc.act(p.to_image()).sum()).backward()
            opt.step()
        ends.append(p.spectrum.detach())
    A, B = ends

    @torch.no_grad()
    def seglengths(chain):
        if arc == "spectrum":
            return (chain[1:] - chain[:-1]).flatten(1).norm(dim=1)
        b = torch.cat(
            [bandpass(image(chain[i : i + mb]), *arc_band) for i in range(0, len(chain), mb)]
        )
        return (b[1:] - b[:-1]).flatten(1).norm(dim=1)

    @torch.no_grad()
    def respace(X):
        chain = torch.cat([A, X, B])
        s = torch.cat([chain.new_zeros(1), seglengths(chain).cumsum(0)])
        s = s / s[-1]
        out = []
        for u in torch.linspace(0, 1, len(X) + 2, device=device)[1:-1]:
            j = int(torch.searchsorted(s, u).clamp(1, len(s) - 1))
            w = (u - s[j - 1]) / (s[j] - s[j - 1] + 1e-12)
            out.append(chain[j - 1] + w * (chain[j] - chain[j - 1]))
        return torch.stack(out)

    X = None
    for level in schedule.split(","):
        n, steps = map(int, level.split(":"))
        if X is None:
            X = A + torch.linspace(0, 1, n + 2, device=device)[1:-1].view(-1, 1, 1, 1, 1) * (B - A)
        else:  # insert midpoints
            chain = torch.cat([A, X, B])
            mids = (chain[1:] + chain[:-1]) / 2
            X = torch.stack([v for pair in zip(chain[:-1], mids) for v in pair] + [chain[-1]])[1:-1]
            assert len(X) == n, (len(X), n)
        X = torch.nn.Parameter(X.clone())
        opt = torch.optim.Adam([X], lr=lr)
        for it in range(steps):
            opt.zero_grad()
            seed = random.randrange(1 << 30)
            for i in range(0, n, mb):
                random.seed(seed)  # same transform for every node this step
                (-sc.act(image(X[i : i + mb])).sum()).backward()
            with torch.no_grad():  # climb only perpendicular to the path
                chain = torch.cat([A, X, B])
                tau = (chain[2:] - chain[:-2]).flatten(1)
                tau = tau / tau.norm(dim=1, keepdim=True)
                g = X.grad.flatten(1)
                X.grad.copy_((g - (g * tau).sum(1, keepdim=True) * tau).view_as(X))
            opt.step()
            if it % 10 == 9:
                X.data = respace(X.data)
        X = X.detach()
    if arc != "spectrum":
        X = respace(X)  # final spacing in the measured metric
    chain = torch.cat([A, X, B])
    with torch.no_grad():
        return torch.cat([image(chain[i : i + mb]) for i in range(0, len(chain), mb)]).cpu()


def morph_path(sc, *, width, height, frames=57, device="cpu", **kw):
    imgs = morph.run_morph(
        sc.bb,
        sc.layer,
        kw.pop("ch_a"),
        kw.pop("ch_b"),
        np.linspace(0, 1, frames),
        width=width,
        height=height,
        warmup=120,
        device=device,
        collect=True,
        **kw,
    )
    return torch.cat(imgs)


def fmt(name, rep, head, secs):
    bands = "  ".join(
        f"{b}: R {rep[b]['R']:.2f} fl {rep[b]['flip']:.2f} CV {rep[b]['CV']:.2f} pk {rep[b]['pk']:.1f}"
        for b, _, _ in BANDS
    )
    return (
        f"{name:>18}  {head}  sat {rep['sat']:5.1%}  detail {rep['detail']:.4f}  {secs:4.0f}s\n"
        f"{'':>18}  {bands}"
    )


MORPH = dict(method="morph", ch_a=100, ch_b=104, layer="features.7")
CONFIGS = {
    "morph": MORPH,
    "morph-ema.9": dict(MORPH, ema=0.9),  # ~10-step window, ~1.7 frames at 6 steps/frame
    "morph-ema.97": dict(MORPH, ema=0.97),  # ~33-step window, ~5.5 frames
    "morph-ema.99": dict(MORPH, ema=0.99),  # ~100-step window, ~17 frames
    "morph-eps5e-5": dict(MORPH, adam_eps=5e-5),  # ~median sqrt(v) after warmup
    "morph-eps5e-4": dict(MORPH, adam_eps=5e-4),  # ~90th percentile
    "string-spectrum": dict(method="string", arc="spectrum"),
    "string-image": dict(method="string", arc="image"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs", nargs="+", default=["string-spectrum", "string-image"], choices=list(CONFIGS)
    )
    ap.add_argument("--layer", default="features.5")
    ap.add_argument("--ch", type=int, default=42)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=216)
    args = ap.parse_args()

    device = pick_device()
    bb = Backbone("tiny", device=device)
    print(f"strobe bench @ {args.width}x{args.height}")
    for name in args.configs:
        cfg = dict(CONFIGS[name])
        method, layer = cfg.pop("method"), cfg.pop("layer", args.layer)
        bb.clear_hooks()
        bb.hook(layer)
        sc = Scorer(bb, layer, args.ch if method == "string" else cfg["ch_a"])
        t0 = time.time()
        if method == "string":
            frames = string_path(sc, width=args.width, height=args.height, device=device, **cfg)
            s = sc.robust(frames.to(device))
            head = f"drop {1 - s.min() / ((s[0] + s[-1]) / 2):5.1%}"
        else:
            ch_b = cfg["ch_b"]
            frames = morph_path(sc, width=args.width, height=args.height, device=device, **cfg)
            bb.hook(layer)  # run_morph clears hooks on exit
            sa = sc.robust(frames.to(device))
            sc.ch = ch_b
            sb = sc.robust(frames.to(device))
            # did the morph arrive? channel activations at the first and last frame
            head = f"A {sa[0]:5.2f}->{sa[-1]:5.2f}  B {sb[0]:5.2f}->{sb[-1]:5.2f}"
        print(fmt(name, temporal_report(frames), head, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
