"""Smooth a ridge-walk loop with a space-time speed limit.

Every frame is a free node of one closed string (no pinned anchors, so no U-turns at
local maxima; frames are the nodes, so no node/blend alternation). Each step: climb the
channel's activation perpendicular to the loop (same random view for every frame), then
project the whole video onto the cone |omega| <= min(v |k|, w_max) of its space-time
spectrum: a feature at spatial frequency k may change at most as fast as motion at v
px/frame would change it, and nothing faster than w_max cycles/frame. Every 10 steps,
respace to equal band-passed image distance around the loop.

Usage: smooth.py --frames-from ring530 --v 1.0 --wmax 0.125 --out DIR
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# hard cap on MPS memory: overshooting raises an OOM instead of pushing the Mac into swap
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, math, random, time
import numpy as np
import torch
from PIL import Image
from striate.param import _rfft2d_freqs, seed_spectrum_from_image
from striate.util import pick_device
from striate.model import Backbone
from zoo import MultiScaleCrop
from ring import Ring, ChunkedAdam, bandpass, temporal_report


def cone(T, size, v, wmax, dev):
    """Keep-mask over (spatial coefficient, temporal bin): (H, W', T/2+1)."""
    k = torch.tensor(_rfft2d_freqs(size, size), dtype=torch.float32, device=dev)  # cycles/px
    w = torch.arange(T // 2 + 1, device=dev, dtype=torch.float32) / T  # cycles/frame
    m = w <= torch.minimum(v * k, torch.tensor(wmax, device=dev))[..., None] + 1e-9
    m[..., 0] = True  # the static image is always allowed
    return m.float()


@torch.no_grad()
def project(X, mask, rows=64):
    """Band-limit the video in time per spatial coefficient, in place, a block of spectrum
    rows at a time (time moved last: MPS FFTs only run on trailing axes)."""
    T = X.shape[0]
    for r in range(0, X.shape[2], rows):
        Y = X[:, :, r : r + rows].permute(1, 2, 3, 4, 0)  # (3, rows, W', 2, T)
        Yf = torch.fft.rfft(Y, dim=4) * mask[None, r : r + rows, :, None, :]
        X[:, :, r : r + rows] = torch.fft.irfft(Yf, n=T, dim=4).permute(4, 0, 1, 2, 3)


def turns(frames):
    """cos between successive frame-change vectors in the 2-32px band (1 = straight on)."""
    f = torch.cat([frames, frames[:2]])
    b = torch.cat([bandpass(f[i : i + 8], 2, 32) for i in range(0, len(f), 8)])
    d1 = (b[1:] - b[:-1]).flatten(1)
    return torch.nn.functional.cosine_similarity(d1[1:], d1[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-from", required=True)
    ap.add_argument("--net", default="resnet50_v1")
    ap.add_argument("--target", default="layer3:530")
    ap.add_argument("--v", type=float, default=1.0)
    ap.add_argument("--wmax", type=float, default=0.125)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--mb", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.frames_from) if f.startswith("f") and f.endswith(".png"))
    T = len(files)
    load = lambda f: (
        torch.from_numpy(np.asarray(Image.open(f"{a.frames_from}/{f}").convert("RGB")))
        .permute(2, 0, 1)
        .float()[None]
        / 255
    )
    size = load(files[0]).shape[-1]
    layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
    net = Backbone(a.net, dev)
    net.hook(layer)
    rg = Ring(net, layer, ch, size, dev)
    # invert the frames back to spectra (exact up to 8-bit rounding)
    X = torch.empty(T, *rg.helper.spectrum.shape[1:], device=dev)
    for t in range(T):
        seed_spectrum_from_image(rg.helper, load(files[t]).to(dev))
        X[t] = rg.helper.spectrum.detach()[0]
    mask = cone(T, size, a.v, a.wmax, dev)
    print(f"cone keeps {mask.mean():.1%} of space-time coefficients", flush=True)
    project(X, mask)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)

    @torch.no_grad()
    def band(spec):
        return torch.cat(
            [bandpass(rg.image(spec[i : i + a.mb]), 2, 32) for i in range(0, len(spec), a.mb)]
        )

    @torch.no_grad()
    def respace(X):
        """In place: equal band-passed image distance around the closed loop."""
        b = band(X)
        d = (torch.roll(b, -1, 0) - b).flatten(1).norm(dim=1)  # d[t] = |b[t+1] - b[t]|, wrapping
        del b
        s = torch.cat([d.new_zeros(1), d.cumsum(0)])
        s = s / s[-1]
        out = torch.empty_like(X)
        for t, u in enumerate(torch.arange(T, device=dev) / T):
            j = int(torch.searchsorted(s, u.clamp(min=1e-9)).clamp(1, T))
            w = (u - s[j - 1]) / (s[j] - s[j - 1] + 1e-12)
            out[t] = X[j - 1] + w * (X[j % T] - X[j - 1])
        X.copy_(out)

    X = torch.nn.Parameter(X)
    opt = ChunkedAdam(X, lr=a.lr)
    for it in range(a.steps):
        for g in opt.param_groups:  # cosine decay: settle at the end
            g["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * it / a.steps))
        opt.zero_grad()
        seed = random.randrange(1 << 30)
        acts = []
        for i in range(0, T, a.mb):
            random.seed(seed)
            loss, act = rg.loss(X[i : i + a.mb], MultiScaleCrop(224 / size, 512))
            loss.backward()
            acts.append(act)
        with torch.no_grad():  # climb only perpendicular to the loop, one frame at a time
            for t in range(T):
                tau = X[(t + 1) % T] - X[t - 1]
                X.grad[t].sub_(tau, alpha=float((X.grad[t] * tau).sum() / (tau * tau).sum()))
        opt.step()
        with torch.no_grad():
            project(X.data, mask)
            if it % 10 == 9:
                respace(X.data)
                project(X.data, mask)
        if it % 25 == 0 or it == a.steps - 1:
            acts = torch.cat(acts)
            log(f"step {it + 1}/{a.steps} act mean {acts.mean():.3f} min {acts.min():.3f}")

    with torch.no_grad():
        X = X.detach()
        frames = torch.cat([rg.image(X[i : i + a.mb]).cpu() for i in range(0, T, a.mb)])
        tot = 0
        for d in range(4):
            r = []
            for i in range(0, T, a.mb):
                random.seed(1000 + d)
                r.append(
                    rg.act(MultiScaleCrop(224 / size, 512)(frames[i : i + a.mb].to(dev))).cpu()
                )
            tot = tot + torch.cat(r)
        act = (tot / 4).numpy()
    for i, f in enumerate(frames):
        Image.fromarray((f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
            f"{a.out}/f{i:03d}.png"
        )
    cos = turns(frames)
    rep = temporal_report(frames)
    rep.update(
        act_mean=float(act.mean()),
        act_min=float(act.min()),
        act_max=float(act.max()),
        turn_min=float(cos.min()),
        turn_mean=float(cos.mean()),
        turns=cos.tolist(),
        act=act.tolist(),
        v=a.v,
        wmax=a.wmax,
        secs=round(time.time() - t0),
    )
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(
        f"done act {act.mean():.3f} [{act.min():.3f}, {act.max():.3f}] turn cos min {cos.min():.2f} mean {cos.mean():.2f} "
        + " ".join(f"{k} R {v['R']:.2f}" for k, v in rep.items() if isinstance(v, dict))
    )


if __name__ == "__main__":
    main()
