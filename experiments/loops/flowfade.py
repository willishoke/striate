"""Motion vs. fade: how much of each frame-to-frame change in a loop is explained by moving pixels.

For frames a = f[t] and b = f[t+gap], RAFT estimates where each pixel of b came from in a; a is
warped onto b along that flow. In the 2-32px luma band (at 1024 scale):
    explained = 1 - |b - warp(a)|^2 / |b - a|^2
1 means the change is pure transport (topology kept); 0 means the flow explains nothing, as in a
crossfade. Also reports the mean flow speed in px/frame at 1024 scale.

Usage: flowfade.py DIR [DIR ...] [--gap 4] [--res 512] [--dev mps]   (DIR holds f000.png ...)
       flowfade.py --selftest SOME_FRAME.png OTHER_FRAME.png [--dev cpu]
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, glob, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from ring import bandpass

MARGIN = 16  # px at working resolution; warps near the border sample outside the frame


def load(path, res):
    x = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).permute(2, 0, 1).float() / 255
    return F.interpolate(x[None], size=(res, res), mode="area")[0]


def warp(a, flow):
    """Sample a at x + flow(x). a: (N, 3, H, W); flow: (N, 2, H, W) in pixels."""
    n, _, h, w = a.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=a.device), torch.arange(w, device=a.device), indexing="ij"
    )
    gx = (xs + flow[:, 0]) / (w - 1) * 2 - 1
    gy = (ys + flow[:, 1]) / (h - 1) * 2 - 1
    return F.grid_sample(
        a, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True
    )


class Meter:
    def __init__(self, dev, res):
        self.dev, self.res = dev, res
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(dev)
        s = res / 1024  # band edges are given at 1024 scale
        self.lo, self.hi = 2 * s, 32 * s

    @torch.no_grad()
    def pairs(self, a, b):
        """a, b: (N, 3, H, W) in [0, 1]. Returns explained fraction and speed (px at 1024) per pair."""
        a, b = a.to(self.dev), b.to(self.dev)
        flow = self.model(b * 2 - 1, a * 2 - 1)[-1]  # for each pixel of b, its offset into a
        wa = warp(a, flow)
        m = slice(MARGIN, -MARGIN)
        band = lambda x: bandpass(x, self.lo, self.hi)[..., m, m]
        res = (band(b) - band(wa)).pow(2).mean((1, 2, 3))
        chg = (band(b) - band(a)).pow(2).mean((1, 2, 3))
        speed = flow[..., m, m].norm(dim=1).mean((1, 2)) * 1024 / self.res
        return (1 - res / chg).cpu(), speed.cpu()


def measure(meter, d, gap, res, chunk=4, max_frames=0):
    files = sorted(
        glob.glob(f"{d}/f[0-9][0-9][0-9].png") + glob.glob(f"{d}/f[0-9][0-9][0-9][0-9].png")
    )
    closed = not max_frames or max_frames >= len(files)
    if not closed:  # an open window of a long loop: pairs stay inside it
        files = files[: max_frames + gap]
    fr = torch.stack([load(f, res) for f in files])
    t = torch.arange(len(fr))
    ex, sp = [], []
    n = len(fr) if closed else len(fr) - gap
    for i in range(0, n, chunk):
        idx = t[i : min(i + chunk, n)]
        e, s = meter.pairs(fr[idx], fr[(idx + gap) % len(fr)])  # closed loop: wrap around
        ex.append(e)
        sp.append(s / gap)
    ex, sp = torch.cat(ex), torch.cat(sp)
    return dict(
        explained_mean=float(ex.mean()),
        explained_min=float(ex.min()),
        explained_p10=float(ex.quantile(0.1)),
        speed_px_per_frame=float(sp.mean()),
        explained=[round(float(x), 3) for x in ex],
    )


def selftest(meter, p1, p2, res, n=8):
    """Known answers: a pure shift should be ~1, a crossfade between unrelated images ~0."""
    a, b = load(p1, res), load(p2, res)
    pad = F.pad(a[None], (16, 16, 16, 16), mode="circular")[0]
    shift = torch.stack(
        [pad[:, 16 : 16 + res, 16 - k : 16 - k + res] for k in range(0, 2 * n + 1, 2)]
    )
    fade = torch.stack([(1 - w) * a + w * b for w in torch.linspace(0, 1, n + 1)])
    for name, fr in [("shift 2px/step", shift), ("crossfade", fade)]:
        e, s = meter.pairs(fr[:-1], fr[1:])
        print(
            f"{name}: explained {e.mean():.3f} (min {e.min():.3f}), flow {s.mean() * res / 1024:.2f}px/step at {res}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--gap", type=int, default=4)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="score only this many leading frames")
    ap.add_argument("--out")
    a = ap.parse_args()
    meter = Meter(a.dev, a.res)
    if a.selftest:
        return selftest(meter, *a.dirs[:2], a.res)
    report = {}
    for d in a.dirs:
        r = measure(meter, d, a.gap, a.res, max_frames=a.max_frames)
        report[os.path.basename(d.rstrip("/"))] = r
        print(
            f"{os.path.basename(d.rstrip('/')):22s} explained mean {r['explained_mean']:.3f} p10 {r['explained_p10']:.3f} "
            f"min {r['explained_min']:.3f}  speed {r['speed_px_per_frame']:.2f}px/frame",
            flush=True,
        )
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
