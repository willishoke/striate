"""Transport loop: every frame is one base image carried by a smooth flow that loops.

frame_t(x) = base(x - d_t(x)) on the torus (FourierImage canvases tile seamlessly), where d_t is
the gradient-rotated stream function psi_t = sum_j c_j cos(2 pi k_j.x / S + 2 pi n_j t / T + phi_j):
a divergence-free swirl of a few low spatial modes, periodic in t, so the loop closes and nothing
is compressed. Topology is kept by construction (the change between frames is pure motion); the
base starts from a ring.py anchor and is re-optimized for the channel averaged over the whole
warp family, so it learns to survive being carried.

Two additions over the first version:
  --shutter N  motion blur: each frame averages N sub-frame warps under a tent over +-1 frame, so
               pixel-scale detail moving ~1 px/frame is low-passed along its motion instead of
               strobing (the base is trained through the same shutter).
  --drift X,Y  a journey: the canvas also slides X, Y whole widths per loop (it wraps seamlessly),
               which needs frames = S / speed for a gentle speed; long loops stream to disk.

Usage: warploop2.py --anchors ring4e198/anchors.pt --net googlenet --target inception4e:198 --out DIR
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# hard cap on MPS memory: overshooting raises an OOM instead of pushing the Mac into swap
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, math, random, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from striate.util import pick_device
from striate.model import Backbone
from zoo import MultiScaleCrop
from octave import LOGIT_RAIL, CHROMA_RATIO, hf_energy
from ring import Ring, temporal_report, bandpass
from smooth import turns
from contourflow import ContourFlow


class Flow:
    """Looping divergence-free displacement field on an S x S torus."""

    def __init__(self, S, T, modes, kmax, speed, seed, dev, drift=(0, 0), tscale=1):
        g = torch.Generator().manual_seed(seed)
        ks = [
            (kx, ky)
            for kx in range(-kmax, kmax + 1)
            for ky in range(-kmax, kmax + 1)
            if 0 < kx * kx + ky * ky <= kmax * kmax and (kx, ky) > (0, 0)
        ]  # one of each +-k pair
        pick = torch.randperm(len(ks), generator=g)[:modes]
        self.k = torch.tensor([ks[i] for i in pick], dtype=torch.float32, device=dev)  # (J, 2)
        self.n = (
            (
                torch.randint(1, 3, (modes,), generator=g)
                * (torch.randint(0, 2, (modes,), generator=g) * 2 - 1)
            )
            .float()
            .to(dev)
        )
        self.n *= (
            tscale  # long loops: same swirl period in frames, so the same amplitude and strain
        )
        self.phi = (torch.rand(modes, generator=g) * 2 * math.pi).to(dev)
        self.c = (1 / self.k.norm(dim=1)).to(dev)  # equal displacement per mode
        self.S, self.T, self.dev = S, T, dev
        self.drift = (0, 0)  # swirl only while calibrating
        # calibrate: mean |d_{t+1} - d_t| over a coarse grid = speed px/frame
        d = self.d(torch.arange(T + 1, device=dev), res=64)
        self.c *= speed / (d[1:] - d[:-1]).norm(dim=1).mean()
        self.max_disp = float(self.d(torch.arange(T, device=dev), res=64).norm(dim=1).max())
        self.strain = float(self._strain(res=128))
        self.drift = drift  # whole canvas widths per loop: a uniform slide, so it adds no strain

    def d(self, t, res=None):
        """Displacement (len(t), 2, res, res) in pixels at full scale."""
        res = res or self.S
        u = torch.arange(res, device=self.dev, dtype=torch.float32) * (self.S / res)
        y, x = torch.meshgrid(u, u, indexing="ij")
        out = torch.zeros(len(t), 2, res, res, device=self.dev)
        for j in range(len(self.k)):
            kx, ky = self.k[j]
            th = 2 * math.pi * (kx * x + ky * y) / self.S  # (res, res)
            ph = 2 * math.pi * self.n[j] * t.float() / self.T + self.phi[j]  # (len(t),)
            s = torch.sin(th[None] + ph[:, None, None])  # dpsi = -c sin(.) * grad(th)
            w = 2 * math.pi * self.c[j] / self.S
            out[:, 0] -= w * ky * s  # d_x = dpsi/dy = -c sin * 2pi ky / S
            out[:, 1] += w * kx * s  # d_y = -dpsi/dx = c sin * 2pi kx / S
        out[:, 0] += (self.drift[0] * self.S * t.float() / self.T)[:, None, None]
        out[:, 1] += (self.drift[1] * self.S * t.float() / self.T)[:, None, None]
        return out

    def _strain(self, res):
        """Largest |grad d| over the loop: below 1 the warp cannot fold (no topology change)."""
        d = self.d(torch.arange(0, self.T, 4, device=self.dev), res)
        h = self.S / res
        gx = (torch.roll(d, -1, 3) - torch.roll(d, 1, 3)) / (2 * h)
        gy = (torch.roll(d, -1, 2) - torch.roll(d, 1, 2)) / (2 * h)
        return torch.cat([gx, gy], 1).norm(dim=1).amax()


def cubic(t):
    """Keys bicubic kernel (a = -0.5)."""
    t = t.abs()
    return torch.where(
        t <= 1,
        (1.5 * t - 2.5) * t * t + 1,
        torch.where(t < 2, ((-0.5 * t + 2.5) * t - 4) * t + 2, 0 * t),
    )


def warp(img, d):
    """img (1, C, S, S) on the torus, sampled at x - d with a bicubic kernel. d: (N, 2, S, S) px.
    Written as wrapped gathers: grid_sample has no MPS backward, and the torus needs no padding."""
    C, S = img.shape[1], img.shape[-1]
    u = torch.arange(S, device=img.device, dtype=torch.float32)
    y, x = torch.meshgrid(u, u, indexing="ij")
    xs, ys = x - d[:, 0], y - d[:, 1]
    x0, y0 = xs.floor(), ys.floor()
    fx, fy = xs - x0, ys - y0
    x0, y0 = x0.long(), y0.long()
    flat = img.reshape(C, S * S)
    out = 0
    for j in range(-1, 3):
        row = ((y0 + j) % S) * S
        wy = cubic(fy - j)
        for i in range(-1, 3):
            idx = row + (x0 + i) % S  # (N, S, S)
            out = out + flat[:, idx.reshape(-1)].view(C, *idx.shape) * (wy * cubic(fx - i))
    return out.transpose(0, 1)  # (N, C, S, S)


def render(pre, flow, ts, shutter):
    """Frames at times ts (float tensor): the warped base through a tent shutter over +-1 frame."""
    if shutter <= 1:
        return torch.sigmoid(warp(pre, flow.d(ts)))
    offs = torch.linspace(-1, 1, shutter + 2)[1:-1].tolist()
    ws = [1 - abs(o) for o in offs]
    out = 0
    for o, w in zip(offs, ws):
        out = out + torch.sigmoid(warp(pre, flow.d(ts + o))) * (w / sum(ws))
    return out


def strobe(frames, closed):
    """temporal_report's R per band and the sharpest turn; open windows are not wrapped."""
    if closed:
        rep, cos = temporal_report(frames), turns(frames)
        return {k: v["R"] for k, v in rep.items() if isinstance(v, dict)}, cos
    out = {}
    for name, lo, hi in [("0-4px", 0.5, 2), ("4-16px", 2, 8), ("16-64px", 8, 32)]:
        b = torch.cat([bandpass(frames[i : i + 8], lo, hi) for i in range(0, len(frames), 8)])
        d1 = b[1:] - b[:-1]
        out[name] = float((d1[1:] - d1[:-1]).pow(2).mean() / d1.pow(2).mean())
    b = torch.cat([bandpass(frames[i : i + 8], 2, 32) for i in range(0, len(frames), 8)])
    d1 = (b[1:] - b[:-1]).flatten(1)
    return out, F.cosine_similarity(d1[1:], d1[:-1])


def penalized(rg, pre, view):
    """Ring.loss on a pre-sigmoid image instead of a spectrum."""
    img = torch.sigmoid(pre)
    a = rg.act(view(img))
    hinge = (torch.relu(pre.abs() - LOGIT_RAIL) ** 2).mean((1, 2, 3))
    y = (img * torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)).sum(
        1, keepdim=True
    )
    ratio = hf_energy(torch.cat([img[:, 2:3] - y, img[:, 0:1] - y], 1)) / hf_energy(y).detach()
    chroma = torch.relu(ratio / CHROMA_RATIO - 1)
    return (-a + a.detach().abs() * (rg.wh * hinge + rg.wc * chroma)).sum(), a.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--which", type=int, default=0)
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--target", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--speed", type=float, default=1.0, help="mean px/frame")
    ap.add_argument("--modes", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=3, help="largest wavenumber, cycles per canvas")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--mb", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shutter", type=int, default=1, help="sub-frame samples per frame (1 = off)")
    ap.add_argument("--drift", default="0,0", help="canvas widths per loop, X,Y")
    ap.add_argument(
        "--window", type=int, default=120, help="frames kept in memory for the strobe metrics"
    )
    ap.add_argument(
        "--swirl-period",
        type=int,
        default=120,
        help="frames per swirl cycle unit; loops longer than this cycle more",
    )
    ap.add_argument(
        "--contour",
        type=float,
        default=0,
        help="contour currents: flow along level sets of the base blurred at this sigma (px) instead of a random swirl",
    )
    ap.add_argument(
        "--cut", type=float, default=0, help="drop detail finer than this period (px) from the base"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
    net = Backbone(a.net, dev)
    net.hook(layer)
    A = torch.load(a.anchors)
    S = A.shape[2]
    rg = Ring(net, layer, ch, S, dev)
    T = a.frames
    drift = tuple(int(v) for v in a.drift.split(","))
    flow = Flow(
        S, T, a.modes, a.kmax, a.speed, a.seed, dev, drift, tscale=max(1, round(T / a.swirl_period))
    )
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)
    log(
        f"flow: {a.modes} modes, k<={a.kmax}, speed {a.speed}px/frame, max displacement {flow.max_disp:.1f}px, "
        f"max strain {flow.strain:.2f}; drift {drift} widths/loop = {math.hypot(*drift) * S / T:.2f}px/frame; shutter {a.shutter}"
    )

    spec = torch.nn.Parameter(A[a.which : a.which + 1].to(dev).clone())
    del A
    if a.contour > 0:
        with torch.no_grad():
            flow = ContourFlow(torch.sigmoid(rg.pre(spec)), T, a.speed, a.contour, dev)
        log(
            f"contour currents sigma {a.contour}px: amplitude {flow.A:.1f} tau, max displacement {flow.max_disp:.1f}px, "
            f"strain {flow.strain:.2f}, measured speed {flow.speed_meas:.2f}px/frame"
        )
    if a.cut > 0:  # zero the spectrum above the cutoff; masking the gradient keeps it zero
        f = torch.sqrt(
            torch.fft.fftfreq(S, device=dev).view(-1, 1) ** 2
            + torch.fft.rfftfreq(S, device=dev).view(1, -1) ** 2
        )
        keep = (f <= 1 / a.cut).float()[None, None, :, :, None]
        with torch.no_grad():
            spec.mul_(keep)
        spec.register_hook(lambda g: g * keep)

    name = (lambda i: f"f{i:04d}.png") if T > 1000 else (lambda i: f"f{i:03d}.png")

    @torch.no_grad()
    def evaluate(write, stride=1):
        """Robust activation per frame from 4 fixed view draws shared by all frames. With write,
        frames stream to disk as PNGs and the first `window` frames (+2) are kept for the metrics."""
        pre = rg.pre(spec)
        ts_all = list(range(0, T, stride))
        acts, keep = [], []
        for i in range(0, len(ts_all), a.mb):
            ts = torch.tensor(ts_all[i : i + a.mb], device=dev, dtype=torch.float32)
            fr = render(pre, flow, ts, a.shutter)
            tot = 0
            for k in range(4):
                random.seed(1000 + k)
                tot = tot + rg.act(MultiScaleCrop(224 / S, 512)(fr)).cpu()
            acts.append(tot / 4)
            if write:
                cpu = fr.cpu()
                for t, f in zip(ts_all[i : i + a.mb], cpu):
                    Image.fromarray(
                        (f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    ).save(f"{a.out}/{name(t)}")
                    if t < a.window + 2:
                        keep.append(f)
        random.seed(time.time_ns())
        return (torch.stack(keep) if keep else None), torch.cat(acts).numpy()

    _, act0 = evaluate(False, stride=max(1, T // 120))
    with torch.no_grad():  # the anchor itself, under the same 4 view draws
        still = 0
        for k in range(4):
            random.seed(1000 + k)
            still += float(rg.act(MultiScaleCrop(224 / S, 512)(torch.sigmoid(rg.pre(spec))))) / 4
        random.seed(time.time_ns())
    log(
        f"anchor still {still:.3f}; carried by the flow before re-optimizing: act {act0.mean():.3f} "
        f"[{act0.min():.3f}, {act0.max():.3f}]"
    )

    opt = torch.optim.Adam([spec], lr=a.lr)
    for it in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * it / a.steps))
        opt.zero_grad()
        ts = torch.tensor(random.sample(range(T), a.mb), device=dev, dtype=torch.float32)
        loss, act = penalized(
            rg,
            torch.logit(render(rg.pre(spec), flow, ts, a.shutter).clamp(1e-6, 1 - 1e-6)),
            MultiScaleCrop(224 / S, 512),
        )
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == a.steps - 1:
            log(f"step {it + 1}/{a.steps} act mean {act.mean():.3f} min {act.min():.3f}")

    frames, act = evaluate(True)
    with torch.no_grad():
        base = torch.sigmoid(rg.pre(spec))[0].cpu()
    Image.fromarray((base.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
        f"{a.out}/base.png"
    )
    torch.save(spec.detach().cpu(), f"{a.out}/base.pt")
    closed = T <= a.window
    R, cos = strobe(frames if closed else frames[: a.window + 2], closed)
    rep = {k: {"R": v} for k, v in R.items()}
    rep.update(
        act_mean=float(act.mean()),
        act_min=float(act.min()),
        act_max=float(act.max()),
        act_before=float(act0.mean()),
        anchor_still=still,
        turn_min=float(cos.min()),
        turn_mean=float(cos.mean()),
        act=act.tolist(),
        speed=a.speed,
        modes=a.modes,
        kmax=a.kmax,
        max_disp=flow.max_disp,
        strain=flow.strain,
        shutter=a.shutter,
        drift=list(drift),
        frames=T,
        metrics_window=None if closed else a.window,
        secs=round(time.time() - t0),
    )
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(
        f"done act {act.mean():.3f} [{act.min():.3f}, {act.max():.3f}] (before {act0.mean():.3f}) turn cos min {cos.min():.2f} "
        + " ".join(f"{k} R {v:.2f}" for k, v in R.items())
    )


if __name__ == "__main__":
    main()
