"""Transport + regrowth loop: every frame is its own image, optimized jointly around a ring.

Each frame must (a) drive the channel and (b) match the previous frame carried one step along a
steady flow:  F_{t+1} ~ F_t(Phi_{-1}(x)),  with F_T = F_0 so the loop closes. The flow is steady
and unbounded (material keeps travelling; it never sloshes back), so over a loop the accumulated
transport is far from the identity; the only way to close the ring is for the network to regrow
detail where the flow stretches and shears the old detail. Where the consistency term wins, matter
is carried (identity kept); where it can't, it is rebuilt continuously rather than crossfaded.

Init (no seam): the two-phase advected crossfade of Neyret's advected textures,
F_t = (1 - t/T) * base(Phi_{-t}(x)) + (t/T) * base(Phi_{T-t}(x)); the optimization then removes
the ghosting of the middle frames.

Flows: "contour" (circulation along level sets of the blurred base, so coarse layout holds still and
detail circulates around it), "swirl" (steady low-mode cellular flow), either plus a uniform drift.

Usage: regrow.py --anchors ring4c244/anchors.pt --net googlenet --target inception4c:244 --flow contour --out DIR
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, math, random, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from striate.util import pick_device
from striate.param import color_correlation_normalized
from striate.model import Backbone
from zoo import MultiScaleCrop
from octave import LOGIT_RAIL, CHROMA_RATIO, hf_energy
from ring import Ring
from warploop2 import warp
from contourflow import _sample

LUMA = [0.299, 0.587, 0.114]


def velocity(kind, base, sigma, modes, kmax, seed, dev):
    """Steady divergence-free velocity (2, S, S), mean speed 1."""
    S = base.shape[-1]
    fy = torch.fft.fftfreq(S, device=dev).view(-1, 1)
    fx = torch.fft.rfftfreq(S, device=dev).view(1, -1)
    if kind == "contour":
        L = (base[0] * torch.tensor(LUMA, device=dev).view(3, 1, 1)).sum(0)
        Z = torch.fft.rfft2(L - L.mean()) * torch.exp(
            -2 * (math.pi * sigma) ** 2 * (fx * fx + fy * fy)
        )
    else:  # random stream function of a few low modes
        g = torch.Generator().manual_seed(seed)
        Z = torch.zeros(S, S // 2 + 1, dtype=torch.complex64, device=dev)
        ks = [
            (kx, ky)
            for kx in range(0, kmax + 1)
            for ky in range(-kmax, kmax + 1)
            if 0 < kx * kx + ky * ky <= kmax * kmax and (kx > 0 or ky > 0)
        ]
        for i in torch.randperm(len(ks), generator=g)[:modes].tolist():
            kx, ky = ks[i]
            ph = float(torch.rand(1, generator=g)) * 2 * math.pi
            Z[ky % S, kx] = complex(math.cos(ph), math.sin(ph)) / math.hypot(kx, ky)
    ux = torch.fft.irfft2(Z * (2j * math.pi * fy), s=(S, S))
    uy = torch.fft.irfft2(Z * (-2j * math.pi * fx), s=(S, S))
    u = torch.stack([ux, uy])
    return u / u.norm(dim=0).mean()


class Steady:
    """Flow maps of a steady velocity (speed px/frame) plus a uniform drift (px/frame)."""

    def __init__(self, u, speed, drift, dev, res=512):
        S = u.shape[-1]
        self.S, self.res, self.dev = S, res, dev
        self.u = F.interpolate(u[None], size=(res, res), mode="area")[0] * speed
        self.u[0] += drift[0]
        self.u[1] += drift[1]
        self.umax = float(self.u.norm(dim=0).max())
        g = torch.arange(res, device=dev, dtype=torch.float32) * (S / res)
        self.yy, self.xx = torch.meshgrid(g, g, indexing="ij")

    def _vel(self, px, py):
        v = _sample(self.u, px, py, self.S)
        return v[0], v[1]

    def maps(self, s_max, sgn):
        """Displacements d_s = x - Phi_{-sgn*s}(x) for s = 1..s_max frames, (s_max, 2, res, res), on CPU."""
        sub = max(2, math.ceil(self.umax / 0.25))
        dt = -sgn / sub
        px, py = self.xx.clone(), self.yy.clone()
        out = []
        for s in range(1, s_max + 1):
            for _ in range(sub):
                k1x, k1y = self._vel(px, py)
                k2x, k2y = self._vel(px + 0.5 * dt * k1x, py + 0.5 * dt * k1y)
                k3x, k3y = self._vel(px + 0.5 * dt * k2x, py + 0.5 * dt * k2y)
                k4x, k4y = self._vel(px + dt * k3x, py + dt * k3y)
                px = px + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
                py = py + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
            out.append(torch.stack([self.xx - px, self.yy - py]).cpu())
        return torch.stack(out)

    def full(self, d):
        """A res-grid displacement (2, res, res) at full resolution (1, 2, S, S) on the device."""
        return F.interpolate(
            d[None].to(self.dev), size=(self.S, self.S), mode="bilinear", align_corners=False
        )


def to_spec(rg, img):
    """Inverse of Ring.pre + sigmoid: spectrum (N, 3, S, S//2+1, 2) whose image is img (N, 3, S, S)."""
    pre = torch.logit(img.clamp(1e-4, 1 - 1e-4))
    mat = color_correlation_normalized.to(img.device)
    p = (pre.permute(0, 2, 3, 1) @ torch.linalg.inv(mat).T).permute(0, 3, 1, 2) * 4.0
    z = torch.fft.rfftn(p, dim=(2, 3))
    return torch.view_as_real(z) / rg.helper.spectrum_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--which", type=int, default=0)
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--target", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--flow", default="contour", choices=["contour", "swirl"])
    ap.add_argument(
        "--sigma", type=float, default=8.0, help="contour flow: blur of the base luminance (px)"
    )
    ap.add_argument("--modes", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument(
        "--speed", type=float, default=0.75, help="mean px/frame of the swirl/contour part"
    )
    ap.add_argument("--drift", default="0,0", help="uniform px/frame")
    ap.add_argument(
        "--lam", type=float, default=30.0, help="consistency weight (relative to activation)"
    )
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--win", type=int, default=4, help="consecutive frames per step")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
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
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)
    spec0 = A[a.which : a.which + 1].to(dev)
    del A
    with torch.no_grad():
        base = torch.sigmoid(rg.pre(spec0))
        u = velocity(a.flow, base, a.sigma, a.modes, a.kmax, a.seed, dev)
        drift = tuple(float(v) for v in a.drift.split(","))
        fl = Steady(u, a.speed, drift, dev)
        step_d = fl.full(fl.maps(1, 1)[0])  # one frame forward: F_{t+1}(x) = F_t(x - step_d(x))
        # init: two-phase advected crossfade, frame by frame (maps forward to t and backward to T - t)
        fwd = fl.maps(T, 1)  # d for Phi_{-s}, s = 1..T
        bwd = fl.maps(T, -1)  # d for Phi_{+s}
        P = torch.zeros(T, 3, S, S // 2 + 1, 2)
        for t in range(T):
            ia = base if t == 0 else warp(base, fl.full(fwd[t - 1]))
            ib = base if t == 0 else warp(base, fl.full(bwd[T - t - 1]))
            w = t / T
            P[t] = to_spec(rg, (1 - w) * ia + w * ib)[0].cpu()
        del fwd, bwd
    P = P.to(dev)
    M, V, NT = torch.zeros_like(P), torch.zeros_like(P), torch.zeros(T)
    log(
        f"flow {a.flow} speed {a.speed} drift {drift}: max speed {fl.umax:.2f}px/frame; init done; P {P.numel() * 4 / 1e9:.2f} GB"
    )

    lum = torch.tensor(LUMA, device=dev).view(1, 3, 1, 1)

    def frames(idx, spec):
        z = torch.view_as_complex((spec * rg.helper.spectrum_scale).contiguous())
        p = torch.fft.irfftn(z, s=(S, S), dim=(2, 3)) / 4.0
        return rg.helper._decorrelate_color(p)

    def carry(img):  # (n, 3, S, S) -> each carried one frame forward
        n = img.shape[0]
        return warp(img.reshape(1, n * 3, S, S), step_d).reshape(n, 3, S, S)

    @torch.no_grad()
    def evaluate(write):
        acts, cons = [], []
        prev0 = None
        for i in range(0, T, 4):
            idx = list(range(i, min(T, i + 4)))
            fr = torch.sigmoid(frames(idx, P[idx]))
            tot = 0
            for k in range(4):
                random.seed(1000 + k)
                tot = tot + rg.act(MultiScaleCrop(224 / S, 512)(fr)).cpu()
            acts.append(tot / 4)
            nxt = torch.sigmoid(frames(None, P[[(j + 1) % T for j in idx]]))
            cons.append((nxt - carry(fr)).pow(2).mean((1, 2, 3)).sqrt().cpu())
            if write:
                for t, f in zip(idx, fr.cpu()):
                    Image.fromarray(
                        (f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    ).save(f"{a.out}/f{t:03d}.png")
        random.seed(time.time_ns())
        return torch.cat(acts).numpy(), torch.cat(cons).numpy()

    act0, cons0 = evaluate(False)
    log(
        f"before: act {act0.mean():.3f} [{act0.min():.3f}, {act0.max():.3f}] consistency rms {cons0.mean():.4f} max {cons0.max():.4f}"
    )
    curve = [(0, float(act0.mean()), float(cons0.mean()))]
    b1, b2 = 0.9, 0.999
    for it in range(a.steps):
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * it / a.steps))
        t = random.randrange(T)
        idx = [(t + j) % T for j in range(a.win)]
        sub = P[idx].clone().requires_grad_(True)
        with torch.no_grad():
            before = torch.sigmoid(frames(None, P[[(t - 1) % T]]))
            after = torch.sigmoid(frames(None, P[[(t + a.win) % T]]))
        pre = frames(idx, sub)
        fr = torch.sigmoid(pre)
        act = rg.act(MultiScaleCrop(224 / S, 512)(fr))
        hinge = (torch.relu(pre.abs() - LOGIT_RAIL) ** 2).mean((1, 2, 3))
        y = (fr * lum).sum(1, keepdim=True)
        ratio = hf_energy(torch.cat([fr[:, 2:3] - y, fr[:, 0:1] - y], 1)) / hf_energy(y).detach()
        chroma = torch.relu(ratio / CHROMA_RATIO - 1)
        chain = torch.cat([before, fr, after])  # consecutive frames t-1 .. t+win
        cons = (chain[1:] - carry(chain[:-1])).pow(2).mean((1, 2, 3))  # win + 1 pairs
        scale = act.detach().abs().mean()
        loss = (
            -act + act.detach().abs() * (rg.wh * hinge + rg.wc * chroma)
        ).sum() + a.lam * scale * cons.sum()
        loss.backward()
        with torch.no_grad():  # Adam on the touched frames only, per-frame bias correction
            g = sub.grad
            NT[idx] += 1
            m = M[idx] * b1 + g * (1 - b1)
            v = V[idx] * b2 + g * g * (1 - b2)
            M[idx], V[idx] = m, v
            n = NT[idx].view(-1, 1, 1, 1, 1).to(dev)
            P[idx] = P[idx] - lr * (m / (1 - b1**n)) / ((v / (1 - b2**n)).sqrt() + 1e-8)
        if it % 250 == 0 or it == a.steps - 1:
            log(f"step {it + 1}/{a.steps} act {act.mean():.3f} cons rms {cons.sqrt().mean():.4f}")
            curve.append((it + 1, float(act.mean()), float(cons.sqrt().mean())))

    act, cons = evaluate(True)
    rep = dict(
        act_mean=float(act.mean()),
        act_min=float(act.min()),
        act_max=float(act.max()),
        act=act.tolist(),
        act_before=float(act0.mean()),
        cons_rms=float(cons.mean()),
        cons_max=float(cons.max()),
        cons=cons.tolist(),
        cons_before=float(cons0.mean()),
        flow=a.flow,
        sigma=a.sigma,
        speed=a.speed,
        drift=drift,
        lam=a.lam,
        frames=T,
        steps=a.steps,
        curve=curve,
        secs=round(time.time() - t0),
    )
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(
        f"done act {act.mean():.3f} [{act.min():.3f}, {act.max():.3f}] (before {act0.mean():.3f}) consistency rms {cons.mean():.4f} (before {cons0.mean():.4f})"
    )


if __name__ == "__main__":
    main()
