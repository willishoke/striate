"""Cell-tissue loop: a periodic power diagram whose cells grow, divide and get extruded, optimized as a loop.

Each of N slots is a cell site s, a power weight w and a texture offset o, each a smooth periodic path
(a few Fourier harmonics). A pixel x belongs to the slot minimizing |x - s|^2 - w (on the torus), and shows
the shared texture at o + (x - s) / rho, so a cell carries its own contents wherever it goes and whatever
shape its neighbours press it into.

The life cycle is a fixed schedule; only the geometry and the contents are optimized. Slot i is born at
phi[i-1] when slot i-1 divides, grows from half to full size over g, divides at phi[i] (the slot carries on
as one daughter, slot i+1 is reborn as the other), and that daughter is squeezed out over the next L and
stays dead until its rebirth. phi[i] = (i p mod N) / N, so every gap phi[i] - phi[i-1] equals g = p / N and
the schedule closes on the loop exactly.

Division: around phi[i] the mother's (s, w, o) and the daughter's are pulled to their common midpoint by a
cos^2 bump, so at the instant of division the two coincide (one region, one texture) and then separate:
the new wall grows from zero length, and the contents tear down the middle into two copies.
Extrusion: the doomed cell's weight sinks until its region shrinks to a point, its contents shrinking with it.

Losses: activation over random frames, plus a "physics" prior: each cell's area follows its age, sites sit
near their cells' centroids (even, mostly hexagonal tissue), and daughters are born a cell-width apart.

Usage: cells.py --base carry4e198/base.pt --target inception4e:198 --out DIR
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
from striate.model import Backbone
from zoo import MultiScaleCrop
from octave import LOGIT_RAIL, CHROMA_RATIO, hf_energy
from ring import Ring

LUMA = [0.299, 0.587, 0.114]
BLK = 16  # candidate search block (px)
MK = 17  # membrane profile knots, 1 px apart
RK, RSTEP = (
    17,
    2.0,
)  # interior profile knots: gain and bias of the colour logits vs distance to the wall


def wrap(d, S):
    return torch.remainder(d + S / 2, S) - S / 2


def blur(img, sigma):
    """Periodic gaussian blur of (..., S, S)."""
    S = img.shape[-1]
    f2 = (
        torch.fft.fftfreq(S, device=img.device).view(-1, 1) ** 2
        + torch.fft.rfftfreq(S, device=img.device).view(1, -1) ** 2
    )
    return torch.fft.irfftn(
        torch.fft.rfftn(img, dim=(-2, -1)) * torch.exp(-2 * math.pi**2 * sigma**2 * f2),
        s=(S, S),
        dim=(-2, -1),
    )


def sample(tex, uv, cubic=True):
    """Periodic texture lookup without grid_sample (its backward falls back to the CPU on MPS).
    tex (C, S, S); uv (..., 2) in px (x, y) -> (C, ...). Catmull-Rom bicubic or bilinear."""
    C, S = tex.shape[0], tex.shape[-1]
    flat = tex.reshape(C, -1)
    sh = uv.shape[:-1]
    u, v = uv[..., 0].reshape(-1), uv[..., 1].reshape(-1)
    x0, y0 = torch.floor(u), torch.floor(v)
    fx, fy = u - x0, v - y0
    x0, y0 = x0.long(), y0.long()
    if cubic:

        def w4(t):
            t2, t3 = t * t, t * t * t
            return [
                -0.5 * t3 + t2 - 0.5 * t,
                1.5 * t3 - 2.5 * t2 + 1,
                -1.5 * t3 + 2 * t2 + 0.5 * t,
                0.5 * t3 - 0.5 * t2,
            ]

        taps = range(-1, 3)
        wx, wy = w4(fx), w4(fy)
    else:
        taps = range(0, 2)
        wx, wy = [1 - fx, fx], [1 - fy, fy]
    out = 0
    for j, dy in enumerate(taps):
        row = torch.remainder(y0 + dy, S) * S
        acc = 0
        for i, dx in enumerate(taps):
            acc = acc + flat[:, row + torch.remainder(x0 + dx, S)] * wx[i]
        out = out + acc * wy[j]
    return out.view(C, *sh)


def cos2(x):
    """1 at 0 falling to 0 at |x| = 1, smooth at both ends."""
    return torch.where(x.abs() < 1, torch.cos(0.5 * math.pi * x) ** 2, torch.zeros_like(x))


class Path:
    """Per-slot periodic paths: c0 (N, D) + sum_k a_k cos(2 pi k u) + b_k sin(2 pi k u)."""

    def __init__(self, c0, K, dev):
        self.c0 = torch.nn.Parameter(c0.clone().float().to(dev))
        self.cc = torch.nn.Parameter(torch.zeros(c0.shape[0], K, 2, c0.shape[1], device=dev))
        self.k = torch.arange(1, K + 1, device=dev, dtype=torch.float32)

    def __call__(self, u):
        """u: (B, N) loop phase per slot -> (B, N, D)."""
        ang = 2 * math.pi * u[..., None] * self.k  # (B, N, K)
        return self.c0 + (
            torch.cos(ang)[..., None] * self.cc[:, :, 0]
            + torch.sin(ang)[..., None] * self.cc[:, :, 1]
        ).sum(2)


class Tissue:
    def __init__(self, N, p, S, dev, K=6, Ko=3, L=0.4, tau=0.03, sigma=0.75):
        self.N, self.S, self.dev = N, S, dev
        self.g, self.L, self.tau, self.sigma = p / N, L, tau, sigma
        assert self.g + L < 1
        self.phi = torch.tensor([(i * p % N) / N for i in range(N)], device=dev)
        self.K, self.Ko = K, Ko
        self.cubic = True
        self.solve = 0
        self.damp = 0.5
        self.strict = False
        self.wcorr = None
        self.mem_floor = 0.0
        self.mem_width = 4.0
        self.A = S * S / (N * (self.g + L)) * 1.0  # rough mean area of a live cell (px^2)
        self.D = math.sqrt(self.A)
        self.W0 = self.A  # weight unit
        self.mem = torch.nn.Parameter(
            torch.cat([torch.full((MK, 1), -5.0), torch.full((MK, 3), 1.5)], 1).to(dev)
        )
        self.rim = torch.nn.Parameter(torch.zeros(RK, 6, device=dev))
        y, x = torch.meshgrid(
            torch.arange(S, device=dev, dtype=torch.float32),
            torch.arange(S, device=dev, dtype=torch.float32),
            indexing="ij",
        )
        G = S // BLK
        self.G = G
        pix = (
            torch.stack([x, y], -1)
            .view(G, BLK, G, BLK, 2)
            .permute(0, 2, 1, 3, 4)
            .reshape(G * G, BLK * BLK, 2)
        )
        self.pix = pix  # (G^2, BLK^2, 2) pixel centres grouped by block
        self.ctr = pix.mean(1)  # (G^2, 2)
        self.ones = torch.ones(8 * S * S, device=dev)

    def init_paths(self, s0, o0, w0):
        self.s = Path(s0, self.K, self.dev)
        self.w = Path(w0[:, None] / self.W0, self.K, self.dev)
        self.o = Path(o0, self.Ko, self.dev)

    def geo_params(self):
        return [self.s.c0, self.s.cc], [self.w.c0, self.w.cc]

    def schedule(self, u):
        """u: (B,) loop phase -> per-slot age since birth ab, alive, area share r, visual share rv, bumps, sink."""
        N, g, L, tau = self.N, self.g, self.L, self.tau
        phim = torch.roll(self.phi, 1)  # birth time of slot i = division of slot i-1
        ab = torch.remainder(u[:, None] - phim[None], 1.0)  # (B, N)
        alive = ab < g + L
        ad = ab - g  # time since own division (doomed daughter phase when >= 0)
        grow = 0.5 + 0.5 * ab / g
        hold = 0.5 + 0.1 * torch.clamp(ad / (0.6 * L), max=1.0)
        shrink = 0.6 * torch.clamp(1 - (ad - 0.6 * L) / (0.4 * L), 0, 1)
        r = torch.where(ab < g, grow, torch.where(ad < 0.6 * L, hold, shrink)) * alive
        bb = cos2(ab / tau) * (ab < tau)  # just born: pulled to the midpoint with the mother
        bd = cos2(ad / tau) * (ad > -tau) * (ad < tau)  # own division, both sides
        rv = r + 0.5 * (bb + bd * (ad >= 0))  # visual scale stays continuous through division
        sink = torch.clamp((ad - 0.6 * L) / (0.4 * L), 0, 1) ** 1.5 * (ad > 0)
        return ab, alive, r, rv, bb, bd, sink

    def midpoints(self):
        """Offsets toward the division midpoint: as mother (with i+1, at phi[i]) and as daughter (with i-1, at phi[i-1])."""
        S = self.S
        tn = self.phi  # (N,)

        def at(path, i_shift):
            # path of slot (i + i_shift) evaluated at phi[i]
            idx = torch.remainder(torch.arange(self.N, device=self.dev) + i_shift, self.N)
            ang = 2 * math.pi * tn[:, None] * path.k  # (N, K)
            c0, cc = path.c0[idx], path.cc[idx]
            return c0 + (
                torch.cos(ang)[..., None] * cc[:, :, 0] + torch.sin(ang)[..., None] * cc[:, :, 1]
            ).sum(1)

        out = {}
        for name, path, per in (("s", self.s, True), ("w", self.w, False), ("o", self.o, True)):
            me, nx = at(path, 0), at(path, 1)
            d = nx - me
            if per:
                d = wrap(d, S)
            out[name] = 0.5 * d  # mother's offset to the midpoint at phi[i]
        out["sep"] = 2 * out["s"].norm(dim=1)  # daughter-mother distance at birth
        return out

    def state(self, ts, T):
        """Effective per-slot geometry at frames ts."""
        u = ts / T
        ab, alive, r, rv, bb, bd, sink = self.schedule(u)
        uu = u[:, None].expand(-1, self.N)
        mp = self.midpoints()
        dm = {k: mp[k] for k in ("s", "w", "o")}
        dd = {k: -torch.roll(mp[k], 1, 0) for k in ("s", "w", "o")}  # as daughter of slot i-1
        s = self.s(uu) + bd[..., None] * dm["s"] + bb[..., None] * dd["s"]
        w = (self.w(uu) + bd[..., None] * dm["w"] + bb[..., None] * dd["w"])[..., 0] * self.W0
        o = self.o(uu) + bd[..., None] * dm["o"] + bb[..., None] * dd["o"]
        if not self.solve:  # with the solve, the shrinking target alone squeezes the cell out
            w = w - 2.5 * self.A * sink
        if (
            self.wcorr is not None
        ):  # smooth fit of the solved weights (wfit.py): areas held without per-frame jitter
            w = w + self.wcorr(uu)[..., 0] * self.W0
        tgt = r / r.sum(1, keepdim=True).clamp(min=1e-6) * self.S * self.S
        rho = torch.sqrt(rv / 0.8).clamp(min=0.05)
        return dict(
            s=torch.remainder(s, self.S), w=w, o=o, alive=alive, tgt=tgt, rho=rho, r=r, mp=mp
        )

    def assign(self, s, w, alive, k=2):
        """The k nearest slots (in power) per pixel: idx (B, G^2, P, k)."""
        S = self.S
        B = s.shape[0]
        with torch.no_grad():
            pw = (wrap(self.ctr[None, :, None] - s[:, None], S) ** 2).sum(-1) - w[
                :, None
            ]  # (B, G^2, N)
            pw = pw.masked_fill(~alive[:, None], float("inf"))
            cand = pw.sort(dim=2).indices[..., : min(8, self.N)]  # (B, G^2, 8)
            sc = torch.gather(
                s[:, None].expand(-1, self.G**2, -1, -1), 2, cand[..., None].expand(-1, -1, -1, 2)
            )
            wc = torch.gather(w[:, None].expand(-1, self.G**2, -1), 2, cand)
            ac = torch.gather(alive[:, None].expand(-1, self.G**2, -1), 2, cand)
            pf = (wrap(self.pix[None, :, :, None] - sc[:, :, None], S) ** 2).sum(-1) - wc[
                :, :, None
            ]  # (B, G^2, P, 8)
            pf = pf.masked_fill(~ac[:, :, None], float("inf"))
            # two argmins, not topk: batched topk on MPS returned wrong indices here (95% of second choices
            # wrong at B = 4 against brute force) and is slow at B = 1
            top = [pf.argmin(3, keepdim=True)]
            for _ in range(k - 1):
                pf = pf.scatter(3, top[-1], float("inf"))
                top.append(pf.argmin(3, keepdim=True))
            top = torch.cat(top, 3)  # (B, G^2, P, k)
            idx = torch.gather(cand[:, :, None].expand(-1, -1, BLK * BLK, -1), 3, top)
        return idx

    def render(self, ts, T, tex_pre, need_stats=True):
        """tex_pre: (1, 3, S, S) pre-sigmoid shared texture. Returns frames (B, 3, S, S) and stats."""
        S, G, P = self.S, self.G, BLK * BLK
        st = self.state(ts, T)
        s, w, o, rho, alive = st["s"], st["w"], st["o"], st["rho"], st["alive"]
        B = s.shape[0]
        if (
            self.solve
        ):  # incompressible cells: solve the weights so each live cell holds its target area
            with torch.no_grad():
                ws = w.clone()
                off = (torch.arange(B, device=self.dev) * self.N).view(B, 1, 1, 1)
                for _ in range(self.solve):
                    if ws.device.type == "mps":
                        torch.mps.synchronize()  # else MPS keeps every iteration's temporaries alive
                    # reinflate: a live cell must at least own its own site (else it has no wall for Newton to move)
                    d2 = (wrap(s[:, :, None] - s[:, None], S) ** 2).sum(-1)  # (B, i, j)
                    dd = d2 - ws[:, None, :]  # power of j at site i
                    # twins (closer than 0.3 D) share a region by design: never reinflate one against the other
                    skip = ~alive[:, None, :] | (d2 < (0.3 * self.D) ** 2)
                    dd = dd.masked_fill(skip, float("inf"))
                    need = (
                        -dd.min(2).values + 0.05 * self.A
                    )  # weight that puts site i inside its own cell
                    ws = torch.where(alive & (st["tgt"] > 0.15 * self.A) & (ws < need), need, ws)
                    idx = self.assign(s, ws, alive)
                    sg = torch.gather(s, 1, idx.view(B, -1, 1).expand(-1, -1, 2)).view(
                        B, G * G, P, 2, 2
                    )
                    wg = torch.gather(ws, 1, idx.view(B, -1)).view(B, G * G, P, 2)
                    pw = (wrap(self.pix[None, :, :, None] - sg, S) ** 2).sum(-1) - wg
                    sep = wrap(sg[:, :, :, 1] - sg[:, :, :, 0], S).norm(dim=-1).clamp(min=1e-3)
                    p0 = torch.sigmoid(
                        ((pw[..., 1] - pw[..., 0]) / (2 * sep)).clamp(min=0) / self.sigma
                    )
                    fi = (idx + off).reshape(-1)
                    ar = (
                        torch.zeros(B * self.N, device=self.dev)
                        .index_add_(0, fi, torch.stack([p0, 1 - p0], -1).reshape(-1))
                        .view(B, self.N)
                    )
                    # full Newton (semi-discrete optimal transport): H is the wall Laplacian, H_ij = -sum p(1-p) / (2 sigma sep)
                    hh = (p0 * (1 - p0) / (2 * self.sigma * sep)).reshape(B, -1)
                    i0, i1 = idx[..., 0].reshape(B, -1), idx[..., 1].reshape(B, -1)
                    lin = (
                        torch.arange(B, device=self.dev)[:, None] * self.N * self.N
                        + i0 * self.N
                        + i1
                    ).view(-1)
                    Hc = (
                        torch.zeros(B * self.N * self.N, device=self.dev)
                        .index_add_(0, lin, hh.view(-1))
                        .view(B, self.N, self.N)
                    )
                    Hc = Hc + Hc.transpose(1, 2)
                    H = torch.diag_embed(Hc.sum(2)) - Hc
                    H = H.cpu().double()
                    dg = H.diagonal(dim1=1, dim2=2)
                    H = H + torch.diag_embed(
                        torch.relu(0.5 - dg) + 1e-3 * dg.mean(1, keepdim=True)
                    )  # vanished cells: a gentle push back
                    rhs = ((st["tgt"] - ar) * alive).cpu().double()
                    dw = torch.linalg.solve(H, rhs[..., None])[..., 0].float().to(self.dev)
                    ws = ws + (self.damp * dw).clamp(-0.5 * self.A, 0.5 * self.A) * alive
            st["wfit"] = ((((w - ws) / self.A) ** 2) * alive).sum(1) / alive.sum(1).clamp(min=1)
            w = (
                ws if self.strict else w + (ws - w).detach()
            )  # strict: activation cannot trade cell areas
            st["w"] = w
        idx3 = self.assign(s, w, alive, k=3)  # (B, G^2, P, 3)
        idx = idx3[..., :2].contiguous()
        flat = idx.view(B, -1)
        gat = lambda v: torch.gather(
            v, 1, flat[..., None].expand(-1, -1, v.shape[-1]) if v.dim() == 3 else flat
        ).view(B, G * G, P, 2, *v.shape[2:])
        sg, wg, og, rg = gat(s), gat(w), gat(o), gat(rho)
        dx = wrap(self.pix[None, :, :, None] - sg, S)  # (B, G^2, P, 2, 2)
        pw = (dx**2).sum(-1) - wg  # (B, G^2, P, 2)
        sep = wrap(sg[:, :, :, 1] - sg[:, :, :, 0], S).norm(dim=-1).clamp(min=1e-3)
        delta = ((pw[..., 1] - pw[..., 0]) / (2 * sep)).clamp(min=0)  # distance to the wall (px)
        p0 = torch.sigmoid(delta / self.sigma)
        # walls as drawn: the wall between twins that have just divided fades in with their separation, and the
        # third-nearest slot supplies the true outer wall while they still coincide
        s2 = torch.gather(s, 1, idx3[..., 2].reshape(B, -1, 1).expand(-1, -1, 2)).view(
            B, G * G, P, 2
        )
        w2 = torch.gather(w, 1, idx3[..., 2].reshape(B, -1)).view(B, G * G, P)
        pw2 = (wrap(self.pix[None] - s2, S) ** 2).sum(-1) - w2
        sep2 = wrap(s2 - sg[:, :, :, 0], S).norm(dim=-1).clamp(min=1e-3)
        a2 = torch.gather(alive, 1, idx3[..., 2].reshape(B, -1)).view(B, G * G, P)
        delta2 = torch.where(
            a2, ((pw2 - pw[..., 0]) / (2 * sep2)).clamp(min=0), torch.full_like(pw2, 1e4)
        )
        f1 = torch.clamp(sep / (0.3 * self.D), 1e-3, 1.0)
        f2 = torch.clamp(sep2 / (0.3 * self.D), 1e-3, 1.0)
        e1 = torch.where(
            f1 > 0.02, delta / f1, torch.full_like(delta, 1e4)
        )  # twins still on top of each other: no wall yet
        e2 = torch.where(f2 > 0.02, delta2 / f2, torch.full_like(delta, 1e4))
        dwall = torch.minimum(e1, e2)  # distance to the nearest drawn wall
        fwall = torch.where(e1 <= e2, f1, f2)
        # texture: o + (x - s) / rho, on the torus
        uv = torch.remainder(og + dx / rg[..., None], S)  # (B, G^2, P, 2, 2)
        pre = sample(tex_pre[0], uv, self.cubic)  # (3, B, G^2, P, 2)
        # interior profile: every cell shades its contents the same way with distance to its wall
        kr = (dwall / RSTEP).clamp(max=RK - 1)
        r0 = kr.floor().long().clamp(max=RK - 2)
        rf = (kr - r0)[..., None]
        rp = (self.rim[r0] * (1 - rf) + self.rim[r0 + 1] * rf).permute(3, 0, 1, 2)[
            ..., None
        ]  # (6, B, G^2, P, 1)
        col = torch.sigmoid(pre * (1 + rp[:3]) + rp[3:])
        rgb = (col[..., 0] * p0 + col[..., 1] * (1 - p0)).permute(1, 0, 2, 3)  # (B, 3, G^2, P)
        # membrane: a learned colour/opacity profile of the distance to the wall
        kx = dwall.clamp(max=MK - 1)
        k0 = kx.floor().long().clamp(max=MK - 2)
        fr = (kx - k0)[..., None]
        prof = self.mem[k0] * (1 - fr) + self.mem[k0 + 1] * fr  # (B, G^2, P, 4)
        ma = torch.sigmoid(prof[..., 0]) * (dwall < MK - 1)
        if self.mem_floor > 0:  # a membrane that is always there, fading over ~4 px
            fl = self.mem_floor * torch.clamp(1 - dwall / self.mem_width, min=0)
            ma = fl + (1 - fl) * ma
        ma = ma * fwall
        mc = torch.sigmoid(prof[..., 1:]).permute(0, 3, 1, 2)
        rgb = rgb * (1 - ma[:, None]) + mc * ma[:, None]
        img = rgb.view(B, 3, G, G, BLK, BLK).permute(0, 1, 2, 4, 3, 5).reshape(B, 3, S, S)
        stats = None
        if need_stats:
            pp = torch.stack([p0, 1 - p0], -1)  # (B, G^2, P, 2)
            off = torch.arange(B, device=self.dev)[:, None] * self.N
            fi = (flat + off).view(-1)
            area = (
                torch.zeros(B * self.N, device=self.dev)
                .index_add_(0, fi, pp.reshape(-1))
                .view(B, self.N)
            )
            mom = (
                torch.zeros(B * self.N, 2, device=self.dev)
                .index_add_(0, fi, (pp[..., None] * dx).reshape(-1, 2))
                .view(B, self.N, 2)
            )
            stats = dict(area=area, cent=mom / area[..., None].clamp(min=1.0))
        return img, st, stats, dwall.view(B, G, G, BLK, BLK).permute(0, 1, 3, 2, 4).reshape(B, S, S)


def seeds_from_image(img, dev, spacing):
    """Cell interiors of a cells-like render: minima of blurred membrane (bright ridge) strength."""
    lum = (img * torch.tensor(LUMA, device=dev).view(1, 3, 1, 1)).sum(1)[0]
    ridge = torch.relu(lum - blur(lum, 12))
    R = blur(ridge, spacing / 4)
    k = int(spacing) | 1
    mn = -F.max_pool2d(F.pad(-R[None, None], (k // 2,) * 4, mode="circular"), k, 1)[0, 0]
    ys, xs = torch.nonzero(R <= mn + 1e-9, as_tuple=True)
    pts = torch.stack([xs, ys], 1).float()
    return pts, blur(ridge, 1.5)


def tour(pts, S):
    """Short closed tour through points on the torus (greedy + 2-opt); returns an index order."""
    n = len(pts)
    d = pts[:, None] - pts[None]
    d = np.abs(d)
    d = np.minimum(d, S - d)
    D = np.sqrt((d**2).sum(-1))
    order = [0]
    left = set(range(1, n))
    while left:
        j = min(left, key=lambda k: D[order[-1], k])
        order.append(j)
        left.remove(j)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b, c, e = order[i - 1], order[i], order[j], order[(j + 1) % n]
                if D[a, c] + D[b, e] < D[a, b] + D[c, e] - 1e-6:
                    order[i : j + 1] = order[i : j + 1][::-1]
                    improved = True
    lens = [D[order[i], order[(i + 1) % n]] for i in range(n)]
    return order, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base", required=True, help="spectrum .pt of a cells-like render (1, 3, S, S//2+1, 2)"
    )
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--target", required=True)
    ap.add_argument("--frames", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--prefit", type=int, default=300)
    ap.add_argument(
        "--spacing", type=float, default=60, help="seed spacing (px), about a cell width"
    )
    ap.add_argument("--life", type=float, default=0.4, help="doomed daughter's life, in loops")
    ap.add_argument(
        "--grow", type=float, default=0.38, help="birth to division, in loops (rounded to p/N)"
    )
    ap.add_argument("--tau", type=float, default=0.03, help="division bump half-width, in loops")
    ap.add_argument("--lr", type=float, default=0.02, help="texture spectrum")
    ap.add_argument(
        "--glr", type=float, default=0.3, help="geometry (px per step; weights in cell areas)"
    )
    ap.add_argument(
        "--wlr", type=float, default=0.004, help="power weights, in cell areas per step"
    )
    ap.add_argument("--olr", type=float, default=0.3)
    ap.add_argument("--mlr", type=float, default=0.03)
    ap.add_argument("--warea", type=float, default=2.0)
    ap.add_argument("--wcvt", type=float, default=1.0)
    ap.add_argument("--wsep", type=float, default=0.5)
    ap.add_argument(
        "--sepf", type=float, default=0.6, help="daughters born this many cell widths apart"
    )
    ap.add_argument(
        "--solve", type=int, default=4, help="weight-solve iterations per frame (0: weights free)"
    )
    ap.add_argument("--damp", type=float, default=0.5, help="Newton step factor in the area solve")
    ap.add_argument("--strict", action="store_true", help="weights come only from the area solve")
    ap.add_argument(
        "--wfit", type=float, default=1.0, help="pull the weight paths onto the solved weights"
    )
    ap.add_argument(
        "--mem-floor", type=float, default=0.0, help="membrane opacity floor at the wall"
    )
    ap.add_argument(
        "--mem-width", type=float, default=4.0, help="membrane floor fades out over this many px"
    )
    ap.add_argument(
        "--mem-rgb", type=float, default=1.5, help="membrane colour logit init (2.5: near white)"
    )
    ap.add_argument("--rlr", type=float, default=0.0, help="interior profile lr (0: off)")
    ap.add_argument("--tex-init", default="base", help="base | noise")
    ap.add_argument(
        "--tex-noise",
        type=float,
        default=0.3,
        help="noise init scale, relative to the base spectrum's std",
    )
    ap.add_argument("--freeze-tex", action="store_true")
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--debug", action="store_true", help="prefit only; write overlays")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
    net = Backbone(a.net, dev)
    net.hook(layer)
    spec0 = torch.load(a.base, map_location="cpu")
    S = spec0.shape[2]
    rg = Ring(net, layer, ch, S, dev)
    T = a.frames
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)
    spec = torch.nn.Parameter(spec0.clone().to(dev))
    with torch.no_grad():
        base = torch.sigmoid(
            rg.pre(spec)
        )  # seeds and the prefit come from the base render's own cells
        if (
            a.tex_init == "noise"
        ):  # but the contents start from scratch, so the only walls are the tissue's
            g_ = torch.Generator().manual_seed(a.seed)
            spec.data.copy_(
                (torch.randn(spec0.shape, generator=g_) * spec0.std() * a.tex_noise).to(dev)
            )
    pts, ridge = seeds_from_image(base, dev, a.spacing)
    n0 = len(pts)
    # slots: N with p coprime, g = p/N ~ --grow; alive-at-0 count must equal the seed count
    best = None
    for N in range(int(n0 / (a.grow + a.life)) - 25, int(n0 / (a.grow + a.life)) + 26):
        p = round(a.grow * N)
        while math.gcd(p, N) != 1:
            p += 1
        tt = Tissue(N, p, S, dev, L=a.life, tau=a.tau)
        n_alive = int(tt.schedule(torch.zeros(1, device=dev))[1].sum())
        if best is None or abs(n_alive - n0) < abs(best[2] - n0):
            best = (N, p, n_alive)
    N, p, n_alive = best
    tis = Tissue(N, p, S, dev, L=a.life, tau=a.tau)
    if n_alive < n0:  # drop the seeds with the most ridge under them
        sc = ridge[pts[:, 1].long(), pts[:, 0].long()]
        pts = pts[sc.argsort()[:n_alive]]
    elif n_alive > n0:  # add seeds at the points farthest from all seeds
        while len(pts) < n_alive:
            yy, xx = torch.meshgrid(
                torch.arange(0, S, 8, device=dev), torch.arange(0, S, 8, device=dev), indexing="ij"
            )
            g = torch.stack([xx, yy], -1).view(-1, 2).float()
            dd = (wrap(g[:, None] - pts[None], S) ** 2).sum(-1).min(1).values
            pts = torch.cat([pts, g[dd.argmax()][None]])
    order, lens = tour(pts.cpu().numpy(), S)
    alive0 = tis.schedule(torch.zeros(1, device=dev))[1][0].cpu().numpy()
    # map the tour onto slots in cycle order: a live slot takes the next tour point; a dead slot waits on its mother's
    s0 = torch.zeros(N, 2, device=dev)
    k, last = 0, pts[order[-1]]
    start = int(np.argmax(alive0))
    for j in range(N):
        i = (start + j) % N
        if alive0[i]:
            last = pts[order[k]]
            k += 1
        s0[i] = last + (0 if alive0[i] else 3.0) * torch.randn(2, device=dev)
    log(
        f"seeds {n0} -> slots N={N} p={p} (g={p / N:.3f}, life {a.life}, alive {n_alive}); tour hops px median {np.median(lens):.0f} max {max(lens):.0f}; cell width ~{tis.D:.0f}px"
    )
    tis.init_paths(s0, s0.clone(), torch.zeros(N, device=dev))
    tis.mem_floor = a.mem_floor
    tis.mem_width = a.mem_width
    tis.mem.data[:, 1:] = a.mem_rgb

    # prefit: walls onto the render's own membranes, at t = 0 (static paths only)
    opt = torch.optim.Adam([{"params": [tis.s.c0], "lr": 1.0}, {"params": [tis.w.c0], "lr": 0.01}])
    with torch.no_grad():
        texp = rg.pre(spec)
    for it in range(a.prefit):
        opt.zero_grad()
        img, st, stt, delta = tis.render(torch.zeros(1, device=dev), T, texp)
        bnd = torch.exp(-0.5 * (delta / 2.0) ** 2)
        on = (ridge * bnd).sum() / bnd.sum()
        al = st["alive"][0]
        av = (stt["area"][0][al] / tis.A - 1) ** 2
        cv = (stt["cent"][0][al] ** 2).sum(-1) / tis.A
        loss = -on / ridge.mean() + 0.05 * av.mean() + 0.2 * cv.mean()
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == a.prefit - 1:
            log(
                f"prefit {it} ridge-on-wall {float(on / ridge.mean()):.3f} area cv {float(av.mean().sqrt()):.3f}"
            )
    tis.solve = a.solve
    tis.strict = a.strict
    tis.damp = a.damp
    with torch.no_grad():
        tis.o.c0.data.copy_(tis.s.c0.data)
        img, st, stt, delta = tis.render(torch.zeros(1, device=dev), T, texp)
        ov = base.clone()
        wall = (delta[None] < 1.0).float()
        ov = ov * (1 - wall) + wall * torch.tensor([1.0, 0.0, 1.0], device=dev).view(1, 3, 1, 1)
        Image.fromarray((ov[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
            f"{a.out}/prefit_overlay.png"
        )
        Image.fromarray((img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
            f"{a.out}/prefit_render.png"
        )
    if a.debug:
        for t in (0, T // 8, T // 4, 3 * T // 8, T // 2):
            with torch.no_grad():
                img, *_ = tis.render(
                    torch.tensor([float(t)], device=dev), T, texp, need_stats=False
                )
            Image.fromarray((img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
                f"{a.out}/dbg{t:04d}.png"
            )
        return

    @torch.no_grad()
    def evaluate(write, stride=1):
        texp = rg.pre(spec)
        ts_all = list(range(0, T, stride))
        acts = []
        for i in range(0, len(ts_all), a.mb):
            ts = torch.tensor(ts_all[i : i + a.mb], device=dev, dtype=torch.float32)
            fr, *_ = tis.render(ts, T, texp, need_stats=False)
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
                    ).save(f"{a.out}/f{t:04d}.png")
        random.seed(time.time_ns())
        return torch.cat(acts).numpy()

    with torch.no_grad():
        alone = 0
        for k in range(4):
            random.seed(1000 + k)
            alone += float(rg.act(MultiScaleCrop(224 / S, 512)(base))) / 4
    act0 = evaluate(False, stride=max(1, T // 64))
    log(
        f"base alone {alone:.3f}; tissue before {act0.mean():.3f} [{act0.min():.3f}, {act0.max():.3f}]"
    )

    sp, wp = tis.geo_params()
    groups = [
        {"params": sp, "lr": a.glr},
        {"params": wp, "lr": a.wlr},
        {"params": [tis.o.c0, tis.o.cc], "lr": a.olr},
        {"params": [tis.mem], "lr": a.mlr},
        {"params": [tis.rim], "lr": a.rlr},
    ]
    if not a.freeze_tex:
        groups.append({"params": [spec], "lr": a.lr})
    opt = torch.optim.Adam(groups)
    base_lr = [g["lr"] for g in groups]
    lum = torch.tensor(LUMA, device=dev).view(1, 3, 1, 1)
    curve = []
    for it in range(a.steps):
        frac = it / a.steps
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * 0.5 * (1 + math.cos(math.pi * frac))
        opt.zero_grad()
        ts = torch.tensor(random.sample(range(T), a.mb), device=dev, dtype=torch.float32)
        texp = rg.pre(spec)
        fr, st, stt, _ = tis.render(ts, T, texp)
        act = rg.act(MultiScaleCrop(224 / S, 512)(fr))
        hinge = (torch.relu(texp.abs() - LOGIT_RAIL) ** 2).mean()
        y = (fr * lum).sum(1, keepdim=True)
        ratio = hf_energy(torch.cat([fr[:, 2:3] - y, fr[:, 0:1] - y], 1)) / hf_energy(y).detach()
        chroma = torch.relu(ratio / CHROMA_RATIO - 1)
        al = st["alive"].float()
        nal = al.sum(1).clamp(min=1)
        area = ((((stt["area"] - st["tgt"]) / tis.A) ** 2) * al).sum(1) / nal
        cvt = (((stt["cent"] ** 2).sum(-1) / tis.A) * al * (st["r"] > 0.3)).sum(1) / nal
        sep = ((st["mp"]["sep"] / (a.sepf * tis.D) - 1) ** 2).mean()
        if "wfit" in st:
            loss_fit = a.wfit * st["wfit"].mean()
        else:
            loss_fit = 0
        pen = a.warea * area + a.wcvt * cvt + a.wsep * sep + rg.wh * hinge + rg.wc * chroma
        loss = (-act + act.detach().abs() * pen).sum() + loss_fit
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == a.steps - 1:
            curve.append((it, float(act.mean())))
            log(
                f"step {it + 1}/{a.steps} act {act.mean():.3f} area {float(area.mean().sqrt()):.3f} cvt {float(cvt.mean().sqrt()):.3f} "
                f"sep {float(st['mp']['sep'].mean() / tis.D):.2f}D membrane rgb {[round(v, 2) for v in torch.sigmoid(tis.mem[0, 1:]).tolist()]} rim0 {[round(v, 2) for v in tis.rim[0].tolist()]}"
            )

    act = evaluate(True, a.eval_stride)
    torch.save(
        {
            "N": N,
            "p": p,
            "life": a.life,
            "tau": a.tau,
            "spec": spec.detach().cpu(),
            "s": [tis.s.c0.detach().cpu(), tis.s.cc.detach().cpu()],
            "w": [tis.w.c0.detach().cpu(), tis.w.cc.detach().cpu()],
            "o": [tis.o.c0.detach().cpu(), tis.o.cc.detach().cpu()],
            "mem": tis.mem.detach().cpu(),
            "rim": tis.rim.detach().cpu(),
            "mem_floor": a.mem_floor,
            "mem_width": a.mem_width,
        },
        f"{a.out}/tissue.pt",
    )
    with torch.no_grad():
        Image.fromarray(
            (torch.sigmoid(rg.pre(spec))[0].permute(1, 2, 0).cpu().numpy() * 255)
            .round()
            .astype(np.uint8)
        ).save(f"{a.out}/texture.png")
    rep = dict(
        act_mean=float(act.mean()),
        act_min=float(act.min()),
        act_max=float(act.max()),
        act=act.tolist(),
        act_before=float(act0.mean()),
        base_alone=alone,
        N=N,
        p=p,
        life=a.life,
        tau=a.tau,
        frames=T,
        steps=a.steps,
        curve=curve,
        args=vars(a),
        secs=round(time.time() - t0),
    )
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(
        f"done act {act.mean():.3f} [{act.min():.3f}, {act.max():.3f}] (before {act0.mean():.3f}; base alone {alone:.2f})"
    )


if __name__ == "__main__":
    main()
