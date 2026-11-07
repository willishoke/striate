"""Tube bodies (gate test): the shells as real 3D bodies, swept tubes, rendered and lit in 3D.

objfield.py and objsurf.py cut the still along the segmentation, so every version inherited its
outlines and its pixels: it read as lasso-tooled. Here each shell is a closed body, and its outline in
the frame is wherever the body's silhouette falls:

  body k    a spine (uniform Catmull-Rom through --nctl 3D control points) swept by a circular cross
            section whose radius runs along it (--nrad controls, interpolated in log), squashed
            vertically by f_k, with an optional periodic ripple (segments). A snail coils its spine, a
            head tapers its radius, a bug arches and ripples. Seeded from the segment: the spine is a
            principal curve through the ridge of the mask's distance transform, the radius is the
            distance transform along it, the body rests on the ground (z = 0) raised by depth.json's
            order (--zs px per unit).
  render    orthographic, any view direction. The tube is a dense union of --nsph squashed spheres
            along the spine; for each pixel and sphere the ray hit is closed form, and the spheres of
            a body are blended by a soft max over depth (smooth tube, analytic normals). Bodies and
            the ground plane are composited by a soft z-buffer. The canvas is a torus in x, y: tilted
            views wrap along the projected lattice.
  texture   per body, in its own coordinates: along the spine (end caps continue past the ends) x
            around it (--tex-around x --tex-along logits), initialised by projecting the still, with our
            own overhead shading divided out, onto the front of the body. The ground has its own
            low-res texture.
  light     Lambert + Blinn specular, exposure-compensated as objsurf.py. Cast shadows are traced
            against every sphere of every body (closed-form ray-sphere distance, penumbra widening
            with distance), so a body shades its neighbours, the ground and its own underside in any
            view. Shadows are computed without gradient.

Training: the channel objective under a fixed light (--train-light) with every body moved and turned
a little each step (--jitter, --yaw), so shadows painted into a texture would land in the wrong place;
a silhouette term pulls each body onto its segment early and is annealed away (--wsil, --sil-until),
so the shapes start where the shells are and are then free; regularisers keep spines smooth and radii
sane; a limb term stops textures darkening toward a body's silhouette (--wrim).

usage: objtube.py --image still.png --seg segdir --target inception4c:7 --out DIR
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, math, random, subprocess, time
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from striate.model import Backbone
from striate.util import pick_device
from zoo import MultiScaleCrop
from ring import Ring
from objsurf import Light
from objfield import build_scene

FFMPEG = ("/private/tmp/claude-501/-Users-lawvere-striate/39cfca31-2c02-4b02-a696-d14df31b10bf/scratchpad/"
          "videnv/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")


# ---------------------------------------------------------------- seeding (numpy)

def cr_matrix(nc, n):
    """(n, nc): uniform Catmull-Rom through nc points, sampled at n even parameters; the ends are
    padded by reflection, so the curve starts and stops at the first and last point."""
    M = np.zeros((n, nc))
    for i, t in enumerate(np.linspace(0, nc - 1, n)):
        j = min(int(math.floor(t)), nc - 2)
        u = t - j
        w = [0.5 * (-u**3 + 2 * u**2 - u), 0.5 * (3 * u**3 - 5 * u**2 + 2), 0.5 * (-3 * u**3 + 4 * u**2 + u), 0.5 * (u**3 - u**2)]
        for q, wq in zip((j - 1, j, j + 1, j + 2), w):
            if q < 0:
                M[i, 0] += 2 * wq; M[i, 1] -= wq
            elif q >= nc:
                M[i, nc - 1] += 2 * wq; M[i, nc - 2] -= wq
            else:
                M[i, q] += wq
    return M


def lin_matrix(nc, n):
    M = np.zeros((n, nc))
    for i, t in enumerate(np.linspace(0, nc - 1, n)):
        j = min(int(math.floor(t)), nc - 2)
        u = t - j
        M[i, j] += 1 - u; M[i, j + 1] += u
    return M


def seed_body(mask, S, nc, nr):
    """Spine control points (nc, 2) in canvas px (x, y; may run past the canvas, it's a torus) and
    radii (nr,) for one segment: a principal curve through the ridge of its distance transform."""
    ys, xs = np.nonzero(mask)
    cy = S / (2 * math.pi) * math.atan2(np.sin(ys * 2 * math.pi / S).mean(), np.cos(ys * 2 * math.pi / S).mean())
    cx = S / (2 * math.pi) * math.atan2(np.sin(xs * 2 * math.pi / S).mean(), np.cos(xs * 2 * math.pi / S).mean())
    cy, cx = cy % S, cx % S
    dy = (ys - cy + S / 2) % S - S / 2
    dx = (xs - cx + S / 2) % S - S / 2
    oy, ox = int(math.floor(dy.min())) - 3, int(math.floor(dx.min())) - 3
    h, w = int(math.ceil(dy.max())) - oy + 4, int(math.ceil(dx.max())) - ox + 4
    loc = np.zeros((h, w), bool)
    loc[np.round(dy - oy).astype(int), np.round(dx - ox).astype(int)] = True
    dt = ndi.distance_transform_edt(loc)
    ridge = (dt == ndi.maximum_filter(dt, 3)) & (dt >= 1.5)
    ry, rx = np.nonzero(ridge)
    if len(ry) < 3:
        ry, rx = np.nonzero(dt >= 0.7 * dt.max())
    wgt = dt[ry, rx]
    P = np.stack([rx, ry], 1).astype(np.float64)
    mu = (P * wgt[:, None]).sum(0) / wgt.sum()
    cov = ((P - mu).T * wgt) @ (P - mu) / wgt.sum()
    ev, evec = np.linalg.eigh(cov)
    ax = evec[:, -1]
    pr = (P - mu) @ ax
    lo, hi = np.quantile(pr, 0.03), np.quantile(pr, 0.97)
    if hi - lo < 4:
        lo, hi = -2.0, 2.0
    ctl = mu + np.linspace(lo, hi, nc)[:, None] * ax[None]
    for _ in range(12 if hi - lo > dt.max() else 0):  # principal curve: each control point moves to the mean of the ridge near it
        seg = np.linspace(0, 1, 64)
        poly = np.concatenate([ctl[j] + seg[:, None] * (ctl[j + 1] - ctl[j]) for j in range(nc - 1)])
        tpar = np.concatenate([(j + seg) / (nc - 1) for j in range(nc - 1)])
        d2 = ((P[:, None] - poly[None]) ** 2).sum(-1)
        tau = tpar[d2.argmin(1)]
        new = ctl.copy()
        for j in range(nc):
            k = np.clip(1 - np.abs(tau * (nc - 1) - j), 0, 1) * wgt
            if k.sum() > 1e-6:
                new[j] = (P * k[:, None]).sum(0) / k.sum()
        sm = new.copy()
        sm[1:-1] = 0.7 * new[1:-1] + 0.15 * (new[:-2] + new[2:])
        ctl = sm
    pts = (cr_matrix(nc, 4 * nr) @ ctl)
    rr = ndi.map_coordinates(dt, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
    rad = np.interp(np.linspace(0, 1, nr), np.linspace(0, 1, len(rr)), ndi.uniform_filter1d(rr, 3))
    ctl = ctl + np.array([ox + cx, oy + cy])
    return ctl, np.maximum(rad, 2.0)


# ---------------------------------------------------------------- bodies

class Tubes(torch.nn.Module):
    def __init__(self, ctl, rad, a, dev):
        super().__init__()
        K = ctl.shape[0]
        self.K, self.N, self.a, self.dev = K, a.nsph, a, dev
        self.ctl = torch.nn.Parameter(torch.tensor(ctl, dtype=torch.float32, device=dev))  # (K, nc, 3)
        self.logr = torch.nn.Parameter(torch.tensor(np.log(rad), dtype=torch.float32, device=dev))  # (K, nr)
        self.fq = torch.nn.Parameter(torch.full((K,), math.log((a.f0 - 0.25) / (1.25 - a.f0)), device=dev))
        self.rip = torch.nn.Parameter(torch.zeros(K, device=dev))  # ripple amplitude (tanh) * 0.25
        self.ripf = torch.nn.Parameter(torch.full((K,), float(a.ripple_freq), device=dev))
        self.ripp = torch.nn.Parameter(torch.zeros(K, device=dev))
        self.tex = torch.nn.Parameter(torch.zeros(K, 3, a.tex_around, a.tex_along, device=dev))
        self.ground = torch.nn.Parameter(torch.zeros(3, a.ground_res, a.ground_res, device=dev))
        self.M = torch.tensor(cr_matrix(ctl.shape[1], a.nsph), dtype=torch.float32, device=dev)
        self.Mr = torch.tensor(lin_matrix(rad.shape[1], a.nsph), dtype=torch.float32, device=dev)
        self.register_buffer("logr0", self.logr.detach().clone())  # the fitted size: growth beyond it is bounded
        self.disp = torch.nn.Parameter(torch.zeros(K, getattr(a, "disp_around", 8), getattr(a, "disp_along", 12), device=dev))

    def f(self):
        return 0.25 + torch.sigmoid(self.fq)

    def base(self):
        """Spine samples (K, N, 3) and radii (K, N) at rest."""
        return torch.einsum("nc,kcd->knd", self.M, self.ctl), torch.exp(torch.einsum("nc,kc->kn", self.Mr, self.logr))

    def geom_params(self):
        """(positions in px, shape parameters, log-scale parameters)"""
        return [self.ctl], [], [self.logr]

    def rip_u(self, s, Lt):
        """Where along a body its ripple runs: arclength for a tube (ripf = ridges per body)."""
        return s / (Lt[:, None] + 1e-3)

    def reg(self, mask=None):
        """Spine curvature and radius smoothness (per body, then averaged over mask)."""
        Cr = torch.einsum("nc,kcd->knd", self.M, self.ctl)
        curv = ((Cr[:, 2:] - 2 * Cr[:, 1:-1] + Cr[:, :-2]) ** 2).sum(-1).mean(1) / 4.0
        rs = ((self.logr[:, 2:] - 2 * self.logr[:, 1:-1] + self.logr[:, :-2]) ** 2).mean(1) / 0.01 * 0.2 + (torch.relu(math.log(2.0) - self.logr) ** 2).mean(1) * 2
        grow = (torch.relu(self.logr - self.logr0 - self.a.size_slack) ** 2).mean(1) * self.a.wsize
        v = curv + rs + grow
        return v.mean() if mask is None else (v * mask).sum() / mask.sum().clamp(min=1)

    def spheres(self, jit=None, yaw=None, wob=None, sub=None):
        """Sphere centres C (B, K, N, 3), radii r (K, N), tangents T (B, K, N, 3), arclength s (K, N),
        texture span (s_lo, s_len) (K,), squash f (K,). jit (B, K, 2) px, yaw (B, K) rad, wob (K, N, 3) px
        or (B, K, N, 3); sub: list of body indices to keep (applied last)."""
        f = self.f()
        C, r = self.base()
        seg = (C[:, 1:] - C[:, :-1]).norm(dim=-1)
        s = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(-1)], -1)  # (K, N), from the rest pose
        Lt = s[:, -1]
        r = r * (1 + 0.25 * torch.tanh(self.rip)[:, None] * torch.sin(2 * math.pi * self.ripf[:, None] * self.rip_u(s, Lt) + self.ripp[:, None]))
        C = C[None]
        if wob is not None:
            C = C + (wob if wob.dim() == 4 else wob[None])
        if jit is not None or yaw is not None:
            B = (jit if jit is not None else yaw).shape[0]
            C = C.expand(B, *C.shape[1:])
            mu = C.mean(2, keepdim=True)
            if yaw is not None:
                cy, sy = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
                dx, dy = C[..., 0] - mu[..., 0], C[..., 1] - mu[..., 1]
                C = torch.stack([mu[..., 0] + cy * dx - sy * dy, mu[..., 1] + sy * dx + cy * dy, C[..., 2]], -1)
            if jit is not None:
                C = C + F.pad(jit, (0, 1))[:, :, None, :]
        T = torch.cat([C[..., 1:2, :] - C[..., 0:1, :], C[..., 2:, :] - C[..., :-2, :], C[..., -1:, :] - C[..., -2:-1, :]], -2)
        T = T / (T.norm(dim=-1, keepdim=True) + 1e-6)
        # frame across the tube, parallel-transported along the spine from "up" at its start, so the
        # around coordinate stays defined where the spine climbs steeply
        zh = torch.tensor([0.0, 0.0, 1.0], device=C.device)
        b0 = zh - T[..., 0, 2:3] * T[..., 0, :]
        alt = torch.tensor([1.0, 0.0, 0.0], device=C.device) - T[..., 0, 0:1] * T[..., 0, :]
        b0 = torch.where(b0.norm(dim=-1, keepdim=True) > 0.2, b0, alt)
        Bs = [b0 / b0.norm(dim=-1, keepdim=True)]
        for i in range(1, T.shape[-2]):
            b = Bs[-1] - (Bs[-1] * T[..., i, :]).sum(-1, keepdim=True) * T[..., i, :]
            Bs.append(b / (b.norm(dim=-1, keepdim=True) + 1e-6))
        Bv = torch.stack(Bs, -2)
        out = dict(C=C, r=r, T=T, Bv=Bv, s=s, s_lo=-r[:, 0], s_len=Lt + r[:, 0] + r[:, -1], f=f,
                   disp=getattr(self.a, "disp_amp", 0.0) * torch.tanh(self.disp) if hasattr(self, "disp") else None)
        if sub is not None:
            out = {k: (None if v is None else v[:, sub] if k in ("C", "T", "Bv") else v[sub]) for k, v in out.items()}
        return out


class Coils(Tubes):
    """Shell bodies: the spine is a logarithmic spiral, the tube radius grows with it (self-similar,
    as a snail's whorls do), and the spire rises toward the apex. Per body: position, size (radius of
    the last whorl), turns, expansion per turn W, tube/spiral ratio c, spire rise h, axis tilt and
    azimuth, start angle, handedness (fixed). Few turns and a large W give a tapered horn (a head or a
    bug); a turn or more gives a coiled shell. Bodies rest on the ground, lifted by depth order."""

    NAMES = ("x", "y", "lrho", "tq", "lw", "cq", "h", "alpha", "beta", "th0", "lift")

    def __init__(self, P, hand, a, dev):
        K = len(hand)
        super().__init__(np.zeros((K, a.nctl, 3)), np.ones((K, a.nrad)), a, dev)
        for n in self.NAMES:
            setattr(self, n, torch.nn.Parameter(torch.tensor(P[n], dtype=torch.float32, device=dev)))
        self.register_buffer("hand", torch.tensor(hand, dtype=torch.float32, device=dev))
        self.register_buffer("lrho0", self.lrho.detach().clone())

    def base(self, f=None):
        u = torch.linspace(0, 1, self.N, device=self.dev)[None]
        turns = 0.05 + F.softplus(self.tq)[:, None]
        Th = 2 * math.pi * turns
        g = torch.exp(self.lw)[:, None] / (2 * math.pi)  # lw = log of the log-expansion per turn
        rho_e = torch.exp(self.lrho)[:, None]
        rho = rho_e * torch.exp(-g * Th * (1 - u))
        ph = self.hand[:, None] * Th * u + self.th0[:, None]
        hh = self.h.clamp(-self.a.coil_max_rise, self.a.coil_max_rise) if hasattr(self.a, "coil_max_rise") else self.h
        loc = torch.stack([rho * torch.cos(ph), rho * torch.sin(ph), hh[:, None] * (rho_e - rho)], -1)
        al = self.alpha.clamp(-self.a.coil_max_tilt, self.a.coil_max_tilt) if hasattr(self.a, "coil_max_tilt") else self.alpha
        ca, sa = torch.cos(al)[:, None], torch.sin(al)[:, None]
        loc = torch.stack([loc[..., 0], ca * loc[..., 1] - sa * loc[..., 2], sa * loc[..., 1] + ca * loc[..., 2]], -1)
        cb, sb = torch.cos(self.beta)[:, None], torch.sin(self.beta)[:, None]
        loc = torch.stack([cb * loc[..., 0] - sb * loc[..., 1], sb * loc[..., 0] + cb * loc[..., 1], loc[..., 2]], -1)
        r = (0.4 + 0.6 * torch.sigmoid(self.cq))[:, None] * rho  # c >= 0.4: successive whorls touch
        mu = loc.mean(1, keepdim=True)
        C = loc - mu * torch.tensor([1.0, 1.0, 0.0], device=self.dev) + torch.stack([self.x, self.y, torch.zeros_like(self.x)], -1)[:, None]
        bottom = (C[..., 2] - (self.f() if f is None else f)[:, None] * r).min(1)[0]
        C = C + torch.stack([torch.zeros_like(bottom), torch.zeros_like(bottom), self.lift - bottom], -1)[:, None]
        return C, r

    def geom_params(self):
        return [self.x, self.y, self.lift], [self.alpha, self.beta, self.th0, self.h, self.tq], [self.lrho, self.lw, self.cq]

    def turns(self):
        return 0.05 + F.softplus(self.tq)

    def rip_u(self, s, Lt):
        """Ridges per turn for a coil: the ripple is phased to the spiral (growth lines and sutures)."""
        return torch.linspace(0, 1, self.N, device=self.dev)[None] * self.turns()[:, None]

    def reg(self, mask=None):
        v = F.relu(self.h.abs() - 1.5) ** 2 + torch.relu(math.log(4.0) - self.lrho) ** 2 + torch.relu(self.lrho - self.lrho0 - self.a.size_slack) ** 2 * self.a.wsize
        return v.mean() if mask is None else (v * mask).sum() / mask.sum().clamp(min=1)


class Mixed(Tubes):
    """Each body is a free tube or a coil, whichever fitted its segment better (is_coil). Both
    families' parameters live here; a body uses one, the other gets no gradient."""

    def __init__(self, ctl, rad, P, hand, is_coil, a, dev):
        super().__init__(ctl, rad, a, dev)
        self.coil = Coils(P, hand, a, dev)
        del self.coil.tex, self.coil.ground, self.coil.disp
        self.register_buffer("is_coil", torch.tensor(is_coil, dtype=torch.bool, device=dev))

    def base(self):
        Ct, rt = Tubes.base(self)
        Cc, rc = self.coil.base(f=self.f())
        m = self.is_coil
        return torch.where(m[:, None, None], Cc, Ct), torch.where(m[:, None], rc, rt)

    def geom_params(self):
        g = self.coil.geom_params()
        return [self.ctl] + g[0], g[1], [self.logr] + g[2]

    def rip_u(self, s, Lt):
        return torch.where(self.is_coil[:, None], self.coil.rip_u(s, Lt), Tubes.rip_u(self, s, Lt))

    def reg(self, mask=None):
        m = self.is_coil.float()
        return Tubes.reg(self, 1 - m) * (1 - m).mean() + self.coil.reg(m) * m.mean()

    @staticmethod
    def from_state(st, a, dev):
        P = {n: st["coil." + n].cpu().numpy() for n in Coils.NAMES}
        mx = Mixed(st["ctl"].cpu().numpy(), np.exp(st["logr"].cpu().numpy()), P, st["coil.hand"].cpu().numpy(), st["is_coil"].cpu().numpy(), a, dev)
        for n in ("capc", "capr", "tex0"):
            if n in st:
                mx.register_buffer(n, st[n].to(dev))
        mx.load_state_dict(st, strict=False)
        return mx


def near_pairs(tb, S, pad=4.0):
    """Body pairs (M, 2) whose rest-pose bounding spheres overlap (plus pad), on the torus."""
    with torch.no_grad():
        C, r = tb.base()
        mu = C.mean(1)
        ext = ((C - mu[:, None]).norm(dim=-1) + r).max(1)[0]
        d = mu[:, None, :2] - mu[None, :, :2]
        d = d - S * torch.round(d / S)
        dist = torch.sqrt((d**2).sum(-1) + (mu[:, None, 2] - mu[None, :, 2]) ** 2)
        hit = (dist < ext[:, None] + ext[None] + pad) & torch.triu(torch.ones_like(dist, dtype=torch.bool), 1)
        return hit.nonzero()


def amodal_masks(sc, S):
    """(K, S, S) bool: each segment's amodal shape from objfield's scene (its outline extended under
    the shells confidently in front of it)."""
    K, Hw = sc["K"], sc["H"]
    out = np.zeros((K, S, S), bool)
    for k in range(1, K + 1):
        oy, ox = sc["origin"][k]
        iy, ix = (oy + np.arange(Hw)) % S, (ox + np.arange(Hw)) % S
        out[k - 1][np.ix_(iy, ix)] = sc["sdf"][k] < 0
    return out


def mask_caps(M, S, slack):
    """Per mask: circular centroid (x, y) and the radius that holds it, times (1 + slack)."""
    cen, rad = np.zeros((len(M), 2), np.float32), np.zeros(len(M), np.float32)
    for k, m in enumerate(M):
        ys, xs = np.nonzero(m)
        cy = (S / (2 * math.pi) * math.atan2(np.sin(ys * 2 * math.pi / S).mean(), np.cos(ys * 2 * math.pi / S).mean())) % S
        cx = (S / (2 * math.pi) * math.atan2(np.sin(xs * 2 * math.pi / S).mean(), np.cos(xs * 2 * math.pi / S).mean())) % S
        dy = (ys - cy + S / 2) % S - S / 2
        dx = (xs - cx + S / 2) % S - S / 2
        cen[k] = cx, cy
        rad[k] = np.sqrt(dx**2 + dy**2).max() * (1 + slack)
    return cen, rad


def cap_loss(tb, cen, rad, S):
    """A body's footprint (spheres in xy) may not reach past its segment's cap circle: squared excess
    in px, averaged over bodies. Hard-ish: the caller weights it strongly."""
    C, r = tb.base()
    d = C[..., :2] - cen[:, None]
    d = d - S * torch.round(d / S)
    reach = (d.norm(dim=-1) + r).max(1)[0]
    return (torch.relu(reach - rad) ** 2).mean()


def stack_bodies(tb, L, zord, a, dev, log):
    """Set each body's height C-style: back to front by depth order, a body is raised until its
    visible top (where the segment shows in the still) clears the tops already placed beneath it by
    --margin (90th percentile), so bodies overlap in depth rather than tile. Bodies first rest on the
    ground (z = 0)."""
    S = L.shape[0]
    top = Cam(np.eye(3), S, S, (S / 2, S / 2, 0.0), S)
    with torch.no_grad():
        # rest every body on the ground
        C, r = tb.base()
        bottom = (C[..., 2] - tb.f()[:, None] * r).min(1)[0]
        if isinstance(tb, Mixed):
            tube = ~tb.is_coil
            # tubes lie flat: each control point at the squashed radius there, so the spine is level
            ir = torch.round(torch.linspace(0, tb.N - 1, tb.ctl.shape[1], device=dev)).long()
            flat = tb.f()[:, None] * r[:, ir]
            tb.ctl[:, :, 2] = torch.where(tube[:, None], flat, tb.ctl[:, :, 2])
            tb.coil.lift.copy_(torch.where(tb.is_coil, torch.zeros_like(tb.coil.lift), tb.coil.lift))
        elif isinstance(tb, Coils):
            tb.lift.zero_()
        else:
            tb.ctl[:, :, 2] -= bottom[:, None]
        sph = tb.spheres()
        pr = build_pairs(sph, top, 2.0)
        pix_all, obj_all = torch.from_numpy(pr["pix"]).to(dev), torch.from_numpy(pr["obj"]).to(dev)
        shf_all = F.pad(torch.from_numpy(pr["shift"]).to(dev), (0, 1))
        zh = torch.tensor([0.0, 0.0, 1.0], device=dev)
        X0_all = torch.stack([(pix_all % S).float(), torch.div(pix_all, S, rounding_mode="floor").float(), torch.zeros_like(pix_all, dtype=torch.float32)], -1) - shf_all
        lab = torch.from_numpy(L.reshape(-1) - 1).to(dev)
        canvas = torch.full((S * S,), -1e3, device=dev)
        dz = torch.zeros(tb.K, device=dev)
        is_tube = (~tb.is_coil) if isinstance(tb, Mixed) else torch.full((tb.K,), not isinstance(tb, Coils), device=dev)
        nc = tb.ctl.shape[1]
        ctl_of = tb.M.argmax(1)  # (N,) the control point that moves each spine sample most

        def hits(k):
            sph = tb.spheres()
            sel = obj_all == int(k)
            Bv = sph["Bv"]
            Nv = torch.cross(sph["T"], Bv, dim=-1)
            hb = body_hits(X0_all[sel], obj_all[sel], sph["C"], sph["r"], sph["T"], Bv, Nv, sph["f"], sph["s"], zh, a.sig, a.tau_s)[0]
            return pix_all[sel], hb[:, 0], hb[:, 1], X0_all[sel], sph

        for k in np.argsort(np.array(zord)[1:]):
            p_, w_, c_, x_, sph = hits(k)
            body = c_ > 0.5
            vis = body & (lab[p_] == int(k))
            under = canvas[p_]
            ok = vis & (under > -1e2)
            need = under - w_ + a.margin
            if bool(is_tube[k]) and ok.sum() > 0:
                # a tube may tilt and bend: each control point rises as far as the visible pixels nearest
                # to it need, and where nothing of it shows it stays down (it ducks under its neighbours)
                d2 = ((x_[ok, None, :2] - sph["C"][0, k, None, :, :2]) ** 2).sum(-1)
                cj = ctl_of[d2.argmin(1)]
                dzc = torch.zeros(nc, device=dev)
                for j in range(nc):
                    nj = need[ok][cj == j]
                    if nj.numel() >= 8:
                        dzc[j] = max(0.0, float(torch.quantile(nj, a.stack_q)))
                # no steeper than --stack-slope: a control point next to a raised one comes part way up
                dxy = (tb.ctl[k, 1:, :2] - tb.ctl[k, :-1, :2]).norm(dim=-1)
                for _ in range(nc):
                    dzc[1:] = torch.maximum(dzc[1:], dzc[:-1] - a.stack_slope * dxy)
                    dzc[:-1] = torch.maximum(dzc[:-1], dzc[1:] - a.stack_slope * dxy)
                tb.ctl[k, :, 2] += dzc
                dz[k] = dzc.max()
            else:
                dz[k] = max(0.0, float(torch.quantile(need[ok], a.stack_q))) if ok.sum() > 0 else 0.0
                if isinstance(tb, Mixed):
                    tb.coil.lift[k] += dz[k]
                elif isinstance(tb, Coils):
                    tb.lift[k] += dz[k]
                else:
                    tb.ctl[k, :, 2] += dz[k]
            p_, w_, c_, _, _ = hits(k)
            body = c_ > 0.5
            canvas[p_[body]] = torch.maximum(canvas[p_[body]], w_[body])
    log(f"stacked back to front: lift median {float(dz.median()):.1f} px, max {float(dz.max()):.1f}; top of pile {float(canvas.max()):.1f} px")
    return dz


def fit_light(tb, sph, cam, pairs, still, lab, a, dev):
    """The light the still looks lit by, given these bodies: over a grid of directions, still luminance
    ~ p * (n.l * visibility) + q by least squares on the pixels each body owns; best R^2 wins.
    Returns (az, el, p, q, r2, the n.l * visibility map)."""
    Y = (still[0] * torch.tensor(LUMA3, device=dev)[:, None, None]).sum(0).reshape(-1)
    best = None
    with torch.no_grad():
        o = render(tb, sph, cam, pairs, [overhead_rig(dev)], a, clay=1.0, shadows=False, aux=True, lab=lab)
        m = o["own"][0] > 0.6
        for el in (30, 50, 70):
            for az in range(0, 360, 30):
                rg = dict(l=torch.tensor([math.cos(math.radians(el)) * math.cos(math.radians(az)), math.cos(math.radians(el)) * math.sin(math.radians(az)),
                                          math.sin(math.radians(el))], device=dev), sh=torch.zeros(4, 3, device=dev), dif=torch.tensor(1.0, device=dev),
                          tint=torch.ones(3, device=dev), spec=0.0, shin=1.0, pen0=a.pen0, pen=a.pen)
                nv = render(tb, sph, cam, pairs, [rg], a, clay=1.0)[0, 0].reshape(-1) * math.sin(math.radians(el))
                x, y = nv[m], Y[m]
                A = torch.stack([x, torch.ones_like(x)], 1)
                sol = torch.linalg.lstsq(A.cpu(), y.cpu()[:, None]).solution[:, 0]
                r2 = 1 - float(((A.cpu() @ sol - y.cpu()) ** 2).mean() / y.var().cpu())
                if best is None or r2 > best[4]:
                    best = (az, el, float(sol[0]), float(sol[1]), r2, nv)
    return best


LUMA3 = (0.299, 0.587, 0.114)


def overhead_rig(dev):
    return dict(l=torch.tensor([0.0, 0.0, 1.0], device=dev), sh=torch.zeros(4, 3, device=dev), dif=torch.tensor(1.0, device=dev),
                tint=torch.ones(3, device=dev), spec=0.0, shin=1.0, pen0=0.6, pen=0.06)


def wrap_texture(tb, sph, cam, pairs, still, lab, a, dev, log):
    """Each body's texture from the still wrapped onto it: every visible pixel of the segment goes
    through the body (ray hit -> along, around) into its texture, with the shading of a light fitted
    to the still divided out; the hidden part is filled by a prior: mirrored around the tube across
    the silhouette (the back copies the front), then for coils continued along the spiral (a hidden
    whorl copies its visible neighbour), and only the rest diffused."""
    az, el, p, q, r2, nv = fit_light(tb, sph, cam, pairs, still, lab, a, dev)
    log(f"light fitted to the still: az {az} el {el}, luminance = {p:.2f} * n.l + {q:.2f} (R^2 {r2:.2f})")
    with torch.no_grad():
        shade = (q + p * nv).clamp(min=1e-3)
        o = render(tb, sph, cam, pairs, [overhead_rig(dev)], a, clay=1.0, shadows=False, aux=True, lab=lab)
        own_px = o["own"][0]
        mean_sh = float(shade[own_px > 0.6].mean())
        sh_ = shade.clamp(min=0.4 * mean_sh) / mean_sh
        albpx = (still[0].reshape(3, -1) / sh_[None]).clamp(1 / 512, 1 - 1 / 512).T  # (HW, 3)
        pix, obj, ep, den = o["pix"], o["obj"], o["ep"][0], o["den"][0]
        own = ep / den[pix]
        # every still pixel goes to the body that shows there in our render (usually its own segment's;
        # where a body's edge overhangs a neighbour, the neighbour's pixels), so the top view reproduces
        # the still at init
        best = torch.zeros(den.shape[0], device=dev).scatter_reduce(0, pix, ep, "amax", include_self=True)
        keep = (ep >= best[pix] - 1e-9) & (o["den"][0][pix] > 0) & ((1 - o["gown"][0])[pix] > 0.5)  # the body on top at each pixel
        K, Ha, Wl = tb.K, a.prior_around, a.prior_along
        x = o["tu"][0][keep] * Ha
        y = o["tv"][0][keep].clamp(0, 1) * (Wl - 1)
        x0, y0 = torch.floor(x), torch.floor(y).clamp(max=Wl - 2)
        fx, fy = x - x0, y - y0
        x0, y0 = x0.long() % Ha, y0.long()
        ob, col = obj[keep], albpx[pix[keep]]
        acc = torch.zeros(K * Ha * Wl, 3, device=dev)
        cnt = torch.zeros(K * Ha * Wl, device=dev)
        for dx_, dy_, w_ in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):  # bilinear splat
            fl = (ob * Ha + (x0 + dx_) % Ha) * Wl + (y0 + dy_)
            acc.index_add_(0, fl, col * w_[:, None])
            cnt.index_add_(0, fl, w_)
        T_ = (acc / cnt.clamp(min=1e-6)[:, None]).reshape(K, Ha, Wl, 3).permute(0, 3, 1, 2).cpu().numpy()
        seen = (cnt > 0.15).reshape(K, Ha, Wl).cpu().numpy()
    # small holes inside the seen patch: one pass of neighbour averaging
    have = seen.copy()
    for _ in range(2):
        nb = (np.roll(have, 1, 1) | np.roll(have, -1, 1)) & np.pad(have, ((0, 0), (0, 0), (1, 1)))[:, :, 2:] & np.pad(have, ((0, 0), (0, 0), (1, 1)))[:, :, :-2]
        hole = ~have & nb
        if hole.any():
            avg = (np.roll(T_, 1, 2) + np.roll(T_, -1, 2) + np.pad(T_, ((0, 0), (0, 0), (0, 0), (1, 1)), mode="edge")[..., 2:] + np.pad(T_, ((0, 0), (0, 0), (0, 0), (1, 1)), mode="edge")[..., :-2]) / 4
            T_ = np.where(hole[:, None], avg, T_)
            have = have | hole
    n_seen = have.copy()
    # mirror around the tube across the silhouette: theta -> pi - theta (the front sits round theta = 0)
    ia = (np.round(Ha / 2 - np.arange(Ha)) % Ha).astype(int)
    mir = have[:, ia, :]
    fill = ~have & mir
    T_ = np.where(fill[:, None], T_[:, :, ia, :], T_)
    have = have | fill
    n_mir = have.copy()
    # coils: a hidden whorl copies the whorl next to it along the spiral
    if isinstance(tb, (Mixed, Coils)):
        coil = tb.is_coil.cpu().numpy() if isinstance(tb, Mixed) else np.ones(K, bool)
        turns = (tb.coil.turns() if isinstance(tb, Mixed) else tb.turns()).detach().cpu().numpy()
        sk = sph["s"].detach().cpu().numpy()
        s_lo, s_len = sph["s_lo"].detach().cpu().numpy(), sph["s_len"].detach().cpu().numpy()
        N = sk.shape[1]
        uu = np.linspace(0, 1, N)
        for k in np.nonzero(coil)[0]:
            if turns[k] < 0.6:
                continue
            sj = s_lo[k] + np.linspace(0, 1, Wl) * s_len[k]
            u = np.interp(sj, sk[k], uu)
            for sgn in (1, -1):
                s2 = np.interp(np.clip(u + sgn / turns[k], 0, 1), uu, sk[k])
                j2 = np.clip(np.round((s2 - s_lo[k]) / s_len[k] * (Wl - 1)), 0, Wl - 1).astype(int)
                ok = np.abs(u + sgn / turns[k] - np.clip(u + sgn / turns[k], 0, 1)) < 1e-6
                src_have = have[k][:, j2] & ok[None]
                f2 = ~have[k] & src_have
                T_[k] = np.where(f2[None], T_[k][:, :, j2], T_[k])
                have[k] = have[k] | f2
    n_per = have.copy()
    # the rest diffuses in
    mean = (T_ * have[:, None]).sum((2, 3), keepdims=True) / np.maximum(have[:, None].sum((2, 3), keepdims=True), 1)
    Tf = np.where(have[:, None], T_, mean)
    for _ in range(60):
        pp = np.concatenate([Tf[:, :, -1:], Tf, Tf[:, :, :1]], 2)
        pp = np.pad(pp, ((0, 0), (0, 0), (0, 0), (1, 1)), mode="edge")
        bl2 = (pp[:, :, :-2, 1:-1] + pp[:, :, 2:, 1:-1] + pp[:, :, 1:-1, :-2] + pp[:, :, 1:-1, 2:]) / 4
        Tf = np.where(have[:, None], T_, bl2)
    lg = lambda x: np.log(np.clip(x, 1 / 512, 1 - 1 / 512) / (1 - np.clip(x, 1 / 512, 1 - 1 / 512)))
    with torch.no_grad():
        # the wrapped still is a fixed prior at its own resolution; the learned texture is a coarser
        # residual on top of it (starts at zero)
        if hasattr(tb, "tex0"):
            del tb.tex0
        tb.register_buffer("tex0", torch.from_numpy(lg(Tf).astype(np.float32)).to(dev))
        tb.tex.data.zero_()
    fr = lambda x: float(x.mean())
    log(f"texture wrapped ({Ha} x {Wl} per body): seen {fr(n_seen):.2f}, + mirrored {fr(n_mir) - fr(n_seen):.2f}, "
        f"+ along the spiral {fr(n_per) - fr(n_mir):.2f}, diffused {1 - fr(n_per):.2f}")
    return dict(az=az, el=el, p=p, q=q, r2=r2, seen=fr(n_seen), mirrored=fr(n_mir) - fr(n_seen), spiral=fr(n_per) - fr(n_mir), diffused=1 - fr(n_per))


def contact_loss(tb, nbr, S, tol):
    """Neighbouring bodies press against each other rather than overlapping deeply: for each pair of
    adjacent segments, the overlap of their spheres (3D, z scaled by the bodies' squash) beyond tol of
    the smaller radius, squared, summed over sphere pairs and averaged over body pairs."""
    C, r = tb.base()
    f = tb.f()
    i, j = nbr[:, 0], nbr[:, 1]
    d = C[i][:, :, None, :] - C[j][:, None, :, :]
    dxy = d[..., :2] - S * torch.round(d[..., :2] / S)
    fz = 2 / (f[i] + f[j])[:, None, None]
    dist = torch.sqrt((dxy**2).sum(-1) + (d[..., 2] * fz) ** 2 + 1e-6)
    ri, rj = r[i][:, :, None], r[j][:, None, :]
    pen = torch.relu(ri + rj - dist - tol * torch.minimum(ri, rj))
    return ((pen**2).sum((1, 2)) / C.shape[1]).mean() / 4.0



def coil_prefit(L, zord, a, dev, log, steps=150, stride=3, target=None, cap=None, spill_w=0.3):
    """Fit every segment with coil bodies from a grid of starting guesses (handedness x heading x
    tilt), on silhouettes alone (no network): coverage outside the segment costs less than coverage
    missing inside it, since a neighbour may hide part of the body. Keeps the best fit per segment."""
    S = L.shape[0]
    K = int(L.max())
    tilts = tuple(float(v) for v in getattr(a, "coil_tilts", "0.3,1.0").split(","))
    hyp = [(hd, bo, al) for hd in (1.0, -1.0) for bo in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi) for al in tilts]
    Hn = len(hyp)
    P = {n: [] for n in Coils.NAMES}
    hand, orig = [], []
    for k in range(1, K + 1):
        m = L == k
        ys, xs = np.nonzero(m)
        cy = (S / (2 * math.pi) * math.atan2(np.sin(ys * 2 * math.pi / S).mean(), np.cos(ys * 2 * math.pi / S).mean())) % S
        cx = (S / (2 * math.pi) * math.atan2(np.sin(xs * 2 * math.pi / S).mean(), np.cos(xs * 2 * math.pi / S).mean())) % S
        dy = (ys - cy + S / 2) % S - S / 2
        dx = (xs - cx + S / 2) % S - S / 2
        ev, evec = np.linalg.eigh(np.cov(np.stack([dx, dy])))
        ax = math.atan2(evec[1, -1], evec[0, -1])
        Req = math.sqrt(m.sum() / math.pi)
        for hd, bo, al in hyp:
            for n, v in (("x", cx), ("y", cy), ("lrho", math.log(0.62 * Req)), ("tq", math.log(math.expm1(getattr(a, "coil_turns", 1.0) - 0.05))), ("lw", math.log(math.log(2.5))),
                         ("cq", 0.0), ("h", 0.3), ("alpha", al), ("beta", ax + bo), ("th0", 0.0), ("lift", a.zs * zord[k])):
                P[n].append(v)
            hand.append(hd); orig.append(k - 1)
    P = {n: np.array(v) for n, v in P.items()}
    cb = Coils(P, np.array(hand), a, dev)
    iou = sil_fit(cb, orig, L, a, dev, log, steps, stride, "coil", target, cap, spill_w).reshape(K, Hn)
    best = iou.argmax(1)
    pick = (torch.arange(K, device=dev) * Hn + best).cpu().numpy()
    log(f"coil prefit: soft IoU of the best guess per segment median {float(iou.max(1)[0].median()):.2f}, "
        f"p10 {float(iou.max(1)[0].quantile(0.1)):.2f}; turns median {float(cb.turns()[pick].median()):.2f}")
    P2 = {n: getattr(cb, n).detach().cpu().numpy()[pick] for n in Coils.NAMES}
    return P2, np.array(hand)[pick], iou.max(1)[0]


def tube_prefit(ctls, rads, L, a, dev, log, steps=150, stride=3, target=None, cap=None, spill_w=0.3):
    """The free tubes fitted to the same silhouettes, for a fair choice between the families."""
    tb = Tubes(ctls, rads, a, dev)
    iou = sil_fit(tb, list(range(tb.K)), L, a, dev, log, steps, stride, "tube", target, cap, spill_w)
    log(f"tube prefit: soft IoU median {float(iou.median()):.2f}, p10 {float(iou.quantile(0.1)):.2f}")
    return tb.ctl.detach().cpu().numpy(), np.exp(tb.logr.detach().cpu().numpy()), iou


def sil_fit(cb, orig, L, a, dev, log, steps, stride, tag, target=None, cap=None, spill_w=0.3):
    """Silhouette-only fit of bodies cb (body i targets segment orig[i]); returns soft IoU per body.
    target (K, S*S) bool on dev: the shape each segment should fill (default: its visible pixels);
    cap (centre (K, 2), radius (K,)) per segment: a body may not reach past it."""
    S = L.shape[0]
    K = int(L.max())
    orig_t = torch.tensor(orig, device=dev)
    top = Cam(np.eye(3), S, S, (S / 2, S / 2, 0.0), S)
    lab = torch.from_numpy(L.reshape(-1) - 1).to(dev)
    gp = cb.geom_params()
    opt = torch.optim.Adam([g for g in [{"params": gp[0], "lr": 0.3}, {"params": gp[1], "lr": 0.03}, {"params": gp[2], "lr": 0.02}] if g["params"]])
    zh = torch.tensor([0.0, 0.0, 1.0], device=dev)
    for it in range(steps):
        if it % 25 == 0:
            with torch.no_grad():
                pr = build_pairs(cb.spheres(), top, 6.0)
                keep = ((pr["pix"] % S) % stride == 0) & ((pr["pix"] // S) % stride == 0)
                pix = torch.from_numpy(pr["pix"][keep]).to(dev)
                obj = torch.from_numpy(pr["obj"][keep]).to(dev)
                shf = F.pad(torch.from_numpy(pr["shift"][keep]).to(dev), (0, 1))
                inside = (lab[pix] == orig_t[obj]).float() if target is None else target[orig_t[obj], pix].float()
                if it == 0:  # the whole segment on the strided grid, not just what the window sees
                    if target is None:
                        sub = torch.from_numpy(L[::stride, ::stride].reshape(-1) - 1).to(dev)
                        area = torch.bincount(sub, minlength=K).float()[orig_t].clamp(min=1)
                    else:
                        grid_ = torch.zeros(S, S, dtype=torch.bool, device=dev)
                        grid_[::stride, ::stride] = True
                        area = (target & grid_.reshape(1, -1)).sum(1).float()[orig_t].clamp(min=1)
        sph = cb.spheres()
        X0 = torch.stack([(pix % S).float(), torch.div(pix, S, rounding_mode="floor").float(), torch.zeros_like(pix, dtype=torch.float32)], -1) - shf
        Bv = sph["Bv"]
        Nv = torch.cross(sph["T"], Bv, dim=-1)
        outs = []
        for i in range(0, pix.shape[0], a.chunk):
            sl_ = slice(i, i + a.chunk)
            outs.append(torch.utils.checkpoint.checkpoint(body_hits, X0[sl_], obj[sl_], sph["C"], sph["r"], sph["T"], Bv, Nv, sph["f"], sph["s"], zh,
                                                          a.sig * 2, a.tau_s, use_reentrant=False))
        ck = torch.cat(outs, 1)[0, :, 1]
        inter = torch.zeros(cb.K, device=dev).index_add_(0, obj, inside * ck)
        miss = area - inter
        spill = torch.zeros(cb.K, device=dev).index_add_(0, obj, (1 - inside) * ck)
        loss = ((miss + spill_w * spill) / area).sum() + cb.K * 0.05 * cb.reg()
        if cap is not None:
            loss = loss + cb.K * cap_loss(cb, cap[0][orig_t], cap[1][orig_t], S)
        opt.zero_grad()
        loss.backward()
        if getattr(cb, "ctl", None) is not None and cb.ctl.grad is not None:
            cb.ctl.grad[..., 2] = 0  # a top-view silhouette says nothing about height: leave z to the stacking
        opt.step()
        if it % 50 == 0 or it == steps - 1:
            log(f"{tag} prefit {it + 1}/{steps}: loss per body {float(loss) / cb.K:.3f}")
    return (inter / (area + spill)).detach()


# ---------------------------------------------------------------- cameras and rendering

def contact_loss_disp(tb, sph, nbr, S, tol):
    """contact_loss, with each sphere's radius widened by the body's relief in the direction of the
    other sphere: a body can then flatten against a neighbour and bulge into a gap, but not through it."""
    C, r, T, Bv = sph["C"][0], sph["r"], sph["T"][0], sph["Bv"][0]
    Nv = torch.cross(T, Bv, dim=-1)
    f = sph["f"]
    disp = sph.get("disp")
    tv = ((sph["s"] - sph["s_lo"][:, None]) / sph["s_len"][:, None]).clamp(0, 1)  # (K, N)
    i, j = nbr[:, 0], nbr[:, 1]
    d = C[j][:, None, :, :] - C[i][:, :, None, :]  # (M, N, N, 3): from sphere a of i to sphere b of j
    d = torch.cat([d[..., :2] - S * torch.round(d[..., :2] / S), d[..., 2:]], -1)
    def reach(k, dd, frame_idx):
        # relief of body k (M,) at its sphere frame_idx toward the unit direction dd (M, N, N, 3)
        if disp is None:
            return 0.0
        Bk = Bv[k][:, :, None, :] if frame_idx == 1 else Bv[k][:, None, :, :]
        Nk = Nv[k][:, :, None, :] if frame_idx == 1 else Nv[k][:, None, :, :]
        th = torch.atan2((dd * Nk).sum(-1), (dd * Bk).sum(-1))
        tu = th / (2 * math.pi) % 1.0
        tvk = (tv[k][:, :, None] if frame_idx == 1 else tv[k][:, None, :]).expand_as(tu)
        kk = k[:, None, None].expand_as(tu)
        return tex_sample(disp[:, None], kk, tu, tvk)[..., 0]
    fz = 2 / (f[i] + f[j])[:, None, None]
    dist = torch.sqrt((d[..., :2] ** 2).sum(-1) + (d[..., 2] * fz) ** 2 + 1e-6)
    u = d / dist[..., None]
    ri, rj = r[i][:, :, None] + reach(i, u, 1), r[j][:, None, :] + reach(j, -u, 2)
    pen = torch.relu(ri + rj - dist - tol * torch.minimum(r[i][:, :, None], r[j][:, None, :]))
    return ((pen ** 2).sum((1, 2)) / C.shape[1]).mean() / 4.0


def fill_loss(wc, H, W):
    """How far each visible point sits below the surface around it (3 and 6 px box means of the
    composited height, top view, on the torus): crevices cost, so bodies inflate into them."""
    d = wc.reshape(-1, 1, H, W)
    acc = 0
    for rr in (3, 6):
        bl = F.avg_pool2d(F.pad(d, (rr, rr, rr, rr), mode="circular"), 2 * rr + 1, stride=1)
        acc = acc + torch.relu(bl - d) / rr
    return acc.mean()


def zband_loss(tb, a):
    """Bodies in one layer: each body's top within --zband lo,hi px and its bottom no lower than
    --zbot, squared px past those, / 25."""
    C, r = tb.base()
    f = tb.f()[:, None]
    tops, bots = (C[..., 2] + f * r).max(1)[0], (C[..., 2] - f * r).min(1)[0]
    lo, hi = map(float, a.zband.split(","))
    return (torch.relu(tops - hi) ** 2 + torch.relu(lo - tops) ** 2 + torch.relu(a.zbot - bots) ** 2).mean() / 25


def rot_x(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Cam:
    """Orthographic. R maps world offsets from `centre` to camera (u right, v down, w toward the
    viewer). Pixel (row, col) of an H x W image sits at u = col - W/2, v = row - H/2. wrap: the world
    is a torus of period S in x and y."""

    def __init__(self, R, H, W, centre, S, wrap=True, zoom=1.0):
        self.R, self.H, self.W, self.centre, self.S, self.wrap = np.asarray(R, float), H, W, np.asarray(centre, float), S, wrap
        self.zoom = zoom  # pixels per world px: 2 renders the same view at twice the resolution


def build_pairs(sph, cam, margin, sub_ids=None):
    """(pixel, body, world shift) pairs: every pixel of the box round every lattice copy of every body."""
    C = sph["C"][0].detach().cpu().numpy()  # (K, N, 3)
    r = sph["r"].detach().cpu().numpy()
    f = sph["f"].detach().cpu().numpy()
    R, S = cam.R, cam.S
    z = getattr(cam, "zoom", 1.0)
    pj = z * np.einsum("ij,knj->kni", R[:2], C - cam.centre)  # (K, N, 2) u, v, in pixels
    mu = pj.mean(1)
    ext = z * r * np.maximum(1.0, f)[:, None]
    half = (np.abs(pj - mu[:, None]).max(-1) + ext).max(1) + margin
    A = z * R[:2, :2] * S  # image of the world lattice vectors (S, 0, 0), (0, S, 0), in pixels
    pix, obj, sh = [], [], []
    rng = range(-3, 4) if cam.wrap else range(0, 1)
    for k in range(C.shape[0]):
        for n1 in rng:
            for n2 in rng:
                c = mu[k] + A @ np.array([n1, n2])
                x0, x1 = int(math.floor(c[0] + cam.W / 2 - half[k])), int(math.ceil(c[0] + cam.W / 2 + half[k]))
                y0, y1 = int(math.floor(c[1] + cam.H / 2 - half[k])), int(math.ceil(c[1] + cam.H / 2 + half[k]))
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, cam.W - 1), min(y1, cam.H - 1)
                if x1 < x0 or y1 < y0:
                    continue
                yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
                pu = xx.ravel() - cam.W / 2 - (c[0] - mu[k, 0])
                pv = yy.ravel() - cam.H / 2 - (c[1] - mu[k, 1])
                dd = np.sqrt((pu[:, None] - pj[k, None, :, 0]) ** 2 + (pv[:, None] - pj[k, None, :, 1]) ** 2) - ext[k][None]
                near = dd.min(1) < margin
                p = (yy * cam.W + xx).ravel()[near]
                pix.append(p); obj.append(np.full(len(p), k)); sh.append(np.tile([n1 * S, n2 * S], (len(p), 1)))
    return dict(pix=np.concatenate(pix), obj=np.concatenate(obj), shift=np.concatenate(sh).astype(np.float32))


def tex_sample(tex, k, tu, tv):
    """Bilinear, wrapping around (tu in turns) and clamped along (tv in [0, 1]). tex (K, 3, Ha, Wl)."""
    K, Cn, Ha, Wl = tex.shape
    flat = tex.permute(0, 2, 3, 1).reshape(-1, Cn)
    x = tu * Ha
    y = (tv.clamp(0, 1) * (Wl - 1))
    x0, y0 = torch.floor(x), torch.floor(y).clamp(max=Wl - 2)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0 = x0.long() % Ha
    x1 = (x0 + 1) % Ha
    y0 = y0.long()
    base = k * (Ha * Wl)
    g = lambda xi, yi: flat[base + xi * Wl + yi]
    return (1 - fx) * ((1 - fy) * g(x0, y0) + fy * g(x0, y0 + 1)) + fx * ((1 - fy) * g(x1, y0) + fy * g(x1, y0 + 1))


def torus_sample(img, x, y):
    """Bilinear sample of img (3, G, G) at (x, y) in canvas periods, wrapping both ways. -> (P, 3)."""
    G = img.shape[-1]
    flat = img.permute(1, 2, 0).reshape(-1, 3)
    x, y = (x % 1.0) * G - 0.5, (y % 1.0) * G - 0.5
    x0, y0 = torch.floor(x), torch.floor(y)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0, y0 = x0.long() % G, y0.long() % G
    x1, y1 = (x0 + 1) % G, (y0 + 1) % G
    g = lambda yi, xi: flat[yi * G + xi]
    return (1 - fy) * ((1 - fx) * g(y0, x0) + fx * g(y0, x1)) + fy * ((1 - fx) * g(y1, x0) + fx * g(y1, x1))


def _occl(rel, dm, rm, fm, val, l, pen0, pen):
    """Visibility of receivers rel (tc, R, 3) against candidate spheres dm (tc, M, 3), radii rm, squash fm."""
    am = l[0] ** 2 + l[1] ** 2 + (l[2] / fm) ** 2
    q = rel[:, :, None, :] - dm[:, None]  # (tc, R, M, 3) point minus centre
    qz = q[..., 2] / fm[:, None]
    b = q[..., 0] * l[0] + q[..., 1] * l[1] + qz * (l[2] / fm[:, None])
    t0 = -b / am[:, None]
    m2 = (q[..., 0] ** 2 + q[..., 1] ** 2 + qz**2 - b * b / am[:, None]).clamp(min=0)
    occl = torch.sigmoid((rm[:, None] - torch.sqrt(m2 + 1e-6)) / (pen0 + pen * t0.clamp(min=0))) * torch.sigmoid((t0 - 1.0) / 0.3)
    return torch.exp((torch.log1p(-0.98 * occl) * val[:, None]).sum(-1))


def shadow(X, n, l, sph, S, wrap, pen0, pen, H, W, ts=16, tchunk=128):
    """Visibility (P,) of the light from world points X (P, 3) of an H x W frame with normals n, traced
    against the spheres: closed-form closest approach of the shadow ray to each (squashed) sphere.
    Receivers go in ts x ts tiles; each tile tests only the spheres whose disc, seen from the light,
    overlaps the tile's footprint (plus the widest penumbra) and that lie toward the light.
    Differentiable in the light (direction l, softness pen0, pen) when those carry gradients; the
    geometry is held fixed."""
    dev = X.device
    need = torch.is_grad_enabled() and any(torch.is_tensor(v) and v.requires_grad for v in (l, pen0, pen))
    with torch.no_grad():
        C = sph["C"].detach().reshape(-1, 3)
        r = sph["r"].detach().reshape(-1)
        f = sph["f"].detach()[:, None].expand(*sph["r"].shape).reshape(-1)
        rr = r * torch.clamp(f, min=1.0)
        Hp, Wp = -(-H // ts) * ts, -(-W // ts) * ts
        def tiles(Y):
            Y = Y.reshape(H, W, -1).permute(2, 0, 1)[None]
            Y = F.pad(Y, (0, Wp - W, 0, Hp - H), mode="replicate")[0]
            return Y.reshape(-1, Hp // ts, ts, Wp // ts, ts).permute(1, 3, 2, 4, 0).reshape(-1, ts * ts, Y.shape[0])
        Xt = tiles(X.detach() + 1.0 * n.detach())  # (nt, ts*ts, 3), lifted off the surface
        ld = l.detach()
        a1 = torch.tensor([-float(ld[1]), float(ld[0]), 0.0], device=dev)
        if a1.norm() < 1e-3:
            a1 = torch.tensor([1.0, 0.0, 0.0], device=dev)
        a1 = a1 / a1.norm()
        a2 = torch.cross(ld, a1, dim=0)
        ctr = Xt.mean(1)  # (nt, 3)
        tmax = float((C[:, 2].max() - X[:, 2].min()).clamp(min=1)) / max(float(ld[2]), 0.1)
        marg = float(pen0) + float(pen) * tmax + 1.5
    outs = []
    for i in range(0, Xt.shape[0], tchunk):
        with torch.no_grad():
            xt, ct = Xt[i:i + tchunk], ctr[i:i + tchunk]
            d = C[None] - ct[:, None]  # (tc, Q, 3) centre minus tile centre
            if wrap:
                d = torch.cat([d[..., :2] - S * torch.round(d[..., :2] / S), d[..., 2:]], -1)
            rel = xt - ct[:, None]  # (tc, ts*ts, 3)
            p1, p2 = rel @ a1, rel @ a2
            c1, c2, cl = d @ a1, d @ a2, d @ ld
            ok = ((c1 + rr + marg) > p1.min(1, keepdim=True)[0]) & ((c1 - rr - marg) < p1.max(1, keepdim=True)[0])
            ok &= ((c2 + rr + marg) > p2.min(1, keepdim=True)[0]) & ((c2 - rr - marg) < p2.max(1, keepdim=True)[0])
            ok &= (cl + rr) > (rel @ ld).min(1, keepdim=True)[0]
            M = max(1, int(ok.sum(1).max()))
            idx = torch.argsort((~ok).to(torch.int8), dim=1, stable=True)[:, :M]
            val = torch.gather(ok, 1, idx).float()
            dm = torch.gather(d, 1, idx[..., None].expand(-1, -1, 3))  # (tc, M, 3)
            rm, fm = r[idx], f[idx]
        if need:
            outs.append(torch.utils.checkpoint.checkpoint(_occl, rel, dm, rm, fm, val, l, pen0, pen, use_reentrant=False))
        else:
            with torch.no_grad():
                outs.append(_occl(rel, dm, rm, fm, val, l, pen0, pen))
    out = torch.cat(outs, 0)
    out = out.reshape(Hp // ts, Wp // ts, ts, ts).permute(0, 2, 1, 3).reshape(Hp, Wp)[:H, :W]
    return out.reshape(-1)


def as_rig(L, a, dev):
    """A light as a dict of tensors: key direction l, order-1 SH ambient sh (4, 3), diffuse dif, colour
    tint (3,), specular, penumbra (pen0 + pen * distance)."""
    if isinstance(L, dict):
        return L
    sh = torch.zeros(4, 3, device=dev)
    sh[0] = L.amb
    return dict(l=torch.tensor(L.vec(), dtype=torch.float32, device=dev), sh=sh, dif=torch.tensor(L.dif, device=dev),
                tint=torch.ones(3, device=dev), spec=L.spec, shin=L.shin, pen0=a.pen0, pen=a.pen)


def with_dir(rig, az, el):
    """The same rig with its key light moved to (az, el), radians."""
    r2 = dict(rig)
    r2["l"] = torch.tensor([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=torch.float32, device=rig["l"].device)
    return r2


class Rig(torch.nn.Module):
    """One learned light rig for the whole scene: key direction, diffuse intensity, colour, penumbra,
    and order-1 spherical-harmonic ambient (per colour channel), all bounded."""

    def __init__(self, az, el, a, dev):
        super().__init__()
        isp = lambda v: math.log(math.expm1(v))
        lgt = lambda v: math.log(v / (1 - v))
        self.az = torch.nn.Parameter(torch.tensor(az, device=dev))
        self.elq = torch.nn.Parameter(torch.tensor(lgt((el - math.radians(10)) / math.radians(80)), device=dev))
        self.difq = torch.nn.Parameter(torch.tensor(lgt((0.78 - 0.3) / 0.9), device=dev))
        self.tintq = torch.nn.Parameter(torch.zeros(3, device=dev))
        self.ambq = torch.nn.Parameter(torch.full((3,), lgt((0.32 - 0.1) / 0.5), device=dev))
        self.sh1 = torch.nn.Parameter(torch.zeros(3, 3, device=dev))
        self.pen0q = torch.nn.Parameter(torch.tensor(isp(a.pen0), device=dev))
        self.penq = torch.nn.Parameter(torch.tensor(isp(a.pen), device=dev))

    def forward(self):
        el = math.radians(10) + math.radians(80) * torch.sigmoid(self.elq)
        l = torch.stack([torch.cos(el) * torch.cos(self.az), torch.cos(el) * torch.sin(self.az), torch.sin(el)])
        sh = torch.cat([(0.1 + 0.5 * torch.sigmoid(self.ambq))[None], 0.3 * torch.tanh(self.sh1)], 0)
        return dict(l=l, sh=sh, dif=0.3 + 0.9 * torch.sigmoid(self.difq), tint=0.6 + 0.8 * torch.sigmoid(self.tintq), spec=0.18, shin=24.0,
                    pen0=F.softplus(self.pen0q), pen=F.softplus(self.penq))

    def describe(self):
        with torch.no_grad():
            d = self()
            el = math.degrees(math.asin(float(d["l"][2])))
            return dict(az_deg=math.degrees(float(self.az)) % 360, el_deg=el, dif=float(d["dif"]), tint=d["tint"].tolist(), sh=d["sh"].tolist(),
                        pen0=float(d["pen0"]), pen=float(d["pen"]))


class BodyLight(torch.nn.Module):
    """Per-body light (ablation): a colour tint and a small offset to the key direction for each body."""

    def __init__(self, K, dev):
        super().__init__()
        self.tintq = torch.nn.Parameter(torch.zeros(K, 3, device=dev))
        self.dlq = torch.nn.Parameter(torch.zeros(K, 3, device=dev))

    def forward(self):
        return 0.6 + 0.8 * torch.sigmoid(self.tintq), 0.3 * torch.tanh(self.dlq)


S_BLEND = 6.0  # px of arclength over which neighbouring spheres share texture coordinates


def body_hits(X0, obj, C, r, T, Bv, Nv, f, s, e, sig, tau_s, disp=None, s_lo=None, s_len=None):
    """Per pixel-body pair (B, P, 8): front depth w, coverage, normal (3), arclength along the spine,
    and the two around-the-tube components of the hit, each blended over the body's spheres.
    disp (K, Ga, Gl) px: a radial displacement in tube coordinates (around x along). A first pass
    finds where each ray meets each sphere; the sphere's radius there is displaced, and the hit,
    coverage and normal recomputed (the normal tilted by the displacement's slope around the tube;
    along the tube the union of spheres carries it)."""
    q = X0[None, :, None, :] - C[:, obj]  # (B, P, N, 3)
    fk = f[obj][None, :, None]
    ez = e[2] / fk
    qz = q[..., 2] / fk
    aa = e[0] ** 2 + e[1] ** 2 + ez**2
    b = q[..., 0] * e[0] + q[..., 1] * e[1] + qz * ez
    m2 = (q[..., 0] ** 2 + q[..., 1] ** 2 + qz**2 - b * b / aa).clamp(min=0)
    rk = r[obj][None]
    Bk, Nk, Tk = Bv[:, obj], Nv[:, obj], T[:, obj]
    w = -b / aa + torch.sqrt(torch.relu(rk**2 - m2) / aa + 1e-4)  # front hit, toward the viewer
    g = q + w[..., None] * e  # hit minus centre, world
    dth = None
    if disp is not None:
        Ga = disp.shape[1]
        th = torch.atan2((g * Nk).sum(-1), (g * Bk).sum(-1))
        tv = ((s - s_lo[:, None]) / s_len[:, None])[obj][None].expand_as(th)
        kk = obj[None, :, None].expand_as(th)
        dk = disp[:, None]
        tu = th / (2 * math.pi) % 1.0
        h = 0.5 / Ga
        dl = tex_sample(dk, kk, tu, tv)[..., 0]
        dth = (tex_sample(dk, kk, (tu + h) % 1.0, tv)[..., 0] - tex_sample(dk, kk, (tu - h) % 1.0, tv)[..., 0]) / (2 * h * 2 * math.pi)
        rk = (rk + dl).clamp(min=0.5)
        w = -b / aa + torch.sqrt(torch.relu(rk**2 - m2) / aa + 1e-4)
        g = q + w[..., None] * e
    cov = torch.sigmoid((rk - torch.sqrt(m2 + 1e-6)) / sig)
    nrm = torch.stack([g[..., 0], g[..., 1], g[..., 2] / fk**2], -1)
    nrm = nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-6)
    if dth is not None:
        gp = g - (g * Tk).sum(-1, keepdim=True) * Tk
        eth = torch.cross(Tk, gp / (gp.norm(dim=-1, keepdim=True) + 1e-6), dim=-1)
        nrm = nrm - (dth / rk)[..., None] * eth
        nrm = nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-6)
    # front-most covering sphere wins, softly. A sphere the ray misses still has a depth (where the ray
    # passes closest), so it must count only by its coverage: else a part of the body rising toward the
    # camera paints its edge-on normal inside the body's own outline (a halo). Covering spheres compete
    # on depth, those behind the front-most fading over tau_s.
    wd = w.detach()
    wref = torch.where(cov.detach() >= 0.5, wd, torch.full_like(wd, -1e9)).amax(-1, keepdim=True)
    wref = torch.where(wref > -1e8, wref, wd.amax(-1, keepdim=True))
    lw = cov * torch.exp(torch.clamp(w - wref, max=0.0) / tau_s) + 1e-12
    lw = lw / lw.sum(-1, keepdim=True)
    wk = (lw * w).sum(-1)
    ck = -torch.expm1(torch.log1p(-cov * (1 - 1e-4)).sum(-1))  # expm1: a faint coverage stays faint, not 0
    nk = (lw[..., None] * nrm).sum(-2)
    nk = nk / (nk.norm(dim=-1, keepdim=True) + 1e-6)
    # texture coordinates blend only along the spine near the front-most sphere: a coil's whorls (or a
    # bent tube's far end) can sit at the same depth, and averaging their arclengths made the texture
    # slide between whorls whenever the weights shifted
    sk = s[obj][None].expand_as(w)
    s_front = torch.gather(sk, -1, lw.detach().argmax(-1, keepdim=True))
    lt = lw * torch.exp(-((sk - s_front) / S_BLEND) ** 2)
    lt = lt / lt.sum(-1, keepdim=True).clamp(min=1e-12)
    sl = (lt * (sk + (g * Tk).sum(-1))).sum(-1)
    ca = (lt * (g * Bk).sum(-1)).sum(-1)
    sa = (lt * (g * Nk).sum(-1)).sum(-1)
    return torch.cat([wk[..., None], ck[..., None], nk, sl[..., None], ca[..., None], sa[..., None]], -1)


def render(tb, sph, cam, pairs, lights, a, clay=None, shadows=True, ground="tex", off=(0.0, 0.0), aux=False, lab=None, tex=None, blight=None, sub=None):
    """Lit frames (B, 3, H, W). clay: grey albedo instead of the textures. ground: 'tex' (learned, on
    the torus), 'flat' (plain grey plane at z = 0) or 'none' (dark backdrop)."""
    dev = tb.dev
    H, W, S = cam.H, cam.W, cam.S
    R = torch.tensor(cam.R, dtype=torch.float32, device=dev)
    ctr = torch.tensor(cam.centre, dtype=torch.float32, device=dev)
    pix = torch.from_numpy(pairs["pix"]).to(dev)
    obj = torch.from_numpy(pairs["obj"]).to(dev)
    shf = F.pad(torch.from_numpy(pairs["shift"]).to(dev), (0, 1))
    C, r, T, f = sph["C"], sph["r"], sph["T"], sph["f"]
    B = C.shape[0]
    e = R[2]  # world direction toward the viewer
    zm = getattr(cam, "zoom", 1.0)
    u = ((pix % W).float() - W / 2 + off[1]) / zm
    v = (torch.div(pix, W, rounding_mode="floor").float() - H / 2 + off[0]) / zm
    X0 = ctr + u[:, None] * R[0] + v[:, None] * R[1] - shf  # (P, 3) pixel ray at w = 0, relative to this copy
    zh = torch.tensor([0.0, 0.0, 1.0], device=dev)
    Bv = sph["Bv"]  # per sphere: across the tube (transported "up"), and the side direction
    Nv = torch.cross(T, Bv, dim=-1)
    args = (C, r, T, Bv, Nv, f, sph["s"], e, a.sig, a.tau_s, sph.get("disp"), sph["s_lo"], sph["s_len"])
    outs = []
    for i in range(0, pix.shape[0], a.chunk):
        sl_ = slice(i, i + a.chunk)
        if torch.is_grad_enabled():
            outs.append(torch.utils.checkpoint.checkpoint(body_hits, X0[sl_], obj[sl_], *args, use_reentrant=False))
        else:
            outs.append(body_hits(X0[sl_], obj[sl_], *args))
    hb = torch.cat(outs, 1)  # (B, P, 8)
    wk, ck, nk, sl, ca, sa = hb[..., 0], hb[..., 1], hb[..., 2:5], hb[..., 5], hb[..., 6], hb[..., 7]
    tu = torch.atan2(sa, ca) / (2 * math.pi) % 1.0
    tv = (sl - sph["s_lo"][obj]) / sph["s_len"][obj]
    if clay is None:
        kb = obj[None].expand(B, -1)
        tus, tvs = tu, tv
        if "tex_shift" in sph:  # per body (turns about the spine, fraction along it): the texture turns and slides on the body
            sh_ = sph["tex_shift"]
            tus, tvs = (tu + sh_[obj, 0]) % 1.0, tv + sh_[obj, 1]
        lgt = tex_sample(tb.tex if tex is None else tex, kb, tus, tvs)
        if hasattr(tb, "tex0"):  # wrapped-still prior + learned residual
            lgt = lgt + tex_sample(tb.tex0 if sub is None else tb.tex0[sub], kb, tus, tvs)
        alb = torch.sigmoid(lgt)  # (B, P, 3)
        if "hue" in sph:  # per body, radians: turn the colour about the grey axis (YIQ), keeping its luminance
            hh = sph["hue"][obj]
            Y_ = alb @ torch.tensor([0.299, 0.587, 0.114], device=dev)
            I_ = alb @ torch.tensor([0.596, -0.274, -0.322], device=dev)
            Q_ = alb @ torch.tensor([0.211, -0.523, 0.312], device=dev)
            c_, s_ = torch.cos(hh), torch.sin(hh)
            I2, Q2 = c_ * I_ - s_ * Q_, s_ * I_ + c_ * Q_
            alb = torch.stack([Y_ + 0.956 * I2 + 0.621 * Q2, Y_ - 0.272 * I2 - 0.647 * Q2, Y_ - 1.106 * I2 + 1.703 * Q2], -1).clamp(0, 1)
    else:
        alb = torch.full((B, pix.shape[0], 3), clay, device=dev)
    # ground: the plane z = 0 seen through every pixel
    yy, xx = torch.meshgrid((torch.arange(H, device=dev).float() - H / 2 + off[0]) / zm, (torch.arange(W, device=dev).float() - W / 2 + off[1]) / zm, indexing="ij")
    uu, vv = xx.reshape(-1), yy.reshape(-1)
    wg = (getattr(a, "ground_z", 0.0) - ctr[2] - R[0, 2] * uu - R[1, 2] * vv) / R[2, 2]  # the ground plane z = --ground-z
    Xg = ctr + uu[:, None] * R[0] + vv[:, None] * R[1] + wg[:, None] * R[2]
    if ground == "tex":
        galb = torch.sigmoid(torus_sample(tb.ground, Xg[:, 0] / S, Xg[:, 1] / S))  # (HW, 3)
        if clay is not None:
            galb = torch.full_like(galb, clay * 0.8)
    else:
        galb = torch.full((H * W, 3), 0.55 if ground == "flat" else 0.08, device=dev)
    if ground == "none":
        wg = torch.full_like(wg, -1e4)
    # bodies: a soft z-buffer decides which body shows; together they cover the pixel with the union of
    # their coverages, and the ground shows through the rest (alpha over the ground)
    if getattr(a, "zbuf", "cover") == "cover":
        # only a body that covers the pixel competes on depth. A ray that misses a body still gets a
        # depth (where it passes closest), and a tall body in front then won pixels outside its own
        # outline, painting its grazing edge over the body behind: grey rings. Here the front-most
        # covering surface sets the reference, surfaces behind it fade over --tau-z, and a pair
        # counts in proportion to its coverage, so a soft edge still blends.
        cov_ok = ck.detach() >= 0.5
        zc = torch.full((B, H * W), -1e9, device=dev).scatter_reduce(1, pix[None].expand(B, -1), torch.where(cov_ok, wk.detach(), torch.full_like(wk, -1e9)), "amax", include_self=True)
        zany = torch.full((B, H * W), -1e9, device=dev).scatter_reduce(1, pix[None].expand(B, -1), wk.detach(), "amax", include_self=True)
        zref = torch.where(zc > -1e8, zc, zany)[:, pix]  # no covering body: the old behaviour, the nearest wins
        ep = ck * torch.exp(torch.clamp(wk - zref, max=0.0) / a.tau_z) + 1e-12
    else:
        key = wk + a.tau_z * torch.log(ck + 1e-6)
        zmax = torch.full((B, H * W), -1e9, device=dev).scatter_reduce(1, pix[None].expand(B, -1), key.detach(), "amax", include_self=True)
        zmax = torch.maximum(zmax, torch.full_like(zmax, -1e8)).detach()
        ep = torch.exp((key - zmax[:, pix]) / a.tau_z)
    logT = torch.zeros(B, H * W, device=dev).index_add_(1, pix, torch.log1p(-ck * (1 - 1e-4)))
    cover = 1 - torch.exp(logT)
    feats, gfeats = [alb, nk, wk[..., None]], [galb.expand(B, -1, -1), zh.expand(B, H * W, 3), wg[None, :, None].expand(B, -1, 1)]
    if blight is not None:  # per-body tint and key-direction offset, composited like albedo
        bt, bd = blight()
        if sub is not None:
            bt, bd = bt[sub], bd[sub]
        feats += [bt[obj][None].expand(B, -1, -1), bd[obj][None].expand(B, -1, -1)]
        gfeats += [torch.ones(B, H * W, 3, device=dev), torch.zeros(B, H * W, 3, device=dev)]
    feat = torch.cat(feats, -1) * ep[..., None]
    nf = feat.shape[-1]
    num = torch.zeros(B, H * W, nf, device=dev).index_add_(1, pix, feat)
    denb = torch.zeros(B, H * W, device=dev).index_add_(1, pix, ep).clamp(min=1e-12)
    comp = cover[..., None] * num / denb[..., None] + (1 - cover[..., None]) * torch.cat(gfeats, -1)
    den = denb / cover.clamp(min=1e-6)  # so that ep / den is a pair's share of the pixel
    albc, nc, wc = comp[..., :3], comp[..., 3:6], comp[..., 6]
    nc = nc / (nc.norm(dim=-1, keepdim=True) + 1e-6)
    Xc = ctr + uu[None, :, None] * R[0] + vv[None, :, None] * R[1] + wc[..., None] * R[2]
    occ = torch.zeros(B, H * W, device=dev)
    if getattr(a, "ao", 0.0) > 0:
        # ambient occlusion from the composited depth (screen space): how far each point sits below the
        # surface around it, at a few scales, so a crevice darkens from the geometry rather than from
        # paint, and the darkness moves with the bodies
        d = wc.reshape(B, 1, H, W)
        acc = 0
        for rr in [int(round(q * zm)) for q in (3, 6, 12)]:
            # the frame tiles only when looking straight down; a tilted view's depth ramps top to bottom
            top_down = cam.wrap and np.allclose(cam.R, np.eye(3))
            bl = F.avg_pool2d(F.pad(d, (rr, rr, rr, rr), mode="circular" if top_down else "replicate"), 2 * rr + 1, stride=1)
            acc = acc + torch.relu(bl - d) / rr
        occ = (1 - torch.exp(-a.ao * acc)).reshape(B, H * W)
    outs = []
    for bi in range(B):
        rg = as_rig(lights[bi if len(lights) > 1 else 0], a, dev)
        l = rg["l"]
        if blight is not None:
            lp = l[None] + comp[bi, :, 10:13]
            lp = lp / lp.norm(dim=-1, keepdim=True)
            tp = comp[bi, :, 7:10]
        else:
            lp, tp = l[None], 1.0
        ndl = torch.relu((nc[bi] * lp).sum(-1))
        vis = shadow(Xc[bi], nc[bi], l, {k_: (v_[bi:bi + 1] if k_ in ("C",) else v_) for k_, v_ in sph.items()},
                     S, cam.wrap, rg["pen0"], rg["pen"], H, W) if shadows else torch.ones_like(ndl)
        hv = lp + e[None]
        hv = hv / hv.norm(dim=-1, keepdim=True)
        ndh = torch.relu((nc[bi] * hv).sum(-1))
        sh = rg["sh"] * getattr(a, "amb_scale", 1.0)
        amb = (sh[0][None] + nc[bi, :, 0:1] * sh[1][None] + nc[bi, :, 1:2] * sh[2][None] + nc[bi, :, 2:3] * sh[3][None]).clamp(min=0)
        a0 = sh[0].mean()
        expo = (a0 + rg["dif"]) / (a0 + rg["dif"] * l[2])
        tint = rg["tint"][None] * tp
        ao = (1 - occ[bi])[:, None]
        if getattr(a, "sky", 0.0) > 0:
            # 3D occlusion: the share of a ring of sky directions each point sees past every sphere
            # (the shadow tracer, soft), weighted by how much the surface faces each; a crevice
            # darkens by the same rule from any camera, and it moves with the bodies
            sv, sw = 0, 0
            for d_ in sky_dirs(dev):
                wgt = torch.relu((nc[bi] * d_).sum(-1))
                v_ = shadow(Xc[bi], nc[bi], d_, {k_: (v_[bi:bi + 1] if k_ in ("C",) else v_) for k_, v_ in sph.items()}, S, cam.wrap, a.sky_pen0, a.sky_pen, H, W)
                sv, sw = sv + wgt * v_, sw + wgt
            sky = (sv / sw.clamp(min=1e-3))[:, None]
            ao = ao * (1 - a.sky * (1 - sky))
        col = albc[bi] * (amb * ao + rg["dif"] * tint * (ndl * vis)[:, None] * (1 - getattr(a, "ao_dif", 0.5) * (1 - ao))) * expo \
            + rg["spec"] * tint * (ndh ** rg["shin"] * vis * (ndl > 0))[:, None] * ao
        outs.append(col.T.reshape(3, H, W))
    img = torch.stack(outs)
    if getattr(a, "outline", 0.0) > 0:
        img = img * (1 - a.outline * outline_band(ep, pix, obj, B, H, W, a.outline_w, cam.wrap and np.allclose(cam.R, np.eye(3))))[:, None]
    img = img.clamp(0, 1)
    if not aux:
        return img
    out = dict(img=img, ep=ep, den=den, pix=pix, obj=obj, alb=alb, nk=nk, ck=ck, tu=tu, tv=tv, gown=1 - cover, wc=wc)
    if lab is not None:
        hit = (obj == lab[pix]).float()
        own = torch.zeros(B, H * W, device=dev).index_add_(1, pix, ep * hit[None]) / den
        out["own"] = own
    return out


def sky_dirs(dev):
    """12 directions over the upper hemisphere: 6 azimuths at 30 deg and 6 (offset) at 60 deg."""
    out = []
    for el, off in ((math.radians(30), 0.0), (math.radians(60), math.pi / 6)):
        for i in range(6):
            az = off + 2 * math.pi * i / 6
            out.append(torch.tensor([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=torch.float32, device=dev))
    return out


@torch.no_grad()
def outline_band(ep, pix, obj, B, H, W, w, wrap):
    """(B, H, W): 1 on a boundary between the bodies that own neighbouring pixels (owner = the pair
    with the largest weight), falling off as exp(-d / w) with distance d px. Not a light: a drawn
    ring of fixed width round every shape, recomputed from the silhouettes each frame (--outline)."""
    out = []
    mode = "circular" if wrap else "replicate"
    for b in range(B):
        best = torch.full((H * W,), -1.0, device=ep.device).scatter_reduce(0, pix, ep[b], "amax", include_self=True)
        win = ep[b] >= best[pix] - 1e-12
        idm = torch.full((H * W,), -1.0, device=ep.device)
        idm[pix[win]] = obj[win].float()
        p = F.pad(idm.reshape(1, 1, H, W), (1, 1, 1, 1), mode=mode)[0, 0]
        c = p[1:-1, 1:-1]
        E = ((c != p[:-2, 1:-1]) | (c != p[2:, 1:-1]) | (c != p[1:-1, :-2]) | (c != p[1:-1, 2:])).float()
        band = E.clone()
        for i in range(1, int(4 * w) + 1):
            d = F.max_pool2d(F.pad(E[None, None], (i, i, i, i), mode=mode), 2 * i + 1, stride=1)[0, 0]
            band = torch.maximum(band, d * math.exp(-i / w))
        out.append(band)
    return torch.stack(out)


def render_ss(tb, sph, cam, pairs, light, a, **kw):
    acc = 0
    for oy in (-0.25, 0.25):
        for ox in (-0.25, 0.25):
            acc = acc + render(tb, sph, cam, pairs, [light], a, off=(oy, ox), **kw)
    return acc / 4


def encode(d, out, fps=24):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{d}/f%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out], check=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--labels", default="labels_display.npy")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--net", default="googlenet")
    # bodies
    ap.add_argument("--body", default="tube", choices=["tube", "coil", "mixed"], help="free spline tubes, coiled shells, or per body whichever fits")
    ap.add_argument("--init-from", help="mixed: bodies (after the prefit) from this file instead of fitting")
    ap.add_argument("--save-init", help="mixed: save the bodies after the prefit here")
    ap.add_argument("--prefit-cache", help="mixed: the two families' silhouette fits, saved here or loaded if present")
    ap.add_argument("--amodal", action="store_true", help="mixed: fit bodies to objfield's amodal outlines, cap their size by them")
    ap.add_argument("--cap-slack", type=float, default=0.15, help="a body may reach this fraction past its amodal outline's radius")
    ap.add_argument("--wcap", type=float, default=5.0, help="cost of reaching past the cap, per px^2")
    ap.add_argument("--spill", type=float, default=0.3, help="prefit: cost of coverage outside the target, relative to missing coverage")
    ap.add_argument("--coil-bias", type=float, default=0.0, help="mixed: choose the coil unless the tube's IoU beats it by more than this")
    ap.add_argument("--stack", action="store_true", help="set heights back to front so each visible top clears what is beneath (C-style)")
    ap.add_argument("--margin", type=float, default=1.0, help="stacking: px a visible top clears the tops beneath it")
    ap.add_argument("--stack-q", type=float, default=0.9, help="stacking: this fraction of a body's visible top must clear what is beneath")
    ap.add_argument("--stack-slope", type=float, default=0.5, help="stacking: largest rise of a tube's spine per px along it")
    ap.add_argument("--coil-turns", type=float, default=1.0, help="coil prefit: starting number of turns")
    ap.add_argument("--coil-max-tilt", type=float, default=0.6, help="coil axis tilt from vertical, rad, at most")
    ap.add_argument("--coil-max-rise", type=float, default=0.8, help="coil spire rise, at most")
    ap.add_argument("--coil-tilts", default="0.3,1.0", help="coil prefit: starting tilts of the coil axis, rad")
    ap.add_argument("--prior-around", type=int, default=64, help="wrapped-still prior texture, around")
    ap.add_argument("--prior-along", type=int, default=144, help="wrapped-still prior texture, along")
    ap.add_argument("--disp-amp", type=float, default=0.0, help="learned radial displacement in tube coordinates, px (tanh cap); 0 = off")
    ap.add_argument("--disp-around", type=int, default=8)
    ap.add_argument("--disp-along", type=int, default=12)
    ap.add_argument("--lr-disp", type=float, default=0.03)
    ap.add_argument("--tex-init", default="project", choices=["project", "wrap"], help="texture from the still: projected and diffused, or wrapped with a prior")
    # objfield scene, for the amodal outlines (as objsdf.py)
    ap.add_argument("--pin", default="38-55,73-74")
    ap.add_argument("--soft", default="64,65,75,77:0.5")
    ap.add_argument("--ext", type=float, default=14)
    ap.add_argument("--reach", type=float, default=16)
    ap.add_argument("--smooth", type=float, default=2.5)
    ap.add_argument("--edge-w", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--wall-sigma", type=float, default=6.0)
    ap.add_argument("--fill", default="mirror")
    ap.add_argument("--inset", type=float, default=1.5)
    ap.add_argument("--min-vote", type=float, default=0.4)
    ap.add_argument("--rip0", type=float, default=0.3, help="mixed: initial ripple (growth lines), tanh units of 25%% of the radius")
    ap.add_argument("--prefit", type=int, default=200, help="coil: silhouette-only fitting steps")
    ap.add_argument("--coil-grow", type=float, default=0.08, help="coil: enlarge each fitted body by this fraction")
    ap.add_argument("--lr-coil", type=float, default=0.01, help="coil: angles, rise, turns")
    ap.add_argument("--nctl", type=int, default=6, help="spine control points")
    ap.add_argument("--nrad", type=int, default=8, help="radius control points")
    ap.add_argument("--nsph", type=int, default=24, help="spheres per body")
    ap.add_argument("--f0", type=float, default=0.7, help="initial vertical squash of the cross section")
    ap.add_argument("--rgrow", type=float, default=1.0, help="initial radius = distance transform * this + 1 px")
    ap.add_argument("--zs", type=float, default=4.0, help="px of lift per unit of depth.json order")
    ap.add_argument("--ripple-freq", type=float, default=4.0)
    ap.add_argument("--tex-around", type=int, default=64)
    ap.add_argument("--tex-along", type=int, default=96)
    ap.add_argument("--ground-res", type=int, default=128)
    ap.add_argument("--ground-z", type=float, default=0.0, help="height of the ground plane, px (raise it to sit just under the pile)")
    ap.add_argument("--ground-dark", type=float, default=1.5, help="logit shift darkening the ground at init")
    # render
    ap.add_argument("--sig", type=float, default=0.5, help="silhouette softness, px")
    ap.add_argument("--tau-s", type=float, default=1.0, help="blend between the spheres of a body, px of depth")
    ap.add_argument("--tau-z", type=float, default=0.75, help="soft z-buffer between bodies, px of depth")
    ap.add_argument("--zbuf", default="cover", choices=["cover", "key"], help="z-buffer: covering surfaces compete on depth (cover), or the old coverage-penalised depth (key)")
    ap.add_argument("--chunk", type=int, default=150000, help="pixel-body pairs per checkpointed chunk")
    ap.add_argument("--pen0", type=float, default=0.6)
    ap.add_argument("--pen", type=float, default=0.06)
    # training
    ap.add_argument("--train-light", default="135,50", help="az,el deg: the fixed light, and the learned rig's start")
    ap.add_argument("--light", default="fixed", choices=["fixed", "random", "learn", "perbody"],
                    help="training light: fixed; a random light per item; half random, half a learned rig; learned rig plus per-body tint/offset")
    ap.add_argument("--lr-light", type=float, default=0.02)
    ap.add_argument("--wcov", type=float, default=0.0, help="visible ground, per 5%% of the frame")
    ap.add_argument("--wcontact", type=float, default=0.0, help="overlap between neighbouring bodies")
    ap.add_argument("--tol", type=float, default=0.3, help="contact: overlap allowed, fraction of the smaller radius")
    ap.add_argument("--size-slack", type=float, default=0.2, help="bodies may grow this much (log) past their fitted size freely")
    ap.add_argument("--wsize", type=float, default=20.0, help="cost of growing past that")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--jitter", type=float, default=3.5)
    ap.add_argument("--yaw", type=float, default=0.08)
    ap.add_argument("--lr-tex", type=float, default=0.04)
    ap.add_argument("--lr-prior", type=float, default=0.0, help="also train the wrapped-still prior texture (tex0)")
    ap.add_argument("--lr-ground", type=float, default=0.03)
    ap.add_argument("--lr-ctl", type=float, default=0.12)
    ap.add_argument("--lr-rad", type=float, default=0.01)
    ap.add_argument("--lr-shape", type=float, default=0.02)
    ap.add_argument("--wsil", type=float, default=1.0)
    ap.add_argument("--sil-floor", type=float, default=0.05, help="fraction of --wsil left after annealing")
    ap.add_argument("--sil-until", type=float, default=0.6, help="anneal over this fraction of the steps")
    ap.add_argument("--wcurv", type=float, default=0.5)
    ap.add_argument("--wrim", type=float, default=1.0)
    ap.add_argument("--wdark", type=float, default=0.0, help="cost of visible texture darker than --dark-floor (luminance)")
    ap.add_argument("--dark-floor", type=float, default=0.15)
    ap.add_argument("--zband", default="30,46", help="lo,hi px: where every body's top should sit")
    ap.add_argument("--zbot", type=float, default=-4.0, help="px: no body's bottom below this")
    ap.add_argument("--contact-disp", type=int, default=0, help="contact term counts each body's relief toward its neighbour (1) or not (0)")
    ap.add_argument("--wfill", type=float, default=0.0, help="cost of visible points sitting below the surface around them: bodies inflate into crevices")
    ap.add_argument("--wzband", type=float, default=0.0, help="cost of tops outside --zband and bottoms below --zbot")
    ap.add_argument("--ao", type=float, default=0.0, help="screen-space ambient occlusion strength (0 = off)")
    ap.add_argument("--amb-scale", type=float, default=1.0, help="scale the rig's ambient light (less fill: darker shadows, more form shading)")
    ap.add_argument("--sky", type=float, default=0.0, help="3D occlusion of the ambient light by the spheres over a ring of sky directions, strength (0 = off)")
    ap.add_argument("--sky-pen0", type=float, default=1.5, help="sky occlusion: penumbra at contact, px")
    ap.add_argument("--sky-pen", type=float, default=0.15, help="sky occlusion: penumbra growth per px of distance")
    ap.add_argument("--outline", type=float, default=0.0, help="drawn dark ring at every body boundary, strength (0 = off)")
    ap.add_argument("--outline-w", type=float, default=2.0, help="its falloff width, px")
    ap.add_argument("--ao-dif", type=float, default=0.5, help="share of the occlusion applied to the key light too")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", help="tube.pt: start from these bodies and textures")
    ap.add_argument("--skip-init-tex", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="chunked training: the first step of this chunk")
    ap.add_argument("--total", type=int, default=0, help="chunked training: steps in the whole schedule (default start + steps)")
    # outputs
    ap.add_argument("--evals", default="lights,clay,turn,orbit,tilt", help="which outputs to render after training")
    ap.add_argument("--loop-max-el", type=float, default=65.0, help="light loop: centre no higher than this, deg")
    ap.add_argument("--turn", default="", help="bodies (label ids) for turntables, comma separated")
    ap.add_argument("--anim", type=int, default=0, help="frames of a wriggle loop (0 = none)")
    ap.add_argument("--anim-amp", type=float, default=2.0, help="wriggle, px at the spine")
    ap.add_argument("--anim-drift", type=float, default=1.5, help="drift, px rms")
    ap.add_argument("--anim-seed", type=int, default=11)
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)
    save = lambda x, p: Image.fromarray((x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(p)

    L = np.load(f"{a.seg}/{a.labels}").astype(np.int64)
    S = L.shape[0]
    K = int(L.max())
    img = np.asarray(Image.open(a.image).convert("RGB")).astype(np.float32) / 255
    still = torch.from_numpy(img.transpose(2, 0, 1))[None].to(dev)
    zord = json.load(open(f"{a.seg}/depth.json"))["z"]
    ctls, rads = [], []
    for k in range(1, K + 1):
        c2, rd = seed_body(L == k, S, a.nctl, a.nrad)
        rd = rd * a.rgrow + 1.0
        rc = np.interp(np.linspace(0, 1, a.nctl), np.linspace(0, 1, a.nrad), rd)
        cz = a.f0 * rc * 0.95 + a.zs * zord[k]
        ctls.append(np.concatenate([c2, cz[:, None]], 1)); rads.append(rd)
    nbr = torch.tensor([[q["i"] - 1, q["j"] - 1] for q in json.load(open(f"{a.seg}/graph.json"))["pairs"]], device=dev)
    st_load = torch.load(a.load, map_location="cpu") if a.load else None
    if st_load is not None and "disp" in st_load["tubes"]:
        d0 = st_load["tubes"]["disp"]
        if tuple(d0.shape[1:]) != (a.disp_around, a.disp_along):  # a finer relief grid: start from the coarse one
            amp0 = st_load.get("args", {}).get("disp_amp", a.disp_amp) or a.disp_amp
            v = (amp0 * torch.tanh(d0))[:, None]
            v = F.interpolate(torch.cat([v[:, :, -1:], v, v[:, :, :1]], 2), size=(a.disp_around + 2 * a.disp_around // d0.shape[1], a.disp_along), mode="bilinear", align_corners=False)
            k_ = a.disp_around // d0.shape[1]
            v = v[:, 0, k_:k_ + a.disp_around]
            st_load["tubes"]["disp"] = torch.atanh((v / a.disp_amp).clamp(-0.99, 0.99))
    if a.body == "mixed" and st_load is not None:
        tb = Mixed.from_state(st_load["tubes"], a, dev)
    elif a.body == "mixed" and a.init_from:
        tb = Mixed.from_state(torch.load(a.init_from, map_location="cpu")["tubes"], a, dev)
        log(f"bodies from {a.init_from}")
    elif a.body == "mixed":
        target, cap = None, None
        if a.amodal:
            lgn = lambda im: np.log(np.clip(im, 1 / 512, 1 - 1 / 512) / (1 - np.clip(im, 1 / 512, 1 - 1 / 512))).transpose(2, 0, 1).astype(np.float32)
            dj = json.load(open(f"{a.seg}/depth.json"))
            votes = [(v["i"], v["j"], v["vote"]) for v in dj["votes"]]
            aff = {(q["i"], q["j"]): q["affinity"] for q in json.load(open(f"{a.seg}/graph.json"))["pairs"]}
            pins = {tuple(sorted(map(int, s_.split("-")))) for s_ in a.pin.split(",") if s_}
            soft = {}
            for grp in a.soft.split(";"):
                if grp:
                    ids, c_ = grp.split(":")
                    for i_ in ids.split(","):
                        soft[int(i_)] = float(c_)
            sc = build_scene(L, lgn(img), votes, aff, pins, soft, a)
            AM = amodal_masks(sc, S)
            target = torch.from_numpy(AM.reshape(K, -1)).to(dev)
            cen, rad = mask_caps(AM, S, a.cap_slack)
            cap = (torch.from_numpy(cen).to(dev), torch.from_numpy(rad).to(dev))
            log(f"amodal outlines: area {AM.sum((1, 2)).mean() / np.bincount(L.ravel())[1:].mean():.2f}x the visible part on average")
        if a.prefit_cache and os.path.exists(a.prefit_cache):
            pc = torch.load(a.prefit_cache, map_location="cpu", weights_only=False)
            P2, hand2, iou_c, ct, rt, iou_t = pc["P2"], pc["hand2"], pc["iou_c"].to(dev), pc["ct"], pc["rt"], pc["iou_t"].to(dev)
            log(f"prefit from {a.prefit_cache}")
        else:
            P2, hand2, iou_c = coil_prefit(L, zord, a, dev, log, steps=a.prefit, target=target, cap=cap, spill_w=a.spill)
            ct, rt, iou_t = tube_prefit(np.stack(ctls), np.stack(rads), L, a, dev, log, steps=a.prefit, target=target, cap=cap, spill_w=a.spill)
            if a.prefit_cache:
                torch.save(dict(P2=P2, hand2=hand2, iou_c=iou_c.cpu(), ct=ct, rt=rt, iou_t=iou_t.cpu()), a.prefit_cache)
        P2["lrho"] = P2["lrho"] + math.log(1 + a.coil_grow)  # a little larger than the visible part: neighbours tuck under
        rt = rt * (1 + a.coil_grow)
        is_coil = (iou_c > iou_t - a.coil_bias).cpu().numpy()
        tb = Mixed(ct, rt, P2, hand2, is_coil, a, dev)
        with torch.no_grad():
            tb.rip.fill_(a.rip0)
        if cap is not None:
            tb.register_buffer("capc", cap[0])
            tb.register_buffer("capr", cap[1])
        log(f"families by IoU (coil bias {a.coil_bias}): {int(is_coil.sum())} coils, {int((~is_coil).sum())} tubes; "
            f"coil IoU median {float(iou_c.median()):.2f}, tube {float(iou_t.median()):.2f}")
        if a.stack:
            stack_bodies(tb, L, zord, a, dev, log)
        if a.save_init:
            torch.save({"tubes": tb.state_dict(), "iou_coil": iou_c.cpu(), "iou_tube": iou_t.cpu()}, a.save_init)
    elif a.body == "coil" and st_load is not None:
        st0 = st_load["tubes"]
        tb = Coils({n: st0[n].numpy() for n in Coils.NAMES}, st0["hand"].numpy(), a, dev)
    elif a.body == "coil":
        P2, hand2, _ = coil_prefit(L, zord, a, dev, log, steps=a.prefit)
        P2["lrho"] = P2["lrho"] + math.log(1 + a.coil_grow)
        tb = Coils(P2, hand2, a, dev)
    else:
        tb = Tubes(np.stack(ctls), np.stack(rads), a, dev)
    lab = torch.from_numpy(L.reshape(-1) - 1).to(dev)  # body index per pixel
    top = Cam(np.eye(3), S, S, (S / 2, S / 2, 0.0), S)
    margin = 2.5 * a.jitter + 4
    sph = tb.spheres()
    pairs = build_pairs(sph, top, margin)
    log(f"{K} bodies, {a.nsph} spheres each; {len(pairs['pix'])} pixel-body pairs")
    az, el = map(float, a.train_light.split(","))
    Ltrain = Light(math.radians(az), math.radians(el))
    overhead = Light(0.0, math.radians(90))
    rig = Rig(math.radians(az), math.radians(el), a, dev) if a.light in ("learn", "perbody") else None
    blight = BodyLight(tb.K, dev) if a.light == "perbody" else None
    if st_load is not None:
        if rig is not None and "rig" in st_load:
            rig.load_state_dict(st_load["rig"])
        if blight is not None and "blight" in st_load:
            blight.load_state_dict(st_load["blight"])

    lg = lambda x: torch.log(x.clamp(1 / 512, 1 - 1 / 512) / (1 - x.clamp(1 / 512, 1 - 1 / 512)))
    with torch.no_grad():
        bl = torch.from_numpy(np.stack([ndi.gaussian_filter(img[..., c], 6, mode="wrap") for c in range(3)])).to(dev)
        tb.ground.copy_(F.interpolate(lg(bl)[None], size=(a.ground_res,) * 2, mode="area")[0] - a.ground_dark)
    wrap_info = None
    if st_load is not None:
        tb.load_state_dict(st_load["tubes"], strict=False)
        log(f"loaded {a.load}")
    elif a.tex_init == "wrap":
        wrap_info = wrap_texture(tb, sph, top, pairs, still, lab, a, dev, log)
    elif not a.skip_init_tex:
        # texture from the still: our own overhead shading divided out, projected onto the front of each body
        with torch.no_grad():
            o = render(tb, sph, top, pairs, [overhead], a, clay=1.0, shadows=False, aux=True, lab=lab)
            shade0 = o["img"][0].mean(0)
            albpx = (still[0] / shade0.clamp(min=0.5)).clamp(1 / 512, 1 - 1 / 512).reshape(3, -1).T  # (HW, 3)
            pix, obj, ep, den = o["pix"], o["obj"], o["ep"][0], o["den"][0]
            own = ep / den[pix]
            keep = (obj == lab[pix]) & (own > 0.5)
            Ha, Wl = a.tex_around, a.tex_along
            ti = torch.round(o["tu"][0] * Ha).long() % Ha
            tj = torch.round(o["tv"][0].clamp(0, 1) * (Wl - 1)).long()
            fl = (obj * Ha + ti) * Wl + tj
            fl, cl = fl[keep], albpx[pix[keep]]
            acc = torch.zeros(tb.K * Ha * Wl, 3, device=dev).index_add_(0, fl, cl)
            cnt = torch.zeros(tb.K * Ha * Wl, device=dev).index_add_(0, fl, torch.ones_like(fl, dtype=torch.float32))
            T_ = (acc / cnt.clamp(min=1)[:, None]).reshape(tb.K, Ha, Wl, 3).permute(0, 3, 1, 2)
            have = (cnt > 0).float().reshape(tb.K, 1, Ha, Wl)
            mean = (T_ * have).sum((2, 3), keepdim=True) / have.sum((2, 3), keepdim=True).clamp(min=1)
            fillv = torch.where(have > 0, T_, mean.expand_as(T_))
            for _ in range(80):  # unseen texels diffuse in from the seen ones
                p = torch.cat([fillv[:, :, -1:], fillv, fillv[:, :, :1]], 2)
                p = F.pad(p, (1, 1, 0, 0), mode="replicate")
                bl2 = (p[:, :, :-2, 1:-1] + p[:, :, 2:, 1:-1] + p[:, :, 1:-1, :-2] + p[:, :, 1:-1, 2:]) / 4
                fillv = torch.where(have > 0, T_, bl2)
            tb.tex.data.copy_(lg(fillv))
            log(f"texture projected: {float(have.mean()):.2f} of texels seen ({Ha} x {Wl} per body)")

    layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
    net = Backbone(a.net, dev)
    net.hook(layer)
    rg = Ring(net, layer, ch, S, dev)

    def score(fr):
        state = random.getstate()
        tot = 0
        for k in range(4):
            random.seed(1000 + k)
            tot = tot + rg.act(MultiScaleCrop(224 / S, 512)(fr))
        random.setstate(state)
        return float(tot / 4)

    ref = (lambda: rig()) if rig is not None else (lambda: as_rig(Ltrain, a, dev))  # the light the frames are shown under
    held = [overhead] + [Light(math.radians(az_), math.radians(50)) for az_ in range(0, 360, 45)]
    with torch.no_grad():
        alone = score(still)
        sph = tb.spheres()
        init_train = score(render(tb, sph, top, pairs, [ref()], a, blight=blight))
    log(f"still {alone:.3f}; init at the reference light {init_train:.3f}")

    groups = [
        {"params": [tb.tex], "lr": a.lr_tex},
        {"params": [tb.ground], "lr": a.lr_ground},
        {"params": tb.geom_params()[0], "lr": a.lr_ctl},
        {"params": tb.geom_params()[1], "lr": a.lr_coil},
        {"params": tb.geom_params()[2], "lr": a.lr_rad},
        {"params": [tb.fq, tb.rip, tb.ripf, tb.ripp], "lr": a.lr_shape},
        {"params": [tb.disp] if a.disp_amp > 0 else [], "lr": a.lr_disp},
    ]
    if a.lr_prior > 0 and hasattr(tb, "tex0"):  # the wrapped-still prior learns too (it carries the fine texture)
        t0_ = tb.tex0.detach().clone()
        del tb.tex0
        tb.tex0 = torch.nn.Parameter(t0_)
        groups.append({"params": [tb.tex0], "lr": a.lr_prior})
    if rig is not None:
        groups.append({"params": list(rig.parameters()), "lr": a.lr_light})
    if blight is not None:
        groups.append({"params": list(blight.parameters()), "lr": a.lr_light})
    opt = torch.optim.Adam([g for g in groups if g["params"]])
    base = [g["lr"] for g in opt.param_groups]
    rng = random.Random(a.seed + 1)
    curve = []
    ctl0 = tb.base()[0].detach().clone()

    def lights_for_step():
        if a.light == "fixed":
            return [Ltrain] * a.mb
        if a.light == "random":
            return [Light.random(rng) for _ in range(a.mb)]
        nr = a.mb // 2  # half the minibatch at random lights keeps the shapes honest, half at the learned rig
        return [Light.random(rng) for _ in range(nr)] + [rig() for _ in range(a.mb - nr)]

    total = a.total or a.start + a.steps
    for it in range(a.steps):
        fr_ = (a.start + it) / max(1, total)  # chunked training: one schedule over --total steps
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * 0.5 * (1 + math.cos(math.pi * fr_))
        if it % 20 == 0:
            with torch.no_grad():
                pairs = build_pairs(tb.spheres(), top, margin)
        jit = (torch.randn(a.mb, tb.K, 2, device=dev) * a.jitter).clamp(-2.2 * a.jitter, 2.2 * a.jitter)
        yaw = torch.randn(a.mb, tb.K, device=dev) * a.yaw
        sph = tb.spheres(jit=jit, yaw=yaw)
        o = render(tb, sph, top, pairs, lights_for_step(), a, aux=True, lab=lab, blight=blight)
        act = rg.act(MultiScaleCrop(224 / S, 512)(o["img"]))
        wsil = a.wsil * (a.sil_floor + (1 - a.sil_floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, fr_ / a.sil_until))))
        sil = -torch.log(o["own"] + 1e-3).mean()
        cov = o["gown"].mean()  # visible ground: the still is packed
        if it % 20 == 0 and a.wcontact > 0:
            nbr = near_pairs(tb, S)
        if a.wcontact > 0 and a.contact_disp:
            con = contact_loss_disp(tb, tb.spheres(), nbr, S, a.tol)
        else:
            con = contact_loss(tb, nbr, S, a.tol) if a.wcontact > 0 else torch.zeros((), device=dev)
        fil = fill_loss(o["wc"], S, S) if a.wfill > 0 else torch.zeros((), device=dev)
        reg = tb.reg()
        ep, den, pix, obj = o["ep"], o["den"], o["pix"], o["obj"]
        own = ep / den[:, pix]
        Y = (o["alb"] * torch.tensor([0.299, 0.587, 0.114], device=dev)).sum(-1)
        fz = o["nk"][..., 2]
        frontw = own * (fz > 0.7).float()
        refY = torch.zeros(a.mb, tb.K, device=dev).index_add_(1, obj, frontw * Y) / torch.zeros(a.mb, tb.K, device=dev).index_add_(1, obj, frontw).clamp(min=1)
        band = own * (fz < 0.4).float()
        rim = (band * torch.relu(0.75 * refY.detach()[:, obj] - Y) ** 2).sum() / band.sum().clamp(min=1) / 0.01
        capl = cap_loss(tb, tb.capc, tb.capr, S) if hasattr(tb, "capc") else torch.zeros((), device=dev)
        dark = (own * torch.relu(a.dark_floor - Y) ** 2).sum() / own.sum().clamp(min=1) / 0.01
        zb = zband_loss(tb, a) if a.wzband > 0 else torch.zeros((), device=dev)
        pen = wsil * sil + a.wcurv * reg + a.wrim * rim + a.wcov * cov / 0.05 + a.wcontact * con + a.wcap * capl + a.wdark * dark + a.wzband * zb + a.wfill * fil
        loss = (-act + act.detach().abs() * pen).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 25 == 0 or it == a.steps - 1:
            moved = (tb.base()[0].detach() - ctl0).norm(dim=-1).mean()
            curve.append((it, float(act.detach().mean())))
            lt = ""
            if rig is not None:
                dsc = rig.describe()
                lt = f" rig az {dsc['az_deg']:.0f} el {dsc['el_deg']:.0f} dif {dsc['dif']:.2f}"
            log(f"step {a.start + it + 1}/{total} act {act.detach().mean():.3f} [{', '.join(f'{float(x):.2f}' for x in act.detach())}] sil {float(sil):.3f} (w {wsil:.2f}) "
                f"ground {float(cov):.3f} contact {float(con):.3f} cap {float(capl):.3f} reg {float(reg):.3f} rim {float(rim):.3f} dark {float(dark):.3f} zband {float(zb):.3f} fill {float(fil):.3f} moved {float(moved):.2f}px r {float(tb.base()[1].mean()):.1f}{lt}")
    if True:  # also after 0 steps: the init (prior only) can be loaded and trained from
        ck = {"tubes": tb.state_dict(), "args": vars(a)}
        if rig is not None:
            ck["rig"] = rig.state_dict()
        if blight is not None:
            ck["blight"] = blight.state_dict()
        torch.save(ck, f"{a.out}/tube.pt")

    rep = dict(still=alone, init_ref_light=init_train, curve=curve, args=vars(a))
    if wrap_info is not None:
        rep["wrap"] = wrap_info
    if rig is not None:
        rep["rig"] = rig.describe()
    if blight is not None:
        with torch.no_grad():
            bt, bd = blight()
        rep["per_body"] = dict(tint_spread=float(bt.std(0).mean()), tint_minmax=[float(bt.min()), float(bt.max())],
                               dir_offset_deg_mean=math.degrees(float(bd.norm(dim=-1).mean())), dir_offset_deg_max=math.degrees(float(bd.norm(dim=-1).max())))
    ev = set(a.evals.split(","))
    if "grad" in ev:
        # where the channel's pressure goes: d act / d params on one frame at the reference light
        probe = rig if rig is not None else Rig(math.radians(az), math.radians(el), a, dev)
        for p_ in list(tb.parameters()) + list(probe.parameters()) + (list(blight.parameters()) if blight is not None else []):
            p_.grad = None
        sph = tb.spheres()
        pairs = build_pairs(sph, top, 4.0)
        g_ = torch.Generator().manual_seed(5)
        acts = []
        for k in range(4):
            random.seed(1000 + k)
            acts.append(rg.act(MultiScaleCrop(224 / S, 512)(render(tb, sph, top, pairs, [probe()], a, blight=blight))))
        torch.stack(acts).mean().backward()
        nrm = lambda ps: math.sqrt(sum(float((p_.grad ** 2).sum()) for p_ in ps if p_.grad is not None))
        rms = lambda ps: nrm(ps) / math.sqrt(max(1, sum(p_.numel() for p_ in ps)))
        gp = tb.geom_params()
        groups_ = {"light_direction (az, el logit)": [probe.az, probe.elq], "light_other (intensity, tint, ambient SH, penumbra)": [probe.difq, probe.tintq, probe.ambq, probe.sh1, probe.pen0q, probe.penq],
                   "geometry_positions_px": gp[0], "geometry_shape_angles": gp[1], "geometry_log_scales": gp[2], "geometry_squash_ripple": [tb.fq, tb.rip, tb.ripf, tb.ripp],
                   "texture": [tb.tex], "ground": [tb.ground]}
        if blight is not None:
            groups_["per_body_light"] = list(blight.parameters())
        rep["grad"] = {k: dict(norm=nrm(v), rms=rms(v), n=sum(p_.numel() for p_ in v)) for k, v in groups_.items()}
        log("grad norms: " + "; ".join(f"{k} {v['norm']:.3g}" for k, v in rep["grad"].items()))
        for p_ in list(tb.parameters()) + list(probe.parameters()):
            p_.grad = None
    with torch.no_grad():
        sph = tb.spheres()
        pairs = build_pairs(sph, top, 4.0)
        if "lights" in ev:
            tiles, acts = [], []
            for L_ in held:
                fr = render(tb, sph, top, pairs, [L_], a, blight=blight)
                acts.append(score(fr)); tiles.append(fr[0])
            save(torch.cat([torch.cat(tiles[r_ * 3:(r_ + 1) * 3], 2) for r_ in range(3)], 1), f"{a.out}/lights.png")
            rep["lights"] = ["overhead"] + [f"az{az_} el50" for az_ in range(0, 360, 45)]
            rep["held_out"] = acts
            fr = render_ss(tb, sph, top, pairs, ref(), a, blight=blight)
            save(fr[0], f"{a.out}/lit_ref.png")
            rep["ref_light_ss"] = score(fr)
            fr = render_ss(tb, sph, top, pairs, Ltrain, a, blight=blight)
            rep["train_light_135_50_ss"] = score(fr)
            log(f"held-out: overhead {acts[0]:.3f}; 8 at 50 deg mean {np.mean(acts[1:]):.3f} [{min(acts[1:]):.3f}, {max(acts[1:]):.3f}]; "
                f"reference light ss {rep['ref_light_ss']:.3f}; 135/50 ss {rep['train_light_135_50_ss']:.3f}")
            o = render(tb, sph, top, pairs, [ref()], a, aux=True, blight=blight)
            rep["visible_ground"] = float(o["gown"].mean())
            rep["contact"] = float(contact_loss(tb, near_pairs(tb, S), S, a.tol))
            save(torch.sigmoid(tb.ground), f"{a.out}/ground.png")
        if "clay" in ev:
            save(render_ss(tb, sph, top, pairs, Ltrain, a, clay=0.72)[0], f"{a.out}/clay_top.png")
            for name, ang in (("clay_tilt60", math.radians(60)), ("clay_back60", math.radians(-60))):
                cam = Cam(rot_x(ang), S, S, (S / 2, S / 2, 0.0), S)
                pr = build_pairs(sph, cam, 4.0)
                save(render(tb, sph, cam, pr, [Ltrain], a, clay=0.72)[0], f"{a.out}/{name}.png")
            cam = Cam(rot_x(math.radians(35)), S, S, (S / 2, S / 2, 0.0), S)
            save(render(tb, sph, cam, build_pairs(sph, cam, 4.0), [ref()], a, blight=blight)[0], f"{a.out}/lit_tilt35.png")
            log("clay renders")
        if "turn" in ev and a.turn:
            ids = [int(x) for x in a.turn.split(",")]
            rows = []
            for k in ids:
                sub = [k - 1]
                s1 = tb.spheres(sub=sub)
                Cn = s1["C"][0, 0].cpu().numpy()
                ctr = (Cn[:, 0].mean(), Cn[:, 1].mean(), Cn[:, 2].mean())
                ext = float((np.abs(Cn[:, :2] - np.array(ctr[:2])).max() + s1["r"].max()).item())
                Z = int(2 * math.ceil(ext * 1.25 + 4))
                ys, xs = np.nonzero(L == k)
                cy, cx = int(np.median(ys)), int(np.median(xs))
                h2 = Z // 2
                crop = torch.roll(still[0], (h2 - cy, h2 - cx), (1, 2))[:, :Z, :Z]
                d = f"{a.out}/turn/b{k}"
                os.makedirs(d, exist_ok=True)
                frames = []
                for i in range(36):
                    Rm = rot_x(math.radians(55)) @ rot_z(2 * math.pi * i / 36)
                    cam = Cam(Rm, Z, Z, ctr, S, wrap=False)
                    pr = build_pairs(s1, cam, 2.0)
                    cl = render(tb, s1, cam, pr, [Ltrain], a, clay=0.72, ground="flat")[0]
                    tx = render(tb, s1, cam, pr, [ref()], a, ground="flat", tex=tb.tex[sub], blight=blight, sub=sub)[0]
                    frames.append(torch.cat([cl, tx], 2))
                    save(F.interpolate(torch.cat([crop, frames[-1]], 2)[None], size=(256, 768), mode="bilinear", antialias=True)[0], f"{d}/f{i:04d}.png")
                encode(d, f"{a.out}/turn_b{k}.mp4", fps=12)
                r_ = torch.cat([crop] + [frames[i] for i in (0, 9, 18, 27)], 2)
                rows.append(F.interpolate(r_[None], size=(200, 200 * r_.shape[2] // r_.shape[1]), mode="bilinear", antialias=True)[0])
            wmax = max(r_.shape[2] for r_ in rows)
            save(torch.cat([F.pad(r_, (0, wmax - r_.shape[2])) for r_ in rows], 1), f"{a.out}/turntables.png")
            log("turntables")
        if "sweep" in ev or "lightloop" in ev:
            azs = list(range(0, 360, 15))
            els = list(range(10, 90, 15))
            grid = np.zeros((len(els), len(azs)))
            base_rig = ref()
            for i, e_ in enumerate(els):
                for j, a_ in enumerate(azs):
                    grid[i, j] = score(render(tb, sph, top, pairs, [with_dir(base_rig, math.radians(a_), math.radians(e_))], a, blight=blight))
            rep["sweep"] = dict(az=azs, el=els, score=grid.tolist())
            save_heat(grid, azs, els, f"{a.out}/sweep.png")
            i_, j_ = np.unravel_index(grid.argmax(), grid.shape)
            log(f"light sweep: max {grid.max():.3f} at az {azs[j_]} el {els[i_]}, min {grid.min():.3f}")
        if "orbit" in ev:
            d = f"{a.out}/orbit"
            os.makedirs(d, exist_ok=True)
            base_rig = ref()
            for i in range(48):
                save(render(tb, sph, top, pairs, [with_dir(base_rig, 2 * math.pi * i / 48, math.radians(45))], a, blight=blight)[0], f"{d}/f{i:04d}.png")
            encode(d, f"{a.out}/orbit.mp4")
            log("light orbit")
        if "tilt" in ev:
            d = f"{a.out}/tilt"
            os.makedirs(d, exist_ok=True)
            for i in range(48):
                ph = 2 * math.pi * i / 48
                b_ = math.radians(20) * math.cos(ph)
                Ry = np.array([[math.cos(b_), 0, math.sin(b_)], [0, 1, 0], [-math.sin(b_), 0, math.cos(b_)]])
                Rm = rot_x(math.radians(20) * math.sin(ph)) @ Ry  # the camera circles 20 deg off vertical
                cam = Cam(Rm, S, S, (S / 2, S / 2, 0.0), S)
                pr = build_pairs(sph, cam, 4.0)
                save(render(tb, sph, cam, pr, [Ltrain], a)[0], f"{d}/f{i:04d}.png")
            encode(d, f"{a.out}/tilt.mp4")
            log("camera tilt")
        if a.anim:
            d = f"{a.out}/anim"
            os.makedirs(d, exist_ok=True)
            g = torch.Generator().manual_seed(a.anim_seed)
            ph = (torch.rand(tb.K, 3, generator=g) * 2 * math.pi).to(dev)
            nw = (1 + torch.rand(tb.K, generator=g) * 1.5).to(dev)  # half-waves along the spine
            cf = torch.randn(tb.K, 2, 2, 2, generator=g).to(dev) * torch.tensor([1.0, 0.5], device=dev).view(1, 2, 1, 1)
            s0 = tb.spheres()
            Tn, sn = s0["T"][0], s0["s"]
            perp = torch.stack([-Tn[..., 1], Tn[..., 0], torch.zeros_like(Tn[..., 0])], -1)
            perp = perp / (perp.norm(dim=-1, keepdim=True) + 1e-6)
            u_ = sn / (sn[:, -1:] + 1e-3)
            acts = []
            for t in range(a.anim):
                tt = 2 * math.pi * t / a.anim
                amp = torch.sin(math.pi * nw[:, None] * u_ + ph[:, :1] + tt) * u_.clamp(0.15, 1)  # travelling bend, ends move most
                wob = perp * (a.anim_amp * amp)[..., None]
                ang = torch.tensor([tt, 2 * tt], device=dev)
                path = (cf[..., 0] * torch.cos(ang)[None, None] + cf[..., 1] * torch.sin(ang)[None, None]).sum(-1)  # (K, 2)
                path = path * a.anim_drift / math.sqrt(1.25)
                sp = tb.spheres(jit=path[None], wob=wob)
                pr = build_pairs(sp, top, 4.0)
                fr = render_ss(tb, sp, top, pr, ref(), a, blight=blight)
                acts.append(score(fr))
                save(fr[0], f"{d}/f{t:04d}.png")
            encode(d, f"{a.out}/anim.mp4")
            rep["anim"] = dict(mean=float(np.mean(acts)), min=float(np.min(acts)), per_frame=acts)
            log(f"wriggle loop: {a.anim} frames, mean {np.mean(acts):.3f} min {np.min(acts):.3f}")
        if "lightloop" in ev:
            # a closed path for the key light through the high-activation region of the sweep: a figure of
            # eight round the best direction, as wide in azimuth as keeps every point within 5% of the peak
            G = np.asarray(grid)
            def fval(az_d, el_d):
                x = (az_d % 360) / 15.0
                y = np.clip((el_d - 10) / 15.0, 0, len(els) - 1 - 1e-6)
                x0, y0 = int(np.floor(x)) % len(azs), int(np.floor(y))
                fx, fy = x - np.floor(x), y - y0
                x1, y1 = (x0 + 1) % len(azs), min(y0 + 1, len(els) - 1)
                return (1 - fy) * ((1 - fx) * G[y0, x0] + fx * G[y0, x1]) + fy * ((1 - fx) * G[y1, x0] + fx * G[y1, x1])
            Gm = np.where(np.array(els)[:, None] <= a.loop_max_el, G, -np.inf)  # a near-overhead light barely moves anything
            i0, j0 = np.unravel_index(Gm.argmax(), G.shape)
            az0, el0 = azs[j0], els[i0]
            ts_ = np.linspace(0, 1, 64, endpoint=False)
            best = (0.0, 0.0)
            for A in (120, 90, 75, 60, 45, 30, 20, 10):
                B = min(15.0, A / 3)
                path = [(az0 + A * math.sin(2 * math.pi * t), float(np.clip(el0 + B * math.sin(4 * math.pi * t), 12, 85))) for t in ts_]
                if min(fval(*p_) for p_ in path) >= 0.95 * G[i0, j0]:
                    best = (A, B)
                    break
            A, B = best if best[0] > 0 else (10.0, 3.0)
            d = f"{a.out}/lightloop"
            os.makedirs(d, exist_ok=True)
            acts = []
            for i in range(96):
                t = i / 96
                az_d, el_d = az0 + A * math.sin(2 * math.pi * t), float(np.clip(el0 + B * math.sin(4 * math.pi * t), 12, 85))
                fr = render_ss(tb, sph, top, pairs, with_dir(base_rig, math.radians(az_d), math.radians(el_d)), a, blight=blight)
                acts.append(score(fr))
                save(fr[0], f"{d}/f{i:04d}.png")
            encode(d, f"{a.out}/lightloop.mp4")
            rep["lightloop"] = dict(centre_az=az0, centre_el=el0, az_amp=A, el_amp=B, mean=float(np.mean(acts)), min=float(np.min(acts)), per_frame=acts)
            log(f"light loop: az {az0}±{A}, el {el0}±{B:.0f}: mean {np.mean(acts):.3f} min {np.min(acts):.3f}")
    rep["secs"] = round(time.time() - t0)
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log("done")


def save_heat(G, azs, els, path, cell=18):
    """Channel score over light azimuth (x) x elevation (y, high at top) as a coloured grid with the
    values written in, the best cell outlined."""
    from PIL import ImageDraw
    lo, hi = float(G.min()), float(G.max())
    stops = np.array([[0.07, 0.05, 0.2], [0.15, 0.35, 0.55], [0.2, 0.65, 0.55], [0.95, 0.85, 0.25]])
    ne, na = G.shape
    cw, ch_ = cell * 2, cell
    im = Image.new("RGB", (na * cw + 60, ne * ch_ + 40), (20, 20, 20))
    dr = ImageDraw.Draw(im)
    for i in range(ne):
        for j in range(na):
            t = (G[i, j] - lo) / max(hi - lo, 1e-6) * (len(stops) - 1)
            k = min(int(t), len(stops) - 2)
            c = stops[k] + (stops[k + 1] - stops[k]) * (t - k)
            y = (ne - 1 - i) * ch_ + 10
            x = j * cw + 50
            dr.rectangle([x, y, x + cw - 1, y + ch_ - 1], fill=tuple(int(255 * v) for v in c))
            dr.text((x + 3, y + 3), f"{G[i, j]:.1f}", fill=(0, 0, 0) if t > 1.5 else (230, 230, 230))
    i0, j0 = np.unravel_index(G.argmax(), G.shape)
    dr.rectangle([j0 * cw + 50, (ne - 1 - i0) * ch_ + 10, j0 * cw + 50 + cw - 1, (ne - 1 - i0) * ch_ + 10 + ch_ - 1], outline=(255, 255, 255), width=2)
    for i, e_ in enumerate(els):
        dr.text((5, (ne - 1 - i) * ch_ + 13), f"el {e_}", fill=(200, 200, 200))
    for j, a_ in enumerate(azs):
        if j % 2 == 0:
            dr.text((j * cw + 52, ne * ch_ + 14), f"{a_}", fill=(200, 200, 200))
    dr.text((50, ne * ch_ + 26), f"azimuth, deg   score {lo:.2f} .. {hi:.2f}", fill=(200, 200, 200))
    im.save(path)


if __name__ == "__main__":
    main()
