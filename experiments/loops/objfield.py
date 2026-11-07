"""Object field loop: every segmented object is its own sprite, wobbling on its own, composited by depth.

Scene (from segobj.py, objgraph.py, objdepth.py):
  sprite k  a square window around object k holding colour (the still's logits plus a learned
            Fourier residual), alpha (see edges) and a fixed depth surface. Its alpha covers
            the object's visible region plus an amodal extension under the neighbours that cover it;
            the colour there starts as the visible texture mirrored across the smoothed outline
            (--fill; see mirror_fill) and is learned only where motion reveals it.
  depth     per pixel and per sprite: 0 on the visible region, -1 on the hidden extension, plus a
            +-delta bump near each wall from that wall's front/back vote. Walls are local, so overlaps
            that loop round (A over B over C over A) are representable.
  motion    per sprite, periodic in the loop phase u (a few harmonics each): drift, rotation, breathing
            and the m = 2, 3 radial wobble modes of a soft blob. A pixel q of sprite k is displaced by
            d + s rel + r perp(rel) + (a2 cos 2t + b2 sin 2t + a3 cos 3t + b3 sin 3t) rel, rel = q - centre.
  edges     alpha is a soft step (--edge-w) across a smooth outline: the signed distance to the segment's
            outline, blurred (--smooth) to iron out pixel wiggles, plus a coarse learned offset. Depth
            surfaces get the same blur, so where two visible shells meet the crossing is smooth too.
  frame     every canvas pixel gathers the few sprites that can reach it, samples each one through
            its displacement, sorts them by depth and composites front to back ("over"), with the
            still, blurred and darkened, as the farthest layer.

Loss: -activation (fixed multi-scale crops, as in cellwarp.py)
      + |activation| * ( wamp  amplitude held at a target (the optimizer would shrink the motion)
                       + wcpl  sum over walls of coupling_kj * |u_k - u_j|^2 at the wall (coupling from the
                               same-object score: pinned pairs 1, others scaled, a small floor everywhere)
                       + wcon  contact: (1 - coupling) * relu((u_k - u_j) . n)^2, separate objects pushing
                               into each other across their wall
                       + wopq  outline offsets stay small (sprites are solid by construction: alpha is a
                               soft step across a smooth outline)
                       + wfid  rest pose (no motion) against the still: the objects stay the still's objects )

usage: objfield.py --image still.png --seg segdir --target inception4c:7 --out DIR [--rest-only]
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, math, random, time
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from striate.model import Backbone
from striate.param import FourierImage
from striate.util import pick_device
from zoo import MultiScaleCrop
from ring import Ring
from cells import Path

def circ_centroid(ys, xs, S):
    a = 2 * np.pi / S
    cy = np.arctan2(np.sin(a * ys).mean(), np.cos(a * ys).mean()) / a % S
    cx = np.arctan2(np.sin(a * xs).mean(), np.cos(a * xs).mean()) / a % S
    return cy, cx


def inpaint(x, m, sigmas=(1.5, 3, 6, 12, 24)):
    """Fill x (C, H, W) outside mask m by normalised convolution at growing scales."""
    out = x * m
    have = m.astype(np.float32).copy()
    for s in sigmas:
        num = np.stack([ndi.gaussian_filter(x[c] * m, s) for c in range(x.shape[0])])
        den = ndi.gaussian_filter(m.astype(np.float32), s)
        fill = (den > 0.05) & (have == 0)
        out[:, fill] = num[:, fill] / den[fill]
        have[fill] = 1
    return out


def mirror_fill(x, m, hidden, fallback, smooth, inset=1.5):
    """Colour for the hidden part of a sprite: the visible texture reflected across the smoothed
    outline, so it carries on past the edge. Switching from the still to a blurred inpaint at the
    segment's own pixel staircase showed up as a jagged seam wherever motion revealed it.
    The mirror is a curve ``inset`` px inside the outline, so the band at the edge (crevice shading and
    pixels of the neighbour that the segment caught) is replaced rather than copied outward. This only
    happens next to ``hidden`` (the amodal extension, the part motion can reveal); elsewhere the edge
    band is the object's own rim and stays as in the still.
    Where the reflection leaves the object (thin or concave parts) it falls back to the inpaint.
    Also returns the visible pixels it replaced, which the rest-pose fidelity term must not pull back
    to the still (the colour residual would repaint the seam into every frame)."""
    sd = ndi.distance_transform_edt(~m) - ndi.distance_transform_edt(m) + 0.5 * (2 * m - 1)
    sd = ndi.gaussian_filter(sd.astype(np.float32), max(smooth, 1.0))
    gy, gx = np.gradient(sd)
    nn = np.sqrt(gy**2 + gx**2) + 1e-6
    yy, xx = np.mgrid[: m.shape[0], : m.shape[1]].astype(np.float32)
    d = np.maximum(sd + inset, 0)
    sy, sx = yy - 2 * d * gy / nn, xx - 2 * d * gx / nn
    mir = np.stack([ndi.map_coordinates(x[c], [sy, sx], order=1, mode="nearest") for c in range(x.shape[0])])
    ok = ndi.map_coordinates(sd, [sy, sx], order=1, mode="constant", cval=1.0) < -inset
    w = ndi.gaussian_filter(ok.astype(np.float32), 1.5) * ok
    filled = np.where(sd <= -inset, x, w * mir + (1 - w) * fallback)
    base = np.where(m, x, fallback)
    if not hidden.any():
        return base.astype(np.float32), np.zeros_like(m)
    # 1 within inset + 1 px of the hidden part, fading to 0 over 3 px along the outline
    wh = np.clip(1 - (ndi.distance_transform_edt(~hidden) - inset - 1) / 3, 0, 1)
    return (base + wh * (filled - base)).astype(np.float32), m & (wh > 0) & (sd > -inset - 1)


def build_scene(L, logits, votes, aff, pins, soft, a):
    """Sprites and per-pixel candidate lists. Returns a dict of numpy arrays."""
    S, n = L.shape[0], int(L.max())
    vote = {(i, j): v for i, j, v in votes}

    def front(j, k):  # is j in front of k?
        v = vote.get((min(j, k), max(j, k)), 0.0)
        # a weak vote (the cues disagree) means side by side: neither one hides under the other
        return (v > 0) == (j < k) if abs(v) >= getattr(a, "min_vote", 0.0) and v != 0 else False

    def coupling(i, j):
        key = (min(i, j), max(i, j))
        if key in pins:
            return 1.0
        c = min(1.0, aff.get(key, 0.0) / 0.25)
        if i in soft and j in soft:
            c = max(c, soft[i])
        return c

    cen, rad = np.zeros((n + 1, 2)), np.zeros(n + 1)
    for k in range(1, n + 1):
        ys, xs = np.nonzero(L == k)
        cy, cx = circ_centroid(ys, xs, S)
        dy = (ys - cy + S / 2) % S - S / 2
        dx = (xs - cx + S / 2) % S - S / 2
        cen[k] = cy, cx
        rad[k] = np.sqrt(dy**2 + dx**2).max()
    H = int(2 * math.ceil(rad.max() + a.ext + a.reach + 2))
    H += H % 2
    K = n
    sdf = np.full((K + 1, H, H), 100.0, np.float32)
    depth = np.full((K + 1, H, H), -1.0, np.float32)
    color = np.zeros((K + 1, 3, H, H), np.float32)
    origin = np.zeros((K + 1, 2), np.int64)
    centre = np.zeros((K + 1, 2), np.float32)  # sprite-local (y, x)
    vis_area = np.zeros(K + 1, np.float32)
    seam = np.zeros((S, S), bool)  # visible pixels whose colour mirror_fill replaced
    cand = {}
    for k in range(1, n + 1):
        oy, ox = int(round(cen[k][0])) - H // 2, int(round(cen[k][1])) - H // 2
        origin[k] = oy, ox
        centre[k] = cen[k][0] - oy, cen[k][1] - ox
        iy, ix = (oy + np.arange(H)) % S, (ox + np.arange(H)) % S
        win = L[np.ix_(iy, ix)]
        mask = win == k
        vis_area[k] = mask.sum()
        nb = [j for j in np.unique(win) if j not in (0, k)]
        fronts = [j for j in nb if front(j, k) or coupling(j, k) >= 0.5]
        dmask = ndi.distance_transform_edt(~mask)
        ext = mask | ((dmask <= a.ext) & np.isin(win, fronts))
        # signed distance to the shape's outline (positive outside), blurred: the outline it encodes
        # is the segmentation's, with the pixel wiggles ironed out
        sd = ndi.distance_transform_edt(~ext) - ndi.distance_transform_edt(ext) + 0.5 * (2 * ext - 1)
        sdf[k] = ndi.gaussian_filter(sd.astype(np.float32), a.smooth) if a.smooth > 0 else sd
        z = np.where(mask, 0.0, -1.0).astype(np.float32)
        for j in nb:
            c = coupling(j, k)
            sgn = 0.0 if c >= 0.5 else (-1.0 if front(j, k) else 1.0)
            if sgn == 0:
                continue
            dj = ndi.distance_transform_edt(win != j)
            z += np.where(mask, a.delta * sgn * np.exp(-(dj**2) / (2 * a.wall_sigma**2)), 0.0)
        depth[k] = ndi.gaussian_filter(z, a.smooth) if a.smooth > 0 else z
        wl = logits[:, iy][:, :, ix]
        color[k] = inpaint(wl, mask)
        if a.fill == "mirror":
            color[k], rep = mirror_fill(wl, mask, ext & ~mask, color[k], a.smooth, a.inset)
            seam[np.ix_(iy, ix)] |= rep
        reach = ndi.distance_transform_edt(~ext) <= a.reach
        ry, rx = np.nonzero(reach)
        for y, x in zip(ry, rx):
            cand.setdefault(int(iy[y] * S + ix[x]), []).append((k, y, x))
    M = max(len(v) for v in cand.values())
    ck = np.zeros((S * S, M), np.int64)  # 0 is an empty sprite (alpha off), used as padding
    cq = np.zeros((S * S, M, 2), np.float32)
    for p, lst in cand.items():
        for m, (k, y, x) in enumerate(lst):
            ck[p, m] = k
            cq[p, m] = y, x
    # wall samples: midpoints between a pixel of i and its neighbour in j, with the normal from i to j
    wl, nrm = [], []
    for ax in (0, 1):
        A, B = L, np.roll(L, -1, ax)
        d = A != B
        ys, xs = np.nonzero(d)
        step = np.array([1.0, 0.0]) if ax == 0 else np.array([0.0, 1.0])
        for y, x in zip(ys, xs):
            i, j = int(A[y, x]), int(B[y, x])
            wl.append((min(i, j), max(i, j), y + 0.5 * step[0], x + 0.5 * step[1]))
            nrm.append(step if i < j else -step)
    wl = np.array(wl)
    cpl = np.array([coupling(int(i), int(j)) for i, j, _, _ in wl], np.float32)
    return dict(
        H=H, K=K, M=M, sdf=sdf, depth=depth, color=color, origin=origin, centre=centre,
        ck=ck, cq=cq, wall=wl, wall_c=cpl, wall_n=np.array(nrm, np.float32), vis_area=vis_area, rad=rad, seam=seam,
    )


def gather_bilinear(spr, k, yx):
    """spr (K, C, H, H); k (...,) long; yx (..., 2) float sprite coords -> (..., C). Clamped at the edge."""
    Kn, C, H, _ = spr.shape
    flat = spr.permute(0, 2, 3, 1).reshape(-1, C)
    y, x = yx[..., 0].clamp(0, H - 1.001), yx[..., 1].clamp(0, H - 1.001)
    y0, x0 = torch.floor(y), torch.floor(x)
    fy, fx = (y - y0)[..., None], (x - x0)[..., None]
    y0, x0 = y0.long(), x0.long()
    base = k * H * H
    i00 = base + y0 * H + x0
    v = (
        flat[i00] * (1 - fy) * (1 - fx)
        + flat[i00 + 1] * (1 - fy) * fx
        + flat[i00 + H] * fy * (1 - fx)
        + flat[i00 + H + 1] * fy * fx
    )
    return v


class Field:
    def __init__(self, sc, logits, dev, Kh, still_bg, aa=0.0, soft=0.5):
        self.dev = dev
        self.soft = soft
        self.aa_k = None
        if aa > 0:
            r = int(math.ceil(3 * aa))
            x = torch.arange(-r, r + 1, dtype=torch.float32)
            k = torch.exp(-(x**2) / (2 * aa**2))
            self.aa_k = (k / k.sum()).to(dev)
        self.K, self.H, self.M = sc["K"], sc["H"], sc["M"]
        K1 = self.K + 1
        self.sdf0 = torch.from_numpy(sc["sdf"]).to(dev)
        g = max(4, self.H // 4)
        self.alpha = torch.nn.Parameter(torch.zeros(K1, 1, g, g, device=dev))  # outline offset, px
        self.depth = torch.from_numpy(sc["depth"]).to(dev)
        self.color0 = torch.from_numpy(sc["color"]).to(dev)
        self.res = FourierImage(self.H, self.H, batch=K1, device=dev)
        with torch.no_grad():
            self.res.spectrum.zero_()
        self.centre = torch.from_numpy(sc["centre"]).to(dev)
        self.ck = torch.from_numpy(sc["ck"]).to(dev)
        self.cq = torch.from_numpy(sc["cq"]).to(dev)
        self.valid = (self.ck > 0).float()
        self.bg = still_bg.to(dev)  # (3, S*S) colours in [0, 1]
        z = lambda d: torch.zeros(K1, d)
        self.paths = {
            "d": Path(z(2), Kh, dev),  # drift, px (y, x)
            "r": Path(z(1), Kh, dev),  # rotation, rad
            "s": Path(z(1), Kh, dev),  # breathing, strain
            "m2": Path(z(2), Kh, dev),  # radial wobble cos 2t, sin 2t, strain
            "m3": Path(z(2), Kh, dev),
        }
        for p in self.paths.values():
            p.c0.requires_grad_(False)
        self.Kh = Kh
        w = sc["wall"]
        self.wi = torch.from_numpy(w[:, 0].astype(np.int64)).to(dev)
        self.wj = torch.from_numpy(w[:, 1].astype(np.int64)).to(dev)
        self.wyx = torch.from_numpy(w[:, 2:].astype(np.float32)).to(dev)
        self.wc = torch.from_numpy(sc["wall_c"]).to(dev)
        self.wn = torch.from_numpy(sc["wall_n"]).to(dev)
        self.origin = torch.from_numpy(sc["origin"].astype(np.float32)).to(dev)
        self.S = int(round(math.sqrt(self.ck.shape[0])))

    def motion_params(self):
        return [p.cc for p in self.paths.values()]

    def at(self, u):
        """Per-sprite motion coefficients at phases u (B,): dict of (B, K+1, D)."""
        B = u.shape[0]
        uu = u[:, None].expand(B, self.K + 1)
        out = {k: p(uu) for k, p in self.paths.items()}
        for k in out:
            out[k] = out[k] * (torch.arange(self.K + 1, device=self.dev) > 0).float()[None, :, None]
        return out

    def disp(self, mo, k, q):
        """Displacement (B, ..., 2) of sprite-local points q (..., 2) of sprites k (...)."""
        rel = q - self.centre[k]  # (..., 2) (y, x)
        th = torch.atan2(rel[..., 0], rel[..., 1])
        perp = torch.stack([rel[..., 1], -rel[..., 0]], -1)
        g = lambda name: mo[name][:, k]  # (B, ..., D)
        radial = (
            g("s")[..., 0]
            + g("m2")[..., 0] * torch.cos(2 * th) + g("m2")[..., 1] * torch.sin(2 * th)
            + g("m3")[..., 0] * torch.cos(3 * th) + g("m3")[..., 1] * torch.sin(3 * th)
        )
        return g("d") + radial[..., None] * rel + g("r")[..., 0][..., None] * perp

    def coverage(self):
        """Alpha as drawn: a soft step (width --edge-w) across the smoothed outline, moved by a coarse,
        smooth learned offset, so training can reshape an outline but not make it jagged again.
        --aa adds a fixed gaussian on top."""
        off = torch.nn.functional.interpolate(self.alpha, size=(self.H, self.H), mode="bicubic", align_corners=False)
        a = torch.sigmoid(-(self.sdf0[:, None] + off) / self.soft)
        if self.aa_k is None:
            return a
        r = self.aa_k.shape[-1] // 2
        a = torch.nn.functional.conv2d(a, self.aa_k.view(1, 1, 1, -1), padding=(0, r))
        return torch.nn.functional.conv2d(a, self.aa_k.view(1, 1, -1, 1), padding=(r, 0))

    def sprites(self):
        col = self.color0 + self.res.preact()
        return torch.cat([self.coverage(), self.depth[:, None], col], 1)  # (K+1, 5, H, H)

    def render(self, u, still=False):
        """u (B,) -> frames (B, 3, S, S) in [0, 1]."""
        B = u.shape[0]
        spr = self.sprites()
        if still:
            src = self.cq.expand(B, *self.cq.shape)
        else:
            mo = self.at(u)
            src = self.cq[None] - self.disp(mo, self.ck, self.cq)
        v = gather_bilinear(spr, self.ck.expand(B, *self.ck.shape), src)  # (B, P, M, 5)
        a = v[..., 0] * self.valid
        z = torch.where(self.valid > 0, v[..., 1], torch.full_like(v[..., 1], -9.0))
        c = torch.sigmoid(v[..., 2:])
        order = torch.argsort(z, -1, descending=True)
        a = torch.gather(a, -1, order)
        c = torch.gather(c, -2, order[..., None].expand_as(c))
        T = torch.cumprod(torch.cat([torch.ones_like(a[..., :1]), 1 - a], -1), -1)
        out = (T[..., :-1, None] * a[..., None] * c).sum(-2) + T[..., -1:] * self.bg.T[None]
        return out.permute(0, 2, 1).reshape(B, 3, self.S, self.S)

    def wall_terms(self, mo, floor):
        """At every wall sample: coupling-weighted squared displacement mismatch (pairs that are one
        object move as one; a small floor keeps the field together), and contact: how far the two
        sides push into each other along the wall normal, for pairs that are separate objects."""
        qi = torch.remainder(self.wyx - self.origin[self.wi], self.S)
        qj = torch.remainder(self.wyx - self.origin[self.wj], self.S)
        di = self.disp(mo, self.wi, qi)
        dj = self.disp(mo, self.wj, qj)
        rel = di - dj
        c = self.wc.clamp(min=floor)
        cpl = (c * (rel**2).sum(-1)).mean()
        push = torch.relu((rel * self.wn).sum(-1))  # i moving into j
        con = ((1 - self.wc) * push**2).mean()
        return cpl, con

    def rms(self, name, n=32):
        u = torch.arange(n, device=self.dev) / n
        x = self.at(u)[name][:, 1:]
        return (x**2).sum(-1).mean().sqrt()

    @torch.no_grad()
    def cap(self, caps, n=64):
        u = torch.arange(n, device=self.dev) / n
        for name, amp in caps.items():
            p = self.paths[name]
            x = p(u[:, None].expand(n, self.K + 1))
            mx = (x**2).sum(-1).max(0).values.sqrt()
            p.cc.mul_((amp / mx.clamp(min=1e-9)).clamp(max=1.0).view(-1, 1, 1, 1))

    @torch.no_grad()
    def hold(self, mins, n=64):
        """Hard amplitude floor: raise each sprite's path so its rms over the loop is at least min."""
        u = torch.arange(n, device=self.dev) / n
        for name, amp in mins.items():
            p = self.paths[name]
            x = p(u[:, None].expand(n, self.K + 1))
            r = (x**2).sum(-1).mean(0).sqrt()
            p.cc.mul_((amp / r.clamp(min=1e-9)).clamp(min=1.0).view(-1, 1, 1, 1))

    @torch.no_grad()
    def init(self, tgts, seed):
        g = torch.Generator().manual_seed(seed)
        for name, tgt in tgts.items():
            p = self.paths[name]
            cc = torch.randn(p.cc.shape, generator=g) / math.sqrt(self.Kh)
            cc *= tgt * (0.6 ** torch.arange(self.Kh).view(1, -1, 1, 1))
            p.cc.data.copy_(cc.to(self.dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--labels", default="labels_display.npy")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--pin", default="38-55,73-74", help="pairs moved as one object")
    ap.add_argument("--soft", default="64,65,75,77:0.5", help="group:coupling for ambiguous objects")
    ap.add_argument("--floor", type=float, default=0.05, help="coupling floor on every wall")
    ap.add_argument("--ext", type=float, default=14, help="amodal extension under front neighbours, px")
    ap.add_argument("--reach", type=float, default=16, help="how far a sprite can move, px")
    ap.add_argument("--smooth", type=float, default=2.5, help="outline smoothing: gaussian on the signed distance, px")
    ap.add_argument("--edge-w", type=float, default=0.5, help="edge width of the alpha step across the outline, px")
    ap.add_argument("--aa", type=float, default=0.0, help="extra fixed gaussian on drawn alpha, px (0 off)")
    ap.add_argument("--fill", default="mirror", choices=["mirror", "blur"],
                    help="hidden-part colour: visible texture mirrored across the smoothed outline, or a blurred inpaint")
    ap.add_argument("--inset", type=float, default=1.5, help="mirror line this far inside the outline, px")
    ap.add_argument("--delta", type=float, default=0.3, help="depth bump at a wall")
    ap.add_argument("--wall-sigma", type=float, default=6.0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--K", type=int, default=3, help="harmonics per path")
    ap.add_argument("--tgt", default="d:6,r:0.03,s:0.03,m2:0.04,m3:0.03")
    ap.add_argument("--cap", default="d:10,r:0.06,s:0.06,m2:0.08,m3:0.06")
    ap.add_argument("--hold", default="", help="hard rms floor per sprite, e.g. d:5 (applied before the cap)")
    ap.add_argument("--wamp", type=float, default=0.5)
    ap.add_argument("--wcpl", type=float, default=0.2)
    ap.add_argument("--wcon", type=float, default=0.5, help="contact: separate objects pushing into each other")
    ap.add_argument("--wopq", type=float, default=0.5)
    ap.add_argument("--wfid", type=float, default=2.0)
    ap.add_argument("--seam-fid", type=float, default=0.0, help="fidelity weight on the edge band mirror_fill replaced")
    ap.add_argument("--lr", type=float, default=0.01, help="motion paths")
    ap.add_argument("--lr-res", type=float, default=0.01, help="colour residual spectrum")
    ap.add_argument("--lr-alpha", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rest-only", action="store_true", help="build the scene, score the rest pose and stop")
    ap.add_argument("--no-train", action="store_true", help="render the random init only")
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)

    L = np.load(f"{a.seg}/{a.labels}").astype(np.int64)
    S = L.shape[0]
    img = np.asarray(Image.open(a.image).convert("RGB")).astype(np.float32) / 255
    p = np.clip(img, 1 / 512, 1 - 1 / 512)
    logits = np.log(p / (1 - p)).transpose(2, 0, 1).astype(np.float32)
    dj = json.load(open(f"{a.seg}/depth.json"))
    votes = [(v["i"], v["j"], v["vote"]) for v in dj["votes"]]
    aff = {(q["i"], q["j"]): q["affinity"] for q in json.load(open(f"{a.seg}/graph.json"))["pairs"]}
    pins = {tuple(sorted(map(int, s.split("-")))) for s in a.pin.split(",") if s}
    soft = {}
    for grp in a.soft.split(";"):
        if grp:
            ids, c = grp.split(":")
            for i in ids.split(","):
                soft[int(i)] = float(c)
    sc = build_scene(L, logits, votes, aff, pins, soft, a)
    log(f"{sc['K']} sprites of {sc['H']} px, up to {sc['M']} per pixel, {len(sc['wall'])} wall samples")

    bgl = np.stack([ndi.gaussian_filter(logits[c], 2, mode="wrap") for c in range(3)]) - 1.0
    bg = torch.from_numpy(1 / (1 + np.exp(-bgl))).reshape(3, -1)
    fld = Field(sc, logits, dev, a.K, bg, a.aa, a.edge_w)
    still = torch.from_numpy(img.transpose(2, 0, 1))[None].to(dev)
    fid_w = torch.from_numpy(np.where(sc["seam"], a.seam_fid, 1.0).astype(np.float32))[None, None].to(dev)
    log(f"{int(sc['seam'].sum())} seam pixels at fidelity weight {a.seam_fid}")

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
        return tot / 4

    with torch.no_grad():
        alone = float(score(still))
        rest = fld.render(torch.zeros(1, device=dev), still=True)
        rest_act = float(score(rest))
        err = (rest - still).abs()
    Image.fromarray((rest[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(f"{a.out}/rest.png")
    log(f"still {alone:.3f}; rest pose {rest_act:.3f}; |rest - still| mean {float(err.mean()):.4f} p99 {float(err.quantile(0.99)):.3f}")
    if a.rest_only:
        json.dump(dict(still=alone, rest=rest_act, err_mean=float(err.mean())), open(f"{a.out}/rest.json", "w"))
        return

    parse = lambda s: {k: float(v) for k, v in (x.split(":") for x in s.split(","))}
    tgts, caps = parse(a.tgt), parse(a.cap)
    holds = parse(a.hold) if a.hold else {}
    fld.init(tgts, a.seed)
    fld.cap(caps)
    T = a.frames

    def evaluate(write, stride=1):
        acts = []
        ts = list(range(0, T, stride))
        with torch.no_grad():
            for i in range(0, len(ts), a.mb):
                u = torch.tensor(ts[i : i + a.mb], device=dev, dtype=torch.float32) / T
                fr = fld.render(u)
                acts.append(score(fr).cpu())
                if write:
                    for t, f in zip(ts[i : i + a.mb], fr.cpu()):
                        Image.fromarray((f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(f"{a.out}/f{t:04d}.png")
        return torch.cat(acts).numpy()

    act0 = evaluate(False, max(1, T // 32))
    log(f"random init: act {act0.mean():.3f} [{act0.min():.3f}, {act0.max():.3f}]")
    opt = torch.optim.Adam([
        {"params": fld.motion_params(), "lr": a.lr},
        {"params": [fld.res.spectrum], "lr": a.lr_res},
        {"params": [fld.alpha], "lr": a.lr_alpha},
    ])
    base_lr = [g["lr"] for g in opt.param_groups]
    steps = 0 if a.no_train else a.steps
    curve = []
    for it in range(steps):
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * 0.5 * (1 + math.cos(math.pi * it / steps))
        opt.zero_grad()
        u = torch.tensor(random.sample(range(T), a.mb), device=dev, dtype=torch.float32) / T
        fr = fld.render(u)
        act = rg.act(MultiScaleCrop(224 / S, 512)(fr))
        mo = fld.at(u)
        amp = sum((fld.rms(k) / v - 1) ** 2 for k, v in tgts.items()) / len(tgts)
        cpl, con = fld.wall_terms(mo, a.floor)
        cpl, con = cpl / tgts["d"] ** 2, con / tgts["d"] ** 2
        opq = (fld.alpha[1:] ** 2).mean() / 4.0  # outline offsets stay small (px^2 / 2 px^2)
        rest = fld.render(torch.zeros(1, device=dev), still=True)
        fid = (((rest - still) ** 2) * fid_w).mean() / 0.01
        pen = a.wamp * amp + a.wcpl * cpl + a.wcon * con + a.wopq * opq + a.wfid * fid
        loss = (-act + act.detach().abs() * pen).sum()
        loss.backward()
        opt.step()
        if holds:
            fld.hold(holds)
        fld.cap(caps)
        if it % 50 == 0 or it == steps - 1:
            curve.append((it, float(act.mean())))
            log(f"step {it + 1}/{steps} act {act.mean():.3f} amp {float(amp):.3f} cpl {float(cpl):.3f} con {float(con):.3f} opq {float(opq):.3f} fid {float(fid):.3f} "
                + " ".join(f"{k} {float(fld.rms(k)):.3f}" for k in tgts))

    act = evaluate(True)
    with torch.no_grad():
        rest = fld.render(torch.zeros(1, device=dev), still=True)
        rest_after = float(score(rest))
    torch.save({"paths": {k: p.cc.detach().cpu() for k, p in fld.paths.items()},
                "alpha": fld.alpha.detach().cpu(), "res": fld.res.spectrum.detach().cpu(), "args": vars(a)},
               f"{a.out}/field.pt")
    rep = dict(act_mean=float(act.mean()), act_min=float(act.min()), act_max=float(act.max()), act=act.tolist(),
               act_init=float(act0.mean()), still=alone, rest_before=rest_act, rest_after=rest_after,
               frames=T, steps=steps, curve=curve, args=vars(a), secs=round(time.time() - t0),
               rms={k: float(fld.rms(k)) for k in tgts})
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(f"done act {act.mean():.3f} [{act.min():.3f}, {act.max():.3f}] (init {act0.mean():.3f}; still {alone:.3f}; rest after {rest_after:.3f})")


if __name__ == "__main__":
    main()
