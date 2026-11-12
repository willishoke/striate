"""The dough (milestone 1): the inception4c:7 still as one continuous surface over the torus.

objtube/objphys modelled the shells as separate swept tubes; their unwrap has poles, and cutting the
picture into bodies caused months of outline trouble. objprobe/objprobe2 found that ch7 has no notion
of an object, only of a new hole: any part of the surface may slide against any other at small cost,
but opening a crevice-coloured gap through continuous surface costs ~0.48. So there are no bodies here,
one surface over the 512 px tile (periodic in x and y), "bubbly pizza dough":

  lobes     one per seg8b object, inflated as in objsdf.py: h solves lap h = -4 inside, the top is
            --kt * sqrt(h) above a common mid-plane (z = 0), the underside --kb * sqrt(h) below it. A
            lobe holds its object's pixels and the -1 pixels nearest it; pixels of unknown owner (-1
            blobs wider than the 1-2 px boundary jitter, and the band seg8b leaves along the tile's
            seam, where the still itself is cut by a straight line) join every object nearly as near,
            so lobes overlap there. 0 stays crevice. Each lobe is an SDF on its own grid (--grid) in its
            own rest frame (a window on the torus + a pose, identity here; milestone 2 moves the pose,
            not the grid) plus a bounded learned residual (objsdf's; its fine per-voxel part is off by
            default: trained, it carves the stipple into the clay).
  sheet     a low periodic height field z = --zb + a bounded learned residual: the valley floor.
  union     the lobes nearest a point (up to 3, ascending) are folded by a quadratic smooth min whose
            radius k(x, y) is a field: near the shared edge of objects i, j it is --k-lo + (--k-hi -
            --k-lo) * g_ij, so strongly grouped neighbours fuse with a big fillet and weak ones keep a
            crease; through pixels of unknown owner it is --k-unknown. Compact support (|d_i - d_j| <
            k), so a lobe's reach ends inside its window. The result fillets into the sheet with
            --k-sheet. Every ray ends on the surface.
  grouping  g_ij from the cached SAM2 masks (segsam.load_cands): of the views eligible to see objects i
            and j whole (segsam.select's rule on their joint box), the share in which one mask covers
            >= --cover of both. groups.json, g.png; --g-override {"i-j": g} (label ids) wins.
  paint     one field in rest space, no ownership: a planar 1 px colour layer over the torus (the still
            with our own shading divided out + a Fourier residual) plus a coarse 3D grid (--c3) for
            sides and undersides, sampled where the surface sits at rest.
  render    each step the union is baked (no grad) onto a 1 px periodic grid; rays march it, the first
            crossing is refined, then the hit and its normal are re-evaluated on the exact union with
            gradients (objsdf's implicit step). Light as objtube's: key with soft shadow, order-1 SH
            ambient, sky occlusion over 12 directions, specular, exposure, S6's learned rig as the
            reference. Shadows and sky march the baked field toward the light, so they tile with it.

Training: ch7 on the top view and a tilted view (--tilt-lo..--tilt deg off vertical, any azimuth) each
step, each with the key light turned at random (--train-light), objtube's decor term on the paint
(regions: the seg8b objects at the hit's rest position) and its dark floor, objsdf's eikonal /
smoothness / volume terms on the geometry residual, and a no-new-hole term: points inside the objects
must stay solid at the mid-plane, so a top may sink but never be cut through to the valley floor.

usage: objdough.py --out DIR [--steps N] [--size 256 --smoke]
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, math, random, subprocess, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import sparse
from striate.model import Backbone
from striate.param import FourierImage
from striate.util import pick_device
from zoo import MultiScaleCrop
from ring import Ring
from objsdf import Bodies, inflate, crop_q, FFMPEG, LUMA
from objtube import Rig, with_dir, decor_loss, sky_dirs, rot_x, rot_z
import segsam

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
BIG = 1e4


# ----------------------------------------------------------------------------- torus helpers (numpy)

def torus_nearest(mask):
    """Distance to, and indices of, the nearest True pixel on the torus."""
    S = mask.shape[0]
    p = S // 2
    d, (iy, ix) = ndi.distance_transform_edt(~np.pad(mask, p, mode="wrap"), return_indices=True)
    c = (slice(p, p + S),) * 2
    return d[c], (iy[c] - p) % S, (ix[c] - p) % S


def circ_span(v, n):
    """Smallest circular interval [s, s + L) of 0..n-1 holding all of v: (s, L)."""
    occ = np.zeros(n, bool)
    occ[np.asarray(v) % n] = True
    p = np.nonzero(occ)[0]
    nxt = np.roll(p, -1)
    nxt[-1] += n
    gap = nxt - p - 1
    i = int(gap.argmax())
    return int(nxt[i] % n), int(n - gap[i])


# ----------------------------------------------------------------------------- owners (numpy)

def uncertain(lab, still, a):
    """Pixels whose owner is unknown: -1 blobs that survive an opening by a disc of --unk-open px (the
    1-2 px -1 jitter along most boundaries is just boundary noise), and the 0 / -1 pixels within
    --seam-band px of the tile's edges that are not crevice-dark in the still (seg8b leaves a band of
    0 / -1 along the seam, where the still itself is cut by a straight line, but no darker than a shell)."""
    S = lab.shape[0]
    r = int(math.ceil(a.unk_open))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    p = 2 * r
    u = ndi.binary_opening(np.pad(lab == -1, p, mode="wrap"), structure=xx**2 + yy**2 <= a.unk_open**2)[p:-p, p:-p]
    if a.seam_band > 0:
        r = np.arange(S)
        e = np.minimum(r, S - 1 - r) < a.seam_band
        lum = still.astype(np.float32) @ np.array(LUMA, np.float32)
        u |= (e[:, None] | e[None, :]) & (lab <= 0) & (lum >= a.seam_dark)
    return u


def owners(lab, u):
    """The partition: lab with every -1 pixel and every pixel of unknown owner given to its nearest
    object (0 stays crevice)."""
    _, iy, ix = torus_nearest(lab > 0)
    return np.where(u | (lab == -1), lab[iy, ix], lab)


# ----------------------------------------------------------------------------- grouping (numpy)

def neighbours(lab, gap):
    """lab (no -1) grown over the 0 pixels within `gap` px of an object (to the nearest one), and the
    4-adjacent pixel pairs where the grown map changes object: {(i, j): (ys, xs)}, both sides."""
    S = lab.shape[0]
    d, iy, ix = torus_nearest(lab > 0)
    gl = np.where(lab > 0, lab, np.where(d <= gap, lab[iy, ix], 0))
    K1 = int(lab.max()) + 1
    keys, ys_, xs_ = [], [], []
    for ax in (0, 1):
        nb = np.roll(gl, -1, ax)
        ys, xs = np.nonzero((gl > 0) & (nb > 0) & (gl != nb))
        a_, b_ = gl[ys, xs], nb[ys, xs]
        k = np.minimum(a_, b_) * K1 + np.maximum(a_, b_)
        y2, x2 = (ys + (ax == 0)) % S, (xs + (ax == 1)) % S
        keys += [k, k]; ys_ += [ys, y2]; xs_ += [xs, x2]
    keys, ys_, xs_ = map(np.concatenate, (keys, ys_, xs_))
    o = np.argsort(keys, kind="stable")
    u, st = np.unique(keys[o], return_index=True)
    edges = {}
    for q, a0, a1 in zip(u, st, list(st[1:]) + [len(o)]):
        edges[(int(q // K1), int(q % K1))] = (ys_[o[a0:a1]], xs_[o[a0:a1]])
    return gl, edges


def grouping(lab, ll, u, lab_hr, cpath, a, log):
    """g_ij per neighbour pair: of the SAM2 views eligible to see i and j whole (their joint box fits
    in some torus copy of the view, and they span two of SAM2's point spacings there; segsam.select's
    rule), the share in which one mask covers >= --cover of each (and, with --mask-cap c, is at most c
    times their joint area). Neighbours from the partition ll; boxes and cover from the objects' own
    pixels (lab, lab_hr)."""
    z, pix, per = segsam.load_cands(cpath)
    S, P, R = int(z["S"]), int(z["P"]), int(z["R"])
    V, view = z["views"], z["view"]
    K = int(lab.max())
    gl, edges = neighbours(ll, a.gap)
    ind = np.concatenate(pix)
    ptr = np.cumsum([0] + [len(p) for p in pix])
    Mm = sparse.csr_matrix((np.ones(len(ind), np.float32), ind, ptr), shape=(len(pix), R * R))
    lf = lab_hr.ravel()
    ok = np.nonzero(lf > 0)[0]
    Lm = sparse.csr_matrix((np.ones(len(ok), np.float32), (ok, lf[ok])), shape=(R * R, K + 1))
    cnt = (Mm @ Lm).toarray()
    area = np.bincount(lf[ok], minlength=K + 1).astype(np.float64)
    frac = cnt / np.maximum(area, 1)[None]
    marea = np.diff(ptr).astype(np.float64)
    spacing = segsam.VIEW / 32
    out = []
    for (i, j), (ys_e, _) in sorted(edges.items()):
        if len(ys_e) < 2 * a.min_edge:
            continue
        ys, xs = np.nonzero((lab == i) | (lab == j))
        by, bh = circ_span(ys, S)
        bx, bw = circ_span(xs, S)
        elig = []
        for vi, (_, l, y0, x0, c) in enumerate(V):
            vy, vx, m = y0 - P, x0 - P, c / segsam.VIEW
            if len(ys) * (segsam.VIEW / c) ** 2 < np.pi * spacing**2:
                continue
            fy = any(vy + m <= by + q * S and by + q * S + bh <= vy + c - m for q in range(-2, 3))
            fx = any(vx + m <= bx + q * S and bx + q * S + bw <= vx + c - m for q in range(-2, 3))
            if fy and fx:
                elig.append(vi)
        both = (frac[:, i] >= a.cover) & (frac[:, j] >= a.cover)
        if a.mask_cap > 0:
            both &= marea <= a.mask_cap * (area[i] + area[j])
        det = np.intersect1d(np.unique(view[both]), elig)
        out.append(dict(i=i, j=j, page=[i - 1, j - 1], edge=int(len(ys_e) // 2), edge_uncertain=float(u[ys_e, edges[(i, j)][1]].mean()),
                        eligible=len(elig), detected=len(det), g=len(det) / len(elig) if elig else 0.0, source="sam2"))
    if a.g_override and os.path.exists(a.g_override):
        ov = json.load(open(a.g_override))
        for p in out:
            for key in (f"{p['i']}-{p['j']}", f"{p['j']}-{p['i']}"):
                if key in ov:
                    p["g"], p["source"] = float(ov[key]), "override"
        log(f"g overrides from {a.g_override}: {sum(p['source'] == 'override' for p in out)} pairs")
    g = np.array([p["g"] for p in out])
    log(f"grouping: {len(out)} neighbour pairs, g mean {g.mean():.2f}, >= 0.5: {(g >= 0.5).sum()}, 0: {(g == 0).sum()}")
    return gl, edges, out


def k_field(S, edges, pairs, u, a):
    """The union's blend radius over the torus: each pixel takes the k of the pair whose shared edge is
    nearest, smoothed by --k-sigma px. Edge pixels of unknown owner (u) take --k-unknown: nothing says
    the surface parts there, so it doesn't."""
    kv = np.full((S, S), np.nan, np.float32)
    for p in sorted(pairs, key=lambda p: p["g"]):  # where two edges meet, the stronger pair's k
        ys, xs = edges[(p["i"], p["j"])]
        kv[ys, xs] = np.where(u[ys, xs], max(a.k_unknown, a.k_lo + (a.k_hi - a.k_lo) * p["g"]), a.k_lo + (a.k_hi - a.k_lo) * p["g"])
    has = ~np.isnan(kv)
    _, iy, ix = torus_nearest(has)
    k = kv[iy, ix] if has.any() else np.full((S, S), a.k_lo, np.float32)
    return ndi.gaussian_filter(k, a.k_sigma, mode="wrap").astype(np.float32)


def draw_groups(still, edges, pairs, u, path):
    """The still (dimmed) at 2x with every shared edge coloured by g (viridis), edges through pixels of
    unknown owner (filleted at --k-unknown whatever g says) in light grey, and a colour bar."""
    vir = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], np.float32) / 255
    cm = lambda v: [np.interp(v, np.linspace(0, 1, 5), vir[:, c]) for c in range(3)]
    S = still.shape[0]
    im = np.kron(still.astype(np.float32) * 0.45, np.ones((2, 2, 1)))
    for p in pairs:
        ys, xs = edges[(p["i"], p["j"])]
        c = np.where(u[ys, xs][:, None], np.float32(200), np.array(cm(p["g"])[:3], np.float32)[None] * 255)
        for dy in (0, 1):
            for dx in (0, 1):
                im[2 * ys + dy, 2 * xs + dx] = c
    bar = np.zeros((44, 2 * S, 3), np.float32) + 20
    ramp = np.array([cm(v)[:3] for v in np.linspace(0, 1, 2 * S - 40)], np.float32) * 255
    bar[6:22, 20:2 * S - 20] = ramp[None]
    out = Image.fromarray(np.concatenate([im, bar]).clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    d.text((20, 2 * S + 26), "g = 0: crease (SAM2 never takes the pair as one)", fill=(230, 230, 230))
    d.text((2 * S - 300, 2 * S + 26), "g = 1: fused (every eligible view does)", fill=(230, 230, 230))
    d.text((2 * S // 2 - 90, 2 * S + 26), "grey: unknown owner, fused", fill=(200, 200, 200))
    out.save(path)


# ----------------------------------------------------------------------------- lobes (numpy)

def lobe_base(m2, a):
    """objsdf's base SDF of one inflated outline, on --grid over the window m2 (no taper, no stacking):
    the body is -kb sqrt(h) <= z <= kt sqrt(h) about the mid-plane z = 0. Returns sdf, lo, hi."""
    Gz, Gy, Gx = a.grid
    Hb, Wb = m2.shape
    h = inflate(m2)
    dout, (iy_, ix_) = ndi.distance_transform_edt(~m2, return_indices=True)
    gy_, gx_ = np.gradient(ndi.gaussian_filter(h, 1.0))
    slope = np.hypot(gy_, gx_)[iy_, ix_]
    hext = ndi.gaussian_filter(np.where(m2, h, -(dout - 0.5).clip(min=0) * slope - 0.5 * slope), 1.0)
    T = np.sqrt(hext.clip(min=0))
    pad = a.res_amp + 2
    lo = np.array([0.0, 0.0, -a.kb * T.max() - pad])
    hi = np.array([float(Wb), float(Hb), a.kt * T.max() + pad])
    n2 = (2 * Gx, 2 * Gy, 2 * Gz)
    ax = [np.linspace(lo[d], hi[d], n2[d]) for d in range(3)]
    Tg = ndi.map_coordinates(T, np.meshgrid(ax[1] - 0.5, ax[0] - 0.5, indexing="ij"), order=1, mode="constant")
    zz = ax[2][:, None, None]
    occ = (Tg[None] > 0.05) & (zz <= a.kt * Tg[None]) & (zz >= -a.kb * Tg[None])
    sp = [(hi[d] - lo[d]) / (n2[d] - 1) for d in (2, 1, 0)]
    sd = ndi.distance_transform_edt(~occ, sampling=sp) - ndi.distance_transform_edt(occ, sampling=sp)
    sd = sd + 0.5 * min(sp) * np.where(occ, 1, -1)
    gpos = [np.linspace(0, n2[d] - 1, (Gx, Gy, Gz)[d]) for d in (2, 1, 0)]
    far = ndi.map_coordinates(sd, np.meshgrid(*gpos, indexing="ij"), order=1)
    # near the surface the smooth implicit (z / c(z))^2 - h, as objsdf
    gx1 = [np.linspace(lo[d], hi[d], (Gx, Gy, Gz)[d]) for d in range(3)]
    hg = ndi.map_coordinates(hext, np.meshgrid(gx1[1] - 0.5, gx1[0] - 0.5, indexing="ij"), order=3, mode="nearest")[None]
    zg = gx1[2][:, None, None]
    c = a.kb + (a.kt - a.kb) / (1 + np.exp(-zg / 1.5))
    G = (zg / c) ** 2 - hg
    dz, dy, dx = np.gradient(G, *[(hi[d] - lo[d]) / ((Gx, Gy, Gz)[d] - 1) for d in (2, 1, 0)])
    near = G / np.sqrt(dz**2 + dy**2 + dx**2 + 1e-6)
    vmax = float(((hi - lo) / (np.array([Gx, Gy, Gz]) - 1)).max())
    w = np.clip(1.5 - np.abs(far) / (2 * vmax), 0, 1)
    return (w * near + (1 - w) * far).astype(np.float32), lo, hi


def build_lobes(lab, ll, u, a, log):
    """One lobe per object: its pixels and every pixel of unknown owner (u) within --share px of being as
    near to it as to the nearest object (so lobes overlap where ownership is unknown, as -1 means), a
    window on the torus padded by --win-pad, and its base SDF. Cached in OUT/lobes.npz by the
    arguments it depends on."""
    key = json.dumps([a.kt, a.kb, list(a.grid), a.res_amp, a.win_pad, a.seg, a.share, a.seam_band, a.seam_dark, a.unk_open])
    path = f"{a.out}/lobes.npz"
    if os.path.exists(path):
        z = np.load(path)
        if str(z["key"]) == key:
            log(f"lobes from {path}")
            return {k: z[k] for k in z.files}
    S = lab.shape[0]
    dn, _, _ = torus_nearest(lab > 0)
    K = int(lab.max())
    Gz, Gy, Gx = a.grid
    base = np.full((K, Gz, Gy, Gx), 50.0, np.float32)
    lo, hi = np.zeros((K, 3), np.float32), np.ones((K, 3), np.float32)
    win = np.zeros((K, 4), np.int64)
    p = a.win_pad
    t0 = time.time()
    for k in range(1, K + 1):
        if (lab == k).sum() < 4:
            continue
        m = (ll == k) | (u & (torus_nearest(lab == k)[0] <= dn + a.share))
        if m.sum() < 4:
            continue
        ys, xs = np.nonzero(m)
        sy, ly = circ_span(ys, S)
        sx, lx = circ_span(xs, S)
        oy, ox, Hb, Wb = sy - p, sx - p, ly + 2 * p, lx + 2 * p
        m2 = m[np.ix_((oy + np.arange(Hb)) % S, (ox + np.arange(Wb)) % S)]
        base[k - 1], lo[k - 1], hi[k - 1] = lobe_base(m2, a)
        win[k - 1] = (oy, ox, Hb, Wb)
    log(f"lobes: {K} base SDFs in {time.time() - t0:.0f}s; windows median {np.median(win[:, 2]):.0f} x {np.median(win[:, 3]):.0f} px")
    out = dict(base=base, lo=lo, hi=hi, win=win, key=np.array(key))
    np.savez_compressed(path, **out)
    return out


# ----------------------------------------------------------------------------- camera

class OCam:
    """Orthographic camera, objtube's convention: the rows of R are the screen's right, down and
    toward-the-viewer directions in the world; W x W pixels of s world px, centred on c."""

    def __init__(self, R, W, s, c):
        self.R, self.W, self.H, self.s, self.c = np.asarray(R, np.float64), W, W, s, np.asarray(c, np.float64)

    @staticmethod
    def top(S, W):
        return OCam(np.eye(3), W, S / W, (S / 2, S / 2, 0.0))

    @staticmethod
    def tilt(S, W, deg, az):
        return OCam(rot_x(math.radians(deg)) @ rot_z(az), W, S / W, (S / 2, S / 2, 0.0))

    def rays(self, dev):
        R = torch.tensor(self.R, dtype=torch.float32, device=dev)
        py, px = torch.meshgrid(torch.arange(self.H, device=dev), torch.arange(self.W, device=dev), indexing="ij")
        u = ((px.reshape(-1).float() + 0.5) - self.W / 2) * self.s
        v = ((py.reshape(-1).float() + 0.5) - self.H / 2) * self.s
        o = torch.tensor(self.c, dtype=torch.float32, device=dev) + u[:, None] * R[0] + v[:, None] * R[1]
        return o, -R[2]


def smin(a, b, k):
    """Quadratic smooth min, compact support |a - b| < k, at most k / 4 below min(a, b)."""
    h = (k - (a - b).abs()).clamp(min=0) / k
    return torch.minimum(a, b) - h * h * k * 0.25


# ----------------------------------------------------------------------------- the dough

class Dough:
    # objsdf's per-body grids: residual, trilinear sampling (exact with grad / fast via grid_sample),
    # and the eikonal / smoothness / volume regularizers, on the lobes' grids
    grid = Bodies.grid
    sdf = Bodies.sdf
    sdf_fast = Bodies.sdf_fast
    _interp = Bodies._interp
    regs = Bodies.regs

    def __init__(self, lb, kmap, own, img, a, dev, log):
        self.a, self.dev = a, dev
        S = self.S = kmap.shape[0]
        K = self.K = lb["base"].shape[0]
        Gz, Gy, Gx = a.grid
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
        self.base = t(lb["base"])[:, None]
        self.lo, self.hi = t(lb["lo"]), t(lb["hi"])
        self.vox = (self.hi - self.lo) / torch.tensor([Gx - 1, Gy - 1, Gz - 1], device=dev)
        win = lb["win"]
        # rest frame of lobe k: local (0, 0, 0) is world (ox, oy, 0); pose (trans, rot about the window
        # centre) is identity at rest, and the paint is addressed in rest space, so milestone 2 moves it
        self.anc = t(np.stack([win[:, 1], win[:, 0], np.zeros(K)], 1))
        self.ctr = t(np.stack([win[:, 3] / 2, win[:, 2] / 2, np.zeros(K)], 1))
        self.pose = None
        g4 = (max(2, Gz // 4), max(2, Gy // 4), max(2, Gx // 4))
        self.rc = torch.nn.Parameter(torch.zeros(K, 1, *g4, device=dev))
        self.rf = torch.nn.Parameter(torch.zeros(K, 1, Gz, Gy, Gx, device=dev))
        self.U = [self._interp(g4[d], (Gz, Gy, Gx)[d]) for d in range(3)]
        self.vol0 = None
        # the baked lattice: 1 px columns x --bake-dz levels over the slab the surface can occupy
        self.zlo = a.zb - a.sheet_amp - 2.0
        self.zhi = float(lb["hi"][:, 2].max()) + 1.0
        self.Nz = int(math.ceil((self.zhi - self.zlo) / a.bake_dz)) + 1
        self.dz = (self.zhi - self.zlo) / (self.Nz - 1)
        bk, bc, bxy = [], [], []
        for k in range(K):
            oy, ox, Hb, Wb = win[k]
            if Hb == 0:
                continue
            rr, cc = np.meshgrid(np.arange(Hb), np.arange(Wb), indexing="ij")
            bk.append(np.full(rr.size, k))
            bc.append((((oy + rr) % S) * S + (ox + cc) % S).ravel())
            bxy.append(np.stack([ox + cc + 0.5, oy + rr + 0.5], -1).reshape(-1, 2))
        self.bk, self.bc, self.bxy = t(np.concatenate(bk), torch.long), t(np.concatenate(bc), torch.long), t(np.concatenate(bxy))
        cy, cx = np.meshgrid(np.arange(S) + 0.5, np.arange(S) + 0.5, indexing="ij")
        self.cxy = t(np.stack([cx.ravel(), cy.ravel()], 1))
        # candidate lobes per --cell square: those whose window overlaps it (padded to a fixed width)
        C = a.cell
        nC = S // C
        cl = [[] for _ in range(nC * nC)]
        for k in range(K):
            oy, ox, Hb, Wb = win[k]
            if Hb == 0:
                continue
            ys = np.unique(((oy + np.arange(Hb)) % S) // C)
            xs = np.unique(((ox + np.arange(Wb)) % S) // C)
            for y in ys:
                for x in xs:
                    cl[y * nC + x].append(k)
        M = max(len(c) for c in cl)
        M = 4 * ((M + 3) // 4)
        self.cells = t([c + [-1] * (M - len(c)) for c in cl], torch.long)
        self.kmap = t(kmap)
        self.kflat = self.kmap.reshape(-1)
        self.sq = torch.nn.Parameter(torch.zeros(a.sheet_grid, a.sheet_grid, device=dev))
        # paint: planar 1 px layer + coarse 3D grid, logits
        lg = lambda im: np.log(np.clip(im, 1 / 512, 1 - 1 / 512) / (1 - np.clip(im, 1 / 512, 1 - 1 / 512)))
        self.tex0 = t(lg(img).transpose(2, 0, 1))
        self.tex = FourierImage(S, S, device=dev)
        Zc, Yc, Xc = a.c3
        self.col3 = torch.nn.Parameter(torch.zeros(3, Zc, Yc, Xc, device=dev))
        with torch.no_grad():
            self.tex.spectrum.zero_()
        self.own = t(own, torch.long)
        self.core = None  # (n, 2) pixel centres inside the objects, for hole_loss (setup sets it)
        log(f"dough: {K} lobes, grid {Gz}x{Gy}x{Gx}, bake {self.Nz} levels z {self.zlo:.1f}..{self.zhi:.1f} "
            f"({len(self.bk) * self.Nz / 1e6:.1f} M lobe samples), {M} candidates per {C} px cell, k {kmap.min():.2f}..{kmap.max():.2f}")

    def params(self):
        return [self.rc], [self.rf], [self.sq], [self.tex.spectrum], [self.col3]

    def hole_loss(self, n, gen):
        """No new holes: n random pixels inside the objects (eroded by 1 px) must stay solid at the
        mid-plane (z = --hole-z), so a top may sink but never be cut through to the valley floor."""
        i = torch.randint(0, len(self.core), (n,), device=self.dev, generator=gen)
        x = torch.cat([self.core[i], torch.full((n, 1), self.a.hole_z, device=self.dev)], 1)
        kq, vq = self.candidates(x)
        f, _ = self.union(x, kq, vq)
        return (torch.relu(f + self.a.hole_margin) ** 2).mean(), float((f > 0).float().mean())

    def state(self):
        return dict(rc=self.rc.detach().cpu(), rf=self.rf.detach().cpu(), sq=self.sq.detach().cpu(), tex=self.tex.spectrum.detach().cpu(),
                    col3=self.col3.detach().cpu(), tex0=self.tex0.cpu())

    def load(self, st):
        with torch.no_grad():
            for n_ in ("rc", "rf", "sq", "col3"):
                getattr(self, n_).copy_(st[n_])
            self.tex.spectrum.copy_(st["tex"])
            self.tex0 = st["tex0"].to(self.dev)

    # ---- per step
    def prep(self, clay=False):
        """Grids (with grad), paint tensors, and (no grad) the bake."""
        Gz, Gy, Gx = self.a.grid
        g = self.grid()
        self.flat = g.reshape(-1)
        self.atlas = g.detach().reshape(1, 1, self.K * Gz, Gy, Gx)
        self.clay = clay
        if not clay:
            self.texl = (self.tex0[None] + self.tex.preact())[0].permute(1, 2, 0).reshape(-1, 3)
            self.c3f = self.col3.permute(1, 2, 3, 0).reshape(-1, 3)
        return g

    def to_local(self, k, x):
        """World points (n, 3) -> lobe k's rest frame, through the torus copy nearest its window."""
        d = x - self.anc[k]
        c = self.ctr[k]
        dxy = d[:, :2] - self.S * torch.round((d[:, :2] - c[:, :2]) / self.S)
        xl = torch.cat([dxy, d[:, 2:]], 1)
        if self.pose is not None:  # milestone 2: (trans (K, 3), rot (K, 3, 3)) about the window centre
            tr, Rm = self.pose
            xl = torch.einsum("nba,nb->na", Rm[k], xl - c - tr[k]) + c
        return xl

    def sheet(self, xy):
        """The valley floor's height at world xy (n, 2): --zb + --sheet-amp tanh(bilinear periodic grid)."""
        G = self.sq.shape[0]
        fx, fy = xy[:, 0] / self.S * G - 0.5, xy[:, 1] / self.S * G - 0.5
        x0, y0 = fx.floor(), fy.floor()
        wx, wy = fx - x0, fy - y0
        x0, y0 = x0.long() % G, y0.long() % G
        x1, y1 = (x0 + 1) % G, (y0 + 1) % G
        q = self.sq.reshape(-1)
        v = (q[y0 * G + x0] * (1 - wx) + q[y0 * G + x1] * wx) * (1 - wy) + (q[y1 * G + x0] * (1 - wx) + q[y1 * G + x1] * wx) * wy
        return self.a.zb + self.a.sheet_amp * torch.tanh(v)

    @torch.no_grad()
    def bake(self):
        """The union on the 1 px periodic lattice: per level, every lobe sampled over its window, the three
        smallest values per column (three scatter-mins), folded as union() does, then the sheet."""
        S2 = self.S * self.S
        k = self.kflat
        hb = self.sheet(self.cxy)
        out = torch.empty(self.Nz, S2, device=self.dev)
        for l in range(self.Nz):
            z = self.zlo + l * self.dz
            xw = torch.cat([self.bxy, torch.full_like(self.bxy[:, :1], z)], 1)
            d = self.sdf_fast(self.bk, self.to_local(self.bk, xw))
            m1 = torch.full((S2,), BIG, device=self.dev).scatter_reduce(0, self.bc, d, "amin")
            d2 = torch.where(d > m1[self.bc], d, torch.full_like(d, BIG))
            m2 = torch.full((S2,), BIG, device=self.dev).scatter_reduce(0, self.bc, d2, "amin")
            d3 = torch.where(d > m2[self.bc], d, torch.full_like(d, BIG))
            m3 = torch.full((S2,), BIG, device=self.dev).scatter_reduce(0, self.bc, d3, "amin")
            u = smin(smin(m1, m2, k), m3, k)
            out[l] = smin(u, z - hb, self.a.k_sheet)
        g = out.view(1, self.Nz, self.S, self.S)
        self.baked = F.pad(g, (1, 1, 1, 1), mode="circular")[None]  # (1, 1, Nz, S + 2, S + 2)

    def field_fast(self, x):
        """No-grad union at world points (n, 3) from the bake (periodic in x, y; clamped in z)."""
        S = self.S
        gx = ((x[:, 0] - 0.5) % S + 1) / (S + 1) * 2 - 1
        gy = ((x[:, 1] - 0.5) % S + 1) / (S + 1) * 2 - 1
        gz = (x[:, 2] - self.zlo) / (self.zhi - self.zlo) * 2 - 1
        v = F.grid_sample(self.baked, torch.stack([gx, gy, gz], -1).view(1, 1, 1, -1, 3), mode="bilinear", padding_mode="border", align_corners=True)
        return v.view(-1)

    @torch.no_grad()
    def candidates(self, x):
        """The (up to) Q lobes nearest each point (n, 3), ascending: (n, Q) ids and validity."""
        C, Q = self.a.cell, self.a.q
        nC = self.S // C
        cx = ((x[:, 0] % self.S) // C).long().clamp(max=nC - 1)
        cy = ((x[:, 1] % self.S) // C).long().clamp(max=nC - 1)
        cand = self.cells[cy * nC + cx]
        M = cand.shape[1]
        ok = cand >= 0
        kc = cand.clamp(min=0)
        xr = x[:, None].expand(-1, M, 3).reshape(-1, 3)
        dd = self.sdf_fast(kc.reshape(-1), self.to_local(kc.reshape(-1), xr)).view(-1, M)
        dd = torch.where(ok, dd, torch.full_like(dd, BIG))
        dq, iq = dd.topk(Q, dim=1, largest=False)
        return kc.gather(1, iq), ok.gather(1, iq) & (dq < BIG / 2)

    def union(self, x, kq, vq):
        """Exact union (with grad) at world points (n, 3) over their candidate lobes (n, Q), as bake():
        the lobes folded ascending with radius k(x, y), then the sheet. Also the sheet's share."""
        n, Q = kq.shape
        xr = x[:, None].expand(-1, Q, 3).reshape(-1, 3)
        d = self.sdf(kq.reshape(-1), self.to_local(kq.reshape(-1), xr)).view(n, Q)
        d = torch.where(vq, d, torch.full_like(d, BIG))
        d = torch.sort(d, 1).values
        S = self.S
        fx, fy = (x[:, 0].detach() - 0.5) % S, (x[:, 1].detach() - 0.5) % S
        x0, y0 = fx.floor(), fy.floor()
        wx, wy = fx - x0, fy - y0
        x0, y0 = x0.long() % S, y0.long() % S
        x1, y1 = (x0 + 1) % S, (y0 + 1) % S
        kf = self.kflat
        k = (kf[y0 * S + x0] * (1 - wx) + kf[y0 * S + x1] * wx) * (1 - wy) + (kf[y1 * S + x0] * (1 - wx) + kf[y1 * S + x1] * wx) * wy
        u = d[:, 0]
        for q in range(1, Q):
            u = smin(u, d[:, q], k)
        db = x[:, 2] - self.sheet(x[:, :2])
        f = smin(u, db, self.a.k_sheet)
        wb = torch.sigmoid((u - db).detach() / 0.5)  # 1 where the sheet is the nearer surface
        return f, wb

    def rest(self, x, kq, vq):
        """Where world points (n, 3) sit at rest, for the paint: with no pose, where they are; with one,
        each candidate lobe's rest-frame position, blended by nearness (softmin of its SDF over 1 px; the
        sheet, which does not move, takes part with its own distance)."""
        if self.pose is None:
            return x
        n, Q = kq.shape
        kf = kq.reshape(-1)
        xr = x[:, None].expand(-1, Q, 3).reshape(-1, 3)
        xl = self.to_local(kf, xr)
        pose, self.pose = self.pose, None
        xl0 = self.to_local(kf, xr)  # the same torus copy, unposed
        self.pose = pose
        d = torch.where(vq, self.sdf(kf, xl).view(n, Q), torch.full((n, Q), BIG, device=self.dev))
        d = torch.cat([d, (x[:, 2] - self.sheet(x[:, :2]))[:, None]], 1)
        w = torch.softmax(-d / 1.0, 1)
        delta = torch.cat([(xl - xl0).view(n, Q, 3), torch.zeros(n, 1, 3, device=self.dev)], 1)
        return x + (w[..., None] * delta).sum(1)

    @torch.no_grad()
    def march(self, o, d):
        """First crossing along each ray through the slab (--samples coarse, --fine refining)."""
        N, N2 = self.a.samples, self.a.fine
        dz = float(d[2])
        t_in, t_out = (self.zhi - o[:, 2]) / dz, (self.zlo - o[:, 2]) / dz
        js = (torch.arange(N, device=self.dev).float() + 0.5) / N
        ts = t_in[:, None] + js[None] * (t_out - t_in)[:, None]
        f = self.field_fast((o[:, None] + ts[..., None] * d).reshape(-1, 3)).view(-1, N)
        neg = f < 0
        hit = neg.any(1)
        j = torch.where(hit, neg.float().argmax(1), torch.full_like(hit, N - 1, dtype=torch.long))
        tb = ts.gather(1, j[:, None])[:, 0]
        ta = torch.where(j > 0, ts.gather(1, (j - 1).clamp(min=0)[:, None])[:, 0], t_in)
        t2 = ta[:, None] + torch.linspace(0, 1, N2, device=self.dev)[None] * (tb - ta)[:, None]
        f2 = self.field_fast((o[:, None] + t2[..., None] * d).reshape(-1, 3)).view(-1, N2)
        neg2 = f2 < 0
        k = torch.where(neg2.any(1), neg2.float().argmax(1), torch.full_like(j, N2 - 1)).clamp(min=1)
        a0, a1 = f2.gather(1, (k - 1)[:, None])[:, 0], f2.gather(1, k[:, None])[:, 0]
        b0, b1 = t2.gather(1, (k - 1)[:, None])[:, 0], t2.gather(1, k[:, None])[:, 0]
        th = b0 + (b1 - b0) * a0 / (a0 - a1).clamp(min=1e-6)
        return torch.where(hit, th, t_out), hit

    @torch.no_grad()
    def visibility(self, p, n, l, pen0, pen, steps):
        """Soft visibility of direction l (3,) from surface points p (n, 3): the baked union sampled toward
        l (geometric spacing, to the top of the slab), each sample sigmoid(F / (pen0 + pen * s)), the min."""
        lz = max(float(l[2]), 0.05)
        smax = ((self.zhi - p[:, 2]) / lz).clamp(min=3.0, max=150.0)
        s0 = 1.0
        fr = torch.arange(steps, device=self.dev).float() / (steps - 1)
        s = s0 * (smax[:, None] / s0) ** fr[None]
        q = p[:, None] + n[:, None] * self.a.normal_offset + s[..., None] * l
        f = self.field_fast(q.reshape(-1, 3)).view(-1, steps)
        return torch.sigmoid(f / (pen0 + pen * s)).min(1).values

    def paint(self, x):
        """Albedo (n, 3) at rest-space points: planar layer (bilinear, periodic) + coarse 3D grid."""
        if self.clay:
            return torch.full((len(x), 3), self.a.clay_grey, device=self.dev)
        S = self.S
        fx, fy = x[:, 0] - 0.5, x[:, 1] - 0.5
        x0, y0 = fx.floor(), fy.floor()
        wx, wy = (fx - x0)[:, None], (fy - y0)[:, None]
        x0, y0 = x0.long() % S, y0.long() % S
        x1, y1 = (x0 + 1) % S, (y0 + 1) % S
        T = self.texl
        t = (T[y0 * S + x0] * (1 - wx) + T[y0 * S + x1] * wx) * (1 - wy) + (T[y1 * S + x0] * (1 - wx) + T[y1 * S + x1] * wx) * wy
        Zc, Yc, Xc = self.a.c3
        qx, qy = x[:, 0] / S * Xc - 0.5, x[:, 1] / S * Yc - 0.5
        qz = ((x[:, 2] - self.zlo) / (self.zhi - self.zlo) * (Zc - 1)).clamp(0, Zc - 1.001)
        ix, iy, iz = qx.floor(), qy.floor(), qz.floor()
        wx3, wy3, wz3 = (qx - ix)[:, None], (qy - iy)[:, None], (qz - iz)[:, None]
        ix, iy, iz = ix.long() % Xc, iy.long() % Yc, iz.long()
        jx, jy = (ix + 1) % Xc, (iy + 1) % Yc
        g = lambda zz, yy, xx: self.c3f[(zz * Yc + yy) * Xc + xx]
        c3 = (((g(iz, iy, ix) * (1 - wx3) + g(iz, iy, jx) * wx3) * (1 - wy3) + (g(iz, jy, ix) * (1 - wx3) + g(iz, jy, jx) * wx3) * wy3) * (1 - wz3)
              + ((g(iz + 1, iy, ix) * (1 - wx3) + g(iz + 1, iy, jx) * wx3) * (1 - wy3) + (g(iz + 1, jy, ix) * (1 - wx3) + g(iz + 1, jy, jx) * wx3) * wy3) * wz3)
        return torch.sigmoid(t + c3)

    def shade(self, alb, n, d, rg, vis, sky):
        """objtube's light: col = alb * mult + add (key with shadow, SH ambient, sky occlusion, specular)."""
        a = self.a
        l = rg["l"]
        ndl = torch.relu(n @ l)
        hv = l - d
        hv = hv / hv.norm()
        ndh = torch.relu(n @ hv)
        sh = rg["sh"] * a.amb_scale
        amb = (sh[0][None] + n[:, 0:1] * sh[1][None] + n[:, 1:2] * sh[2][None] + n[:, 2:3] * sh[3][None]).clamp(min=0)
        a0 = sh[0].mean()
        expo = (a0 + rg["dif"]) / (a0 + rg["dif"] * l[2])
        tint = rg["tint"][None]
        ao = (1 - a.sky * (1 - sky))[:, None]
        mult = (amb * ao + rg["dif"] * tint * (ndl * vis)[:, None] * (1 - a.ao_dif * (1 - ao))) * expo
        add = rg["spec"] * tint * (ndh ** rg["shin"] * vis * (ndl > 0))[:, None] * ao
        return alb * mult + add, mult, add

    def render_rays(self, o, d, rg, aux):
        a = self.a
        n_ = len(o)
        t0, hit = self.march(o, d)
        x0 = o + t0[:, None] * d
        kq, vq = self.candidates(x0)
        h = a.fd_h
        E = torch.eye(3, device=self.dev) * h
        X = torch.cat([x0] + [x0 + s_ * E[i] for i in range(3) for s_ in (1, -1)])
        f7, wb = self.union(X, kq.repeat(7, 1), vq.repeat(7, 1))
        f7 = f7.view(7, n_)
        g = torch.stack([f7[1] - f7[2], f7[3] - f7[4], f7[5] - f7[6]], -1) / (2 * h)
        nrm = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        gd = (g.detach() @ d).clamp(max=-0.05)
        t = t0 - f7[0] / gd  # implicit hit: moves with the field
        x = o + t[:, None] * d
        nrm = torch.where((nrm.detach() @ d)[:, None] > 0, -nrm, nrm)
        alb = self.paint(self.rest(x, kq, vq))
        with torch.no_grad():
            xd, nd = x.detach(), nrm.detach()
            vis = self.visibility(xd, nd, rg["l"], rg["pen0"], rg["pen"], a.shadow_steps)
            if a.sky > 0:
                sv, sw = 0, 0
                for dr in self.skyd:
                    wgt = torch.relu(nd @ dr)
                    sv = sv + wgt * self.visibility(xd, nd, dr, a.sky_pen0, a.sky_pen, a.sky_steps)
                    sw = sw + wgt
                sky = sv / sw.clamp(min=1e-3)
            else:
                sky = torch.ones_like(vis)
        col, mult, add = self.shade(alb, nrm, d, rg, vis, sky)
        out = dict(col=col)
        if aux:
            S = self.S
            xi = (xd[:, 0].floor().long() % S)
            yi = (xd[:, 1].floor().long() % S)
            out.update(alb=alb, n=nrm, sky=sky, vis=vis, mult=mult.detach(), add=add.detach(), hit=hit, wb=wb[:n_].detach(),
                       region=self.own[yi, xi], xw=xd, ndl=torch.relu(nd @ rg["l"]))
        return out

    def render(self, cam, rg, clay=False, aux=False, chunk=65536):
        """(3, H, W) frame (and per-pixel aux). prep() and bake() must have run."""
        self.clay = clay
        o, d = cam.rays(self.dev)
        self.skyd = sky_dirs(self.dev)
        P = len(o)
        ch = min(chunk, P)
        parts = [self.render_rays(o[s:s + ch], d, rg, aux) for s in range(0, P, ch)]
        res = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        img = res["col"].T.reshape(3, cam.H, cam.W).clamp(0, 1)
        if aux:
            res["img"] = img
            res["d"] = d
            return res
        return img


# ----------------------------------------------------------------------------- main

def decor(o, rg, K, chroma):
    """objtube.decor_loss on a dough frame: one sample per pixel, regions = objects at the rest position."""
    P = len(o["col"])
    one = torch.ones(1, P, device=o["col"].device)
    q = dict(pix=torch.arange(P, device=one.device), obj=o["region"], ep=one, den=one, alb=o["alb"][None], nk=o["n"][None],
             lvec=rg["l"].detach()[None], gown=torch.zeros_like(one), sky=o["sky"][None])
    return decor_loss(q, K, chroma=chroma)


def video(frames_dir, out, fps=24, crf=18):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{frames_dir}/f%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), out], check=True)


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=f"{ROOT}/outputs/sheet-4c-2026-10-31/googlenet_inception4c_7.png")
    ap.add_argument("--seg", default=f"{ROOT}/outputs/obj3d-2026-11-01/seg8b")
    ap.add_argument("--rig", default=f"{ROOT}/outputs/obj3d-2026-11-01/phys/edge/S6/tube.pt", help="checkpoint whose learned light rig is the reference light")
    ap.add_argument("--target", default="inception4c:7")
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--out", required=True)
    # grouping
    ap.add_argument("--gap", type=int, default=4, help="max 0 / -1 band between neighbours, px (objprobe's)")
    ap.add_argument("--min-edge", type=int, default=3)
    ap.add_argument("--cover", type=float, default=0.5, help="a mask groups i and j when it covers this share of each")
    ap.add_argument("--mask-cap", type=float, default=0.0, help="... and is at most this times their joint area (0: any size)")
    ap.add_argument("--g-override", default="", help='JSON {"i-j": g} by label id (page id + 1)')
    ap.add_argument("--seam-band", type=int, default=6, help="0 pixels this close to the tile's edges are of unknown owner unless crevice-dark")
    ap.add_argument("--seam-dark", type=float, default=48.0, help="crevice-dark: still luminance (0-255) below this (p75 of seg8b's 0 pixels)")
    ap.add_argument("--unk-open", type=float, default=2.0, help="-1 pixels are of unknown owner where they survive an opening by a disc this wide, px")
    ap.add_argument("--share", type=float, default=6.0, help="an unknown-owner pixel joins every object within this many px of its nearest")
    ap.add_argument("--k-unknown", type=float, default=10.0, help="union radius on shared edges through unknown-owner pixels, px")
    # geometry
    ap.add_argument("--grid", default="32,72,72", help="SDF grid per lobe, z,y,x")
    ap.add_argument("--kt", type=float, default=0.7)
    ap.add_argument("--kb", type=float, default=0.4)
    ap.add_argument("--res-amp", type=float, default=4.0, help="largest SDF change the network may make, px")
    ap.add_argument("--k-lo", type=float, default=1.0, help="union radius for g = 0 (a crease), px")
    ap.add_argument("--k-hi", type=float, default=10.0, help="union radius for g = 1 (a fillet), px")
    ap.add_argument("--k-sigma", type=float, default=3.0)
    ap.add_argument("--k-sheet", type=float, default=3.0, help="fillet of every lobe into the sheet, px")
    ap.add_argument("--zb", type=float, default=-3.0, help="the sheet's height (lobes' mid-plane is 0)")
    ap.add_argument("--sheet-amp", type=float, default=2.0)
    ap.add_argument("--sheet-grid", type=int, default=32)
    ap.add_argument("--win-pad", type=int, default=0, help="px around each lobe's pixels (default k-hi + res-amp + 2)")
    ap.add_argument("--bake-dz", type=float, default=1.0)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--q", type=int, default=3, help="lobes folded per point")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--fine", type=int, default=12)
    ap.add_argument("--fd-h", type=float, default=1.0, help="normals: central differences this wide, px")
    # paint
    ap.add_argument("--c3", default="8,64,64", help="coarse 3D colour grid, z,y,x")
    ap.add_argument("--deshade-lo", type=float, default=0.5)
    ap.add_argument("--clay-grey", type=float, default=0.72)
    # light (objtube / S6 values)
    ap.add_argument("--sky", type=float, default=1.0)
    ap.add_argument("--sky-pen0", type=float, default=1.5)
    ap.add_argument("--sky-pen", type=float, default=0.15)
    ap.add_argument("--sky-steps", type=int, default=10)
    ap.add_argument("--shadow-steps", type=int, default=24)
    ap.add_argument("--normal-offset", type=float, default=1.0)
    ap.add_argument("--ao-dif", type=float, default=1.0)
    ap.add_argument("--amb-scale", type=float, default=0.7)
    # training
    ap.add_argument("--size", type=int, default=512, help="training / scoring frame size (the tile is 512 world px)")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--train-light", type=float, default=60.0, help="key light turned up to this many deg in azimuth, dropped up to half of it")
    ap.add_argument("--tilt-lo", type=float, default=20.0)
    ap.add_argument("--tilt", type=float, default=45.0)
    ap.add_argument("--lr-geo", type=float, default=0.02)
    ap.add_argument("--lr-fine", type=float, default=0.0, help="the residual's fine (per-voxel) part: 0 keeps geometry to forms (it carves the stipple otherwise)")
    ap.add_argument("--lr-sheet", type=float, default=0.02)
    ap.add_argument("--lr-tex", type=float, default=0.01)
    ap.add_argument("--lr-c3", type=float, default=0.02)
    ap.add_argument("--wdecor", type=float, default=4.0)
    ap.add_argument("--decor-chroma", type=float, default=0.25)
    ap.add_argument("--wdark", type=float, default=1.0)
    ap.add_argument("--dark-floor", type=float, default=0.15)
    ap.add_argument("--weik", type=float, default=0.2)
    ap.add_argument("--wlap", type=float, default=2.0)
    ap.add_argument("--wvol", type=float, default=1.0)
    ap.add_argument("--vol-tol", type=float, default=0.25)
    ap.add_argument("--whole", type=float, default=10.0, help="no-new-hole term (hole_loss)")
    ap.add_argument("--hole-n", type=int, default=16384)
    ap.add_argument("--hole-z", type=float, default=0.0)
    ap.add_argument("--hole-margin", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", help="state .pt to start from")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--evals", default="all", help="comma list: scores,renders,video or all / none")
    ap.add_argument("--video-frames", type=int, default=72)
    ap.add_argument("--smoke", action="store_true", help="time a render and a step, write a few frames, stop")
    a = ap.parse_args(argv)
    a.grid = tuple(int(v) for v in a.grid.split(","))
    a.c3 = tuple(int(v) for v in a.c3.split(","))
    a.win_pad = a.win_pad or int(math.ceil(a.k_hi + a.res_amp + 2))
    return a


def save(x, p):
    Image.fromarray((x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(p)


def setup(a, dev, log):
    """Grouping (groups.json, g.png), k field, lobes, the reference light, the dough with its paint
    deshaded (or loaded from --load). Returns the dough, the reference rig and its az / el, the still
    and the labels."""
    os.makedirs(a.out, exist_ok=True)
    still_np = np.asarray(Image.open(a.image).convert("RGB"))
    img = still_np.astype(np.float32) / 255
    S = img.shape[0]
    z = np.load(f"{a.seg}/labels.npz")
    lab, lab_hr = z["labels"].astype(np.int64), z["labels_hr"].astype(np.int64)

    # ---- grouping, k field, lobes (CPU)
    u = uncertain(lab, still_np, a)
    ll = owners(lab, u)
    log(f"owners: {int((lab == -1).sum())} px -1 given to the nearest object; unknown owner (lobes overlap, edges fuse): {int(u.sum())} px")
    gl, edges, pairs = grouping(lab, ll, u, lab_hr, f"{a.seg}/cands.npz", a, log)
    json.dump(dict(rule=f"g = share of eligible SAM2 views with one mask covering >= {a.cover} of both objects"
                        + (f", mask <= {a.mask_cap} x their area" if a.mask_cap else ""),
                   ids="i, j: label ids in seg8b/labels.npz (page ids = label - 1)", pairs=pairs),
              open(f"{a.out}/groups.json", "w"), indent=1)
    draw_groups(still_np, edges, pairs, u, f"{a.out}/g.png")
    kmap = k_field(S, edges, pairs, u, a)
    lb = build_lobes(lab, ll, u, a, log)
    _, iy, ix = torus_nearest(ll > 0)
    own = ll[iy, ix]  # every pixel's nearest object: decor's regions

    # ---- reference light (S6's learned rig)
    ck = torch.load(a.rig, map_location="cpu", weights_only=False)
    rig = Rig(0.0, 0.8, argparse.Namespace(pen0=0.6, pen=0.06), dev)
    rig.load_state_dict(ck["rig"])
    del ck
    with torch.no_grad():
        ref = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in rig().items()}
    ref_az = math.atan2(float(ref["l"][1]), float(ref["l"][0]))
    ref_el = math.asin(float(ref["l"][2]))
    log(f"reference light: az {math.degrees(ref_az) % 360:.0f} el {math.degrees(ref_el):.0f} (S6 rig)")

    dg = Dough(lb, kmap, own, img, a, dev, log)
    p = 2
    core = ndi.binary_erosion(np.pad(lab > 0, p, mode="wrap"))[p:-p, p:-p]
    ys, xs = np.nonzero(core)
    dg.core = torch.tensor(np.stack([xs + 0.5, ys + 0.5], 1), dtype=torch.float32, device=dev)
    # ---- deshade: the still over our own clay shading at the reference light (albedo 1)
    with torch.no_grad():
        dg.prep(clay=True)
        dg.bake()
        cg = a.clay_grey
        a.clay_grey = 1.0
        t1 = time.time()
        o = dg.render(OCam.top(S, S), ref, clay=True, aux=True)
        if str(dev) == "mps":
            torch.mps.synchronize()
        log(f"clay render at {S}: {time.time() - t1:.2f}s; coverage {float(o['hit'].float().mean()):.5f}")
        a.clay_grey = cg
        still_t = torch.from_numpy(img.reshape(-1, 3)).to(dev)
        alb = ((still_t - o["add"]) / o["mult"].clamp(min=a.deshade_lo)).clamp(1 / 512, 1 - 1 / 512)
        if not a.load:
            dg.tex0 = torch.logit(alb).T.reshape(3, S, S).contiguous()
        sh = o["mult"].mean(1).reshape(S, S)
        save((sh / sh.max())[None].expand(3, -1, -1), f"{a.out}/shade0.png")
        log(f"deshaded: our shading at init {float(sh.min()):.2f}..{float(sh.max()):.2f}, mean {float(sh.mean()):.2f}")
        del o
    if a.load:
        dg.load(torch.load(a.load, map_location="cpu"))
        log(f"loaded {a.load}")
    return dg, ref, ref_az, ref_el, img, lab


def main():
    a = build_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    t00 = time.time()
    LOG = []

    def log(s):
        s = f"[{time.time() - t00:.0f}s] {s}"
        print(s, flush=True)
        LOG.append(s)

    dev = pick_device()
    dg, ref, ref_az, ref_el, img, lab = setup(a, dev, log)
    S = dg.S
    W = a.size
    top = OCam.top(S, W)
    tilts = [OCam.tilt(S, W, 40.0, az) for az in (0.3, 2.4, 4.5)]  # objeval's tilt
    rep = dict(args=vars(a), still={}, init={}, trained={})
    core1 = ndi.binary_erosion(np.pad(lab > 0, 2, mode="wrap"))[2:-2, 2:-2]
    gj = json.load(open(f"{a.out}/groups.json"))["pairs"]
    rep["groups"] = dict(pairs=len(gj), g_mean=float(np.mean([p["g"] for p in gj])), g_ge_half=int(sum(p["g"] >= 0.5 for p in gj)),
                         g_zero=int(sum(p["g"] == 0 for p in gj)), k_min=float(dg.kmap.min()), k_max=float(dg.kmap.max()),
                         pink59_coil71=[p for p in gj if (p["i"], p["j"]) == (60, 72)])

    layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
    net = Backbone(a.net, dev)
    net.hook(layer)
    rg_ = Ring(net, layer, ch, S, dev)

    def score(fr):
        st = random.getstate()
        tot = 0
        for k in range(4):
            random.seed(1000 + k)
            tot = tot + rg_.act(MultiScaleCrop(224 / 512, 512)(fr[None]))
        random.setstate(st)
        return float(tot / 4)

    def frame(cam, light, clay=False, ss=2):
        """A frame of cam's size, rendered at ss x ss and box-filtered."""
        c2 = OCam(cam.R, cam.W * ss, cam.s / ss, cam.c)
        return F.avg_pool2d(dg.render(c2, light, clay=clay)[None], ss)[0]

    def evaluate(tag):
        """Scores (2x2 supersampled, objeval's), coverage, the sheet's share, decor; frames to OUT."""
        r = {}
        with torch.no_grad():
            dg.prep()
            dg.bake()
            fr = frame(top, ref)
            r["rest"] = score(fr)
            save(fr, f"{a.out}/{tag}_rest.png")
            r["rest_1spp"] = score(dg.render(top, ref))
            ts = []
            for i, c in enumerate(tilts):
                ft = frame(c, ref)
                ts.append(score(ft))
                save(ft, f"{a.out}/{tag}_tilt{i}.png")
            r["tilt"] = float(np.mean(ts))
            r["tilt_each"] = ts
            # coverage and the valley floor, and decor (objdecor's four views), at 1 spp
            cov, dc = {}, []
            for nm, c in [("top", OCam.top(S, S))] + [(f"tilt{i}", OCam(c.R, S, 1.0, c.c)) for i, c in enumerate(tilts)]:
                o = dg.render(c, ref, aux=True)
                cov[nm] = float(o["hit"].float().mean())
                if nm == "top":
                    lw = torch.from_numpy(lab.reshape(-1)).to(dev)
                    floor = o["wb"] > 0.5
                    r["sheet_share_top"] = float(floor.float().mean())
                    r["sheet_in_objects_top"] = float((floor & (lw > 0)).float().sum() / (lw > 0).float().sum())
                    cw = torch.from_numpy(core1.reshape(-1)).to(dev)  # objects less their outermost px: a hole, not a wider crevice
                    r["sheet_in_object_cores_top"] = float((floor & cw).float().sum() / cw.float().sum())
                    r["sheet_in_crevice_top"] = float((floor & (lw == 0)).float().sum() / (lw == 0).float().sum().clamp(min=1))
                    al = o["alb"].T.reshape(3, S, S)
                    save(al, f"{a.out}/{tag}_albedo.png")
                d_, sg = decor(o, ref, dg.K + 1, a.decor_chroma)
                dc.append(dict(term=float(d_), **sg))
                del o
            r["coverage"] = cov
            r["coverage_min"] = min(cov.values())
            r["decor"] = {k: float(np.mean([v[k] for v in dc])) for k in dc[0]}
            r["decor_top"] = dc[0]
            dg.prep(clay=True)
            save(frame(top, ref, clay=True), f"{a.out}/{tag}_clay_top.png")
            save(frame(tilts[0], ref, clay=True), f"{a.out}/{tag}_clay_tilt0.png")
            dg.prep()
        log(f"{tag}: rest {r['rest']:.3f} (1 spp {r['rest_1spp']:.3f}), tilt {r['tilt']:.3f} {[round(v, 3) for v in ts]}, "
            f"coverage min {r['coverage_min']:.5f}, sheet {r['sheet_share_top']:.3f} of top (in objects {r['sheet_in_objects_top']:.4f}, "
            f"cores {r['sheet_in_object_cores_top']:.5f}, "
            f"crevice {r['sheet_in_crevice_top']:.3f}), decor {r['decor']['term']:.4f} "
            + " ".join(f"{k} {v:+.3f}" for k, v in r["decor"].items() if k != "term"))
        return r

    with torch.no_grad():
        rep["still"]["score"] = score(torch.from_numpy(img.transpose(2, 0, 1)).to(dev))
        log(f"still {rep['still']['score']:.3f}")

    if a.smoke:
        with torch.no_grad():
            for clay in (True, False):
                dg.prep(clay=clay)
                dg.bake()
                t1 = time.time()
                fr = dg.render(top, ref, clay=clay)
                if str(dev) == "mps":
                    torch.mps.synchronize()
                log(f"render {W} {'clay' if clay else 'paint'}: {time.time() - t1:.2f}s, score {score(fr):.3f}")
                save(fr, f"{a.out}/smoke_{'clay' if clay else 'paint'}_top.png")
                save(dg.render(tilts[0], ref, clay=clay), f"{a.out}/smoke_{'clay' if clay else 'paint'}_tilt0.png")
            t1 = time.time()
            dg.bake()
            if str(dev) == "mps":
                torch.mps.synchronize()
            log(f"bake {time.time() - t1:.3f}s")

    # ---- training
    if a.steps or a.smoke:
        if not a.smoke and (a.evals == "all" or "scores" in a.evals):
            rep["init"] = evaluate("init")
            json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
        geo, fine, shp, tex, c3 = dg.params()
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        opt = torch.optim.Adam([{"params": geo, "lr": a.lr_geo}, {"params": fine, "lr": a.lr_fine}, {"params": shp, "lr": a.lr_sheet},
                                {"params": tex, "lr": a.lr_tex}, {"params": c3, "lr": a.lr_c3}])
        base_lr = [g["lr"] for g in opt.param_groups]
        rng = random.Random(a.seed + 7)
        steps = 2 if a.smoke else a.steps
        curve, tt = [], time.time()
        for it in range(steps):
            ts_ = time.time()
            for g, b in zip(opt.param_groups, base_lr):
                g["lr"] = b * 0.5 * (1 + math.cos(math.pi * it / steps))
            opt.zero_grad()
            with torch.no_grad():
                dg.prep()
                dg.bake()
            acts, pens = [], []
            for b in range(2):
                dg.prep()
                cam = top if b == 0 else OCam.tilt(S, W, rng.uniform(a.tilt_lo, a.tilt), rng.uniform(0, 2 * math.pi))
                lt = ref
                if a.train_light > 0:
                    lt = with_dir(ref, ref_az + math.radians(rng.uniform(-a.train_light, a.train_light)),
                                  max(ref_el - math.radians(rng.uniform(0, a.train_light / 2)), math.radians(15)))
                o = dg.render(cam, lt, aux=True)
                act = rg_.act(crop_q(o["img"][None]))[0]
                Y = o["alb"] @ torch.tensor(LUMA, device=dev)
                dark = (torch.relu(a.dark_floor - Y) ** 2).mean() / 0.01
                pen = a.wdark * dark
                if a.wdecor > 0:
                    dc_, _ = decor(o, lt, dg.K + 1, a.decor_chroma)
                    pen = pen + a.wdecor * dc_
                ((-act + act.detach().abs() * pen) / 2).backward()
                acts.append(float(act)); pens.append(float(pen))
                del o
            dg.prep()
            eik, lap, volp = dg.regs()
            hole, holes = dg.hole_loss(a.hole_n, gen) if a.whole > 0 else (torch.zeros((), device=dev), 0.0)
            ((abs(np.mean(acts))) * (a.weik * eik + a.wlap * lap + a.wvol * volp + a.whole * hole)).backward()
            opt.step()
            if str(dev) == "mps":
                torch.mps.empty_cache()
            if it % a.log_every == 0 or it == steps - 1 or a.smoke:
                res = dg._res.detach()
                curve.append((it, float(np.mean(acts)), acts))
                mem = f"mps {torch.mps.current_allocated_memory() / 2**30:.1f}/{torch.mps.driver_allocated_memory() / 2**30:.1f} GB" if str(dev) == "mps" else ""
                log(f"step {it + 1}/{steps} act top {acts[0]:.3f} tilt {acts[1]:.3f} pen {np.mean(pens):.3f} eik {float(eik):.3f} lap {float(lap):.4f} "
                    f"vol {float(volp):.4f} hole {float(hole):.4f} ({holes:.4f} open) | res rms {float(res.pow(2).mean().sqrt()):.2f} max {float(res.abs().max()):.2f} px "
                    f"| {time.time() - ts_:.2f}s/step, {mem}")
        rep["curve"] = curve
        rep["train_secs"] = time.time() - tt
        if not a.smoke:
            torch.save(dg.state(), f"{a.out}/state.pt")
            log(f"saved {a.out}/state.pt")
        if a.smoke:
            log("smoke done")
            return
    else:
        torch.save(dg.state(), f"{a.out}/state.pt")

    ev = set("scores,renders,video".split(",")) if a.evals == "all" else set(a.evals.split(",")) - {"none", ""}
    key = "trained" if a.steps or a.load else "init"
    if "scores" in ev:
        rep[key] = evaluate(key)
        json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    with torch.no_grad():
        if "renders" in ev:
            dg.prep()
            dg.bake()
            save(frame(top, ref), f"{a.out}/paint_rest_512.png")
            save(frame(OCam.top(S, 1024), ref), f"{a.out}/paint_rest_1024.png")
            for i, c in enumerate(tilts):
                save(frame(c, ref), f"{a.out}/paint_tilt{i}.png")
            dg.prep(clay=True)
            save(frame(top, ref, clay=True), f"{a.out}/clay_rest_512.png")
            save(frame(OCam.top(S, 1024), ref, clay=True), f"{a.out}/clay_rest_1024.png")
            for i, c in enumerate(tilts):
                save(frame(c, ref, clay=True), f"{a.out}/clay_tilt{i}.png")
            dg.prep()
            log("renders")
        if "video" in ev:
            # tilt from straight down to 45 deg, then once round at 45 deg: clay | paint
            d = f"{a.out}/orbit"
            os.makedirs(d, exist_ok=True)
            n1, n2 = a.video_frames // 3, a.video_frames
            views = [(45.0 * j / n1, 0.3) for j in range(n1)] + [(45.0, 0.3 + 2 * math.pi * j / n2) for j in range(n2)]
            for j, (deg, az) in enumerate(views):
                c = OCam.tilt(S, 512, deg, az)
                dg.prep(clay=True)
                fc = frame(c, ref, clay=True)
                dg.prep()
                fp = frame(c, ref)
                save(torch.cat([fc, fp], 2), f"{d}/f{j:04d}.png")
            video(d, f"{a.out}/orbit.mp4")
            log("orbit video")
    rep["secs"] = time.time() - t00
    rep["log"] = LOG
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
