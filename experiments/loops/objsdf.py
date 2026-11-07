"""Learned 3D bodies (gate test, option C): every shell is a free-form signed distance field.

objsurf.py lit the still's own cut-outs as height fields, and every version looked lasso-tooled:
outlines and pixels came from the segmentation. Here each shell is a closed solid in 3D whose
silhouette, shading, cast shadows and occlusion all come from the body, and the network reshapes
the body itself:

  body k    SDF on a per-object grid (--grid, over the object's box; px units) = base + bounded
            learned residual, A * tanh(coarse + fine) with A = --res-amp px. The base inflates the
            object's outline (objfield's amodal shape, smoothed): h solves lap h = -4 inside, so
            sqrt(h) is a sphere over a disc and a tube over a strip; the top is --kt * sqrt(h) above
            a mid-plane, the underside --kb * sqrt(h) below it. Bodies are stacked back to front by
            depth.json: each rests on the floor or on the tops of the bodies already placed.
  colour    a planar texture in the body's own xy at 1 px (the still, with our own shading divided
            out, mirror-filled under neighbours, + a learned Fourier residual) plus a coarse 3D
            colour grid for sides and underside.
  render    orthographic, per body window: N samples along each ray through the body's box (no
            grad), then the first sign change refined with an implicit-function gradient step, so
            the hit point moves with the SDF. Coverage = sigmoid(-min SDF along the ray / eps):
            an antialiased silhouette with gradients. Per pixel the bodies are sorted by depth and
            composited front to back over a floor plane.
  light     Lambert + Blinn specular; cast shadows from a shadow map (the same renderer seen from
            the light, PCF-softened), so undersides and drop shadows are real; ambient occlusion
            from the union SDF sampled along the normal.

Training: the channel under the loop's light, every body jittered a few px and a few degrees and
the camera tilted up to --tilt deg each step, so contact shading can't be painted into the colour
and the sides and back get some pressure. Eikonal, smoothness and volume terms keep the residual a
body rather than foam; a silhouette term holds the outline early and is annealed away.

usage: objsdf.py --image still.png --seg segdir --target inception4c:7 --out DIR
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, math, random, subprocess, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from striate.model import Backbone
from striate.param import FourierImage
from striate.util import pick_device
from zoo import MultiScaleCrop
from striate.transforms import TransformRobustness
from ring import Ring
from objfield import build_scene
from objsurf import Light

FFMPEG = "/private/tmp/claude-501/-Users-lawvere-striate/39cfca31-2c02-4b02-a696-d14df31b10bf/scratchpad/videnv/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1"
LUMA = (0.299, 0.587, 0.114)


# ----------------------------------------------------------------------------- geometry helpers

def inflate(mask):
    """h with lap h = -4 inside mask, 0 outside (unit grid): sqrt(h) is a sphere over a disc."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    n = len(ys)
    idx = -np.ones(mask.shape, np.int64)
    idx[ys, xs] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yy, xx = ys + dy, xs + dx
        inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        j = np.full(n, -1)
        j[inb] = idx[yy[inb], xx[inb]]
        ok = j >= 0
        rows.append(np.nonzero(ok)[0]); cols.append(j[ok]); vals.append(np.full(ok.sum(), -1.0))
    A = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    h = np.zeros(mask.shape)
    h[ys, xs] = spsolve(A, np.full(n, 4.0))
    return h


def rodrigues(axis, ang):
    """(..., 3) unit axes, (...) angles -> (..., 3, 3)."""
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(ang), torch.sin(ang)
    C = 1 - c
    return torch.stack([
        torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
        torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], -2)


class Cam:
    """Orthographic camera: rays travel along f; image x along r, image rows along u (r x u = -f)."""

    def __init__(self, f, r, u, c, W, H, s=1.0):
        self.f, self.r, self.u, self.c = f, r, u, c
        self.W, self.H, self.s = W, H, s

    @staticmethod
    def top(S, dev, s=1.0, zc=10.0):
        t = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)
        return Cam(t([0, 0, -1]), t([1, 0, 0]), t([0, 1, 0]), t([S / 2, S / 2, zc]), int(S / s), int(S / s), s)

    def tilted(self, pitch, yaw):
        """This camera rotated by pitch (rad) about the horizontal axis at angle yaw, about its centre."""
        ax = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=self.f.device)
        Rm = rodrigues(ax, torch.tensor(pitch, device=self.f.device))
        return Cam(Rm @ self.f, Rm @ self.r, Rm @ self.u, self.c, self.W, self.H, self.s)

    @staticmethod
    def orbit(c, el, az, W, s, dev):
        """Looking at c from elevation el and azimuth az (rad), world z up on screen."""
        f = -torch.tensor([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], device=dev)
        z = torch.tensor([0.0, 0.0, 1.0], device=dev)
        if abs(math.sin(el)) > 0.999:
            z = torch.tensor([-math.sin(az), math.cos(az), 0.0], device=dev) * (-1 if el > 0 else 1)
        u = -(z - (z @ f) * f)
        u = u / u.norm()
        r = torch.linalg.cross(u, -f)
        return Cam(f, r, u, c, W, W, s)

    @staticmethod
    def light(Lv, S, m, zmax, dev, s=1.0):
        """Shadow-map camera looking along -Lv over the field [-m, S+m]^2 x [0, zmax]."""
        f = -torch.tensor(Lv, dtype=torch.float32, device=dev)
        z = torch.tensor([0.0, 0.0, 1.0], device=dev)
        if abs(float(f[2])) > 0.999:
            z = torch.tensor([0.0, -1.0, 0.0], device=dev)
        u = -(z - (z @ f) * f)
        u = u / u.norm()
        r = torch.linalg.cross(u, -f)
        c = torch.tensor([S / 2, S / 2, zmax / 2], device=dev)
        g = torch.tensor([[x, y, zz] for x in (-m, S + m) for y in (-m, S + m) for zz in (0, zmax)], dtype=torch.float32, device=dev)
        ex = ((g - c) @ r).abs().max().item(), ((g - c) @ u).abs().max().item()
        return Cam(f, r, u, c, int(2 * ex[0] / s) + 4, int(2 * ex[1] / s) + 4, s)


def bucket(n):
    """Round a count up to one of a few sizes (2^k times 1, 1.25, 1.5, 1.75): MPS compiles a graph per
    tensor shape, and ray counts change every frame, so shapes are padded to these buckets."""
    if n <= 4096:
        return 4096
    e = 2 ** int(math.floor(math.log2(n)))
    return next(int(e * m) for m in (1, 1.25, 1.5, 1.75, 2) if e * m >= n)


def select(mask):
    """Indices of the True entries of a (bucket-sized) mask, padded to bucket(count), and the mask of
    real entries. Built by a stable sort so that no tensor ever takes the count's own size."""
    n = int(mask.sum())
    nb = min(bucket(n), len(mask))
    order = torch.argsort((~mask).to(torch.int32), stable=True)[:nb]
    return order, torch.arange(nb, device=mask.device) < n


def crop_q(x, tf=TransformRobustness()):
    """MultiScaleCrop(224/512, 512) with the output size rounded to a multiple of 32: the same views,
    from 10 sizes instead of ~290, so MPS compiles the network's graphs once each."""
    s = math.exp(random.uniform(math.log(224 / 512), 0.0))
    o = max(224, int(round(512 * s / 32)) * 32)
    return tf(F.interpolate(x, size=(o, o), mode="bilinear", antialias=True, align_corners=False))


# ----------------------------------------------------------------------------- the bodies

class Bodies:
    def __init__(self, sc, still_np, alb_np, L, z, dev, a, log):
        self.dev, self.a, self.S = dev, a, L.shape[0]
        S = self.S
        K, Hw = sc["K"], sc["H"]
        self.K, self.Hw = K, Hw
        Gz, Gy, Gx = a.grid
        ks = range(1, K + 1)
        # ---- per-object outline, inflation, taper under the shells in front, box
        lo = np.zeros((K, 3), np.float32); hi = np.zeros((K, 3), np.float32)
        T = np.zeros((K, Hw, Hw), np.float32)
        tap = np.ones((K, Hw, Hw), np.float32)
        hext = np.zeros((K, Hw, Hw), np.float32)
        vis = np.zeros((K, Hw, Hw), bool)
        self.dwin = np.zeros((K, Hw, Hw), np.float32)  # inner distance to the base outline, px
        for i, k in enumerate(ks):
            m2 = sc["sdf"][k] < 0
            oy, ox = sc["origin"][k]
            vis[i] = L[np.ix_((oy + np.arange(Hw)) % S, (ox + np.arange(Hw)) % S)] == k
            h = inflate(m2)
            # continue h outside the outline with its boundary slope, so h = 0 is a smooth curve
            dout, (iy_, ix_) = ndi.distance_transform_edt(~m2, return_indices=True)
            gy_, gx_ = np.gradient(ndi.gaussian_filter(h, 1.0))
            slope = np.hypot(gy_, gx_)[iy_, ix_]
            hx = np.where(m2, h, -(dout - 0.5).clip(min=0) * slope - 0.5 * slope)
            hext[i] = ndi.gaussian_filter(hx, 1.0)
            T[i] = np.sqrt(hext[i].clip(min=0))
            self.dwin[i] = np.maximum(-sc["sdf"][k], 0)
            # the top ducks under the shells in front: from --taper of full height at the visible
            # edge, rising to full height --taper-w px into the visible part, falling further beneath
            hid = m2 & ~vis[i]
            if hid.any():
                sgn = ndi.distance_transform_edt(~hid) - ndi.distance_transform_edt(hid)
                tap[i] = np.clip(a.taper + (1 - a.taper) * sgn / a.taper_w, 0.05, 1.0)
                tap[i] = ndi.gaussian_filter(tap[i], 1.0)
        # ---- heights, back to front: a body's visible top clears the (tapered) tops beneath it
        order = np.argsort(np.array(z)[1:])
        top = np.full((S, S), -1e3, np.float32)
        cz = np.zeros(K, np.float32)
        for i in order:
            oy, ox = sc["origin"][i + 1]
            iy, ix = (oy + np.arange(Hw)) % S, (ox + np.arange(Hw)) % S
            sub = top[np.ix_(iy, ix)]
            rest = a.kb * T[i].max() * 0.8  # underside sinks a little into the floor
            need = (sub - a.kt * T[i] * tap[i] + a.margin)[vis[i] & (sub > -1e2)]
            cz[i] = max(rest, np.quantile(need, 0.9)) if need.size else rest
            fp = T[i] > 0
            top[np.ix_(iy, ix)] = np.where(fp, np.maximum(sub, cz[i] + a.kt * T[i] * tap[i]), sub)
        self.zmax = float(top.max()) + 4
        pad = a.res_amp + 2
        for i in range(K):
            ys, xs = np.nonzero(T[i] > 0)
            lo[i] = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), -a.kb * T[i].max() - pad)
            hi[i] = (min(xs.max() + 1 + pad, Hw), min(ys.max() + 1 + pad, Hw), a.kt * (T[i] * tap[i]).max() + pad)
        # ---- base SDF on the grid: occupancy at 2x, exact distance transform, sampled at the grid
        base = np.zeros((K, Gz, Gy, Gx), np.float32)
        for i in range(K):
            n2 = (2 * Gx, 2 * Gy, 2 * Gz)
            ax = [np.linspace(lo[i, d], hi[i, d], n2[d]) for d in range(3)]
            mg = np.meshgrid(ax[1] - 0.5, ax[0] - 0.5, indexing="ij")
            Tg = ndi.map_coordinates(T[i], mg, order=1, mode="constant")
            Pg = ndi.map_coordinates(T[i] * tap[i], mg, order=1, mode="constant")
            zz = ax[2][:, None, None]
            occ = (Tg[None] > 0.05) & (zz <= a.kt * Pg[None]) & (zz >= -a.kb * Tg[None])
            sp = [(hi[i, d] - lo[i, d]) / (n2[d] - 1) for d in (2, 1, 0)]
            sd = ndi.distance_transform_edt(~occ, sampling=sp) - ndi.distance_transform_edt(occ, sampling=sp)
            sd = sd + 0.5 * min(sp) * np.where(occ, 1, -1)
            gpos = [np.linspace(0, n2[d] - 1, (Gx, Gy, Gz)[d]) for d in (2, 1, 0)]
            far = ndi.map_coordinates(sd, np.meshgrid(*gpos, indexing="ij"), order=1)
            # near the surface a smooth implicit instead of the voxel staircase: the body is
            # (z / c(z))^2 < h with c from --kb below to --kt * taper above, and G / |grad G| is
            # its distance to first order
            gx_ = [np.linspace(lo[i, d], hi[i, d], (Gx, Gy, Gz)[d]) for d in range(3)]
            mg1 = np.meshgrid(gx_[1] - 0.5, gx_[0] - 0.5, indexing="ij")
            hg = ndi.map_coordinates(hext[i], mg1, order=3, mode="nearest")[None]
            tg = ndi.map_coordinates(tap[i], mg1, order=1, mode="nearest")[None]
            zg = gx_[2][:, None, None]
            c = a.kb + (a.kt * tg - a.kb) / (1 + np.exp(-zg / 1.5))
            G = (zg / c) ** 2 - hg
            dz, dy, dx = np.gradient(G, *[(hi[i, d] - lo[i, d]) / ((Gx, Gy, Gz)[d] - 1) for d in (2, 1, 0)])
            near = G / np.sqrt(dz**2 + dy**2 + dx**2 + 1e-6)
            vmax = float(((hi[i] - lo[i]) / (np.array([Gx, Gy, Gz]) - 1)).max())
            w = np.clip(1.5 - np.abs(far) / (2 * vmax), 0, 1)
            base[i] = w * near + (1 - w) * far
        self.tap = tap
        log(f"bodies: grid {Gz}x{Gy}x{Gx}, box xy median {np.median(hi[:, 0] - lo[:, 0]):.0f} px, "
            f"heights {cz.min():.1f}..{cz.max():.1f}, top of pile {self.zmax:.1f} px")
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
        self.base = t(base)[:, None]  # (K, 1, Gz, Gy, Gx) px
        self.lo, self.hi = t(lo), t(hi)
        self.vox = (self.hi - self.lo) / torch.tensor([Gx - 1, Gy - 1, Gz - 1], device=dev)  # (K, 3) px per voxel (x, y, z)
        self.anchor0 = t(np.stack([sc["origin"][1:, 1], sc["origin"][1:, 0], cz], 1).astype(np.float32))  # world of local (0,0,0)
        self.cz = cz
        self.T = T
        # wrap duplicates: an instance for every torus copy whose box reaches into the field + margin
        inst = []
        m = a.wrap_margin
        for i in range(K):
            for dy in (-S, 0, S):
                for dx in (-S, 0, S):
                    x0, x1 = self.anchor0[i, 0].item() + lo[i, 0] + dx, self.anchor0[i, 0].item() + hi[i, 0] + dx
                    y0, y1 = self.anchor0[i, 1].item() + lo[i, 1] + dy, self.anchor0[i, 1].item() + hi[i, 1] + dy
                    if x1 > -m and x0 < S + m and y1 > -m and y0 < S + m:
                        inst.append((i, dx, dy))
        self.ik = t([q[0] for q in inst], torch.long)
        self.ioff = t([[q[1], q[2], 0] for q in inst])
        log(f"{len(inst)} instances ({len(inst) - K} wrap copies)")
        # ---- learned parameters
        g4 = (max(2, Gz // 4), max(2, Gy // 4), max(2, Gx // 4))
        self.rc = torch.nn.Parameter(torch.zeros(K, 1, *g4, device=dev))
        self.rf = torch.nn.Parameter(torch.zeros(K, 1, Gz, Gy, Gx, device=dev))
        self.U = [self._interp(g4[d], (Gz, Gy, Gx)[d]) for d in range(3)]
        lg = lambda im: np.log(np.clip(im, 1 / 512, 1 - 1 / 512) / (1 - np.clip(im, 1 / 512, 1 - 1 / 512)))
        self.tex0 = t(sc["color"][1:])  # (K, 3, Hw, Hw) logits
        self.tex = FourierImage(Hw, Hw, batch=K, device=dev)
        self.col3 = torch.nn.Parameter(torch.zeros(K, 3, 6, 10, 10, device=dev))
        fb = np.stack([ndi.gaussian_filter(lg(alb_np)[..., c], 8, mode="wrap") for c in range(3)]) - a.floor_dark
        self.floor0 = t(fb.astype(np.float32))
        self.floor = FourierImage(S, S, device=dev)
        with torch.no_grad():
            self.tex.spectrum.zero_()
            self.floor.spectrum.zero_()
        self.dwin_t = t(self.dwin)
        self.vol0 = None

    def _interp(self, n, N):
        """(N, n) linear interpolation matrix, align_corners."""
        Mx = torch.zeros(N, n)
        p = torch.linspace(0, n - 1, N)
        i0 = p.floor().clamp(max=n - 2).long()
        w = p - i0
        Mx[torch.arange(N), i0] = 1 - w
        Mx[torch.arange(N), i0 + 1] = w
        return Mx.to(self.dev)

    def params(self):
        return [self.rc, self.rf], [self.tex.spectrum, self.col3, self.floor.spectrum]

    def grid(self):
        """(K, 1, Gz, Gy, Gx) SDF in px."""
        c = torch.einsum("kczyx,Zz,Yy,Xx->kcZYX", self.rc, *self.U)
        self._res = self.a.res_amp * torch.tanh(c + self.rf)
        return self.base + self._res

    def prep(self, clay=False):
        """Per-step tensors: the SDF grid flattened, textures."""
        g = self.grid()
        self.flat = g.reshape(-1)
        Gz, Gy, Gx = self.a.grid
        self.atlas = g.detach().reshape(1, 1, self.K * Gz, Gy, Gx)
        self.clay = clay
        if not clay:
            self.texl = (self.tex0 + self.tex.preact()).permute(0, 2, 3, 1).reshape(-1, 3)  # (K*Hw*Hw, 3)
            self.c3 = self.col3.permute(0, 2, 3, 4, 1).reshape(-1, 3)
            self.flr = torch.sigmoid(self.floor0[None] + self.floor.preact())[0]  # (3, S, S)
        return g

    # ---- sampling
    def sdf(self, k, xl):
        """k (R,), xl (R, 3) local px -> (R,) SDF px; outside the box: box distance + clamped sample."""
        Gz, Gy, Gx = self.a.grid
        lo, hi, vox = self.lo[k], self.hi[k], self.vox[k]
        xc = torch.minimum(torch.maximum(xl, lo), hi)
        out = (xl - xc).norm(dim=-1)
        f = (xc - lo) / vox
        nx = torch.tensor([Gx - 1, Gy - 1, Gz - 1], device=self.dev)
        f = torch.minimum(f, nx.float())
        i0 = torch.minimum(f.floor().long(), nx - 1)
        w = f - i0
        base = k * (Gz * Gy * Gx)
        ix, iy, iz = i0.unbind(-1)
        wx, wy, wz = w.unbind(-1)
        g = lambda dz, dy, dx: self.flat[base + (iz + dz) * (Gy * Gx) + (iy + dy) * Gx + ix + dx]
        c00 = g(0, 0, 0) * (1 - wx) + g(0, 0, 1) * wx
        c01 = g(0, 1, 0) * (1 - wx) + g(0, 1, 1) * wx
        c10 = g(1, 0, 0) * (1 - wx) + g(1, 0, 1) * wx
        c11 = g(1, 1, 0) * (1 - wx) + g(1, 1, 1) * wx
        v = (c00 * (1 - wy) + c01 * wy) * (1 - wz) + (c10 * (1 - wy) + c11 * wy) * wz
        return v + out

    def sdf_fast(self, k, xl):
        """No-grad sdf() through one grid_sample over all grids stacked along z."""
        Gz, Gy, Gx = self.a.grid
        lo, hi, vox = self.lo[k], self.hi[k], self.vox[k]
        xc = torch.minimum(torch.maximum(xl, lo), hi)
        out = (xl - xc).norm(dim=-1)
        f = (xc - lo) / vox
        n = torch.tensor([Gx - 1, Gy - 1, self.K * Gz - 1], device=self.dev, dtype=torch.float32)
        f = torch.stack([f[:, 0], f[:, 1], f[:, 2].clamp(max=Gz - 1) + k * Gz], -1) / n * 2 - 1
        v = F.grid_sample(self.atlas, f.view(1, 1, 1, -1, 3), mode="bilinear", align_corners=True).view(-1)
        return v + out

    def grad(self, k, xl):
        """Central differences one voxel wide: the slope of trilinear interpolation averaged over two
        cells, so normals vary continuously instead of jumping at every cell face."""
        h = self.vox[k]  # (R, 3)
        e = torch.eye(3, device=self.dev)
        return torch.stack([(self.sdf(k, xl + e[i] * h[:, i:i + 1]) - self.sdf(k, xl - e[i] * h[:, i:i + 1])) / (2 * h[:, i]) for i in range(3)], -1)

    def colour(self, k, xl):
        """Albedo (R, 3) at local points: planar texture in the body's xy + coarse 3D grid."""
        if self.clay:
            return torch.full((len(k), 3), self.a.clay_grey, device=self.dev)
        Hw = self.Hw
        fx = (xl[:, 0] - 0.5).clamp(0, Hw - 1.001)
        fy = (xl[:, 1] - 0.5).clamp(0, Hw - 1.001)
        x0, y0 = fx.floor().long(), fy.floor().long()
        wx, wy = (fx - x0)[:, None], (fy - y0)[:, None]
        b = k * Hw * Hw
        g = lambda dy, dx: self.texl[b + (y0 + dy) * Hw + x0 + dx]
        t = (g(0, 0) * (1 - wx) + g(0, 1) * wx) * (1 - wy) + (g(1, 0) * (1 - wx) + g(1, 1) * wx) * wy
        # coarse 3D colour grid over the box
        q = ((xl - self.lo[k]) / (self.hi[k] - self.lo[k])).clamp(0, 1)
        n = torch.tensor([9, 9, 5], device=self.dev)
        f = q * n
        i0 = torch.minimum(f.floor().long(), n - 1)
        w = f - i0
        ix, iy, iz = i0.unbind(-1)
        wx3, wy3, wz3 = [w[:, j:j + 1] for j in range(3)]
        bb = k * 600
        h = lambda dz, dy, dx: self.c3[bb + (iz + dz) * 100 + (iy + dy) * 10 + ix + dx]
        c3 = (((h(0, 0, 0) * (1 - wx3) + h(0, 0, 1) * wx3) * (1 - wy3) + (h(0, 1, 0) * (1 - wx3) + h(0, 1, 1) * wx3) * wy3) * (1 - wz3)
              + ((h(1, 0, 0) * (1 - wx3) + h(1, 0, 1) * wx3) * (1 - wy3) + (h(1, 1, 0) * (1 - wx3) + h(1, 1, 1) * wx3) * wy3) * wz3)
        return torch.sigmoid(t + c3)

    def floor_colour(self, xw):
        if self.clay:
            return torch.full((len(xw), 3), self.a.clay_floor, device=self.dev)
        S = self.S
        fx, fy = xw[:, 0] - 0.5, xw[:, 1] - 0.5
        x0, y0 = fx.floor(), fy.floor()
        wx, wy = (fx - x0)[:, None], (fy - y0)[:, None]
        x0, y0 = x0.long() % S, y0.long() % S
        x1, y1 = (x0 + 1) % S, (y0 + 1) % S
        im = self.flr.permute(1, 2, 0).reshape(-1, 3)
        g = lambda yy, xx: im[yy * S + xx]
        return (g(y0, x0) * (1 - wx) + g(y0, x1) * wx) * (1 - wy) + (g(y1, x0) * (1 - wx) + g(y1, x1) * wx) * wy

    # ---- instances and rays
    def instances(self, trans=None, rot=None, objs=None):
        """World pose of every instance: k, anchor (I, 3), R (I, 3, 3). trans (K, 3), rot (K, 3, 3)."""
        k, off = self.ik, self.ioff
        if objs is not None:
            keep = torch.isin(k, torch.as_tensor(objs, device=self.dev)) & (off.abs().sum(1) == 0)
            k, off = k[keep], off[keep]
        anc = self.anchor0[k] + off
        if trans is not None:
            anc = anc + trans[k]
        R = rot[k] if rot is not None else torch.eye(3, device=self.dev).expand(len(k), 3, 3)
        # rotate about the body's centre, not its window corner
        ctr = 0.5 * (self.lo[k] + self.hi[k])
        ctr[:, 2] = 0
        anc = anc + ctr - torch.einsum("iab,ib->ia", R, ctr)
        return dict(k=k, anc=anc, R=R)

    def rays(self, cam, inst):
        k, anc, R = inst["k"], inst["anc"], inst["R"]
        lo, hi = self.lo[k], self.hi[k]
        corners = torch.stack([torch.stack([torch.where(torch.tensor(b >> d & 1, device=self.dev).bool(), hi[:, d], lo[:, d]) for d in range(3)], -1) for b in range(8)], 1)
        wc = anc[:, None] + torch.einsum("iab,icb->ica", R, corners)
        sx = ((wc - cam.c) @ cam.r) / cam.s + cam.W / 2
        sy = ((wc - cam.c) @ cam.u) / cam.s + cam.H / 2
        x0 = sx.min(1).values.floor().clamp(0, cam.W).long(); x1 = sx.max(1).values.ceil().clamp(0, cam.W).long()
        y0 = sy.min(1).values.floor().clamp(0, cam.H).long(); y1 = sy.max(1).values.ceil().clamp(0, cam.H).long()
        w, h = (x1 - x0).clamp(min=0), (y1 - y0).clamp(min=0)
        n = w * h
        end = torch.cumsum(n, 0)
        Rn = int(end[-1])
        j = torch.arange(bucket(Rn), device=self.dev)
        valid = j < Rn
        ii = torch.searchsorted(end, j, right=True).clamp(max=len(k) - 1)  # ray j belongs to instance ii
        start = end - n
        li = (torch.arange(len(ii), device=self.dev) - start[ii]).clamp(min=0)
        wc_ = w[ii].clamp(min=1)
        px, py = (x0[ii] + li % wc_).clamp(max=cam.W - 1), (y0[ii] + li // wc_).clamp(max=cam.H - 1)
        o = cam.c + (((px.float() + 0.5) - cam.W / 2) * cam.s)[:, None] * cam.r + (((py.float() + 0.5) - cam.H / 2) * cam.s)[:, None] * cam.u - 400 * cam.f
        Rt = R[ii].transpose(1, 2)
        ol = torch.einsum("rab,rb->ra", Rt, o - anc[ii])
        dl = torch.einsum("rab,b->ra", Rt, cam.f)
        dl = torch.where(dl.abs() < 1e-6, torch.full_like(dl, 1e-6), dl)
        t1, t2 = (lo[ii] - ol) / dl, (hi[ii] - ol) / dl
        tn = torch.minimum(t1, t2).max(1).values
        tf = torch.maximum(t1, t2).min(1).values
        valid = valid & (tf > tn + 1e-3)
        tf = torch.where(valid, tf, tn + 1.0)
        pix = torch.where(valid, py * cam.W + px, torch.full_like(px, cam.W * cam.H))  # invalid -> spare pixel P
        return dict(i=ii, k=k[ii], pix=pix, ol=ol, dl=dl, tn=tn, tf=tf, valid=valid)

    @torch.no_grad()
    def march(self, ry, N, N2=12, chunk=262144):
        """Samples along each ray through its box: N coarse, then N2 around the first crossing (or the
        closest approach) spanning two coarse intervals. Returns the first sign change and the deepest point."""
        R = len(ry["k"])
        while chunk < R and R % chunk:
            chunk //= 2
        hit = torch.zeros(R, dtype=torch.bool, device=self.dev)
        th = torch.zeros(R, device=self.dev); tm = torch.zeros(R, device=self.dev); fm = torch.zeros(R, device=self.dev)
        js = (torch.arange(N, device=self.dev).float() + 0.5) / N
        j2 = torch.linspace(-1, 1, N2, device=self.dev)
        for s in range(0, R, chunk):
            e = min(R, s + chunk)
            k, ol, dl, tn, tf = ry["k"][s:e], ry["ol"][s:e], ry["dl"][s:e], ry["tn"][s:e], ry["tf"][s:e]
            dt = (tf - tn) / N
            ts = tn[:, None] + js[None] * (tf - tn)[:, None]  # (r, N)
            kk = k[:, None].expand(-1, N).reshape(-1)
            f = self.sdf_fast(kk, (ol[:, None] + ts[..., None] * dl[:, None]).reshape(-1, 3)).view(-1, N)
            neg = f < 0
            jn = torch.where(neg.any(1), neg.float().argmax(1), f.argmin(1))
            tc = ts.gather(1, jn[:, None])[:, 0]
            t2 = tc[:, None] + j2[None] * dt[:, None]  # (r, N2)
            k2 = k[:, None].expand(-1, N2).reshape(-1)
            f2 = self.sdf_fast(k2, (ol[:, None] + t2[..., None] * dl[:, None]).reshape(-1, 3)).view(-1, N2)
            neg2 = f2 < 0
            first = torch.where(neg2.any(1), neg2.float().argmax(1), torch.full_like(k, -1))
            h = first > 0
            j1 = first.clamp(min=1)
            a0, a1 = f2.gather(1, (j1 - 1)[:, None])[:, 0], f2.gather(1, j1[:, None])[:, 0]
            b0, b1 = t2.gather(1, (j1 - 1)[:, None])[:, 0], t2.gather(1, j1[:, None])[:, 0]
            th[s:e] = b0 + (b1 - b0) * a0 / (a0 - a1).clamp(min=1e-6)
            hit[s:e] = h
            m1, i1 = f.min(1)
            m2, i2 = f2.min(1)
            use2 = m2 < m1
            fm[s:e] = torch.where(use2, m2, m1)
            tm[s:e] = torch.where(use2, t2.gather(1, i2[:, None])[:, 0], ts.gather(1, i1[:, None])[:, 0])
        v = ry["valid"]
        return hit & v, th, tm, torch.where(v, fm, torch.full_like(fm, 1e9))

    def surface(self, ry, inst, hit, th, tm, eps):
        """With gradients: hit/closest point (world), normal (world), coverage, local point."""
        k, ol, dl = ry["k"], ry["ol"], ry["dl"]
        R = inst["R"][ry["i"]]
        t0 = torch.where(hit, th, tm)
        x0 = ol + t0[:, None] * dl
        f0 = self.sdf(k, x0)
        g = self.grad(k, x0)
        gd = (g * dl).sum(-1).detach()
        # implicit hit point: moves with the SDF as -df / (grad f . d); misses stay at closest approach
        t = torch.where(hit, t0 - (f0 - f0.detach()) / gd.clamp(max=-0.05), t0)
        xl = ol + t[:, None] * dl
        fmin = torch.where(hit, self.sdf(k, ol + tm[:, None] * dl), f0)
        cov = torch.sigmoid(-fmin / eps)
        n = torch.einsum("rab,rb->ra", R, g)
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        xw = inst["anc"][ry["i"]] + torch.einsum("rab,rb->ra", R, xl)
        return xw, n, cov, xl

    # ---- one frame
    def slots(self, pix, P):
        order = torch.argsort(pix, stable=True)
        ps = pix[order]
        new = torch.ones_like(ps, dtype=torch.bool)
        new[1:] = ps[1:] != ps[:-1]
        gid = torch.cumsum(new.long(), 0) - 1
        starts = torch.nonzero(new)[:, 0]
        rank = torch.empty_like(pix)
        rank[order] = torch.arange(len(ps), device=self.dev) - starts[gid]
        return torch.where(pix < P, rank, torch.zeros_like(rank))

    def shadow_map(self, inst, Lv):
        a = self.a
        cam = Cam.light(Lv, self.S, a.wrap_margin, self.zmax, self.dev, a.smap_s)
        ry = self.rays(cam, inst)
        hit, th, tm, fm = self.march(ry, a.light_samples, 8)
        P = cam.W * cam.H
        hk, hv = select(hit)
        self._lrays = (len(ry["k"]), len(hk))
        sub = {q: v[hk] for q, v in ry.items()}
        R = inst["R"][sub["i"]]
        x0 = sub["ol"] + th[hk, None] * sub["dl"]
        f0 = self.sdf(sub["k"], x0)
        with torch.no_grad():
            h = 0.75
            gd = (self.sdf_fast(sub["k"], x0 + h * sub["dl"]) - self.sdf_fast(sub["k"], x0 - h * sub["dl"])) / (2 * h)
        t = th[hk] - (f0 - f0.detach()) / gd.clamp(max=-0.05)
        xw = inst["anc"][sub["i"]] + torch.einsum("rab,rb->ra", R, sub["ol"] + t[:, None] * sub["dl"])
        Lt = torch.tensor(Lv, device=self.dev)
        dep = torch.where(hv, xw @ Lt, torch.full_like(t, -1e9))  # toward the light
        # floor depth for every light pixel
        py, px = torch.meshgrid(torch.arange(cam.H, device=self.dev), torch.arange(cam.W, device=self.dev), indexing="ij")
        o = cam.c + (((px.reshape(-1).float() + 0.5) - cam.W / 2) * cam.s)[:, None] * cam.r + (((py.reshape(-1).float() + 0.5) - cam.H / 2) * cam.s)[:, None] * cam.u
        tf = -o[:, 2] / cam.f[2]
        fdep = (o + tf[:, None] * cam.f) @ Lt
        smap = torch.cat([fdep, fdep[:1]])  # + the spare pixel P of invalid rays
        smap = smap.scatter_reduce(0, sub["pix"], dep, reduce="amax", include_self=True)
        return cam, smap[:P].view(cam.H, cam.W), Lt

    def lit(self, xw, n, cam, smap, Lt):
        """Shadow visibility (R,) of world points from a shadow map, PCF-softened."""
        a = self.a
        p = xw + n * a.normal_offset
        u = ((p - cam.c) @ cam.r) / cam.s + cam.W / 2 - 0.5
        v = ((p - cam.c) @ cam.u) / cam.s + cam.H / 2 - 0.5
        d = p @ Lt
        vis = 0
        offs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        flat = smap.reshape(-1)
        for dx, dy in offs:
            fx = (u + dx * a.pcf / cam.s).clamp(0, cam.W - 1.001)
            fy = (v + dy * a.pcf / cam.s).clamp(0, cam.H - 1.001)
            x0, y0 = fx.floor().long(), fy.floor().long()
            wx, wy = fx - x0, fy - y0
            g = lambda yy, xx: flat[yy * cam.W + xx]
            sm = (g(y0, x0) * (1 - wx) + g(y0, x0 + 1) * wx) * (1 - wy) + (g(y0 + 1, x0) * (1 - wx) + g(y0 + 1, x0 + 1) * wx) * wy
            vis = vis + torch.sigmoid((d - sm + a.bias) / a.shadow_soft)
        return vis / len(offs)

    @torch.no_grad()
    def occlusion(self, xw, n, pix, table, inst):
        """Ambient occlusion from the union SDF (candidate bodies at this pixel + floor) along n."""
        a = self.a
        occ = torch.zeros(len(xw), device=self.dev)
        M = table.shape[0]
        for dlt, w in ((2.0, 1.0), (5.0, 0.6), (10.0, 0.35)):
            q = xw + n * dlt
            un = q[:, 2].clone()  # floor
            for m in range(M):
                ii = table[m, pix]
                ok = ii >= 0
                iv = ii.clamp(min=0)
                ql = torch.einsum("rba,rb->ra", inst["R"][iv], q - inst["anc"][iv])
                un = torch.where(ok, torch.minimum(un, self.sdf_fast(inst["k"][iv], ql)), un)
            occ = occ + w * torch.relu(dlt - un) / dlt
        return torch.exp(-a.ao * occ)

    def shade(self, alb, n, vis, ao, light, view):
        lx, ly, lz = light.vec()
        Lt = torch.tensor([lx, ly, lz], device=self.dev)
        ndl = torch.relu(n @ Lt)
        hv = Lt - view
        hv = hv / hv.norm()
        ndh = torch.relu(n @ hv)
        expo = (light.amb + light.dif) / (light.amb + light.dif * max(lz, 0.05))
        sh = (light.amb * ao + light.dif * ndl * vis) * expo
        spec = light.spec * ndh ** light.shin * vis
        return alb * sh[:, None] + spec[:, None], sh, spec

    def render(self, cam, light, trans=None, rot=None, objs=None, floor=True, clay=False, parts=False, sil=False):
        """(3, H, W) frame (plus parts / silhouette loss). prep() must have run."""
        a = self.a
        self.clay = clay
        inst = self.instances(trans, rot, objs)
        Lv = light.vec()
        lcam, smap, Lt = self.shadow_map(inst, Lv)
        ry = self.rays(cam, inst)
        hit, th, tm, fm = self.march(ry, a.samples)
        P = cam.W * cam.H
        rank = self.slots(ry["pix"], P)
        M = int(rank.max()) + 1
        M = 4 * ((M + 3) // 4)  # bucketed like the ray counts
        table = torch.full((M, P + 1), -1, dtype=torch.long, device=self.dev)
        table[rank, ry["pix"]] = ry["i"]
        table = table[:, :P]
        act, av = select(fm < 4 * a.eps)
        self.shapes = getattr(self, "shapes", []) + [(len(ry["k"]), len(act), M, self._lrays)]
        sub = {q: v[act] for q, v in ry.items()}
        xw, n, cov, xl = self.surface(sub, inst, hit[act], th[act], tm[act], a.eps)
        n = torch.where((n @ cam.f)[:, None] > 0, -n, n)  # grazing misses: face the camera
        cov = cov * av
        alb = self.colour(sub["k"], xl)
        vis = self.lit(xw, n, lcam, smap, Lt)
        ao = self.occlusion(xw, n, sub["pix"].clamp(max=P - 1), table, inst)
        col, sh, sp = self.shade(alb, n, vis, ao, light, cam.f)
        if parts:
            col = torch.cat([col, sh[:, None], sp[:, None]], 1)
        C = col.shape[1]
        dep = -(xw @ cam.f)
        idx = torch.where(av, rank[act] * P + sub["pix"], torch.full_like(act, M * P))  # padding -> spare slot
        D = torch.full((M * P + 1,), -1e9, device=self.dev).index_put((idx,), dep.detach())[:-1]
        A = torch.zeros(M * P + 1, device=self.dev).index_put((idx,), cov)[:-1]
        Cl = torch.zeros(M * P + 1, C, device=self.dev).index_put((idx,), col)[:-1]
        order = torch.argsort(D.view(M, P), 0, descending=True)
        A = torch.gather(A.view(M, P), 0, order)
        Cl = torch.gather(Cl.view(M, P, C), 0, order[..., None].expand(M, P, C))
        Tr = torch.cumprod(torch.cat([torch.ones_like(A[:1]), 1 - A], 0), 0)
        w = Tr[:-1] * A
        out = (w[..., None] * Cl).sum(0)
        # floor
        py, px = torch.meshgrid(torch.arange(cam.H, device=self.dev), torch.arange(cam.W, device=self.dev), indexing="ij")
        o = cam.c + (((px.reshape(-1).float() + 0.5) - cam.W / 2) * cam.s)[:, None] * cam.r + (((py.reshape(-1).float() + 0.5) - cam.H / 2) * cam.s)[:, None] * cam.u - 400 * cam.f
        if floor and float(cam.f[2]) < -0.05:
            tf = -o[:, 2] / cam.f[2]
            xf = o + tf[:, None] * cam.f
            nf = torch.tensor([0.0, 0.0, 1.0], device=self.dev).expand(P, 3)
            visf = self.lit(xf, nf, lcam, smap, Lt)
            aof = self.occlusion(xf, nf, torch.arange(P, device=self.dev), table, inst)
            cf, shf, spf = self.shade(self.floor_colour(xf), nf, visf, aof, light, cam.f)
            if parts:
                cf = torch.cat([cf, shf[:, None], spf[:, None]], 1)
        else:
            cf = torch.full((P, C), a.bg_grey, device=self.dev)
        out = out + Tr[-1][:, None] * cf
        img = out.T.reshape(C, cam.H, cam.W)
        if not parts:
            img = img.clamp(0, 1)
        if sil:
            # weight each pixel gives to the body the segmentation puts there
            kk = torch.full((M * P + 1,), -1, dtype=torch.long, device=self.dev).index_put((idx,), sub["k"])[:-1]
            kk = torch.gather(kk.view(M, P), 0, order)
            return img, w, kk
        return img

    # ---- regularizers
    def regs(self):
        a = self.a
        g = self.base + self._res
        v = self.vox  # (K, 3) x, y, z
        gx = (g[..., 2:] - g[..., :-2]) / (2 * v[:, 0].view(-1, 1, 1, 1, 1))
        gy = (g[..., 2:, :] - g[..., :-2, :]) / (2 * v[:, 1].view(-1, 1, 1, 1, 1))
        gz = (g[:, :, 2:] - g[:, :, :-2]) / (2 * v[:, 2].view(-1, 1, 1, 1, 1))
        gm = torch.sqrt(gx[:, :, 1:-1, 1:-1] ** 2 + gy[:, :, 1:-1, :, 1:-1] ** 2 + gz[:, :, :, 1:-1, 1:-1] ** 2 + 1e-8)
        near = (g[:, :, 1:-1, 1:-1, 1:-1].abs() < 4).float()
        eik = ((gm - 1) ** 2 * near).sum() / near.sum().clamp(min=1)
        r = self._res
        lap = sum(((r.roll(1, d) + r.roll(-1, d) - 2 * r) ** 2)[:, :, 1:-1, 1:-1, 1:-1].mean() for d in (2, 3, 4))
        vol = torch.sigmoid(-g / 1.0).sum((1, 2, 3, 4))
        if self.vol0 is None:
            self.vol0 = torch.sigmoid(-self.base / 1.0).sum((1, 2, 3, 4))
        volp = (torch.relu((vol / self.vol0 - 1).abs() - a.vol_tol) ** 2).mean()
        return eik, lap, volp

    def rim_loss(self):
        """No darkening of the texture toward the body's base outline (contact darkness is the light's)."""
        t = torch.sigmoid(self.tex0 + self.tex.preact())
        y = (t * torch.tensor(LUMA, device=self.dev).view(1, 3, 1, 1)).sum(1)
        d = self.dwin_t
        inner = (d > 6).float()
        ref = (y * inner).sum((1, 2), keepdim=True) / inner.sum((1, 2), keepdim=True).clamp(min=1)
        band = ((d > 0) & (d < 3)).float()
        return ((torch.relu(0.75 * ref.detach() - y) ** 2) * band).sum() / band.sum().clamp(min=1) / 0.01

    def state(self):
        return dict(rc=self.rc.detach().cpu(), rf=self.rf.detach().cpu(), tex=self.tex.spectrum.detach().cpu(),
                    col3=self.col3.detach().cpu(), floor=self.floor.spectrum.detach().cpu())

    def load(self, st):
        with torch.no_grad():
            self.rc.copy_(st["rc"]); self.rf.copy_(st["rf"]); self.tex.spectrum.copy_(st["tex"])
            self.col3.copy_(st["col3"]); self.floor.spectrum.copy_(st["floor"])


# ----------------------------------------------------------------------------- main

def deshade(bd, still, light, cam, lo=0.5):
    with torch.no_grad():
        bd.prep(clay=True)
        a0 = bd.a.clay_grey, bd.a.clay_floor
        bd.a.clay_grey = bd.a.clay_floor = 1.0
        pr = bd.render(cam, light, clay=True, parts=True)
        bd.a.clay_grey, bd.a.clay_floor = a0
        sh, sp = pr[3], pr[4]
        alb = ((still[0] - sp) / sh.clamp(min=lo)).clamp(1 / 512, 1 - 1 / 512)
    return alb, sh


def video(frames_dir, out, fps=24, crf=20):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{frames_dir}/f%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), out], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--labels", default="labels_display.npy")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--net", default="googlenet")
    # objfield scene (outlines, amodal extension under confident fronts, mirror-filled colour)
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
    # bodies
    ap.add_argument("--grid", default="32,48,48", help="SDF grid per body, z,y,x")
    ap.add_argument("--kt", type=float, default=0.7, help="top height = kt * sqrt(h)")
    ap.add_argument("--kb", type=float, default=0.4, help="underside depth = kb * sqrt(h)")
    ap.add_argument("--margin", type=float, default=1.0, help="px a visible top clears the tops beneath it")
    ap.add_argument("--taper", type=float, default=0.3, help="height of a shell's top where it goes under a front shell")
    ap.add_argument("--taper-w", type=float, default=12.0, help="px over which it rises back to full height")
    ap.add_argument("--res-amp", type=float, default=6.0, help="largest SDF change the network may make, px")
    ap.add_argument("--wrap-margin", type=float, default=24.0)
    ap.add_argument("--samples", type=int, default=16, help="coarse samples per ray through a body's box (then 12 fine)")
    ap.add_argument("--light-samples", type=int, default=12)
    ap.add_argument("--eps", type=float, default=0.5, help="silhouette softness, px")
    # light
    ap.add_argument("--ao", type=float, default=0.9)
    ap.add_argument("--smap-s", type=float, default=1.0, help="shadow map px per texel")
    ap.add_argument("--pcf", type=float, default=1.25)
    ap.add_argument("--bias", type=float, default=1.5)
    ap.add_argument("--soft-shadow", dest="shadow_soft", type=float, default=0.5)
    ap.add_argument("--normal-offset", type=float, default=1.0)
    ap.add_argument("--floor-dark", type=float, default=1.0)
    ap.add_argument("--clay-grey", type=float, default=0.72)
    ap.add_argument("--clay-floor", type=float, default=0.5)
    ap.add_argument("--bg-grey", type=float, default=0.12)
    # training
    ap.add_argument("--train-light", default="135,50")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--jitter", type=float, default=3.5, help="per-body translation in training, px rms")
    ap.add_argument("--rot-jitter", type=float, default=4.0, help="per-body rotation in training, deg rms")
    ap.add_argument("--tilt", type=float, default=10.0, help="camera tilt in training, deg (uniform)")
    ap.add_argument("--lr-geo", type=float, default=0.02)
    ap.add_argument("--lr-tex", type=float, default=0.01)
    ap.add_argument("--wsil", type=float, default=2.0, help="silhouette term at step 0, annealed to 0 by --sil-until")
    ap.add_argument("--sil-until", type=float, default=0.5)
    ap.add_argument("--weik", type=float, default=0.2)
    ap.add_argument("--wlap", type=float, default=2.0)
    ap.add_argument("--wvol", type=float, default=1.0)
    ap.add_argument("--vol-tol", type=float, default=0.25)
    ap.add_argument("--wrim", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", help="state .pt: resume from it")
    ap.add_argument("--save", help="state .pt to write after training (default OUT/state.pt)")
    ap.add_argument("--start", type=int, default=0, help="first step (for chunked training: cosine schedule over --total)")
    ap.add_argument("--total", type=int, default=0, help="steps in the whole schedule (default --steps)")
    ap.add_argument("--evals", default="all", help="comma list: lights,clay,turn,orbit,tilt,anim or all / none")
    ap.add_argument("--objs", default="", help="objects for turntables (1-based labels); default auto")
    ap.add_argument("--anim", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--smoke", action="store_true", help="time one render and one step, then stop")
    a = ap.parse_args()
    a.grid = tuple(int(v) for v in a.grid.split(","))
    a.total = a.total or a.start + a.steps
    dev = pick_device()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed + a.start)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)
    save = lambda x, p: Image.fromarray((x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(p)

    L = np.load(f"{a.seg}/{a.labels}").astype(np.int64)
    S = L.shape[0]
    img = np.asarray(Image.open(a.image).convert("RGB")).astype(np.float32) / 255
    lg = lambda im: np.log(np.clip(im, 1 / 512, 1 - 1 / 512) / (1 - np.clip(im, 1 / 512, 1 - 1 / 512))).transpose(2, 0, 1).astype(np.float32)
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
    still = torch.from_numpy(img.transpose(2, 0, 1))[None].to(dev)
    overhead = Light(0.0, math.radians(90))
    az, el = map(float, a.train_light.split(","))
    tl = Light(math.radians(az), math.radians(el))

    sc = build_scene(L, lg(img), votes, aff, pins, soft, a)
    log("scene")
    bd = Bodies(sc, img, img, L, dj["z"], dev, a, log)
    cam = Cam.top(S, dev, zc=bd.zmax / 2)
    alb, sh0 = deshade(bd, still, overhead, cam)
    alb_np = alb.permute(1, 2, 0).cpu().numpy()
    save(sh0[None].expand(3, -1, -1) / sh0.max(), f"{a.out}/shade0.png")
    sc2 = build_scene(L, lg(alb_np), votes, aff, pins, soft, a)
    bd.tex0 = torch.from_numpy(sc2["color"][1:]).to(dev)
    fb = np.stack([ndi.gaussian_filter(lg(alb_np)[c], 8, mode="wrap") for c in range(3)]) - a.floor_dark
    bd.floor0 = torch.from_numpy(fb.astype(np.float32)).to(dev)
    log(f"deshaded: shading at init {float(sh0.min()):.2f}..{float(sh0.max()):.2f}, mean {float(sh0.mean()):.2f}")

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

    held = [overhead] + [Light(math.radians(z_), math.radians(50)) for z_ in range(0, 360, 45)]
    names = ["overhead"] + [f"az{z_} el50" for z_ in range(0, 360, 45)]
    rep = {"args": vars(a)}
    if os.path.exists(f"{a.out}/report.json"):
        rep = json.load(open(f"{a.out}/report.json")) | {"args": vars(a)}

    if a.load:
        bd.load(torch.load(a.load))
        log(f"loaded {a.load}")

    with torch.no_grad():
        rep["still"] = score(still)
        bd.prep()
        t1 = time.time()
        fr0 = bd.render(cam, tl)
        log(f"render {time.time() - t1:.2f}s; still {rep['still']:.3f}, this state at the training light {score(fr0[None]):.3f}")
        save(fr0, f"{a.out}/start_{a.start}.png")

    geo, col = bd.params()
    opt = torch.optim.Adam([{"params": geo, "lr": a.lr_geo}, {"params": col, "lr": a.lr_tex}])
    if a.load and os.path.exists(a.load.replace(".pt", "_opt.pt")):
        opt.load_state_dict(torch.load(a.load.replace(".pt", "_opt.pt")))
    base_lr = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(a.seed + a.start)
    curve = rep.get("curve", [])
    for it in range(a.start, a.start + a.steps):
        ts = time.time()
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * 0.5 * (1 + math.cos(math.pi * it / a.total))
        opt.zero_grad()
        bd.prep()
        wsil = a.wsil * max(0.0, 1 - it / (a.sil_until * a.total))
        # one frame at a time, each with its own graph and backward, so only one frame's rays are held
        acts, sil = [], torch.zeros((), device=dev)
        for b in range(a.mb):
            bd.prep()
            trans = torch.from_numpy(rng.normal(0, a.jitter / math.sqrt(2), (bd.K, 3))).float().to(dev)
            trans[:, 2] = 0
            ax = torch.nn.functional.normalize(torch.randn(bd.K, 3, device=dev), dim=1)
            ang = torch.from_numpy(rng.normal(0, math.radians(a.rot_jitter), bd.K)).float().to(dev)
            rot = rodrigues(ax, ang)
            c = cam
            if a.tilt > 0 and b > 0:
                c = cam.tilted(math.radians(rng.uniform(-a.tilt, a.tilt)), rng.uniform(0, math.pi))
            extra = 0
            if b == 0 and wsil > 0:
                fr, w, kk = bd.render(c, tl, sil=True)
                lab = torch.from_numpy(L.reshape(-1) - 1).to(dev)
                sil = ((1 - (w * (kk == lab[None]).float()).sum(0)) ** 2).mean()
                extra = wsil * sil
            else:
                fr = bd.render(c, tl, trans, rot)
            act = rg.act(crop_q(fr[None]))[0]
            (-act + act.detach().abs() * extra).backward()
            acts.append(act.detach())
            del fr
        act = torch.stack(acts)
        bd.prep()
        eik, lap, volp = bd.regs()
        rim = bd.rim_loss()
        pen = a.weik * eik + a.wlap * lap + a.wvol * volp + a.wrim * rim
        (act.abs().sum() * pen).backward()
        opt.step()
        torch.mps.empty_cache()
        if a.log_every == 1:
            print(f"  step {it} {time.time() - ts:.1f}s shapes {bd.shapes}", flush=True)
        bd.shapes = []
        if it % a.log_every == 0 or it == a.start + a.steps - 1 or a.smoke:
            res = bd._res.detach()
            curve.append((it, float(act.mean())))
            log(f"step {it + 1}/{a.total} act {act.mean():.3f} eik {float(eik):.3f} lap {float(lap):.4f} vol {float(volp):.4f} "
                f"rim {float(rim):.3f} sil {float(sil):.3f} | res rms {float(res.pow(2).mean().sqrt()):.2f} max {float(res.abs().max()):.2f} px "
                f"| {time.time() - ts:.2f}s/step, mps {torch.mps.current_allocated_memory() / 2**30:.1f}/{torch.mps.driver_allocated_memory() / 2**30:.1f} GB")
        if a.smoke:
            return
    rep["curve"] = curve
    if a.steps:
        st = a.save or f"{a.out}/state.pt"
        torch.save(bd.state(), st)
        torch.save(opt.state_dict(), st.replace(".pt", "_opt.pt"))
        log(f"saved {st}")
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)

    ev = set("lights,clay,turn,orbit,tilt,anim".split(",")) if a.evals == "all" else set(a.evals.split(",")) - {"none", ""}
    ss = Cam.top(S, dev, s=0.5, zc=bd.zmax / 2)
    down = lambda x: F.avg_pool2d(x[None], 2)[0]
    with torch.no_grad():
        bd.prep()
        if "lights" in ev:
            acts, tiles = [], []
            for L_ in held:
                fr = bd.render(cam, L_)
                acts.append(score(fr[None]))
                tiles.append(fr)
            rep["lights"] = dict(zip(names, acts))
            rep["train_light"] = score(bd.render(cam, tl)[None])
            log(f"lights: train {rep['train_light']:.3f}; overhead {acts[0]:.3f}; 8 at 50 deg mean {np.mean(acts[1:]):.3f} [{min(acts[1:]):.3f}, {max(acts[1:]):.3f}]")
            grid = torch.cat([torch.cat(tiles[r * 3:(r + 1) * 3], 2) for r in range(3)], 1)
            save(grid, f"{a.out}/lights.png")
            save(down(bd.render(ss, tl)), f"{a.out}/lit_train.png")
            save(torch.sigmoid(bd.tex0 + bd.tex.preact())[0], f"{a.out}/tex_example.png")
        if "clay" in ev:
            bd.prep(clay=True)
            save(down(bd.render(ss, tl, clay=True)), f"{a.out}/clay_top.png")
            for deg in (30, 60):
                c = ss.tilted(math.radians(deg), 0.0)
                save(down(bd.render(c, tl, clay=True)), f"{a.out}/clay_tilt{deg}.png")
            bd.prep()
        if "turn" in ev or "orbit" in ev or "tilt" in ev or "anim" in ev:
            json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
        if "turn" in ev:
            objs = [int(o) for o in a.objs.split(",")] if a.objs else []
            for o in objs:
                i = o - 1
                ctr = bd.anchor0[i] + 0.5 * (bd.lo[i] + bd.hi[i])
                ctr[2] = bd.anchor0[i, 2]
                ext = float((bd.hi[i] - bd.lo[i]).max()) * 1.3
                Wt = 256
                d = f"{a.out}/turn_{o}"
                os.makedirs(d, exist_ok=True)
                for clay in (True, False):
                    bd.prep(clay=clay)
                    row = []
                    for j, (e_, a_) in enumerate([(90, 0), (35, 0), (35, 90), (35, 180), (35, 270), (-60, 45)]):
                        c = Cam.orbit(ctr, math.radians(e_), math.radians(a_), Wt, ext / Wt, dev)
                        row.append(bd.render(c, tl, objs=[i], floor=False, clay=clay))
                    save(torch.cat(row, 2), f"{d}/{'clay' if clay else 'lit'}.png")
                bd.prep(clay=True)
                for j in range(48):
                    c = Cam.orbit(ctr, math.radians(25), 2 * math.pi * j / 48, Wt, ext / Wt, dev)
                    save(bd.render(c, tl, objs=[i], floor=False, clay=True), f"{d}/f{j:04d}.png")
                video(d, f"{a.out}/turn_{o}.mp4")
                bd.prep()
            log(f"turntables: {objs}")
        if "orbit" in ev:
            d = f"{a.out}/orbit"
            os.makedirs(d, exist_ok=True)
            for j in range(48):
                save(down(bd.render(ss, Light(2 * math.pi * j / 48, math.radians(45)))), f"{d}/f{j:04d}.png")
            video(d, f"{a.out}/orbit.mp4")
            log("light orbit")
        if "tilt" in ev:
            d = f"{a.out}/tilt"
            os.makedirs(d, exist_ok=True)
            for j in range(48):
                c = ss.tilted(math.radians(20), 2 * math.pi * j / 48)
                save(down(bd.render(c, tl)), f"{d}/f{j:04d}.png")
            video(d, f"{a.out}/tilt.mp4")
            log("camera tilt")
        if "anim" in ev and a.anim:
            d = f"{a.out}/anim"
            os.makedirs(d, exist_ok=True)
            g = np.random.default_rng(11)
            ph = g.uniform(0, 2 * np.pi, (bd.K, 4, 2))
            amp = g.normal(0, 1, (bd.K, 4, 2))
            acts = []
            for j in range(a.anim):
                th_ = 2 * np.pi * j / a.anim
                v = (amp * np.sin(np.array([1, 2])[None, None] * th_ + ph)).sum(-1)  # (K, 4)
                trans = torch.zeros(bd.K, 3, device=dev)
                trans[:, :2] = torch.from_numpy(v[:, :2] * 2.5 / math.sqrt(2)).float()
                ax = torch.nn.functional.normalize(torch.from_numpy(np.c_[v[:, 2], v[:, 3], np.ones(bd.K) * 0.3]).float().to(dev), dim=1)
                ang = torch.from_numpy(np.hypot(v[:, 2], v[:, 3]) * math.radians(4)).float().to(dev)
                fr = down(bd.render(ss, tl, trans, rodrigues(ax, ang)))
                acts.append(score(fr[None]))
                save(fr, f"{d}/f{j:04d}.png")
            video(d, f"{a.out}/anim.mp4")
            rep["anim"] = dict(mean=float(np.mean(acts)), min=float(np.min(acts)), frames=acts)
            log(f"anim: mean {np.mean(acts):.3f} min {np.min(acts):.3f}")
    rep["secs"] = rep.get("secs", 0) + time.time() - t0
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
