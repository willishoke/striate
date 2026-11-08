"""A generative prior for the shells' skins: a small convolutional VAE trained only on the skin the still
shows sharply (the tops), flattened to a distortion-free plane, so that skins decoded from its latents
can only be the kind of texture the tops have, on every side of every shell.

The skin of a body (objtube.py) is a tube unwrap: rows run around (wrapping), columns along the spine
and over the end caps (geodesic, --caps disc). A texel there spans 2 pi r / Ha px around and
s_len / (Wl - 1) px along, and r varies along the body, so the unwrap is stretched differently
everywhere. Here every body gets a chart, a canvas whose texels are --delta px on the surface:

  rows    Hc = 2 pi r_ref / delta around (rounded to a multiple of 16; wrapping), r_ref the body's
          geometric-mean radius
  cols    unwrap column j sits at canvas column c_j = sum (s_len / (Wl - 1)) / delta * (r_ref / r_j)^alpha,
          r_j the radius there (caps: r cos(geodesic angle), floored at half the end radius)

alpha = 1 is conformal (isotropic everywhere, the pattern's scale following the radius, as on a real
shell); alpha = 0 keeps the scale and squeezes the pattern around on thin parts; the default 0.5 splits
the difference (a body whose radius varies 2.5x is off by at most 1.26x either way from its mean).
The inside of a coil's whorl is compressed along by 1 - kappa r cos(theta); that is ignored.

Stages:
  data    from a checkpoint: render the rest pose on the CPU (2x, footprints, as objphys.py --wskin
          does), keep the texels seen at <= --fp texels per still px (majority-smoothed), and cut 32 x 32
          patches of the paint albedo, sigmoid(tex0 + upsampled tex), in each texel's local isotropic
          chart (--delta px per texel), at least 90% sharp. Saves patches.pt and a contact sheet.
  train   a VAE on those patches (dihedral augmentation: orientation on the skin is arbitrary): fully
          convolutional, stride 4 latents, padding that wraps around and repeats along, so a whole
          canvas decodes seamlessly. L1 on albedo and its gradients, + beta KL. CPU, ~20 min.
          Saves model.pt and sheets (reconstructions; random latents, patch-sized and canvas-sized).
  objphys.py --skin-prior model.pt: each body's latents (encoded from its current skin, so step 0 is
          close to the checkpoint) are the trained parameters; the decoded canvases, resampled to the
          unwrap, are tex0 (the residual paint is zeroed). See SkinPrior.

usage: skinprior.py data --load tube.pt --out DIR
       skinprior.py train --data DIR/patches.pt --out DIR [--steps 8000]
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------- charts

def radius_along(s, r, s_lo, s_len, Wl):
    """Each unwrap column's tube radius (K, Wl): the spine's (interpolated) within it, r cos(angle) over
    the geodesic caps, floored at half the end radius (nearer the pole the cap's disc takes over)."""
    K, N = s.shape
    sg = s_lo[:, None] + torch.linspace(0, 1, Wl, device=s.device)[None] * s_len[:, None]  # (K, Wl) px along
    L, r0, re = s[:, -1:], r[:, :1], r[:, -1:]
    idx = torch.searchsorted(s.contiguous(), sg.clamp(min=0).contiguous()).clamp(1, N - 1)
    sa, sb = torch.gather(s, 1, idx - 1), torch.gather(s, 1, idx)
    ra, rb = torch.gather(r, 1, idx - 1), torch.gather(r, 1, idx)
    w = ((sg - sa) / (sb - sa).clamp(min=1e-6)).clamp(0, 1)
    rt = ra + w * (rb - ra)
    rc0 = r0 * torch.cos((-sg / r0).clamp(0, math.pi / 2))
    rc1 = re * torch.cos(((sg - L) / re).clamp(0, math.pi / 2))
    rr = torch.where(sg < 0, torch.maximum(rc0, 0.5 * r0), torch.where(sg > L, torch.maximum(rc1, 0.5 * re), rt))
    return rr, sg


def charts(s, r, s_lo, s_len, Wl, delta=0.5, alpha=0.5, mult=16, wmult=16):
    """Per body: (Hc, Wc, c (Wl,) canvas column of each unwrap column, r_ref). Lists of python/tensors. Hc is a
    multiple of mult, Wc of wmult (fewer distinct canvas shapes: MPS keeps buffers and graphs per shape)."""
    rr, sg = radius_along(s, r, s_lo, s_len, Wl)
    out = []
    for k in range(s.shape[0]):
        L = float(s[k, -1])
        tube = (sg[k] >= 0) & (sg[k] <= L)
        rref = float(torch.exp(torch.log(rr[k][tube]).mean()))
        Hc = max(mult, int(round(2 * math.pi * rref / delta / mult)) * mult)
        rref = Hc * delta / (2 * math.pi)  # the period is what the rounding made it
        dcol = float(s_len[k]) / (Wl - 1) / delta * (rref / rr[k]) ** alpha
        c = torch.cat([torch.zeros(1, device=s.device), torch.cumsum(0.5 * (dcol[1:] + dcol[:-1]), 0)])
        Wc = int(math.ceil((float(c[-1]) + 1) / wmult)) * wmult
        c = c + (Wc - 1 - float(c[-1])) / 2
        out.append((Hc, Wc, c, rref))
    return out


def bilerp(x, rows, cols):
    """x (C, Hc, Wc) at fractional texel positions rows, cols (any shape): bilinear, rows wrapping, columns
    clamped. Gathers, not grid_sample, whose backward MPS runs on the CPU."""
    Cn, Hc, Wc = x.shape
    cols = cols.clamp(0, Wc - 1)
    r0f, c0f = torch.floor(rows), torch.floor(cols).clamp(max=Wc - 2)
    fr, fc = rows - r0f, cols - c0f
    r0 = r0f.long() % Hc
    r1 = (r0 + 1) % Hc
    c0 = c0f.long()
    xf = x.reshape(Cn, -1)
    g = lambda r_, c_: xf[:, (r_ * Wc + c_).flatten()].reshape(Cn, *rows.shape)
    return (1 - fr) * ((1 - fc) * g(r0, c0) + fc * g(r0, c0 + 1)) + fr * ((1 - fc) * g(r1, c0) + fc * g(r1, c0 + 1))


def canvas_to_unwrap(x, c, Ha, ss=2):
    """A canvas x (C, Hc, Wc) read at every unwrap texel (C, Ha, Wl): row i at turn i / Ha, column j at
    canvas column c[j]; bilinear, wrapping around, ss x ss samples per texel (the canvas is often finer)."""
    Cn, Hc, Wc = x.shape
    Wl = c.shape[0]
    dc = torch.cat([c[1:2] - c[0:1], (c[2:] - c[:-2]) / 2, c[-1:] - c[-2:-1]])
    acc = 0
    for oi in range(ss):
        for oj in range(ss):
            fi, fj = (oi + 0.5) / ss - 0.5, (oj + 0.5) / ss - 0.5
            rows = ((torch.arange(Ha, device=x.device).float() + fi) * Hc / Ha)[:, None].expand(Ha, Wl)
            cols = (c + fj * dc)[None, :].expand(Ha, Wl)
            acc = acc + bilerp(x, rows, cols)
    return acc / (ss * ss)


def unwrap_to_canvas(t, c, Hc, Wc):
    """An unwrap map t (C, Ha, Wl) read at every canvas texel (C, Hc, Wc): the inverse of canvas_to_unwrap."""
    Cn, Ha, Wl = t.shape
    cn = c.detach().cpu().numpy()
    jf = torch.tensor(np.interp(np.arange(Wc), cn, np.arange(Wl)), dtype=torch.float32, device=t.device)  # clamps outside
    tp = torch.cat([t[:, -1:], t, t[:, :1]], 1)[None]
    rows = torch.arange(Hc, device=t.device).float() * Ha / Hc + 1
    gy = (rows / (Ha + 1) * 2 - 1)[:, None].expand(Hc, Wc)
    gx = (jf / (Wl - 1) * 2 - 1)[None, :].expand(Hc, Wc)
    return F.grid_sample(tp, torch.stack([gx, gy], -1)[None], mode="bilinear", padding_mode="border", align_corners=True)[0]


# ---------------------------------------------------------------- model

class PConv(nn.Conv2d):
    """3x3 conv whose padding wraps rows (around the tube) and repeats columns (along), or (wrap=False,
    for training patches) repeats both."""
    wrap = True

    def __init__(self, ci, co, stride=1):
        super().__init__(ci, co, 3, stride=stride, padding=0)

    def forward(self, x):
        if PConv.wrap:
            x = torch.cat([x[..., -1:, :], x, x[..., :1, :]], -2)
            x = F.pad(x, (1, 1, 0, 0), mode="replicate")
        else:
            x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return super().forward(x)


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = PConv(c, c), PConv(c, c)

    def forward(self, x):
        return x + self.b(F.silu(self.a(F.silu(x))))


class VAE(nn.Module):
    """nq = 0: a VAE (encode gives mean and log variance); nq > 0: a VQ-VAE with nq codes (encode gives the
    continuous latents and None; quantize snaps them to the nearest code, straight through), so any latent
    map decodes to an arrangement of the codes the tops were made of."""

    def __init__(self, zc=8, w=(32, 64, 96), ci=4, nq=0):
        super().__init__()
        self.cfg = dict(zc=zc, w=list(w), ci=ci, nq=nq)
        self.nq = nq
        if nq:
            self.code = nn.Parameter(torch.randn(nq, zc) * 0.1)
        c1, c2, c3 = w
        self.enc = nn.Sequential(PConv(ci, c1), nn.SiLU(), PConv(c1, c2, 2), nn.SiLU(), Res(c2), PConv(c2, c3, 2), nn.SiLU(), Res(c3), PConv(c3, (1 if nq else 2) * zc))
        self.dec = nn.Sequential(PConv(zc, c3), Res(c3), Res(c3), nn.Upsample(scale_factor=2), PConv(c3, c2), Res(c2),
                                 nn.Upsample(scale_factor=2), PConv(c2, c1), nn.SiLU(), PConv(c1, c1), nn.SiLU(), PConv(c1, ci))

    def encode(self, x):
        """skin (B, 4, H, W) in [0, 1]: albedo and relief (tanh(bump) + 1) / 2 -> (mu, logvar), (B, zc, H / 4, W / 4)"""
        if self.nq:
            return self.enc(x * 2 - 1), None
        return self.enc(x * 2 - 1).chunk(2, 1)

    def quantize(self, z):
        """VQ: (straight-through quantized z, code indices (B, h, w), codebook loss, commitment loss)"""
        B, C, h, w = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, C)
        idx = torch.cdist(zf, self.code).argmin(1)
        q = self.code[idx].reshape(B, h, w, C).permute(0, 3, 1, 2)
        return z + (q - z).detach(), idx.reshape(B, h, w), ((q - z.detach()) ** 2).mean(), ((z - q.detach()) ** 2).mean()

    def decode(self, z):
        """-> logits (B, 4, 4h, 4w): the paint's (tex0) and twice the relief's (bump = logit / 2, as 2 sigmoid(v) - 1 = tanh(v / 2))"""
        return self.dec(z)


def load_vae(path, dev):
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = VAE(**d["cfg"]).to(dev)
    m.load_state_dict(d["state"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, d


class SkinPrior(nn.Module):
    """objphys.py --skin-prior: each body's skin is decoded from its own latent canvas z_k (zc, Hc/4, Wc/4),
    the trained parameters. init(logit, bump, sph): charts from the rest pose, and z encoded from the
    current paint logits (K, 3, Ha, Wl) and relief parameters (K, Hb, Wb), read into each canvas.
    decode(): (tex0 logits (K, 3, Ha, Wl), bump parameters (K, Hb, Wb)). kl(): the latents' distance from
    the prior N(0, 1), per element (VQ: from their nearest codes)."""

    def __init__(self, path, dev):
        super().__init__()
        self.vae, d = load_vae(path, dev)
        self.delta, self.alpha = d["delta"], d["alpha"]
        self.dev = dev

    def init(self, logit, bump, sph):
        K, _, Ha, Wl = logit.shape
        self.Ha, self.Wl, self.Hb, self.Wb = Ha, Wl, bump.shape[1], bump.shape[2]
        self.ch = charts(sph["s"], sph["r"], sph["s_lo"], sph["s_len"], Wl, self.delta, self.alpha, wmult=64)
        jb = torch.linspace(0, Wl - 1, self.Wb, device=logit.device)
        self.cb = []
        PConv.wrap = True
        zs = []
        with torch.no_grad():
            bu = F.interpolate(torch.cat([bump[:, -1:], bump, bump[:, :1]], 1)[:, None], size=(Ha + 2 * Ha // self.Hb, Wl), mode="bilinear", align_corners=False)
            bu = bu[:, :, Ha // self.Hb:Ha // self.Hb + Ha]
            for k, (Hc, Wc, c, _) in enumerate(self.ch):
                j0 = jb.floor().long().clamp(max=Wl - 2)
                self.cb.append(c[j0] + (jb - j0) * (c[j0 + 1] - c[j0]))
                x = unwrap_to_canvas(torch.cat([torch.sigmoid(logit[k]), (torch.tanh(bu[k]) + 1) / 2]), c, Hc, Wc)[None]
                zs.append(self.vae.encode(x)[0][0])
        self.z = nn.ParameterList([nn.Parameter(z_) for z_ in zs])

    def decode_k(self, k):
        PConv.wrap = True
        z = self.z[k][None]
        if self.vae.nq:  # snapped to the codes, the gradient passing straight through
            z = self.vae.quantize(z)[0]
        x = self.vae.decode(z)[0]
        return canvas_to_unwrap(x[:3], self.ch[k][2], self.Ha), canvas_to_unwrap(x[3:], self.cb[k], self.Hb)[0] / 2

    def decode(self):
        t, b = zip(*[self.decode_k(k) for k in range(len(self.ch))])
        out = torch.stack(t), torch.stack(b)
        self._flush()
        return out

    def _flush(self):  # the canvases come in dozens of shapes: hand their cached buffers back every step
        if str(self.dev).startswith("mps"):
            torch.mps.empty_cache()

    def backward(self, gt, gb):
        """Push the gradients on the decoded tex0 (K, 3, Ha, Wl) and bump (K, Hb, Wb) (either None) into the
        latents, body by body (one body's decoder graph alive at a time)."""
        for k in range(len(self.ch)):
            t, b = self.decode_k(k)
            tot = 0
            if gt is not None:
                tot = tot + (t * gt[k]).sum()
            if gb is not None:
                tot = tot + (b * gb[k]).sum()
            if torch.is_tensor(tot):
                tot.backward()
        self._flush()

    def kl(self):
        if self.vae.nq:  # VQ: the commitment, how far the latents sit from their codes
            return torch.stack([self.vae.quantize(z_[None])[3] for z_ in self.z]).mean()
        return torch.stack([(z_ ** 2).mean() for z_ in self.z]).mean() * 0.5

    def state(self):
        return dict(z=[z_.detach().cpu() for z_ in self.z], delta=self.delta, alpha=self.alpha, charts=[(h, w, c.cpu(), rr) for h, w, c, rr in self.ch])


# ---------------------------------------------------------------- stages

def sharp_mask(ck, fp, dev):
    """(Mixed bodies, paint logits (K, 3, Ha, Wl), sharp (K, 1, Ha, Wl), rest spheres): texels the rest
    view shows at <= fp texels per still px, as objphys.py --spec-fill finds them."""
    import sys
    from objtube import Mixed, Rig, Cam, build_pairs, render
    ta = argparse.Namespace(**ck["args"])
    ta.chunk = 60000
    tb = Mixed.from_state(ck["tubes"], ta, dev)
    rig = Rig(0.0, 0.8, ta, dev)
    rig.load_state_dict(ck["rig"])
    S = 512
    ta2 = argparse.Namespace(**vars(ta)); ta2.mip = 1
    Z2 = 2 * S
    cam2 = Cam(np.eye(3), Z2, Z2, (S / 2, S / 2, 0.0), S, zoom=2.0)
    with torch.no_grad():
        Kt, _, Ha, Wl = tb.tex0.shape
        k_ = Ha // tb.tex.shape[2]
        up = F.interpolate(torch.cat([tb.tex[:, :, -1:], tb.tex, tb.tex[:, :, :1]], 2), size=(Ha + 2 * k_, Wl), mode="bilinear", align_corners=False)
        logit = tb.tex0 + up[:, :, k_:k_ + Ha]
        sp = tb.spheres()
        o = render(tb, sp, cam2, build_pairs(sp, cam2, 4.0), [rig()], ta2, aux=True)
        pix_, obj_ = o["pix"], o["obj"]
        share = o["ep"][0] / o["den"][0][pix_]
        top_ = torch.zeros(Z2 * Z2, device=dev).scatter_reduce(0, pix_, share, "amax", include_self=True)
        dom = (share >= top_[pix_] - 1e-9) & (share > 0.3)
        fp_ = torch.maximum(o["fpa"][0] * Ha, o["fpl"][0] * (Wl - 1)) * 2.0
        idx = (obj_ * Ha + (o["tu"][0] * Ha).long() % Ha) * Wl + (o["tv"][0].clamp(0, 1) * (Wl - 1)).round().long()
        best = torch.full((Kt * Ha * Wl,), 1e3, device=dev).scatter_reduce(0, idx[dom], fp_[dom], "amin", include_self=True)
        q = (best.reshape(Kt, 1, Ha, Wl) <= fp).float()
        sharp = (F.avg_pool2d(F.pad(torch.cat([q[:, :, -1:], q, q[:, :, :1]], 2), (1, 1, 0, 0), mode="replicate"), 3, stride=1) > 0.4).float()
    return tb, logit, sharp, sp


def cut_patches(alb, sharp, sph, ps=32, delta=0.5, stride=6, keep=0.9):
    """Patches (M, 3, ps, ps) of albedo alb (K, 3, Ha, Wl) in each centre's local isotropic chart (delta px
    per patch texel), at least `keep` sharp, centres on a grid of `stride` texels over the tube and the
    caps' outer quarter (inside it the disc takes over)."""
    K, _, Ha, Wl = alb.shape
    rr, sg = radius_along(sph["s"], sph["r"], sph["s_lo"], sph["s_len"], Wl)
    out, src = [], []
    d = (torch.arange(ps).float() - (ps - 1) / 2) * delta  # px offsets
    for k in range(K):
        L, r0, re = float(sph["s"][k, -1]), float(sph["r"][k, 0]), float(sph["r"][k, -1])
        dpx = float(sph["s_len"][k]) / (Wl - 1)  # px per column
        xm = torch.cat([alb[k], sharp[k]], 0)
        xp = torch.cat([xm[:, -Ha // 2:], xm, xm[:, :Ha // 2]], 1)[None]  # rows wrap (half a turn of slack)
        ii, jj = torch.meshgrid(torch.arange(0, Ha, stride), torch.arange(0, Wl, stride), indexing="ij")
        ii, jj = ii.flatten(), jj.flatten()
        ok = (sharp[k, 0, ii, jj] > 0) & (sg[k, jj] > -0.3 * r0) & (sg[k, jj] < L + 0.3 * re)
        ii, jj = ii[ok], jj[ok]
        if len(ii) == 0:
            continue
        rows = ii[:, None, None].float() + (d[None, :, None] * Ha / (2 * math.pi * rr[k, jj]).clamp(min=1e-3)[:, None, None]) + Ha // 2
        cols = jj[:, None, None].float() + (d[None, None, :] / dpx)
        rows, cols = rows.expand(-1, ps, ps), cols.expand(-1, ps, ps)
        g = torch.stack([cols / (Wl - 1) * 2 - 1, rows / (2 * Ha - 1) * 2 - 1], -1)
        smp = F.grid_sample(xp.expand(len(ii), -1, -1, -1), g, mode="bilinear", padding_mode="zeros", align_corners=True)
        inb = ((cols >= 0) & (cols <= Wl - 1)).float().mean((1, 2))
        good = (smp[:, -1].mean((1, 2)) >= keep) & (inb > 0.999)
        out.append(smp[good, :-1])
        src.append(torch.stack([torch.full_like(ii[good], k), ii[good], jj[good]], 1))
    return torch.cat(out), torch.cat(src)


def sheet(x, n, path, pad=2):
    """Grid of up to n images x (M, 3, h, w) in [0, 1]."""
    x = x[:n].detach().clamp(0, 1).cpu()
    M, _, h, w = x.shape
    cols = int(math.ceil(math.sqrt(M)))
    rows = int(math.ceil(M / cols))
    g = torch.ones(3, rows * (h + pad) + pad, cols * (w + pad) + pad)
    for i in range(M):
        r_, c_ = divmod(i, cols)
        g[:, pad + r_ * (h + pad):pad + r_ * (h + pad) + h, pad + c_ * (w + pad):pad + c_ * (w + pad) + w] = x[i]
    Image.fromarray((g.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path)


def dihedral(x, g):
    """One of the 8 symmetries of the square, per batch (same for all here)."""
    if g & 1:
        x = x.flip(-1)
    if g & 2:
        x = x.flip(-2)
    if g & 4:
        x = x.transpose(-1, -2)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["data", "train"])
    ap.add_argument("--load", help="data: objtube/objphys checkpoint (tube.pt)")
    ap.add_argument("--data", help="train: patches.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp", type=float, default=4.0, help="data: sharp = at most this many texels per still px at rest")
    ap.add_argument("--delta", type=float, default=0.5, help="px on the surface per rectified texel")
    ap.add_argument("--alpha", type=float, default=0.5, help="chart: 1 conformal (scale follows the radius), 0 scale kept")
    ap.add_argument("--ps", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beta", type=float, default=0.003, help="KL weight (per latent element, against L1 per texel)")
    ap.add_argument("--zc", type=int, default=8)
    ap.add_argument("--nq", type=int, default=0, help="> 0: a VQ-VAE with this many codes instead of a VAE")
    ap.add_argument("--wgrad", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)
    cpu = torch.device("cpu")
    if a.stage == "data":
        ck = torch.load(a.load, map_location="cpu", weights_only=False)
        tb, logit, sharp, sp = sharp_mask(ck, a.fp, cpu)
        log(f"rest render: {float(sharp.mean()):.1%} of texels sharp (<= {a.fp} texels per still px)")
        alb = torch.sigmoid(logit)
        Ha, Wl = alb.shape[2:]
        bm = tb.bump.detach()  # relief on its own (coarser) grid of the same span, to the paint's
        kb = Ha // bm.shape[1]
        bu = F.interpolate(torch.cat([bm[:, -1:], bm, bm[:, :1]], 1)[:, None], size=(Ha + 2 * kb, Wl), mode="bilinear", align_corners=False)[:, :, kb:kb + Ha]
        P, src = cut_patches(torch.cat([alb, (torch.tanh(bu) + 1) / 2], 1), sharp, sp, a.ps, a.delta)
        log(f"{len(P)} patches of {a.ps} x {a.ps} at {a.delta} px per texel, from {len(src[:, 0].unique())} bodies")
        torch.save(dict(patches=P, src=src, delta=a.delta, alpha=a.alpha, load=a.load), f"{a.out}/patches.pt")
        g = torch.Generator().manual_seed(0)
        pi = torch.randperm(len(P), generator=g)[:144]
        sheet(P[pi, :3], 144, f"{a.out}/patches.png")
        sheet(P[pi, 3:].expand(-1, 3, -1, -1), 144, f"{a.out}/patches_relief.png")
        # raw unwraps (paint and sharp mask) of a few bodies, for comparison
        pk = list(range(0, alb.shape[0], max(1, alb.shape[0] // 6)))[:6]
        sheet(torch.cat([alb[pk], sharp[pk].expand(-1, 3, -1, -1)]), 12, f"{a.out}/unwraps.png")
        return
    d = torch.load(a.data)
    P = d["patches"]
    n_val = max(64, len(P) // 20)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(P), generator=g)
    Pv, Pt = P[perm[:n_val]], P[perm[n_val:]]
    m = VAE(a.zc, ci=P.shape[1], nq=a.nq)
    if a.nq:  # codes start as encodings of random patches
        with torch.no_grad():
            z0 = m.encode(Pt[torch.randperm(len(Pt), generator=g)[:64]])[0].permute(0, 2, 3, 1).reshape(-1, a.zc)
            m.code.copy_(z0[torch.randperm(len(z0), generator=g)[:a.nq]])
    PConv.wrap = False
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    log(f"train {len(Pt)} / val {len(Pv)} patches; {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params")
    rng = torch.Generator().manual_seed(1)
    cr = 4  # loss on the interior (the patch border's padding is not the canvas's)

    usage = torch.zeros(max(a.nq, 1))

    def losses(x, sample=True):
        mu, lv = m.encode(x)
        if a.nq:  # kl: codebook + 0.25 commitment
            z, idx, cb, cm = m.quantize(mu)
            usage.mul_(0.99).index_add_(0, idx.flatten(), torch.full((idx.numel(),), 0.01))
            kl = cb + 0.25 * cm
        else:
            lv = lv.clamp(-10, 4)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if sample else mu
            kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).mean()
        y = torch.sigmoid(m.decode(z))
        xi, yi = x[..., cr:-cr, cr:-cr], y[..., cr:-cr, cr:-cr]
        l1 = (yi - xi).abs().mean()
        gr = ((yi.diff(dim=-1) - xi.diff(dim=-1)).abs().mean() + (yi.diff(dim=-2) - xi.diff(dim=-2)).abs().mean())
        return l1, gr, kl, y

    for it in range(a.steps):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * it / a.steps))
        idx = torch.randint(0, len(Pt), (a.bs,), generator=rng)
        x = dihedral(Pt[idx], int(torch.randint(0, 8, (1,), generator=rng)))
        x = x * (1 + 0.1 * (torch.rand(a.bs, 1, 1, 1, generator=rng) - 0.5))  # slight brightness jitter
        l1, gr, kl, _ = losses(x.clamp(0, 1))
        beta = 1.0 if a.nq else a.beta * min(1.0, it / (0.2 * a.steps))  # warm up the KL
        loss = l1 + a.wgrad * gr + beta * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        if a.nq and it % 200 == 199 and it < 0.8 * a.steps:  # dead codes restart at current encodings
            with torch.no_grad():
                dead = (usage < 1e-3).nonzero()[:, 0]
                if len(dead):
                    ze = m.encode(x.clamp(0, 1))[0].permute(0, 2, 3, 1).reshape(-1, a.zc)
                    m.code[dead] = ze[torch.randint(0, len(ze), (len(dead),), generator=rng)] + 0.01 * torch.randn(len(dead), a.zc, generator=rng)
                    usage[dead] = 0.01
        if it % 500 == 0 or it == a.steps - 1:
            with torch.no_grad():
                vl1, vgr, vkl, _ = losses(Pv, sample=False)
            log(f"step {it} l1 {float(l1):.4f} grad {float(gr):.4f} kl {float(kl):.3f} | val l1 {float(vl1):.4f} grad {float(vgr):.4f} kl {float(vkl):.3f}"
                + (f" | codes used {int((usage > 1e-3).sum())}/{a.nq}" if a.nq else ""))
    torch.save(dict(cfg=m.cfg, state=m.state_dict(), delta=d["delta"], alpha=d["alpha"], args=vars(a)), f"{a.out}/model.pt")
    m.eval()
    with torch.no_grad():
        _, _, _, y = losses(Pv[:48], sample=False)
        rgbh = lambda q: torch.cat([q[:, :3], q[:, 3:].expand(-1, 3, -1, -1)], -1)  # paint | relief
        sheet(torch.stack([rgbh(Pv[:48]), rgbh(y)], 1).flatten(0, 1), 96, f"{a.out}/recon.png")
        h = a.ps // 4
        if a.nq:  # random latents: codes drawn independently by how often the tops use them
            rz = lambda n, hh, ww: m.code[torch.multinomial(usage / usage.sum(), n * hh * ww, replacement=True, generator=rng)].reshape(n, hh, ww, a.zc).permute(0, 3, 1, 2)
        else:
            rz = lambda n, hh, ww: torch.randn(n, a.zc, hh, ww, generator=rng)
        sheet(rgbh(torch.sigmoid(m.decode(rz(64, h, h)))), 64, f"{a.out}/random.png")
        PConv.wrap = True  # a canvas-sized decode: wraps around (rows), as on a body
        big = rgbh(torch.sigmoid(m.decode(rz(4, 40, 64))))
        sheet(torch.cat([big, big], 2), 4, f"{a.out}/random_canvas.png")  # stacked twice: the wrap shows no seam
    log("done")


if __name__ == "__main__":
    main()
