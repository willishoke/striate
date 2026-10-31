"""Cell-aware warp loop: the render's own cells drift and breathe, its walls deform between them.

Input: a render (spectrum .pt) and its segmentation (labels.npz with dom, cells). Every cell c gets
periodic paths (a few Fourier harmonics over the loop): a drift d_c(t) in px on top of its domain's drift
D_dom(t), and a breathing scale s_c(t). A pixel x belongs softly to the cells near it (weights from the
distance to each cell's region: 1 inside, blended across the walls, more widely across the membrane band),
and the frame is the base image sampled at x + sum_c w_c(x) (D + d_c - s_c (x - cen_c)): cell interiors move
rigidly and grow or shrink about their centroid; the walls take the strain.

Losses: channel activation of the warped frames; rms amplitude of each path held near a target (so the
loop keeps moving); breathing zero-sum within each domain (one cell grows, its neighbours give way);
drift coherence between neighbours. Amplitudes are hard-capped by projection after every step.

Usage: cellwarp.py --base assets/base.pt --labels assets/labels_4e198.npz --target inception4e:198 --out DIR
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
from striate.util import pick_device
from zoo import MultiScaleCrop
from ring import Ring
from cells import wrap, sample, Path, blur


def membership(cells, band, S, sig_wall, sig_band, K, R=16):
    """Per-pixel top-K soft memberships. Weight of cell c at x: exp(-d(x, c)^2 / 2 sigma^2), d = distance
    to c's region (periodic), sigma = sig_band inside the membrane band else sig_wall. Returns idx (S,S,K),
    w (S,S,K) normalised, cell centroids (nC, 2) in (x, y), cell areas."""
    nC = int(cells.max())
    Lp = np.pad(cells, R, mode="wrap")
    sig = np.where(band, sig_band, sig_wall).astype(np.float32)
    objs = ndi.find_objects(Lp)
    px, cl, wt = [], [], []
    yy, xx = np.mgrid[0 : S + 2 * R, 0 : S + 2 * R]
    for c, sl in enumerate(objs, 1):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - R, 0), min(sl[0].stop + R, S + 2 * R)
        x0, x1 = max(sl[1].start - R, 0), min(sl[1].stop + R, S + 2 * R)
        crop = Lp[y0:y1, x0:x1]
        d = ndi.distance_transform_edt(crop != c)
        ys, xs = np.nonzero(d < R)
        gy, gx = (ys + y0 - R) % S, (xs + x0 - R) % S
        w = np.exp(-(d[ys, xs] ** 2) / (2 * sig[gy, gx] ** 2))
        keep = w > 1e-3
        px.append(gy[keep] * S + gx[keep])
        cl.append(np.full(keep.sum(), c))
        wt.append(w[keep])
    px, cl, wt = np.concatenate(px), np.concatenate(cl), np.concatenate(wt)
    order = np.lexsort((-wt, cl, px))
    px, cl, wt = px[order], cl[order], wt[order]
    first = np.ones(len(px), bool)
    first[1:] = (px[1:] != px[:-1]) | (cl[1:] != cl[:-1])
    px, cl, wt = (
        px[first],
        cl[first],
        wt[first],
    )  # a pixel seen twice through the wrap keeps its best weight
    order = np.lexsort((-wt, px))
    px, cl, wt = px[order], cl[order], wt[order]
    start = np.searchsorted(px, np.arange(S * S))
    rank = np.arange(len(px)) - start[px]
    keep = rank < K
    idx = np.zeros((S * S, K), np.int64)
    W = np.zeros((S * S, K), np.float32)
    idx[px[keep], rank[keep]] = cl[keep] - 1
    W[px[keep], rank[keep]] = wt[keep]
    W /= W.sum(1, keepdims=True).clip(1e-6)
    # centroids (circular mean) and areas
    ys, xs = np.mgrid[0:S, 0:S]
    ang = 2 * np.pi / S
    cen = np.zeros((nC, 2))
    for k, g in enumerate((xs, ys)):
        cs = ndi.sum(np.cos(ang * g), cells, range(1, nC + 1))
        sn = ndi.sum(np.sin(ang * g), cells, range(1, nC + 1))
        cen[:, k] = np.arctan2(sn, cs) / ang % S
    area = np.bincount(cells.ravel(), minlength=nC + 1)[1:].astype(np.float32)
    return idx.reshape(S, S, K), W.reshape(S, S, K), cen.astype(np.float32), area


def adjacency(cells):
    a = np.stack([cells, np.roll(cells, 1, 0)], -1).reshape(-1, 2)
    b = np.stack([cells, np.roll(cells, 1, 1)], -1).reshape(-1, 2)
    e = np.concatenate([a, b])
    e = e[e[:, 0] != e[:, 1]]
    e = np.sort(e, 1)
    e, n = np.unique(e, axis=0, return_counts=True)
    return e - 1, n.astype(np.float32)


class Warp:
    def __init__(self, idx, W, cen, area, dom_of, nD, S, dev, K):
        self.S, self.dev = S, dev
        self.idx = torch.from_numpy(idx).to(dev)
        self.W = torch.from_numpy(W).to(dev)
        self.cen = torch.from_numpy(cen).to(dev)
        self.area = torch.from_numpy(area).to(dev)
        self.dom_of = torch.from_numpy(dom_of).to(dev)
        nC = len(cen)
        g = torch.stack(
            torch.meshgrid(
                torch.arange(S, dtype=torch.float32),
                torch.arange(S, dtype=torch.float32),
                indexing="xy",
            ),
            -1,
        ).to(dev)
        self.grid = g  # (S, S, 2) as (x, y)
        self.rel = wrap(g[:, :, None] - self.cen[self.idx], S)  # (S, S, K, 2)
        self.D = Path(torch.zeros(nD, 2), K, dev)
        self.d = Path(torch.zeros(nC, 2), K, dev)
        self.s = Path(torch.zeros(nC, 1), K, dev)
        self.h = Path(torch.zeros(nC, 1), K, dev)  # hue rotation about the grey axis (rad)
        self.o = Path(torch.zeros(nC, 2), K, dev)  # scroll of the fine texture inside the core (px)
        for p in (self.D, self.d, self.s, self.h, self.o):
            p.c0.requires_grad_(False)  # no DC: a constant shift is just a different still
        self.K = K
        self.use_hue = self.use_scroll = False
        self.core = ((self.W[..., 0] - 0.6) / 0.3).clamp(
            0, 1
        )  # 1 deep inside a cell, 0 within ~3 px of a wall

    def params(self):
        return (
            [self.D.cc, self.d.cc, self.s.cc]
            + ([self.h.cc] if self.use_hue else [])
            + ([self.o.cc] if self.use_scroll else [])
        )

    def pix(self, path, u):
        """Membership-weighted per-pixel value of a per-cell path: (B, S, S, D)."""
        B = u.shape[0]
        v = path(u[:, None].expand(B, path.c0.shape[0]))
        return (self.W[..., None] * v[:, self.idx]).sum(3)

    @staticmethod
    def rotate_hue(v, theta):
        """Rotate RGB logits (B, 3, S, S) about the grey axis by theta (B, S, S)."""
        k = 1 / math.sqrt(3)
        c, s = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
        r, g, b = v[:, 0:1], v[:, 1:2], v[:, 2:3]
        cross = torch.cat([b - g, r - b, g - r], 1) * k
        dot = (r + g + b) * k
        return v * c + cross * s + k * dot * (1 - c)

    def paths(self, u):
        """u (B,) -> drift (B, nC, 2), scale (B, nC, 1)."""
        B = u.shape[0]
        Dd = self.D(u[:, None].expand(B, self.D.c0.shape[0]))[:, self.dom_of] + self.d(
            u[:, None].expand(B, self.d.c0.shape[0])
        )
        s = self.s(u[:, None].expand(B, self.s.c0.shape[0]))
        return Dd, s

    def field(self, u):
        Dd, s = self.paths(u)
        v = Dd[:, self.idx] - s[:, self.idx] * self.rel  # (B, S, S, K, 2)
        return (self.W[..., None] * v).sum(3), Dd, s  # (B, S, S, 2)

    def render(self, u, texp, hf=None, res=None, cubic=True):
        """hf: fine texture (3, S, S) to scroll inside cores; res: callable u -> (3, S, S) time-varying residual
        added to the scrolled fine texture. Both only act where core > 0."""
        f, Dd, s = self.field(u)
        uv = self.grid + f
        out = sample(texp, uv, cubic).permute(1, 0, 2, 3)  # (B, 3, S, S)
        if self.use_hue:
            out = self.rotate_hue(out, self.pix(self.h, u)[..., 0])
        if hf is not None and (self.use_scroll or res is not None):
            o = self.pix(self.o, u) if self.use_scroll else torch.zeros_like(uv)
            still = sample(hf, uv, False).permute(1, 0, 2, 3)
            moved = []
            for b in range(u.shape[0]):
                tex = hf + res(u[b]) if res is not None else hf
                moved.append(sample(tex, uv[b] + o[b], False))
            out = out + self.core * (torch.stack(moved) - still)
        return out, Dd, s

    def init(self, tgt_D, tgt_d, tgt_s, seed, tgt_h=0.0, tgt_o=0.0):
        g = torch.Generator().manual_seed(seed)
        for p, tgt in (
            (self.D, tgt_D),
            (self.d, tgt_d),
            (self.s, tgt_s),
            (self.h, tgt_h),
            (self.o, tgt_o),
        ):
            cc = torch.randn(p.cc.shape, generator=g) / math.sqrt(
                self.K
            )  # rms over the loop ~ 1 per coordinate
            cc *= tgt * (0.6 ** torch.arange(self.K).view(1, -1, 1, 1))  # gentler high harmonics
            p.cc.data.copy_(cc.to(self.dev))

    def rms(self, p, n=32):
        u = torch.arange(n, device=self.dev) / n
        x = p(u[:, None].expand(n, p.c0.shape[0]))  # (n, N, D)
        return (x**2).sum(-1).mean(0).sqrt()  # (N,)

    @torch.no_grad()
    def cap(self, amp_D, amp_d, amp_s, amp_h=1e9, amp_o=1e9, n=64):
        """Hard amplitude cap: scale each slot's harmonics so its path never exceeds amp."""
        u = torch.arange(n, device=self.dev) / n
        for p, amp in (
            (self.D, amp_D),
            (self.d, amp_d),
            (self.s, amp_s),
            (self.h, amp_h),
            (self.o, amp_o),
        ):
            x = p(u[:, None].expand(n, p.c0.shape[0]))
            mx = (x**2).sum(-1).max(0).values.sqrt()
            p.cc.mul_((amp / mx.clamp(min=1e-9)).clamp(max=1.0).view(-1, 1, 1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--K", type=int, default=3, help="harmonics per path")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--sig-wall", type=float, default=3.0)
    ap.add_argument("--sig-band", type=float, default=7.0)
    ap.add_argument("--amp-D", type=float, default=6.0, help="cap on domain drift (px)")
    ap.add_argument(
        "--amp-d", type=float, default=2.5, help="cap on cell drift relative to its domain (px)"
    )
    ap.add_argument(
        "--amp-s", type=float, default=0.08, help="cap on breathing (fraction of radius)"
    )
    ap.add_argument(
        "--hue", type=float, default=0.0, help="target rms hue rotation per cell (rad); 0 off"
    )
    ap.add_argument("--hue-cap", type=float, default=0.3)
    ap.add_argument("--whue", type=float, default=2.0, help="hue zero-sum per domain")
    ap.add_argument(
        "--scroll",
        type=float,
        default=0.0,
        help="target rms scroll of the fine texture inside cores (px); 0 off",
    )
    ap.add_argument("--scroll-cap", type=float, default=4.0)
    ap.add_argument(
        "--hf-sigma", type=float, default=4.0, help="fine texture = base logits minus this blur"
    )
    ap.add_argument(
        "--res",
        type=float,
        default=0.0,
        help="target rms of the time-varying residual, as a fraction of the fine texture rms; 0 off",
    )
    ap.add_argument("--res-K", type=int, default=2)
    ap.add_argument("--res-lr", type=float, default=0.02)
    ap.add_argument("--wres", type=float, default=2.0)
    ap.add_argument("--tgt-D", type=float, default=3.0)
    ap.add_argument("--tgt-d", type=float, default=1.2)
    ap.add_argument("--tgt-s", type=float, default=0.04)
    ap.add_argument("--wamp", type=float, default=0.5)
    ap.add_argument("--warea", type=float, default=2.0, help="breathing zero-sum per domain")
    ap.add_argument(
        "--wcoh", type=float, default=0.2, help="drift coherence between neighbouring cells"
    )
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--no-train", action="store_true", help="render the random init only")
    ap.add_argument(
        "--paths",
        default="",
        help="render the paths saved in a warp.pt (trained with the same flags) instead of training",
    )
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)

    layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
    net = Backbone(a.net, dev)
    net.hook(layer)
    spec0 = torch.load(a.base, map_location="cpu")
    S = spec0.shape[2]
    rg = Ring(net, layer, ch, S, dev)
    T = a.frames

    z = np.load(a.labels)
    dom, cells = z["dom"].astype(np.int64), z["cells"].astype(np.int64)
    band = z["band"] if "band" in z else np.zeros_like(cells, bool)
    nD, nC = int(dom.max()), int(cells.max())
    dom_of = np.array(ndi.maximum(dom, cells, range(1, nC + 1))).astype(np.int64) - 1
    cache = a.labels.replace(".npz", f"_m{a.topk}_{a.sig_wall:g}_{a.sig_band:g}.npz")
    if os.path.exists(cache):
        m = np.load(cache)
        idx, W, cen, area = m["idx"], m["W"], m["cen"], m["area"]
    else:
        idx, W, cen, area = membership(cells, band, S, a.sig_wall, a.sig_band, a.topk)
        np.savez_compressed(cache, idx=idx, W=W, cen=cen, area=area)
    edges, elen = adjacency(cells)
    log(f"{nD} domains, {nC} cells, {len(edges)} walls; membership ready; band {band.mean():.2f}")

    wp = Warp(idx, W, cen, area, dom_of, nD, S, dev, a.K)
    wp.use_hue, wp.use_scroll = a.hue > 0, a.scroll > 0
    wp.init(a.tgt_D, a.tgt_d, a.tgt_s, a.seed, a.hue, a.scroll)
    caps = (a.amp_D, a.amp_d, a.amp_s, a.hue_cap, a.scroll_cap)
    wp.cap(*caps)
    loaded = torch.load(a.paths, map_location="cpu") if a.paths else {}
    for k, p in (("D", wp.D), ("d", wp.d), ("s", wp.s), ("h", wp.h), ("o", wp.o)):
        if k in loaded:
            p.cc.data.copy_(loaded[k].to(dev))
    E = torch.from_numpy(edges).to(dev)
    EL = torch.from_numpy(elen).to(dev)
    EL = EL / EL.mean()
    darea = torch.zeros(nD, device=dev).index_add_(0, wp.dom_of, wp.area)

    with torch.no_grad():
        texp0 = rg.pre(spec0.to(dev))[0]
        base = torch.sigmoid(texp0)[None]
        alone = 0
        for k in range(4):
            random.seed(1000 + k)
            alone += float(rg.act(MultiScaleCrop(224 / S, 512)(base))) / 4
        random.seed(a.seed)
        hf = (texp0 - blur(texp0, a.hf_sigma)) if (a.scroll > 0 or a.res > 0) else None
        hf_rms = float(hf.pow(2).mean().sqrt()) if hf is not None else 0.0
    log(f"base alone {alone:.3f}; fine texture rms {hf_rms:.3f}")
    # time-varying residual inside the cores: sum_k A_k cos(2 pi k u) + B_k sin(2 pi k u), A, B Fourier-parameterised images
    res_spec = None
    if a.res > 0:
        g_ = torch.Generator().manual_seed(a.seed + 1)
        res_spec = torch.nn.Parameter(
            (
                torch.randn((2 * a.res_K,) + tuple(spec0.shape[1:]), generator=g_)
                * spec0.std()
                * 0.1
            ).to(dev)
        )
    res_k = torch.arange(1, a.res_K + 1, device=dev, dtype=torch.float32)

    def res_maps():
        return rg.pre(res_spec)  # (2K, 3, S, S)

    def res_at(maps):
        def f(u):
            ang = 2 * math.pi * u * res_k
            return (
                maps[: a.res_K] * torch.cos(ang).view(-1, 1, 1, 1)
                + maps[a.res_K :] * torch.sin(ang).view(-1, 1, 1, 1)
            ).sum(0)

        return f

    def res_rms(maps):
        return (maps**2).sum(0).mean().sqrt() / math.sqrt(2)  # rms over the loop of the residual

    if res_spec is not None:
        with torch.no_grad():
            if loaded.get("res_spec") is not None:
                res_spec.copy_(loaded["res_spec"].to(dev))
            else:
                res_spec.mul_(a.res * hf_rms / float(res_rms(res_maps())))

    def evaluate(write, stride=1):
        acts, state = (
            [],
            random.getstate(),
        )  # fixed crops for scoring; training randomness resumes after
        ts_all = list(range(0, T, stride))
        with torch.no_grad():
            rf = res_at(res_maps()) if res_spec is not None else None
            for i in range(0, len(ts_all), a.mb):
                u = torch.tensor(ts_all[i : i + a.mb], device=dev, dtype=torch.float32) / T
                fr = torch.sigmoid(wp.render(u, texp0, hf, rf)[0])
                tot = 0
                for k in range(4):
                    random.seed(1000 + k)
                    tot = tot + rg.act(MultiScaleCrop(224 / S, 512)(fr)).cpu()
                acts.append(tot / 4)
                if write:
                    for t, f in zip(ts_all[i : i + a.mb], fr.cpu()):
                        Image.fromarray(
                            (f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                        ).save(f"{a.out}/f{t:04d}.png")
                torch.mps.synchronize() if str(dev).startswith("mps") else None
        random.setstate(state)
        return torch.cat(acts).numpy()

    act0 = evaluate(False, stride=max(1, T // 32))
    log(f"random init: act {act0.mean():.3f} [{act0.min():.3f}, {act0.max():.3f}]")
    groups = [{"params": wp.params(), "lr": a.lr}]
    if res_spec is not None:
        groups.append({"params": [res_spec], "lr": a.res_lr})
    opt = torch.optim.Adam(groups)
    base_lr = [g["lr"] for g in groups]
    curve = []
    steps = 0 if (a.no_train or a.paths) else a.steps
    for it in range(steps):
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * 0.5 * (1 + math.cos(math.pi * it / steps))
        opt.zero_grad()
        u = torch.tensor(random.sample(range(T), a.mb), device=dev, dtype=torch.float32) / T
        maps = res_maps() if res_spec is not None else None
        pre, Dd, s = wp.render(u, texp0, hf, res_at(maps) if maps is not None else None)
        fr = torch.sigmoid(pre)
        act = rg.act(MultiScaleCrop(224 / S, 512)(fr))
        # priors
        targets = (
            [(wp.D, a.tgt_D), (wp.d, a.tgt_d), (wp.s, a.tgt_s)]
            + ([(wp.h, a.hue)] if a.hue > 0 else [])
            + ([(wp.o, a.scroll)] if a.scroll > 0 else [])
        )
        amp = sum(((wp.rms(p) / tgt - 1) ** 2).mean() for p, tgt in targets) / len(targets)
        dsum = (
            torch.zeros(a.mb, nD, device=dev).index_add_(1, wp.dom_of, s[..., 0] * wp.area) / darea
        )
        area = ((dsum / a.tgt_s) ** 2).mean()
        dd = wp.d(u[:, None].expand(a.mb, nC))
        coh = ((((dd[:, E[:, 0]] - dd[:, E[:, 1]]) ** 2).sum(-1) / a.tgt_d**2) * EL).mean()
        pen = a.wamp * amp + a.warea * area + a.wcoh * coh
        hue0 = res0 = torch.zeros((), device=dev)
        if a.hue > 0:  # hue zero-sum per domain, area-weighted: the local mean colour stays put
            hh = wp.h(u[:, None].expand(a.mb, nC))[..., 0]
            hsum = torch.zeros(a.mb, nD, device=dev).index_add_(1, wp.dom_of, hh * wp.area) / darea
            hue0 = ((hsum / a.hue) ** 2).mean()
            pen = pen + a.whue * hue0
        if maps is not None:
            res0 = (res_rms(maps) / (a.res * hf_rms) - 1) ** 2
            pen = pen + a.wres * res0
        loss = (-act + act.detach().abs() * pen).sum()
        loss.backward()
        opt.step()
        wp.cap(*caps)
        if it % 50 == 0 or it == steps - 1:
            curve.append((it, float(act.mean())))
            log(
                f"step {it + 1}/{steps} act {act.mean():.3f} amp {float(amp):.3f} area {float(area):.3f} coh {float(coh):.3f} hue0 {float(hue0):.3f} res0 {float(res0):.3f} "
                f"rmsD {float(wp.rms(wp.D).mean()):.2f} rmsd {float(wp.rms(wp.d).mean()):.2f} rmss {float(wp.rms(wp.s).mean()):.3f} "
                f"rmsh {float(wp.rms(wp.h).mean()):.3f} rmso {float(wp.rms(wp.o).mean()):.2f}"
            )

    act = evaluate(True, a.eval_stride)
    torch.save(
        {
            "D": wp.D.cc.detach().cpu(),
            "d": wp.d.cc.detach().cpu(),
            "s": wp.s.cc.detach().cpu(),
            "h": wp.h.cc.detach().cpu(),
            "o": wp.o.cc.detach().cpu(),
            "res_spec": None if res_spec is None else res_spec.detach().cpu(),
            "args": vars(a),
        },
        f"{a.out}/warp.pt",
    )
    rep = dict(
        act_mean=float(act.mean()),
        act_min=float(act.min()),
        act_max=float(act.max()),
        act=act.tolist(),
        act_before=float(act0.mean()),
        base_alone=alone,
        frames=T,
        steps=steps,
        curve=curve,
        args=vars(a),
        secs=round(time.time() - t0),
        rmsD=float(wp.rms(wp.D).mean()),
        rmsd=float(wp.rms(wp.d).mean()),
        rmss=float(wp.rms(wp.s).mean()),
        rmsh=float(wp.rms(wp.h).mean()),
        rmso=float(wp.rms(wp.o).mean()),
        hf_rms=hf_rms,
        res_rms=float(res_rms(res_maps())) if res_spec is not None else 0.0,
    )
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(
        f"done act {act.mean():.3f} [{act.min():.3f}, {act.max():.3f}] (init {act0.mean():.3f}; base alone {alone:.2f})"
    )


if __name__ == "__main__":
    main()
