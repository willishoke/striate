"""Multiplane loop: a stack of periodic layers, each sliding rigidly at its own speed, composited
back to front with learned alpha (and an optional drop shadow from each layer onto those behind it).

frame_t = over(layer_L(x - v_L t), ..., layer_1(x - v_1 t), layer_0(x - v_0 t)),  v_l = drift_l * S / T

Every layer is translated by whole canvas widths per loop, so the loop closes exactly, and within a
layer nothing deforms: an object keeps its identity for the whole loop. Topology changes happen only
where they do physically, at occlusion boundaries, when a nearer layer passes over a farther one.
Translations are exact phase shifts of each layer's Fourier spectrum (no resampling blur). The layers
start from ring.py anchors and are optimized jointly so that the composite at every relative offset
along the loop drives the channel; alpha is pushed to a target coverage and to 0/1.

Usage: layers.py --anchors ring4c244/anchors.pt --net googlenet --target inception4c:244 \
         --drifts "1,0;2,0" --cover 0.4 --frames 1024 --out DIR
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
from warploop2 import Flow, warp

LUMA = [0.299, 0.587, 0.114]


def swirl_d(fl, ts, off):
    """warploop2.Flow displacement (N, 2, S, S) evaluated in a sliding layer's own coordinates
    (x - off), so each layer's swirl travels with it."""
    S = fl.S
    u = torch.arange(S, device=fl.dev, dtype=torch.float32)
    y, x = torch.meshgrid(u, u, indexing="ij")
    out = torch.zeros(len(ts), 2, S, S, device=fl.dev)
    for j in range(len(fl.k)):
        kx, ky = fl.k[j]
        th = (
            2
            * math.pi
            * (kx * (x[None] - off[:, 0, None, None]) + ky * (y[None] - off[:, 1, None, None]))
            / S
        )
        ph = 2 * math.pi * fl.n[j] * ts / fl.T + fl.phi[j]
        s = torch.sin(th + ph[:, None, None])
        w = 2 * math.pi * fl.c[j] / S
        out[:, 0] -= w * ky * s
        out[:, 1] += w * kx * s
    return out


def warp_each(x, d):
    return torch.cat([warp(x[i : i + 1], d[i : i + 1]) for i in range(len(x))])


class Stack:
    def __init__(
        self,
        rg,
        specs,
        drifts,
        cover,
        T,
        dev,
        cut=0.0,
        shadow=0.0,
        shadow_off=(6, 9),
        shadow_sigma=5.0,
        amax=0.0,
        amin=0.0,
    ):
        S = rg.size
        self.rg, self.S, self.T, self.dev = rg, S, T, dev
        self.specs = [torch.nn.Parameter(s.clone().to(dev)) for s in specs]
        self.v = torch.tensor(drifts, dtype=torch.float32, device=dev) * S / T  # (L, 2) px/frame
        fy = torch.fft.fftfreq(S, device=dev).view(-1, 1)
        fx = torch.fft.rfftfreq(S, device=dev).view(1, -1)
        self.KX, self.KY = fx * S, fy * S  # cycles per canvas
        f = torch.sqrt(fx * fx + fy * fy)
        self.mask = (f <= 1 / cut).float() if cut > 0 else None
        # alpha spectra: 1/f like the colour layers, one channel per front layer
        self.ascale = (1.0 / torch.clamp(f, min=1.0 / S)) * S / 4.0
        if (
            amax > 0
        ):  # alpha may not hold structure coarser than amax px (except its mean): objects, not continents
            self.ascale = self.ascale * ((f >= 1 / amax) | (f == 0)).float()
        if amin > 0:  # nor detail finer than amin px: no speckle, clean edges
            self.ascale = self.ascale * (f <= 1 / amin).float()
        self.alphas = [
            torch.nn.Parameter(torch.zeros(1, 1, S, S // 2 + 1, 2, device=dev)) for _ in specs[1:]
        ]
        self.cover, self.shadow = cover, shadow
        self.flows = []
        self.vol = None  # optional trainable back layer: a volslice.Volume swept through depth once per `vol_cycles` loops
        self.vol_cycles = 1
        self.back = None  # optional frozen back layer: a looping frame sequence (N, 3, S, S), played at 1 frame per frame
        self.sh_kernel = (
            torch.exp(-2 * (math.pi * shadow_sigma) ** 2 * f * f)
            * self._phase(torch.tensor([shadow_off], dtype=torch.float32, device=dev))[0]
        )

    def params(self):
        return self.specs, self.alphas

    def _phase(self, sh):
        """exp(-2 pi i k.sh / S) for shifts sh (N, 2) px: (N, S, S//2+1) complex."""
        a = (
            -2
            * math.pi
            * (sh[:, 0, None, None] * self.KX + sh[:, 1, None, None] * self.KY)
            / self.S
        )
        return torch.complex(torch.cos(a), torch.sin(a))

    def _color(self, spec, ph):
        z = torch.view_as_complex(
            (spec * self.rg.helper.spectrum_scale).contiguous()
        )  # (1, 3, S, W)
        if self.mask is not None:
            z = z * self.mask
        p = torch.fft.irfftn(z * ph[:, None], s=(self.S, self.S), dim=(2, 3)) / 4.0
        return self.rg.helper._decorrelate_color(p)

    def _alpha_pre(self, a, ph):
        z = torch.view_as_complex((a * self.ascale[None, None, ..., None]).contiguous())
        return torch.fft.irfftn(z * ph[:, None], s=(self.S, self.S), dim=(2, 3))

    def init_alpha(self, l, target):
        """Set front layer l's alpha logits (S, S tensor) exactly (inverse of _alpha_pre)."""
        z = torch.fft.rfftn(target.to(self.dev)[None, None], dim=(2, 3)) / torch.where(
            self.ascale > 0, self.ascale, torch.ones_like(self.ascale)
        )
        z = z * (self.ascale > 0)
        self.alphas[l - 1].data.copy_(torch.view_as_real(z))

    def render(self, ts, parts=False):
        """Composite frames at times ts (float tensor). Returns (frames, pres, alphas)."""
        pres, als = [], []
        out = None
        for l, spec in enumerate(self.specs):
            off = self.v[l][None] * ts[:, None]
            ph = self._phase(off)
            pre = self._color(spec, ph)
            d = swirl_d(self.flows[l], ts, off) if self.flows else None
            if d is not None:
                pre = warp_each(pre, d)
            pres.append(pre)
            rgb = torch.sigmoid(pre)
            if l == 0:
                if self.vol is not None:
                    z = (ts * self.vol_cycles / self.T % 1.0) * self.vol.Z
                    pres[-1] = self.vol.pre(z)
                    out = torch.sigmoid(pres[-1])
                elif self.back is not None:
                    idx = (ts.long() % len(self.back)).tolist()
                    out = self.back[idx].float()
                    pres.pop()
                else:
                    out = rgb
                continue
            ap_ = self._alpha_pre(self.alphas[l - 1], ph)
            al = torch.sigmoid(warp_each(ap_, d) if d is not None else ap_)
            als.append(al)
            if self.shadow > 0:
                z = torch.fft.rfftn(al, dim=(2, 3)) * self.sh_kernel
                out = out * (
                    1
                    - self.shadow * torch.fft.irfftn(z, s=(self.S, self.S), dim=(2, 3)).clamp(0, 1)
                )
            out = out * (1 - al) + rgb * al
        return out, pres, als


def act_map(net, layer, ch, img, pad=128):
    """Channel map of a periodic canvas at full resolution, free of border effects (circular pad, then
    crop), high-passed so it marks individual feature instances rather than broad regions."""
    S = img.shape[-1]
    with torch.no_grad():
        m = net(F.pad(img, (pad,) * 4, mode="circular"))[layer][0, ch]
        r = m.shape[-1] / (S + 2 * pad)
        c = round(pad * r)
        m = m[c : m.shape[0] - c, c : m.shape[1] - c]
        m = F.interpolate(m[None, None], size=(S, S), mode="bicubic", align_corners=False)
        k = 61  # subtract the local mean over ~60 px
        low = F.avg_pool2d(F.pad(m, (k // 2,) * 4, mode="circular"), k, 1)
        return (m - low)[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--which", default="0,1", help="anchor index per layer, back to front")
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--target", required=True)
    ap.add_argument(
        "--drifts", default="1,0;2,0", help="canvas widths per loop per layer, back to front"
    )
    ap.add_argument("--cover", default="0.4", help="target alpha coverage per front layer")
    ap.add_argument("--frames", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--alr", type=float, default=0.05)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--wbin", type=float, default=0.5, help="alpha 0/1 penalty, ramps in")
    ap.add_argument("--wcov", type=float, default=4.0)
    ap.add_argument("--shadow", type=float, default=0.0)
    ap.add_argument(
        "--cut", type=float, default=0.0, help="drop detail finer than this period (px)"
    )
    ap.add_argument(
        "--alpha-max-period",
        type=float,
        default=0,
        help="band-limit alpha to features finer than this (px)",
    )
    ap.add_argument(
        "--swirl",
        type=float,
        default=0,
        help="per-layer carried swirl, mean px/frame (0 = rigid layers)",
    )
    ap.add_argument("--alpha-min-period", type=float, default=0)
    ap.add_argument(
        "--back-volume",
        default="",
        help="trainable back layer: a volslice volume.pt, swept once per loop",
    )
    ap.add_argument("--vol-lr", type=float, default=0.01)
    ap.add_argument(
        "--back-frames",
        default="",
        help="frozen back layer: a directory of looping frames f000.png ... (frames must be a multiple of their count)",
    )
    ap.add_argument(
        "--alpha-init",
        default="act",
        help="act | noise | PATH to a 0/1 mask PNG (one per front layer, comma-separated)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-stride", type=int, default=1)
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
    which = [int(v) for v in a.which.split(",")]
    drifts = [tuple(float(v) for v in d.split(",")) for d in a.drifts.split(";")]
    covers = [float(v) for v in a.cover.split(",")]
    covers = covers + covers[-1:] * (len(which) - 1 - len(covers))
    assert len(drifts) == len(which)
    st = Stack(
        rg,
        [A[w : w + 1] for w in which],
        drifts,
        covers,
        T,
        dev,
        a.cut,
        a.shadow,
        amax=a.alpha_max_period,
        amin=a.alpha_min_period,
    )
    if a.back_volume:
        from volslice import Volume

        vs = torch.load(a.back_volume)
        st.vol = Volume(vs["S"], vs["Z"], vs["Lz"], dev, stretch=vs["stretch"], norm=vs["norm"])
        st.vol.P.data.copy_(vs["P"].to(dev))
        del vs
        st.specs[0].requires_grad_(False)
    if a.back_frames:
        import glob

        fs = sorted(glob.glob(f"{a.back_frames}/f[0-9][0-9][0-9].png"))
        st.back = (
            torch.stack(
                [
                    torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).permute(2, 0, 1)
                    for f in fs
                ]
            )
            .to(dev)
            .half()
            / 255
        )
        assert T % len(fs) == 0, "loop length must be a multiple of the back sequence"
        st.specs[0].requires_grad_(False)
    if a.swirl > 0:
        st.flows = [
            Flow(S, T, 4, 3, a.swirl, a.seed + 17 * l, dev, tscale=max(1, round(T / 120)))
            for l in range(len(which))
        ]
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)

    # alpha init: each front layer shows its own strongest feature instances (top `cover` of its
    # activation map), or smooth noise
    for l in range(1, len(which)):
        if a.alpha_init not in ("act", "noise"):
            mk = (
                np.asarray(Image.open(a.alpha_init.split(",")[l - 1]).convert("L"), np.float32)
                / 255
            )
            mk = torch.from_numpy(mk).to(dev)[None, None]
            mk = F.avg_pool2d(F.pad(mk, (1, 1, 1, 1), mode="circular"), 3, 1)[0, 0]  # 1 px feather
            st.init_alpha(l, 12 * (mk - 0.5))
            continue
        if a.alpha_init == "act":
            m = act_map(net, layer, ch, torch.sigmoid(rg.pre(A[which[l] : which[l] + 1].to(dev))))
        else:
            g = torch.Generator().manual_seed(a.seed + l)
            m = F.interpolate(
                torch.randn(1, 1, 16, 16, generator=g),
                size=(S, S),
                mode="bicubic",
                align_corners=False,
            )[0, 0].to(dev)
        q = torch.quantile(m.flatten()[::7].float().cpu(), 1 - covers[l - 1]).item()
        st.init_alpha(l, 4 * (m - q) / m.std())

    @torch.no_grad()
    def evaluate(write, stride=1):
        ts_all = list(range(0, T, stride))
        acts = []
        for i in range(0, len(ts_all), a.mb):
            ts = torch.tensor(ts_all[i : i + a.mb], device=dev, dtype=torch.float32)
            fr, _, _ = st.render(ts)
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

    with torch.no_grad():  # each anchor alone, same 4 views
        alone = []
        for w in which:
            img = torch.sigmoid(rg.pre(A[w : w + 1].to(dev)))
            s = 0
            for k in range(4):
                random.seed(1000 + k)
                s += float(rg.act(MultiScaleCrop(224 / S, 512)(img))) / 4
            alone.append(s)
    act0 = evaluate(False, stride=max(1, T // 64))
    log(
        f"layers {len(which)} drifts {drifts} cover {covers} speeds {[round(float(n), 2) for n in st.v.norm(dim=1)]} px/frame; "
        f"anchors alone {[round(v, 3) for v in alone]}; composite before {act0.mean():.3f} [{act0.min():.3f}, {act0.max():.3f}]"
    )

    specs, alphas = st.params()
    groups = [
        {"params": [p for p in specs if p.requires_grad], "lr": a.lr},
        {"params": alphas, "lr": a.alr},
    ]
    base_lr = [a.lr, a.alr]
    if st.vol is not None:
        groups.append({"params": [st.vol.P], "lr": a.vol_lr})
        base_lr.append(a.vol_lr)
    opt = torch.optim.Adam(groups)
    lum = torch.tensor(LUMA, device=dev).view(1, 3, 1, 1)
    curve = []
    for it in range(a.steps):
        frac = it / a.steps
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * 0.5 * (1 + math.cos(math.pi * frac))
        opt.zero_grad()
        ts = torch.tensor(random.sample(range(T), a.mb), device=dev, dtype=torch.float32)
        fr, pres, als = st.render(ts)
        act = rg.act(MultiScaleCrop(224 / S, 512)(fr))
        hinge = sum((torch.relu(p.abs() - LOGIT_RAIL) ** 2).mean((1, 2, 3)) for p in pres)
        y = (fr * lum).sum(1, keepdim=True)
        ratio = hf_energy(torch.cat([fr[:, 2:3] - y, fr[:, 0:1] - y], 1)) / hf_energy(y).detach()
        chroma = torch.relu(ratio / CHROMA_RATIO - 1)
        apen = 0
        for al, c in zip(als, covers):
            apen = (
                apen
                + a.wcov * (al.mean((1, 2, 3)) - c) ** 2
                + a.wbin * min(1.0, 2 * frac) * (al * (1 - al)).mean((1, 2, 3))
            )
        loss = (-act + act.detach().abs() * (rg.wh * hinge + rg.wc * chroma + apen)).sum()
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == a.steps - 1:
            cov = [round(float(al.mean()), 3) for al in als]
            soft = [round(float((al * (1 - al)).mean() * 4), 3) for al in als]
            curve.append((it, float(act.mean())))
            log(f"step {it + 1}/{a.steps} act {act.mean():.3f} cover {cov} softness {soft}")

    act = evaluate(True, a.eval_stride)
    with torch.no_grad():
        for l, spec in enumerate(specs):
            img = torch.sigmoid(rg.pre(spec))[0].cpu()
            Image.fromarray((img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
                f"{a.out}/layer{l}.png"
            )
            if l:
                al = torch.sigmoid(
                    st._alpha_pre(alphas[l - 1], st._phase(torch.zeros(1, 2, device=dev)))
                )[0, 0].cpu()
                Image.fromarray((al.numpy() * 255).round().astype(np.uint8)).save(
                    f"{a.out}/alpha{l}.png"
                )
        torch.save(
            {
                "specs": [s.detach().cpu() for s in specs],
                "alphas": [x.detach().cpu() for x in alphas],
            },
            f"{a.out}/stack.pt",
        )
    rep = dict(
        act_mean=float(act.mean()),
        act_min=float(act.min()),
        act_max=float(act.max()),
        act=act.tolist(),
        act_before=float(act0.mean()),
        anchors_alone=alone,
        drifts=drifts,
        cover=covers,
        frames=T,
        speeds=[float(n) for n in st.v.norm(dim=1)],
        shadow=a.shadow,
        cut=a.cut,
        steps=a.steps,
        curve=curve,
        secs=round(time.time() - t0),
    )
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(
        f"done act {act.mean():.3f} [{act.min():.3f}, {act.max():.3f}] (before {act0.mean():.3f}; anchors alone {[round(v, 2) for v in alone]})"
    )


if __name__ == "__main__":
    main()
