"""B4 at high resolution through octaves, with a hue field in the spirit of the Cell Content Loop.

Stage 0 (small, 16:9): the lucid parameterization with shared init, brightness constancy at gaps
1/4/16 with an L1 residual under free per-pair flows, positional diversity by loop distance, plus a
low-res hue-rotation field (zero-mean over the loop and over space, rms fixed) applied before the
sigmoid. Full batch, as in B4.
Stages 1..K: the composite is folded into one Fourier spectrum per frame, upsampled x2, and refined
with minibatches of 8 frames (two windows {t, t+1, t+4, t+16}). The spectra and their Adam states
live in CPU memory; only the minibatch touches the GPU. Activation is scored on multi-scale crops,
the constancy term on a random window, so the per-step cost is bounded regardless of canvas size.
Output: frames at the top octave, downsampled copies, and a report.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, math, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from zoo import MultiScaleCrop
from striate.model import Backbone
from striate.param import FourierImage, color_correlation_normalized
from striate.transforms import TransformRobustness
from lucid import batch_weights, pos_cos, pair_weights  # noqa: F401

LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def psample(tex, uv, cubic=False):
    """Periodic lookup, bilinear (4 taps) or Catmull-Rom (16). tex (C, H, W); uv (..., 2) px (x, y) -> (C, ...)."""
    C, H, W = tex.shape
    flat = tex.reshape(C, -1)
    sh = uv.shape[:-1]
    u, v = uv[..., 0].reshape(-1), uv[..., 1].reshape(-1)
    x0, y0 = torch.floor(u), torch.floor(v)
    fx, fy = u - x0, v - y0
    x0, y0 = x0.long(), y0.long()

    def w4(t):
        t2, t3 = t * t, t * t * t
        return [
            -0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1,
            -1.5 * t3 + 2 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2,
        ]

    if cubic:
        wx, wy, taps = w4(fx), w4(fy), range(-1, 3)
    else:
        wx, wy, taps = [1 - fx, fx], [1 - fy, fy], range(0, 2)
    out = 0
    for j, dy in enumerate(taps):
        row = torch.remainder(y0 + dy, H) * W
        acc = 0
        for i, dx in enumerate(taps):
            acc = acc + flat[:, row + torch.remainder(x0 + dx, W)] * wx[i]
        out = out + acc * wy[j]
    return out.reshape(C, *sh)


class Knots(torch.nn.Module):
    """Low-res tensor with k knots in time, interpolated circularly across N frames and bilinearly in space."""

    def __init__(self, N, k, C, kh, kw, loop, sd, dev):
        super().__init__()
        self.t = torch.nn.Parameter(sd * torch.randn(k, C, kh, kw, device=dev))
        self.register_buffer("W", batch_weights(N, k, loop).to(dev))

    def forward(self, h, w, idx=None):
        if (
            self.t.shape[0] == self.W.shape[0] and idx is not None
        ):  # one knot per frame: upsample only those
            return F.interpolate(self.t[idx], size=(h, w), mode="bilinear", align_corners=False)
        W = self.W if idx is None else self.W[idx]
        low = torch.einsum("nk,kchw->nchw", W, self.t)
        return F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)


class Affine:
    def __init__(self, h, w, dev):
        gy = torch.arange(h, device=dev, dtype=torch.float32) - h / 2
        gx = torch.arange(w, device=dev, dtype=torch.float32) - w / 2
        y, x = torch.meshgrid(gy, gx, indexing="ij")
        self.B = torch.stack([torch.ones_like(x), x, y], -1).reshape(-1, 3)
        self.P = torch.linalg.pinv(self.B.cpu()).to(dev)

    def remove(self, u):
        N = u.shape[0]
        flat = u.reshape(N, 2, -1)
        coef = torch.einsum("kp,ncp->nck", self.P, flat)
        return (flat - torch.einsum("nck,pk->ncp", coef, self.B)).reshape_as(u)


def decor_sigmoid(x, hue=None):
    """x (B, 3, H, W) un-decorrelated preact -> image. hue (B, 1, H, W) rotation about the grey axis."""
    y = F.conv2d(x, color_correlation_normalized.to(x.device)[:, :, None, None])
    if hue is not None:
        a = torch.full((1, 3, 1, 1), 1 / math.sqrt(3), device=x.device)
        c, s = torch.cos(hue), torch.sin(hue)
        cross = torch.cross(a.expand_as(y), y, dim=1)
        dot = (y * a).sum(1, keepdim=True)
        y = y * c + cross * s + a * dot * (1 - c)
    return torch.sigmoid(y)


def fold_to_spectra(preact, h, w, dev):
    """Un-decorrelated preact (N, 3, h0, w0) -> per-frame spectra (list of CPU fp32 tensors) at (h, w)."""
    ref = FourierImage(h, w, batch=1, device=dev, decorrelate=False)
    scale = ref.spectrum_scale  # (1, 1, h, w//2+1, 1)
    out = []
    for n in range(preact.shape[0]):
        p = preact[n : n + 1].to(dev)
        if p.shape[2:] != (h, w):
            p = F.interpolate(p, size=(h, w), mode="bicubic", align_corners=False)
        spec = torch.view_as_real(torch.fft.rfftn(p * 4.0, dim=(2, 3))) / scale
        out.append(spec[0].detach().cpu().contiguous())
    return out, scale


def spectra_to_preact(specs, scale, h, w):
    """specs (B, 3, h, w//2+1, 2) on device -> preact (B, 3, h, w)."""
    c = torch.view_as_complex((specs * scale).contiguous())
    return torch.fft.irfftn(c, s=(h, w), dim=(2, 3)) / 4.0


class Fields(torch.nn.Module):
    """Per-(gap, frame) constancy flows and the hue field, resolution-independent (knot counts fixed)."""

    def __init__(self, N, gaps, kw_flow, kw_hue, aspect, loop, dev, hue_k=4):
        super().__init__()
        self.N, self.gaps, self.loop = N, gaps, loop
        kh_f, kh_h = max(2, round(kw_flow * aspect)), max(2, round(kw_hue * aspect))
        self.flows = torch.nn.ModuleList(
            [Knots(N, N, 2, kh_f, kw_flow, False, 0.01, dev) for _ in gaps]
        )
        self.hue = Knots(N, hue_k, 1, kh_h, kw_hue, loop, 0.01, dev)
        self.flow_scale = 4.0  # px per unit at the current octave

    def flow(self, gi, h, w, idx):
        return self.flows[gi](h, w, idx) * self.flow_scale

    def hue_field(self, h, w, idx, fix):
        """Zero-mean over the loop and over space, rms fixed; all done at knot resolution, since the
        interpolation is linear and circular with equal spacing."""
        k = self.hue.t
        k = k - k.mean(0, keepdim=True)
        k = k - k.mean((2, 3), keepdim=True)
        low = torch.einsum("nk,kchw->nchw", self.hue.W, k)  # (N, 1, kh, kw)
        k = k * (fix / (low.pow(2).mean().sqrt() + 1e-8))
        W = self.hue.W if idx is None else self.hue.W[idx]
        low = torch.einsum("nk,kchw->nchw", W, k)
        return F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)


def clip_pen(img, lo=0.05, hi=0.95):
    """Pixels outside [lo, hi] are saturated, change nothing between frames, and so cheat the constancy term."""
    return (F.relu(lo - img) + F.relu(img - hi)).mean()


class ScaleCrop(MultiScaleCrop):
    """MultiScaleCrop with an upper scale bound: at an upper octave the structure is right only when the
    network sees the canvas at <= smax, so it should not be shown the image at native magnification."""

    def __init__(self, min_scale, crop, smax):
        super().__init__(min_scale, crop)
        self.smax = smax

    def __call__(self, x):
        _, _, h, w = x.shape
        s = math.exp(random.uniform(math.log(self.m), math.log(self.smax)))
        wh, ww = min(h, round(self.crop / s)), min(w, round(self.crop / s))
        top, left = random.randint(0, h - wh), random.randint(0, w - ww)
        x = x[:, :, top : top + wh, left : left + ww]
        oh, ow = max(1, round(wh * s)), max(1, round(ww * s))
        if (oh, ow) != (wh, ww):
            x = F.interpolate(
                x, size=(oh, ow), mode="bilinear", antialias=True, align_corners=False
            )
        return self.tf(x)


def bc_term(img_prev, img_next, v, grid, win):
    """L1-type residual of next vs prev warped by v on a window. img (3, H, W); v (2, H, W); grid (H, W, 2)."""
    y0, x0, wh, ww = win
    uv = grid[y0 : y0 + wh, x0 : x0 + ww] + v[:, y0 : y0 + wh, x0 : x0 + ww].permute(1, 2, 0)
    prev = psample(img_prev, uv)
    nxt = img_next[:, y0 : y0 + wh, x0 : x0 + ww]
    r2 = (nxt - prev) ** 2
    return (r2 + 1e-4).sqrt().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--target", default="inception4e:198")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--w0", type=int, default=960)
    ap.add_argument("--h0", type=int, default=540)
    ap.add_argument(
        "--octaves", type=int, default=3, help="stage 0 plus this many minus one doublings"
    )
    ap.add_argument("--steps", default="1000,600,600")
    ap.add_argument("--lrs", default="0.05,0.02,0.01")
    ap.add_argument("--mb", default="8,8,4", help="minibatch frames per stage (multiples of 4)")
    ap.add_argument("--wbc", type=float, default=50.0, help="constancy weight")
    ap.add_argument(
        "--wclip", type=float, default=100.0, help="penalty on pixels outside [0.05, 0.95]"
    )
    ap.add_argument(
        "--smax0", type=float, default=1.5, help="upper crop scale at stage k is smax0 * 2^-k"
    )
    ap.add_argument(
        "--mb0", type=int, default=20, help="stage 0: contiguous block of frames per step"
    )
    ap.add_argument("--gaps", default="1,4,16")
    ap.add_argument("--flow-knots", type=int, default=64, help="flow knots across the width")
    ap.add_argument("--wpos", type=float, default=8.0)
    ap.add_argument("--pos-layer", default="inception3b")
    ap.add_argument(
        "--hue-fix", type=float, default=0.12, help="rms hue rotation in radians; 0 disables"
    )
    ap.add_argument("--hue-knots", type=int, default=32)
    ap.add_argument("--bc-win", type=int, default=1024)
    ap.add_argument(
        "--bc-sub",
        type=int,
        default=8,
        help="stage 0: frames per gap per step in the constancy term",
    )
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--stateless-from",
        type=int,
        default=2,
        help="stage index from which spectra use a stateless step",
    )
    ap.add_argument("--resume", default="", help="stage checkpoint to continue from")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("mps")
    layer, ch = a.target.split(":")
    ch = int(ch)
    net = Backbone("googlenet", dev)
    net.hook(layer)
    net.hook(a.pos_layer)
    N, gaps = a.n, [int(g) for g in a.gaps.split(",")]
    steps = [int(s) for s in a.steps.split(",")]
    lrs = [float(x) for x in a.lrs.split(",")]
    aspect = a.h0 / a.w0
    fields = Fields(N, gaps, a.flow_knots, a.hue_knots, aspect, True, dev)
    pw_full = pair_weights(N, True, "loop").to(dev)
    t_all = time.time()

    start_stage = 1
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu")
        fields.load_state_dict(ck["fields"])
        fields.flow_scale = ck["flow_scale"]
        h, w, start_stage = ck["h"], ck["w"], ck["stage"] + 1
        specs = [t.contiguous() for t in ck["specs"]]
        log(f"resumed from {a.resume}: stage {ck['stage']} at {w}x{h}")
    # ---------------- stage 0: lucid parameterization, full batch ----------------
    if a.resume:
        steps_0 = 0
    else:
        steps_0 = steps[0]
    h, w = (h, w) if a.resume else (a.h0, a.w0)
    unique = FourierImage(h, w, batch=N if not a.resume else 1, device=dev, decorrelate=False)
    with torch.no_grad():
        unique.spectrum.copy_(unique.spectrum[:1].expand_as(unique.spectrum).clone())
    shared = torch.nn.ModuleList(
        [
            Knots(N, k, 3, max(2, h // s), max(2, w // s), True, 0.01, dev)
            for k, s in [(1, 2), (1, 4), (1, 8), (3, 8), (1, 16), (3, 16)]
        ]
    )
    opt = torch.optim.Adam(
        [
            {"params": [unique.spectrum] + list(shared.parameters()), "lr": lrs[0]},
            {"params": list(fields.parameters()), "lr": 0.02},
        ]
    )
    msc0 = MultiScaleCrop(224 / min(h, w), a.crop)
    gy = torch.arange(h, device=dev, dtype=torch.float32)
    gx = torch.arange(w, device=dev, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1)  # (h, w, 2) x, y
    idx_all = torch.arange(N, device=dev)
    t0 = time.time()
    for i in range(steps_0):
        opt.zero_grad()
        start = random.randrange(N)
        blk = [(start + r) % N for r in range(min(a.mb0, N))]
        bi = torch.tensor(blk, device=dev)
        pre = unique.preact(unique.spectrum[bi]) + sum(
            m(h, w, bi) for m in shared
        )  # only the block's frames
        hue = fields.hue_field(h, w, bi, a.hue_fix) if a.hue_fix else None
        img = decor_sigmoid(pre, hue)
        acts = net(
            msc0(img)
        )  # one multi-scale window for the block: a 512 crop, not the whole frame
        act = acts[layer][:, ch].mean((1, 2))
        pos = (pos_cos(acts[a.pos_layer]) * pw_full[bi][:, bi]).sum() / pw_full[bi][:, bi].sum()
        bc = 0
        span = len(blk) - max(gaps) if len(blk) < N else len(blk)
        sub = random.sample(
            range(span), min(a.bc_sub, span)
        )  # block positions whose gap partners are in the block
        for gi, g in enumerate(gaps):
            v = fields.flow(gi, h, w, bi[sub])
            bc = bc + sum(
                bc_term(img[r0], img[(r0 + g) % len(blk)], v[r], grid, (0, 0, h, w))
                for r, r0 in enumerate(sub)
            ) / len(sub) / len(gaps)
        loss = -act.mean() + a.wpos * pos + a.wbc * bc + a.wclip * clip_pen(img)
        loss.backward()
        opt.step()
        bc_v, act_m, act_n, pos_v = float(bc), float(act.mean()), float(act.min()), float(pos)
        del pre, hue, img, acts, loss, bc, act, pos
        torch.mps.empty_cache()  # crops of varying size fragment the caching allocator
        if (i + 1) % max(1, steps_0 // 10) == 0 or i == 0:
            log(
                f"stage0 {w}x{h} step {i + 1}/{steps_0} act {act_m:.3f} min {act_n:.3f} pos {pos_v:.3f} bc {bc_v:.4f} {time.time() - t0:.0f}s"
            )
    with torch.no_grad():
        if a.resume:
            scale = FourierImage(h, w, batch=1, device=dev, decorrelate=False).spectrum_scale
            pre_cpu = torch.cat(
                [spectra_to_preact(specs[j][None].to(dev), scale, h, w).cpu() for j in range(N)]
            )
            del specs
        else:
            pre_cpu = (unique.preact() + sum(m(h, w) for m in shared)).detach().cpu()
    del unique, shared, opt
    torch.mps.empty_cache()
    if not a.resume:
        specs, scale = fold_to_spectra(pre_cpu, h, w, dev)
        torch.save(
            {
                "specs": specs,
                "h": h,
                "w": w,
                "stage": 0,
                "fields": fields.state_dict(),
                "flow_scale": fields.flow_scale,
            },
            f"{a.out}/stage0.pt",
        )
        if a.octaves > 1:
            del specs

    # ---------------- stages 1..: per-frame spectra on the CPU, minibatches ----------------
    for stage in range(start_stage, a.octaves):
        h, w = h * 2, w * 2
        specs, scale = fold_to_spectra(pre_cpu, h, w, dev)
        del pre_cpu
        m_state = [None] * N
        v_state = [None] * N
        n_step = [0] * N
        lr = lrs[min(stage, len(lrs) - 1)]
        b1, b2, eps = 0.9, 0.999, 1e-8
        fields.flow_scale *= 2
        fopt = torch.optim.Adam(fields.parameters(), lr=0.02)
        msc = ScaleCrop(224 / min(h, w), a.crop, min(1.0, a.smax0 * 2**-stage))
        gy = torch.arange(h, device=dev, dtype=torch.float32)
        gx = torch.arange(w, device=dev, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1)
        mbs = [int(x) for x in a.mb.split(",")]
        nwin = mbs[min(stage, len(mbs) - 1)] // 4
        t0 = time.time()
        st = steps[min(stage, len(steps) - 1)]
        for i in range(st):
            base = [random.randrange(N) for _ in range(nwin)]
            idx = [(b + d) % N for b in base for d in (0, 1, 4, 16)]
            ti = torch.tensor(idx, device=dev)
            sp = torch.stack([specs[j] for j in idx]).to(dev).requires_grad_(True)
            pre = spectra_to_preact(sp, scale, h, w)
            hue = fields.hue_field(h, w, ti, a.hue_fix) if a.hue_fix else None
            img = decor_sigmoid(pre, hue)
            crops = msc(img)  # one window for the whole minibatch, so positions are comparable
            acts = net(crops)
            act = acts[layer][:, ch].mean((1, 2))
            d = (ti[:, None] - ti[None, :]).abs()
            d = torch.minimum(d, N - d).float()
            pw = (1 - torch.cos(2 * math.pi * d / N)) / 2
            pw.fill_diagonal_(0)
            pw = pw / pw.sum()
            pos = (pos_cos(acts[a.pos_layer]) * pw).sum()
            wh, ww = min(a.bc_win, h), min(a.bc_win, w)
            win = (random.randrange(h - wh + 1), random.randrange(w - ww + 1), wh, ww)
            bc = 0
            for k in range(nwin):
                b = 4 * k
                for gi, (g, off) in enumerate(zip(gaps, (1, 2, 3))):
                    v = fields.flow(gi, h, w, ti[b : b + 1])[0]
                    bc = bc + bc_term(img[b], img[b + off], v, grid, win) / nwin / len(gaps)
            loss = -act.mean() + a.wpos * pos + a.wbc * bc + a.wclip * clip_pen(img)
            fopt.zero_grad()
            loss.backward()
            fopt.step()
            with (
                torch.no_grad()
            ):  # Adam on the CPU for the touched spectra; stateless at the top octaves
                g_all = sp.grad.cpu()
                for r, j in enumerate(idx):
                    g = g_all[r]
                    if (
                        stage >= a.stateless_from
                    ):  # normalized gradient step, no per-frame state (saves 2x the spectra)
                        specs[j].add_(-lr * (g / (g.pow(2).mean().sqrt() + eps)).clamp_(-3, 3))
                        continue
                    if m_state[j] is None:
                        m_state[j] = torch.zeros_like(g)
                        v_state[j] = torch.zeros_like(g)
                    n_step[j] += 1
                    m_state[j].mul_(b1).add_(g, alpha=1 - b1)
                    v_state[j].mul_(b2).addcmul_(g, g, value=1 - b2)
                    mh = m_state[j] / (1 - b1 ** n_step[j])
                    vh = v_state[j] / (1 - b2 ** n_step[j])
                    specs[j].add_(-lr * mh / (vh.sqrt() + eps))
            del sp, pre, img, crops, acts, loss
            torch.mps.empty_cache()  # the caching allocator otherwise holds the previous stage's blocks
            if (i + 1) % max(1, st // 10) == 0 or i == 0:
                log(
                    f"stage{stage} {w}x{h} step {i + 1}/{st} act {act.mean():.3f} pos {pos:.3f} bc {float(bc):.4f} {time.time() - t0:.0f}s"
                )
        del m_state, v_state
        torch.mps.empty_cache()
        if stage < a.octaves - 1:
            torch.save(
                {
                    "specs": specs,
                    "h": h,
                    "w": w,
                    "stage": stage,
                    "fields": fields.state_dict(),
                    "flow_scale": fields.flow_scale,
                },
                f"{a.out}/stage{stage}.pt",
            )
            with torch.no_grad():
                pre_cpu = torch.cat(
                    [spectra_to_preact(specs[j][None].to(dev), scale, h, w).cpu() for j in range(N)]
                )
            del specs
    # ---------------- final render ----------------
    torch.save(
        {
            "fields": fields.state_dict(),
            "flow_scale": fields.flow_scale,
            "args": vars(a),
            "h": h,
            "w": w,
        },
        f"{a.out}/fields.pt",
    )
    os.makedirs(f"{a.out}/full", exist_ok=True)
    acts_f = []
    with torch.no_grad():
        hue_all = fields.hue_field(h, w, None, a.hue_fix).cpu() if a.hue_fix else None
        for j in range(N):
            pre = spectra_to_preact(specs[j][None].to(dev), scale, h, w)
            img = decor_sigmoid(pre, hue_all[j : j + 1].to(dev) if hue_all is not None else None)
            tot = 0
            for k in range(4):
                random.seed(1000 + k)
                tot += float(net(MultiScaleCrop(224 / min(h, w), a.crop)(img))[layer][:, ch].mean())
            acts_f.append(tot / 4)
            arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(arr).save(f"{a.out}/full/f{j:03d}.png")
            Image.fromarray(arr).resize((w // 2, h // 2), Image.LANCZOS).save(
                f"{a.out}/f{j:03d}.png"
            )
            del pre, img
    rep = dict(
        args=vars(a),
        h=h,
        w=w,
        act_frames=[round(x, 3) for x in acts_f],
        act_mean=float(np.mean(acts_f)),
        act_min=float(np.min(acts_f)),
        seconds=time.time() - t_all,
        log=LOG,
    )
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(f"done {w}x{h} act {rep['act_mean']:.3f} min {rep['act_min']:.3f} {rep['seconds']:.0f}s")


if __name__ == "__main__":
    main()
