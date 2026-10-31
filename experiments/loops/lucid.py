"""Faithful port of the lucid neuron-interaction recipe, extended to loops within one channel,
plus two transport-aware couplings:

  --wbc W    brightness constancy between adjacent frames under a free flow: the change from frame t
             to t+1 must be explained by a displacement v_t (optimized jointly); only the residual is
             penalized. Flow knots every --bc-s px; --bc-tv puts a total-variation prior on the flow;
             --bc-noaff projects the affine part out of it so global slides and turns are not available.
  --warp     carrier + shared displacement field (one Fourier image every frame shares, warped per
             frame). --warp-s knots in space, --warp-k knots in time (circular), --warp-fix rms in px
             (fixed by normalization), --warp-tv total-variation prior, --warp-noaff removes the
             per-frame affine part, --warp-nomean removes the per-frame mean.

Parameterization (lucid interpolate_param_f): per-frame Fourier image ("unique") plus shared
low-resolution tensors at scales /2 /4 /8 /16, each with a small batch dimension that is linearly
interpolated across the frames. Colour decorrelation and sigmoid act on the sum. --shared-init
starts every unique image from the same spectrum.

Pressure to change: channel_interpolate (--a to --b), lucid's Gram diversity (--wdiv, --divw loop
weights pairs by loop distance), or per-position diversity (--wpos: cosine similarity of activation
maps at the same location, which transport changes and the Gram does not).
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, math, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from cells import sample
from striate.model import Backbone
from striate.param import FourierImage, color_correlation_normalized
from striate.transforms import TransformRobustness


def batch_weights(N, k, loop):
    """(N, k) interpolation weights from k knots to N frames; circular if loop."""
    W = torch.zeros(N, k)
    if k == 1:
        W[:] = 1
        return W
    for i in range(N):
        p = i * k / N if loop else i * (k - 1) / (N - 1)
        j = int(math.floor(p))
        f = p - j
        W[i, j % k] += 1 - f
        W[i, (j + 1) % k] += f
    return W


class Lowres(torch.nn.Module):
    def __init__(self, N, k, C, h, w, s, loop, sd, dev):
        super().__init__()
        self.t = torch.nn.Parameter(sd * torch.randn(k, C, h // s, w // s, device=dev))
        self.register_buffer("W", batch_weights(N, k, loop).to(dev))
        self.h, self.w, self.s = h, w, s

    def forward(self):
        up = (
            self.t
            if self.s == 1
            else F.interpolate(self.t, size=(self.h, self.w), mode="bilinear", align_corners=False)
        )
        return torch.einsum("nk,kchw->nchw", self.W, up)


def tv(f):
    """Mean absolute spatial gradient of a (N, C, S, S) field, periodic."""
    return (f - f.roll(1, 2)).abs().mean() + (f - f.roll(1, 3)).abs().mean()


class Affine:
    """Least-squares affine part of a displacement field on the pixel grid, to project out."""

    def __init__(self, size, dev):
        g = torch.arange(size, device=dev, dtype=torch.float32) - size / 2
        y, x = torch.meshgrid(g, g, indexing="ij")
        B = torch.stack([torch.ones_like(x), x, y], -1).reshape(-1, 3)  # (S*S, 3)
        self.B = B
        self.P = torch.linalg.pinv(B.cpu()).to(dev)  # (3, S*S)

    def remove(self, u):  # u (N, 2, S, S)
        N = u.shape[0]
        flat = u.reshape(N, 2, -1)  # (N, 2, S*S)
        coef = torch.einsum("kp,ncp->nck", self.P, flat)
        fit = torch.einsum("nck,pk->ncp", coef, self.B)
        return (flat - fit).reshape_as(u)


class Param(torch.nn.Module):
    def __init__(
        self,
        N,
        size,
        loop,
        dev,
        shared=True,
        unique=True,
        shared_init=False,
        warp=False,
        warp_s=16,
        warp_k=2,
        warp_amp=8.0,
        warp_fix=0.0,
        warp_nomean=False,
        warp_noaff=False,
    ):
        super().__init__()
        self.N, self.size, self.loop, self.dev = N, size, loop, dev
        self.unique = (
            FourierImage(size, size, batch=N, device=dev, decorrelate=False) if unique else None
        )
        if unique and shared_init:
            with torch.no_grad():
                self.unique.spectrum.copy_(
                    self.unique.spectrum[:1].expand_as(self.unique.spectrum).clone()
                )
        specs = [(1, 2), (1, 4), (1, 8), (2, 8), (1, 16), (2, 16)] if shared else []
        if loop:  # a crossfade between two knots can't close; use 3 knots where lucid used 2
            specs = [(3 if k == 2 else k, s) for k, s in specs]
        self.shared = torch.nn.ModuleList(
            [Lowres(N, k, 3, size, size, s, loop, 0.01, dev) for k, s in specs]
        )
        self.warp = warp
        self.warp_fix, self.warp_nomean, self.warp_noaff = warp_fix, warp_nomean, warp_noaff
        g = torch.arange(size, device=dev, dtype=torch.float32)
        self.register_buffer(
            "grid", torch.stack(torch.meshgrid(g, g, indexing="xy"), -1)
        )  # (S, S, 2) x, y
        if warp:
            self.carrier = FourierImage(size, size, batch=1, device=dev, decorrelate=False)
            self.field = Lowres(N, warp_k, 2, size, size, warp_s, loop, 0.01, dev)
            self.warp_amp = warp_amp
            self.aff = Affine(size, dev) if warp_noaff else None

    def displacement(self):
        f = self.field()
        if self.loop:  # no constant term over the loop: a static warp is not motion
            f = f - f.mean(0, keepdim=True)
        if self.warp_nomean:
            f = f - f.mean((2, 3), keepdim=True)
        if self.warp_noaff:
            f = self.aff.remove(f)
        if (
            self.warp_fix
        ):  # fixed rms amplitude by construction: the optimizer chooses only the shape
            return f * (self.warp_fix / (f.pow(2).sum(1).mean().sqrt() + 1e-8))
        return f * self.warp_amp

    def preact(self):
        x = sum(m() for m in self.shared) if len(self.shared) else 0
        if self.unique is not None:
            x = x + self.unique.preact()
        if self.warp:
            u = self.displacement().permute(0, 2, 3, 1)  # (N, S, S, 2)
            car = self.carrier.preact()[0]
            x = x + torch.stack([sample(car, self.grid + u[b], True) for b in range(self.N)])
        return x

    def image(self):
        x = self.preact()
        b, _, h, w = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, 3) @ color_correlation_normalized.to(x.device).T
        return torch.sigmoid(flat.reshape(b, h, w, 3).permute(0, 3, 1, 2))


class Flow(torch.nn.Module):
    """One free displacement field per (gap, frame) pair, for the brightness-constancy coupling.
    A dissolve spread over N frames costs 1/N under an L2 residual at gap 1, so longer gaps and an
    L1 residual are needed to make a dissolve cost what it should."""

    def __init__(self, N, size, s, loop, dev, noaff, gaps, amp=4.0):
        super().__init__()
        self.gaps = gaps
        self.pairs = [N if loop else N - g for g in gaps]
        self.v = torch.nn.ModuleList(
            [Lowres(p, p, 2, size, size, s, False, 0.01, dev) for p in self.pairs]
        )
        self.amp = amp
        self.aff = Affine(size, dev) if noaff else None

    def forward(self):
        out = []
        for m in self.v:
            v = m() * self.amp
            out.append(self.aff.remove(v) if self.aff is not None else v)
        return out


def gram_cos(act):
    """act (N, C, H, W) -> (N, N) cosine similarity of normalized Gram matrices (lucid diversity)."""
    N, C = act.shape[:2]
    f = act.reshape(N, C, -1)
    g = torch.bmm(f, f.transpose(1, 2)).reshape(N, -1)
    g = g / (g.norm(dim=1, keepdim=True) + 1e-10)
    return g @ g.T


def pos_cos(act):
    """(N, N) mean over positions of the cosine similarity of the channel vectors at that position."""
    a = act / (act.norm(dim=1, keepdim=True) + 1e-6)
    return torch.einsum("ichw,jchw->ij", a, a) / (a.shape[2] * a.shape[3])


def pair_weights(N, loop, mode):
    """(N, N) weights for a diversity term. 'uniform' = lucid; 'loop' = (1 - cos(2 pi d / N)) / 2."""
    i = torch.arange(N)
    d = (i[:, None] - i[None, :]).abs().float()
    if loop:
        d = torch.minimum(d, N - d)
    if mode == "uniform":
        w = torch.ones(N, N)
    else:
        w = (
            (1 - torch.cos(2 * math.pi * d / N)) / 2
            if loop
            else (1 - torch.cos(math.pi * d / (N - 1))) / 2
        )
    w.fill_diagonal_(0)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--a", required=True, help="layer:channel at frame 0")
    ap.add_argument(
        "--b", default="", help="layer:channel at the last frame (default: same as --a)"
    )
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-shared", action="store_true")
    ap.add_argument(
        "--no-unique", action="store_true", help="no per-frame component: carrier + field only"
    )
    ap.add_argument(
        "--shared-init",
        action="store_true",
        help="start every per-frame image from the same spectrum",
    )
    ap.add_argument("--wdiv", type=float, default=0.0, help="Gram diversity weight")
    ap.add_argument("--wpos", type=float, default=0.0, help="per-position diversity weight")
    ap.add_argument("--divw", default="uniform", choices=["uniform", "loop"])
    ap.add_argument("--div-layer", default="", help="layer for the Gram (default: layer of --a)")
    ap.add_argument("--pos-layer", default="inception3b")
    ap.add_argument("--walign", type=float, default=0.0, help="lucid alignment weight")
    ap.add_argument("--align-layer", default="inception3b")
    ap.add_argument("--wbc", type=float, default=0.0, help="brightness-constancy weight")
    ap.add_argument("--bc-s", type=int, default=4, help="flow knot spacing in px")
    ap.add_argument("--bc-tv", type=float, default=0.0)
    ap.add_argument("--bc-noaff", action="store_true")
    ap.add_argument("--bc-lr", type=float, default=0.02)
    ap.add_argument(
        "--bc-gaps", default="1", help="comma list of frame gaps for the constancy term"
    )
    ap.add_argument(
        "--bc-l1", action="store_true", help="Charbonnier (L1-like) residual instead of L2"
    )
    ap.add_argument("--warp", action="store_true")
    ap.add_argument("--warp-s", type=int, default=16)
    ap.add_argument("--warp-k", type=int, default=2)
    ap.add_argument("--warp-amp", type=float, default=8.0)
    ap.add_argument("--warp-lr", type=float, default=0.01)
    ap.add_argument("--warp-fix", type=float, default=0.0, help="fix the field rms at this many px")
    ap.add_argument("--warp-tv", type=float, default=0.0)
    ap.add_argument("--warp-nomean", action="store_true")
    ap.add_argument("--warp-noaff", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    la, ca = a.a.split(":")
    ca = int(ca)
    lb, cb = (a.b or a.a).split(":")
    cb = int(cb)
    dl = a.div_layer or la
    net = Backbone(a.net, dev)
    for layer in {la, lb, dl, a.align_layer, a.pos_layer}:
        net.hook(layer)
    N = a.n
    P = Param(
        N,
        a.size,
        a.loop,
        dev,
        shared=not a.no_shared,
        unique=not a.no_unique,
        shared_init=a.shared_init,
        warp=a.warp,
        warp_s=a.warp_s,
        warp_k=a.warp_k,
        warp_amp=a.warp_amp,
        warp_fix=a.warp_fix,
        warp_nomean=a.warp_nomean,
        warp_noaff=a.warp_noaff,
    )
    groups = [
        {"params": [p for n_, p in P.named_parameters() if not n_.startswith("field")], "lr": a.lr}
    ]
    if a.warp:
        groups.append({"params": list(P.field.parameters()), "lr": a.warp_lr})
    flow = None
    if a.wbc:
        flow = Flow(
            N, a.size, a.bc_s, a.loop, dev, a.bc_noaff, [int(g) for g in a.bc_gaps.split(",")]
        )
        groups.append({"params": list(flow.parameters()), "lr": a.bc_lr})
    opt = torch.optim.Adam(groups)
    t = torch.linspace(0, 1, N, device=dev) if (a.b and a.b != a.a) else torch.zeros(N, device=dev)
    tf = TransformRobustness(jitter=16, pad=16)
    pw = pair_weights(N, a.loop, a.divw).to(dev)
    grid = P.grid
    t0 = time.time()
    log = []
    for i in range(a.steps):
        opt.zero_grad()
        img = P.image()
        acts = net(tf(img))
        act = (1 - t) * acts[la][:, ca].mean((1, 2)) + t * acts[lb][:, cb].mean((1, 2))
        loss = -act.mean()
        div = (gram_cos(acts[dl]) * pw).sum()
        if a.wdiv:
            loss = loss + a.wdiv * div
        pos = (pos_cos(acts[a.pos_layer]) * pw).sum()
        if a.wpos:
            loss = loss + a.wpos * pos
        al = torch.zeros((), device=dev)
        if a.walign:
            x = acts[a.align_layer]
            for d in (1, 2, 3, 4):
                if a.loop:
                    al = al + ((x - x.roll(-d, 0)) ** 2).mean() / 2**d
                elif d < N:
                    al = al + ((x[d:] - x[:-d]) ** 2).mean() / 2**d
            loss = loss + a.walign * al
        bcres = torch.zeros((), device=dev)
        if flow is not None:
            bcres = torch.zeros((), device=dev)
            bctv = torch.zeros((), device=dev)
            for g, vf, npair in zip(flow.gaps, flow(), flow.pairs):
                v = vf.permute(
                    0, 2, 3, 1
                )  # (pairs, S, S, 2): for each pixel of frame t+g, its offset into frame t
                nxt = img.roll(-g, 0) if a.loop else img[g:]
                prev = torch.stack([sample(img[b], grid + v[b], True) for b in range(npair)])
                r2 = (nxt - prev) ** 2
                bcres = bcres + ((r2 + 1e-4).sqrt().mean() if a.bc_l1 else r2.mean()) / len(
                    flow.gaps
                )
                bctv = bctv + tv(vf) / len(flow.gaps)
            loss = loss + a.wbc * (bcres + (a.bc_tv * bctv if a.bc_tv else 0))
        if a.warp and a.warp_tv:
            loss = loss + a.warp_tv * tv(P.displacement())
        loss.backward()
        opt.step()
        if (i + 1) % max(1, a.steps // 10) == 0 or i == 0:
            msg = (
                f"step {i + 1}/{a.steps} act {act.mean():.3f} min {act.min():.3f} div {div:.3f} pos {pos:.3f} "
                f"align {al:.4f} bc {bcres:.5f} {time.time() - t0:.0f}s"
            )
            print(msg, flush=True)
            log.append(msg)
    with torch.no_grad():
        img = P.image().detach()
        acts = net(img)
        act_c = (1 - t) * acts[la][:, ca].mean((1, 2)) + t * acts[lb][:, cb].mean((1, 2))
        gc = gram_cos(acts[dl]).cpu()
        pix = [
            float(((img[(j + 1) % N] - img[j]) ** 2).mean().sqrt())
            for j in range(N if a.loop else N - 1)
        ]
        rep = dict(
            args=vars(a),
            act_frames=[round(float(x), 3) for x in act_c],
            act_mean=float(act_c.mean()),
            act_min=float(act_c.min()),
            gram_cos=[[round(float(x), 3) for x in r] for r in gc],
            gram_cos_adjacent=float(
                np.mean([gc[j, (j + 1) % N] for j in range(N if a.loop else N - 1)])
            ),
            gram_cos_far=float(gc[0, N // 2]),
            pix_rms_adjacent=float(np.mean(pix)),
            pos_cos_far=float(pos_cos(acts[a.pos_layer])[0, N // 2]),
            log=log,
            seconds=time.time() - t0,
        )
        if a.warp:
            u = P.displacement()
            rep["warp_px_rms"] = float(u.norm(dim=1).pow(2).mean().sqrt())
            rep["warp_px_max"] = float(u.norm(dim=1).max())
            np.save(f"{a.out}/field.npy", u.cpu().numpy().astype(np.float32))
        if flow is not None:
            vf = flow()[0]
            rep["bc_residual"] = float(bcres)
            rep["flow_px_rms"] = float(vf.norm(dim=1).pow(2).mean().sqrt())
            np.save(f"{a.out}/flow.npy", vf.cpu().numpy().astype(np.float32))
        json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
        arr = (img.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        for j in range(N):
            Image.fromarray(arr[j]).save(f"{a.out}/f{j:03d}.png")
        Image.fromarray(np.concatenate(list(arr[:: max(1, N // 8)]), 1)).save(
            f"{a.out}/strip.jpg", quality=92
        )
        print(
            f"done act {rep['act_mean']:.3f} gram adj {rep['gram_cos_adjacent']:.3f} far {rep['gram_cos_far']:.3f} "
            f"pix adj {rep['pix_rms_adjacent']:.4f} {rep['seconds']:.0f}s"
        )


if __name__ == "__main__":
    main()
