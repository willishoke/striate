"""Ridge-walk cycle: a closed loop of near-maximal images of one channel.

K anchor facets (joint octave renders of different seeds) are joined in a ring by K
open strings (the string method from striate's strobe_bench.py): nodes climb only
perpendicular to the path, every node sees the same random view each step, and nodes
are respaced to equal band-passed image distance. Coarse to fine in node count; the
last level decays the learning rate so frames settle instead of jittering. A final
equal-speed respacing around the whole loop gives the frames.

Usage: ring.py --net resnet50_v1 --target layer3:530 --size 1024 --out DIR
(PYTHONPATH must include the repo root and this folder; see ../ridge-landscape-handoff.md)
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# hard cap on MPS memory: overshooting raises an OOM instead of pushing the Mac into swap
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, math, random, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from striate.param import FourierImage, seed_spectrum_from_image
from striate.util import pick_device
from zoo import Net, MultiScaleCrop
from octave import truncate, Stop, LOGIT_RAIL, CHROMA_RATIO, hf_energy

LUMA = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)


def blur(x, s):
    k = int(3 * s) * 2 + 1
    g = torch.exp(-((torch.arange(k, device=x.device) - k // 2) ** 2) / (2 * s * s))
    g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


def bandpass(frames, lo, hi):
    lum = (frames * LUMA.to(frames.device)).sum(1, keepdim=True)
    return blur(lum, lo) - blur(lum, hi)


def temporal_report(frames):
    """Strobe metric from the README, on a closed loop. frames: (T, 3, H, W) in [0, 1]."""
    out = {"sat": float(((frames < 0.02) | (frames > 0.98)).float().mean())}
    for name, lo, hi in [("0-4px", 0.5, 2), ("4-16px", 2, 8), ("16-64px", 8, 32)]:
        b = torch.cat([bandpass(frames[i : i + 8], lo, hi) for i in range(0, len(frames), 8)])
        b = torch.cat([b, b[:2]])  # wrap: the loop has no seam
        d1 = b[1:] - b[:-1]
        d2 = d1[1:] - d1[:-1]
        v = d1.pow(2).mean((1, 2, 3)).sqrt()
        out[name] = dict(
            R=float(d2.pow(2).mean() / d1.pow(2).mean()),
            CV=float(v.std() / v.mean()),
            pk=float(v.max() / v.median()),
        )
    return out


class ChunkedAdam:
    """Adam over one big tensor of frames, updated a few frames at a time so the step's
    temporaries are a small fraction of the state (torch's Adam allocates full-size ones)."""

    def __init__(self, p, lr, betas=(0.9, 0.999), eps=1e-8, chunk=8):
        self.p, self.b1, self.b2, self.eps, self.chunk, self.t = p, *betas, eps, chunk, 0
        self.param_groups = [{"lr": lr}]
        self.m, self.v = torch.zeros_like(p), torch.zeros_like(p)

    def zero_grad(self):
        self.p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        lr = self.param_groups[0]["lr"]
        bc1, bc2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        n = self.p.shape[0] * (self.p.shape[1] if self.p.dim() == 6 else 1)
        P, G = self.p.data.view(n, *self.p.shape[-4:]), self.p.grad.view(n, *self.p.shape[-4:])
        M, V = self.m.view_as(P), self.v.view_as(P)
        for i in range(0, n, self.chunk):
            g, m, v = G[i : i + self.chunk], M[i : i + self.chunk], V[i : i + self.chunk]
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            P[i : i + self.chunk].addcdiv_(
                m, (v.sqrt() / math.sqrt(bc2)).add_(self.eps), value=-lr / bc1
            )


class Ring:
    def __init__(self, net, layer, ch, size, dev, wh=4.0, wc=4.0):
        self.net, self.layer, self.ch, self.dev = net, layer, ch, dev
        self.wh, self.wc = wh, wc
        self.helper = FourierImage(size, size, device=dev)
        self.size = size

    def pre(self, spec):
        z = torch.view_as_complex((spec * self.helper.spectrum_scale).contiguous())
        p = torch.fft.irfftn(z, s=(self.size, self.size), dim=(2, 3)) / 4.0
        return self.helper._decorrelate_color(p)

    def image(self, spec):
        return torch.sigmoid(self.pre(spec))

    def act(self, x):
        self.net.acts = {}
        try:
            self.net(x)
        except Stop:
            pass
        return self.net.acts[self.layer][:, self.ch].mean((1, 2))

    def loss(self, spec, view):
        """-activation + activation-scaled clip hinge and chroma cap, per item, summed."""
        pre = self.pre(spec)
        img = torch.sigmoid(pre)
        a = self.act(view(img))
        hinge = (torch.relu(pre.abs() - LOGIT_RAIL) ** 2).mean((1, 2, 3))
        y = (img * torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)).sum(
            1, keepdim=True
        )
        ratio = hf_energy(torch.cat([img[:, 2:3] - y, img[:, 0:1] - y], 1)) / hf_energy(y).detach()
        chroma = torch.relu(ratio / CHROMA_RATIO - 1)
        return (-a + a.detach().abs() * (self.wh * hinge + self.wc * chroma)).sum(), a.detach()


def anchors(rg, k, sizes, steps, seed, dev):
    """K seeds rendered together with the joint octave recipe; returns spectra at the last size."""
    torch.manual_seed(seed)
    random.seed(seed)
    img = None
    for size in sizes:
        param = FourierImage(size, size, batch=k, device=dev)
        if img is not None:
            up = F.interpolate(img, size=(size, size), mode="bicubic", align_corners=False).clamp(
                0, 1
            )
            seed_spectrum_from_image(param, up)
        sub = Ring(rg.net, rg.layer, rg.ch, size, dev, rg.wh, rg.wc)
        opt = torch.optim.Adam([param.spectrum], lr=0.05)
        view = MultiScaleCrop(224 / size, 512)
        for _ in range(steps):
            opt.zero_grad()
            loss, a = sub.loss(param.spectrum, view)
            loss.backward()
            opt.step()
        img = param.to_image().detach()
        print(f"anchors {size}px act {[round(float(v), 3) for v in a]}", flush=True)
    return param.spectrum.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="resnet50_v1")
    ap.add_argument("--target", default="layer3:530")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--schedule", default="4:300,9:100,19:50,39:40")
    ap.add_argument("--anchor-steps", type=int, default=384)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--mb", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--anchors-from",
        default="",
        help="corpus.pt from landscape.py: use its attractors as anchors",
    )
    ap.add_argument("--anchor-ids", default="", help="comma-separated indices into --anchors-from")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(a.out, exist_ok=True)
    layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
    net = Net(a.net, dev)
    net.hook(layer)
    truncate(net, {layer})
    rg = Ring(net, layer, ch, a.size, dev)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)

    sizes = [s for s in (224, 320, 448, 640, 896) if s < a.size] + [a.size]
    if a.anchors_from:  # waypoints chosen from a measured corpus (handoff step 5, string tours)
        C = torch.load(a.anchors_from, weights_only=False)
        assert C["size"] == a.size, (C["size"], a.size)
        A = C["spectra"][[int(i) for i in a.anchor_ids.split(",")]].to(dev)
    else:
        A = anchors(rg, a.k, sizes, a.anchor_steps, a.seed, dev)  # (K, 3, H, W', 2)
    torch.save(A.cpu(), f"{a.out}/anchors.pt")
    K = len(A)
    log("anchors done")

    @torch.no_grad()
    def band(spec):
        return torch.cat(
            [bandpass(rg.image(spec[i : i + a.mb]), 2, 32) for i in range(0, len(spec), a.mb)]
        )

    @torch.no_grad()
    def resample(chain, n, closed):
        """n points at equal band-passed image distance along chain (closed: back to chain[0])."""
        if closed:
            chain = torch.cat([chain, chain[:1]])
        b = band(chain)
        s = torch.cat([chain.new_zeros(1), (b[1:] - b[:-1]).flatten(1).norm(dim=1).cumsum(0)])
        s = s / s[-1]
        us = (
            torch.arange(n, device=dev) / n
            if closed
            else torch.linspace(0, 1, n + 2, device=dev)[1:-1]
        )
        out = chain.new_empty(len(us), *chain.shape[1:])  # filled in place: no list + stack copy
        for i, u in enumerate(us):
            j = int(torch.searchsorted(s, u.clamp(min=1e-9)).clamp(1, len(s) - 1))
            w = (u - s[j - 1]) / (s[j] - s[j - 1] + 1e-12)
            torch.lerp(chain[j - 1], chain[j], w, out=out[i])
        return out

    X = None  # (K, m, 3, H, W', 2): interior nodes of each segment k -> k+1
    levels = [tuple(map(int, l.split(":"))) for l in a.schedule.split(",")]
    for li, (m, steps) in enumerate(levels):
        if X is None:
            t = torch.linspace(0, 1, m + 2, device=dev)[1:-1].view(-1, 1, 1, 1, 1)
            X = torch.stack([A[k] + t * (A[(k + 1) % K] - A[k]) for k in range(K)])
        else:  # insert midpoints within each segment, straight into the new tensor
            assert m == 2 * X.shape[1] + 1, (X.shape, m)
            new = X.new_empty(K, m, *X.shape[2:])
            for k in range(K):
                chain = [A[k]] + list(X[k]) + [A[(k + 1) % K]]
                for i in range(m):  # even i: midpoint of chain[i/2], chain[i/2+1]; odd: old node
                    if i % 2 == 0:
                        torch.lerp(chain[i // 2], chain[i // 2 + 1], 0.5, out=new[k, i])
                    else:
                        new[k, i].copy_(chain[i // 2 + 1])
            X = new
            del new
        X = torch.nn.Parameter(X)
        opt = ChunkedAdam(X, lr=a.lr)
        last = li == len(levels) - 1
        for it in range(steps):
            if last:  # cosine decay so the final frames settle
                for g in opt.param_groups:
                    g["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * it / steps))
            opt.zero_grad()
            seed = random.randrange(1 << 30)
            flat = X.view(K * m, *X.shape[2:])
            acts = []
            for i in range(0, K * m, a.mb):
                random.seed(seed)  # same view for every node this step
                view = MultiScaleCrop(224 / a.size, 512)
                loss, act = rg.loss(flat[i : i + a.mb], view)
                loss.backward()
                acts.append(act)
            with torch.no_grad():  # climb only perpendicular to the path
                for k in range(K):
                    chain = torch.cat([A[k : k + 1], X[k], A[(k + 1) % K][None]])
                    tau = (chain[2:] - chain[:-2]).flatten(1)
                    tau = tau / tau.norm(dim=1, keepdim=True)
                    g = X.grad[k].flatten(1)
                    X.grad[k].copy_((g - (g * tau).sum(1, keepdim=True) * tau).view_as(X[k]))
            opt.step()
            if it % 10 == 9:
                with torch.no_grad():
                    for k in range(K):
                        chain = torch.cat([A[k : k + 1], X.data[k], A[(k + 1) % K][None]])
                        X.data[k] = resample(chain, m, closed=False)
            if it % max(1, steps // 4) == 0 or it == steps - 1:
                acts = torch.cat(acts)
                log(
                    f"level {m}x{K} step {it + 1}/{steps} act mean {acts.mean():.3f} min {acts.min():.3f}"
                )
        X = X.detach()
        del opt  # drop this level's Adam state and gradient before the next level allocates
        if dev == "mps":
            torch.mps.empty_cache()

    loop = torch.cat([torch.cat([A[k : k + 1], X[k]]) for k in range(K)])  # K*(m+1) nodes, closed
    del X
    frames_spec = resample(loop, a.frames, closed=True)
    del loop
    with torch.no_grad():
        frames = torch.cat(
            [rg.image(frames_spec[i : i + a.mb]).cpu() for i in range(0, a.frames, a.mb)]
        )
        # robust activation per frame: fixed view draws shared by all frames
        tot = 0
        for d in range(4):
            r = []
            for i in range(0, a.frames, a.mb):
                random.seed(1000 + d)
                r.append(
                    rg.act(MultiScaleCrop(224 / a.size, 512)(frames[i : i + a.mb].to(dev))).cpu()
                )
            tot = tot + torch.cat(r)
        act = (tot / 4).numpy()
    for i, f in enumerate(frames):
        Image.fromarray((f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
            f"{a.out}/f{i:03d}.png"
        )
    anchor_idx = [
        int(np.argmin([(frames_spec[j] - A[k]).norm().item() for j in range(a.frames)]))
        for k in range(K)
    ]
    rep = temporal_report(frames)
    rep.update(
        act_mean=float(act.mean()),
        act_min=float(act.min()),
        act_max=float(act.max()),
        anchor_frames=anchor_idx,
        act=act.tolist(),
        secs=round(time.time() - t0),
    )
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(
        f"done act mean {act.mean():.3f} min {act.min():.3f} max {act.max():.3f} "
        + " ".join(
            f"{k} R {v['R']:.2f} CV {v['CV']:.2f}" for k, v in rep.items() if isinstance(v, dict)
        )
    )


if __name__ == "__main__":
    main()
