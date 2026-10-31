"""Slice loop: render a seamless 3D volume for one channel, then sweep a cutting plane through it.

The volume is a FourierImage lifted to 3D: coefficients over (kz, ky, kx) with the same 1/|k|
preconditioning, where kz counts cycles per Lz pixels of depth. Every horizontal slice is an image;
each step scores a few random slices against the channel. Frame t is the slice at depth Lz * t / T,
so the loop closes (the volume wraps in z) and features change the way grain does in a cut block:
rings and cells split, merge, appear and close as they pass through the plane, never cross-fade.

Octaves 224 -> 448 -> 1024 as in the stills; between octaves the whole volume is re-fit at the new
resolution (render every z sample, upsample, transform back).

Usage: volslice.py --net googlenet --target inception3b:142 --out DIR [--lz 128] [--z 64]
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
from striate.param import color_correlation_normalized
from striate.util import pick_device
from striate.model import Backbone
from zoo import MultiScaleCrop
from ring import Ring, temporal_report
from smooth import turns
from warploop import penalized


class Volume:
    """Real 3D field on an S x S x Z torus; Z samples span Lz pixels of depth."""

    def __init__(self, S, Z, Lz, dev, sd=0.01, stretch=1.0, norm=False):
        self.S, self.Z, self.Lz, self.dev, self.stretch, self.norm = S, Z, Lz, dev, stretch, norm
        fy = np.fft.fftfreq(S)[None, :, None]
        fx = np.fft.rfftfreq(S)[None, None, :]
        # cycles per pixel of depth, weighted as if depth were `stretch` times finer: structure comes out
        # that many times longer along depth than across the image, so each feature lasts longer in the sweep
        fz = (np.fft.fftfreq(Z) * Z / Lz * stretch)[:, None, None]
        f = np.sqrt(fx * fx + fy * fy + fz * fz)
        scale = 1.0 / np.maximum(f, 1.0 / S) * S  # = FourierImage's scale on the kz = 0 plane
        if norm:
            # A slice sums all depth planes. At fine scales every plane contributes nearly equally, at coarse
            # scales only the lowest few, so the raw 3D weighting makes each slice's prior far whiter than a
            # still's 1/f. Rescale per spatial frequency so the planes' summed power equals the 2D weight:
            # every slice then sees exactly FourierImage's spectrum, and depth only decides how it varies.
            s2 = 1.0 / np.maximum(np.sqrt(fx * fx + fy * fy), 1.0 / S) * S
            scale = scale * s2 / np.sqrt((scale**2).sum(0, keepdims=True))
        self.W = S // 2 + 1
        self.N = 3 * S * self.W
        # (Z, 1, S, W) -> broadcast over colour, flattened to (Z, N) with the colour axis outermost in N
        sc = (
            torch.tensor(scale, dtype=torch.float32)[:, None]
            .expand(Z, 3, S, self.W)
            .reshape(Z, self.N)
        )
        self.scale = sc.to(dev)
        self.kz = (torch.fft.fftfreq(Z) * Z).to(dev)
        self.P = torch.nn.Parameter((torch.randn(2, Z, self.N) * sd).to(dev))

    def _spec(self, z):
        """(B, 3, S, W) complex 2D spectrum of the slices at depths z (sample units, float)."""
        th = 2 * math.pi * z[:, None] * self.kz[None] / self.Z  # (B, Z)
        c, s = torch.cos(th), torch.sin(th)
        Pr, Pi = self.P[0] * self.scale, self.P[1] * self.scale
        Gr, Gi = c @ Pr - s @ Pi, s @ Pr + c @ Pi
        return torch.complex(Gr, Gi).view(len(z), 3, self.S, self.W)

    def pre(self, z):
        img = torch.fft.irfft2(self._spec(z), s=(self.S, self.S)) / 4.0
        mat = color_correlation_normalized.to(img.device)
        return torch.einsum("bchw,dc->bdhw", img, mat)

    @torch.no_grad()
    def refit(self, S2, chunk=8):
        """A new Volume at resolution S2 reproducing this one (upsampled), z sampling unchanged."""
        new = Volume(
            S2, self.Z, self.Lz * S2 / self.S, self.dev, stretch=self.stretch, norm=self.norm
        )
        inv = torch.linalg.inv(color_correlation_normalized.to(self.dev))
        G = torch.empty(self.Z, 3, S2, new.W, dtype=torch.complex64, device=self.dev)
        z = torch.arange(self.Z, device=self.dev, dtype=torch.float32)
        for i in range(0, self.Z, chunk):
            x = torch.sigmoid(self.pre(z[i : i + chunk]))
            x = F.interpolate(x, size=(S2, S2), mode="bicubic", align_corners=False).clamp(
                1e-4, 1 - 1e-4
            )
            p = torch.logit(x)
            p = torch.einsum("bchw,dc->bdhw", p, inv) * 4.0
            G[i : i + chunk] = torch.fft.rfft2(p)
        C = torch.fft.fft(G, dim=0) / self.Z  # G(z) = sum_kz C[kz] e^{2 pi i kz z / Z}
        del G
        C = C.reshape(self.Z, new.N)
        new.P.data[0] = C.real / new.scale
        new.P.data[1] = C.imag / new.scale
        return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--target", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--z", type=int, default=64, help="depth samples")
    ap.add_argument(
        "--lz", type=float, default=128, help="depth period in pixels at the final size"
    )
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument(
        "--stretch", type=float, default=1.0, help="features this many times longer along depth"
    )
    ap.add_argument(
        "--final-steps", type=int, default=0, help="steps at the last octave (default: --steps)"
    )
    ap.add_argument(
        "--norm",
        action="store_true",
        help="per-frequency normalization: every slice gets the stills' 1/f prior",
    )
    ap.add_argument(
        "--lr-scale",
        type=float,
        default=0,
        help="lr multiplier (default 1/sqrt(z), or 0.5 with --norm)",
    )
    ap.add_argument(
        "--resume",
        help="volume.pt from an earlier run: skip the octaves and continue at its resolution",
    )
    ap.add_argument(
        "--sharpen-steps",
        type=int,
        default=0,
        help="final stage updating only detail finer than --sharpen-cut",
    )
    ap.add_argument(
        "--sharpen-cut",
        type=float,
        default=16,
        help="period in px below which the sharpening stage may change detail",
    )
    ap.add_argument(
        "--sharpen-lr", type=float, default=2.0, help="sharpening lr as a multiple of --lr"
    )
    ap.add_argument("--octaves", default="224,448,1024")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="2D-equivalent; divided by sqrt(z), since a slice sums z planes",
    )
    ap.add_argument("--mb", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
    net = Backbone(a.net, dev)
    net.hook(layer)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)
    sizes = [int(s) for s in a.octaves.split(",")]
    vol, curve = None, []
    if a.resume:
        st = torch.load(a.resume)
        vol = Volume(st["S"], st["Z"], st["Lz"], dev, stretch=st["stretch"], norm=st["norm"])
        vol.P.data.copy_(st["P"].to(dev))
        sizes = [st["S"]]
        log(f"resumed {a.resume}: {st['S']}px, {st['Z']} depth samples")
    for S in sizes if not a.resume else []:
        vol = (
            Volume(S, a.z, a.lz * S / a.size, dev, stretch=a.stretch, norm=a.norm)
            if vol is None
            else vol.refit(S)
        )
        steps = a.final_steps if (S == sizes[-1] and a.final_steps) else a.steps
        rg = Ring(net, layer, ch, S, dev)
        lr = a.lr * (a.lr_scale or (0.5 if a.norm else 1 / math.sqrt(a.z)))
        opt = torch.optim.Adam([vol.P], lr=lr)
        for it in range(steps):
            for g in opt.param_groups:  # cosine decay within each octave: settle before re-fitting
                g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * it / steps))
            opt.zero_grad()
            z = torch.rand(a.mb, device=dev) * a.z
            loss, act = penalized(rg, vol.pre(z), MultiScaleCrop(224 / S, 512))
            loss.backward()
            opt.step()
            if it % 128 == 0 or it == steps - 1:
                log(f"{S}px step {it + 1}/{steps} act mean {act.mean():.3f} min {act.min():.3f}")
            if S == sizes[-1] and (it % 256 == 255 or it == steps - 1):
                with torch.no_grad():
                    zs = torch.arange(12, device=dev, dtype=torch.float32) * a.z / 12
                    x = torch.sigmoid(vol.pre(zs))
                    c = 0
                    for k in range(2):
                        random.seed(2000 + k)
                        c = c + rg.act(MultiScaleCrop(224 / S, 512)(x)) / 2
                    random.seed(time.time_ns())
                curve.append((it + 1, float(c.mean())))
                log(f"curve step {it + 1}: fixed-slice act {c.mean():.3f}")
        del opt
        if dev == "mps":
            torch.mps.empty_cache()

    S, T = sizes[-1], a.frames
    rg = Ring(net, layer, ch, S, dev)
    if a.sharpen_steps:
        # only fine detail may change: mask the gradient to spatial periods below sharpen_cut, all depth planes
        fy = torch.fft.fftfreq(S, device=dev)[:, None]
        fx = torch.fft.rfftfreq(S, device=dev)[None, :]
        fine = (
            ((fx * fx + fy * fy).sqrt() > 1 / a.sharpen_cut)
            .float()
            .expand(3, S, vol.W)
            .reshape(1, 1, -1)
        )
        lr = a.lr * a.sharpen_lr
        opt = torch.optim.Adam([vol.P], lr=lr)
        for it in range(a.sharpen_steps):
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * it / a.sharpen_steps))
            opt.zero_grad()
            z = torch.rand(a.mb, device=dev) * a.z
            loss, act = penalized(rg, vol.pre(z), MultiScaleCrop(224 / S, 512))
            loss.backward()
            vol.P.grad.mul_(fine)
            opt.step()
            if it % 256 == 255 or it == a.sharpen_steps - 1:
                with torch.no_grad():
                    zs = torch.arange(12, device=dev, dtype=torch.float32) * a.z / 12
                    x = torch.sigmoid(vol.pre(zs))
                    c = 0
                    for k in range(2):
                        random.seed(2000 + k)
                        c = c + rg.act(MultiScaleCrop(224 / S, 512)(x)) / 2
                    random.seed(time.time_ns())
                curve.append((f"sharpen {it + 1}", float(c.mean())))
                log(f"sharpen step {it + 1}: fixed-slice act {c.mean():.3f}")
        del opt
    with torch.no_grad():
        frames = torch.cat(
            [
                torch.sigmoid(
                    vol.pre(
                        torch.arange(i, min(i + a.mb, T), device=dev, dtype=torch.float32) * a.z / T
                    )
                ).cpu()
                for i in range(0, T, a.mb)
            ]
        )
        tot = 0
        for k in range(4):
            r = []
            for i in range(0, T, a.mb):
                random.seed(1000 + k)
                r.append(rg.act(MultiScaleCrop(224 / S, 512)(frames[i : i + a.mb].to(dev))).cpu())
            tot = tot + torch.cat(r)
        act = (tot / 4).numpy()
        mean = frames.mean(0, keepdim=True)
        zvar = float(
            (frames - mean).norm() / mean.expand_as(frames).norm()
        )  # 0 = every slice the same
    torch.save(
        {
            "P": vol.P.detach().cpu(),
            "S": S,
            "Z": a.z,
            "Lz": vol.Lz,
            "stretch": a.stretch,
            "norm": a.norm,
        },
        f"{a.out}/volume.pt",
    )
    for i, f in enumerate(frames):
        Image.fromarray((f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
            f"{a.out}/f{i:03d}.png"
        )
    cos = turns(frames)
    rep = temporal_report(frames)
    rep.update(
        act_mean=float(act.mean()),
        act_min=float(act.min()),
        act_max=float(act.max()),
        zvar=zvar,
        turn_min=float(cos.min()),
        turn_mean=float(cos.mean()),
        act=act.tolist(),
        z=a.z,
        lz=a.lz,
        stretch=a.stretch,
        final_steps=a.final_steps or a.steps,
        norm=a.norm,
        curve=curve,
        sharpen_steps=a.sharpen_steps,
        sharpen_cut=a.sharpen_cut,
        sharpen_lr=a.sharpen_lr,
        secs=round(time.time() - t0),
    )
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(
        f"done act {act.mean():.3f} [{act.min():.3f}, {act.max():.3f}] slice variation {zvar:.3f} turn cos min {cos.min():.2f} "
        + " ".join(f"{k} R {v['R']:.2f}" for k, v in rep.items() if isinstance(v, dict))
    )


if __name__ == "__main__":
    main()
