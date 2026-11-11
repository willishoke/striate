"""Re-render a sheet still at 2-4x, keeping its composition, so the segmenter sees clean pixels.

The 512 px stills (zoo.py sheet) carry the network's pixel-scale checkerboard. Here the still is
carried up by joint octaves (as in sheet.py / octave.py): each octave seeds a FourierImage from
the periodic bicubic upsample of the previous one, with the seed's top octave removed at the first
step (that is where the checkerboard lives), then keeps maximizing the same channel. The network
sees the canvas at most at --detail times the still's own scale: at 1 (default) it redraws the
still's own detail as clean hires pixels; at 2 it adds an octave of new structure (on ch7 that cuts
shells into chambers, which a segmenter reads as parts). An anchor ties the blurred downsample to
the blurred still, so objects stay where the 512 px labels say they are.

usage: stillup.py still.png out.png --target inception4c:7 [--net googlenet] [--factor 4]
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
from striate.model import Backbone
from striate.param import FourierImage, seed_spectrum_from_image
from striate.util import pick_device
from striate.transforms import TransformRobustness


def pup(x, size):
    """Periodic bicubic resize (x tiles)."""
    h = x.shape[-1]
    p = 4
    xp = torch.cat([x[..., -p:, :], x, x[..., :p, :]], -2)
    xp = torch.cat([xp[..., -p:], xp, xp[..., :p]], -1)
    k = size / h
    y = F.interpolate(xp, scale_factor=k, mode="bicubic", align_corners=False)
    q = round(p * k)
    return y[..., q : q + size, q : q + size]


def pdown(x, size):
    """Periodic antialiased downsample."""
    h = x.shape[-1]
    p = 8 * (h // size)
    xp = torch.cat([x[..., -p:, :], x, x[..., :p, :]], -2)
    xp = torch.cat([xp[..., -p:], xp, xp[..., :p]], -1)
    k = size / h
    y = F.interpolate(xp, scale_factor=k, mode="bilinear", antialias=True, align_corners=False)
    q = round(p * k)
    return y[..., q : q + size, q : q + size]


def lowpass(x, keep):
    """Keep spatial frequencies below `keep` x Nyquist (periodic, raised-cosine roll-off)."""
    H, W = x.shape[-2:]
    fy = torch.fft.fftfreq(H, device=x.device).abs()[:, None] * 2
    fx = torch.fft.rfftfreq(W, device=x.device)[None, :] * 2
    f = torch.maximum(fy, fx)
    m = (0.5 + 0.5 * torch.cos(math.pi * ((f - 0.8 * keep) / (0.4 * keep)).clamp(0, 1)))
    return torch.fft.irfft2(torch.fft.rfft2(x) * m, s=(H, W))


def blur(x, sigma):
    r = int(3 * sigma + 0.5)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-(t**2) / (2 * sigma**2))
    g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="circular"), g.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="circular"), g.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("still")
    ap.add_argument("out")
    ap.add_argument("--target", required=True, help="layer:channel")
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--steps", type=int, default=256, help="per octave")
    ap.add_argument("--detail", type=float, default=1.0, help="max view zoom relative to the still")
    ap.add_argument("--keep", type=float, default=0.5, help="share of the still's band kept in the seed")
    ap.add_argument("--anchor", type=float, default=4.0)
    ap.add_argument("--anchor-sigma", type=float, default=1.5, help="px at the still's scale")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = pick_device()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    layer, ch = a.target.split(":")
    ch = int(ch)
    net = Backbone(a.net, dev)
    net.hook(layer)

    still = torch.tensor(np.asarray(Image.open(a.still).convert("RGB")), dtype=torch.float32, device=dev)
    still = still.permute(2, 0, 1)[None] / 255
    S = still.shape[-1]
    ref = blur(still, a.anchor_sigma)
    img = lowpass(still, a.keep)  # the still's top octave (the checkerboard) is redrawn, not carried
    log = {"args": vars(a), "octaves": []}
    t0 = time.time()
    size = S
    while size < a.factor * S:
        size *= 2
        k = size / S
        param = FourierImage(size, size, device=dev)
        seed_spectrum_from_image(param, pup(img, size).clamp(0.002, 0.998))
        opt = torch.optim.Adam([param.spectrum], lr=0.05)
        tf, smin, smax = TransformRobustness(), 224 / size, min(1.0, a.detail / k)
        for it in range(a.steps):
            opt.zero_grad()
            x = param.to_image()
            # zoo.MultiScaleCrop, with the scale capped at smax
            s = math.exp(random.uniform(math.log(smin), math.log(smax)))
            wh = min(size, round(512 / s))
            top, left = random.randint(0, size - wh), random.randint(0, size - wh)
            v = x[:, :, top : top + wh, left : left + wh]
            o = max(1, round(wh * s))
            if o != wh:
                v = F.interpolate(v, size=(o, o), mode="bilinear", antialias=True, align_corners=False)
            act = net(tf(v))[layer][0, ch].mean()
            anc = ((blur(pdown(x, S), a.anchor_sigma) - ref) ** 2).mean()
            (-act + act.detach().abs() * a.anchor * anc * 100).backward()
            opt.step()
            if (it + 1) % 64 == 0:
                print(f"[{time.time() - t0:.0f}s] {size}px step {it + 1} act {act.item():.3f} anchor rms {anc.sqrt().item():.4f}", flush=True)
        img = param.to_image().detach()
        log["octaves"].append({"size": size, "act": round(act.item(), 4), "anchor_rms": round(anc.sqrt().item(), 5)})
        del param, opt
        if dev == "mps":
            torch.mps.empty_cache()
    out = (img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(out).save(a.out)
    with torch.no_grad():  # fidelity: the result shrunk back to the still's size, against the still
        d = pdown(img, S) - still
        log["rms_vs_still"] = round(d.pow(2).mean().sqrt().item(), 4)
        log["rms_vs_still_blur"] = round((blur(pdown(img, S), a.anchor_sigma) - ref).pow(2).mean().sqrt().item(), 4)
    json.dump(log, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
