"""Octave renders across backbones: 224 -> 1792 in sqrt(2) steps, one objective throughout.

Each octave seeds a FourierImage from the bicubic-upsampled previous result, then keeps
maximizing the same channel with the network looking at the canvas at 1:1 (random
crops of at most 512 once the canvas outgrows that, which is an unbiased estimate of
the full-image channel mean). The composition is set at 224; every later octave is the
network adding its own native-scale detail inside the enlarged forms.

  free:   all frequencies stay trainable (what striate's render_progressive does)
  joint:  all frequencies free, but each step views the canvas at a random scale
          (whole canvas at 224 up to 1:1), so composition and detail are both rewarded
  banded: frequencies the previous octave already resolved well (below half its
          Nyquist) are frozen, so later octaves can only add detail, not repaint

Usage: octave.py --net convnext --targets features.3:191,features.5:308 --mode banded --out DIR
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, math, random, time
import numpy as np
import torch
import torch.nn.functional as F
from striate.param import FourierImage, seed_spectrum_from_image
from striate.transforms import TransformRobustness
from striate.util import pick_device, save_image
from zoo import Net, MultiScaleCrop

SIZES = [224, 320, 448, 640, 896, 1280, 1792]
CROP = 512
LOGIT_RAIL = 2.944  # logit(0.95)
CHROMA_RATIO = 0.2  # (pixel-scale chroma / luma energy) of the 224 starts, ~0.44^2


def hf_energy(x):
    """Per-item energy of the pixel-scale (x - 3x3 mean) residual."""
    return ((x - F.avg_pool2d(F.pad(x, [1] * 4, mode="reflect"), 3, 1)) ** 2).mean((1, 2, 3))


def penalties(param, img):
    """Logit hinge (pixels driven past the rails) and pixel-scale chroma relative to luma."""
    hinge = (torch.relu(param.preact().abs() - LOGIT_RAIL) ** 2).mean((1, 2, 3))
    y = (img * torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)).sum(
        1, keepdim=True
    )
    ratio = hf_energy(torch.cat([img[:, 2:3] - y, img[:, 0:1] - y], 1)) / hf_energy(y).detach()
    return hinge, torch.relu(ratio / CHROMA_RATIO - 1)


class Stop(Exception):
    pass


def truncate(net, layers):
    """Stop the forward pass once every target layer has fired."""
    want = set(layers)

    def done(m, i, o):
        if want <= net.acts.keys():
            raise Stop

    for l in layers:
        net.hooks.append(net.mods[l].register_forward_hook(done))


def acts_of(net, x, targets):
    net.acts = {}
    try:
        net(x)
    except Stop:
        pass
    return [net.acts[l][b, c].mean() for b, (l, c) in enumerate(targets)]


def band_mask(param, prev):
    """1 where the spectrum may move, 0 below half the previous octave's Nyquist."""
    fy = np.abs(np.fft.fftfreq(param.height))[:, None]
    fx = np.fft.rfftfreq(param.width)[None, :]
    lim = 0.25 * prev / param.height
    keep = ~((fy < lim) & (fx < lim))
    return torch.tensor(keep, dtype=torch.float32)[None, None, ..., None]


def crop(x, tf):
    _, _, h, w = x.shape
    if h > CROP:
        top, left = random.randint(0, h - CROP), random.randint(0, w - CROP)
        x = x[:, :, top : top + CROP, left : left + CROP]
    return tf(x)


@torch.no_grad()
def measure(net, img, targets, tf, n=16):
    """Overview: whole canvas shown at 224. Native: mean over 1:1 crops."""
    small = F.interpolate(
        img, size=(224, 224), mode="bilinear", antialias=True, align_corners=False
    )
    over = torch.stack([torch.stack(acts_of(net, tf(small), targets)) for _ in range(n)]).mean(0)
    nat = torch.stack([torch.stack(acts_of(net, crop(img, tf), targets)) for _ in range(n)]).mean(0)
    return over.tolist(), nat.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--mode", choices=["free", "banded", "joint"], required=True)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument(
        "--variants",
        default="",
        help="a:b,... one batch item per (hinge, chroma) weight pair; needs one target",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = pick_device()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    targets = [(t.split(":")[0], int(t.split(":")[1])) for t in a.targets.split(",")]
    variants = (
        [tuple(map(float, v.split(":"))) for v in a.variants.split(",")] if a.variants else []
    )
    if variants:
        targets = targets[:1] * len(variants)
        wh = torch.tensor([v[0] for v in variants], device=dev)
        wc = torch.tensor([v[1] for v in variants], device=dev)
    sizes = [int(s) for s in a.sizes.split(",")]
    net = Net(a.net, dev)
    for l in {l for l, _ in targets}:
        net.hook(l)
    truncate(net, {l for l, _ in targets})
    tf = TransformRobustness()
    os.makedirs(a.out, exist_ok=True)
    stats, img, t0 = {}, None, time.time()
    for k, size in enumerate(sizes):
        param = FourierImage(size, size, batch=len(targets), device=dev)
        mask = None
        if img is not None:
            up = F.interpolate(img, size=(size, size), mode="bicubic", align_corners=False).clamp(
                0, 1
            )
            seed_spectrum_from_image(param, up)
            if a.mode == "banded":
                mask = band_mask(param, sizes[k - 1]).to(dev)
        opt = torch.optim.Adam([param.spectrum], lr=a.lr)
        view = MultiScaleCrop(224 / size, CROP) if a.mode == "joint" else (lambda x: crop(x, tf))
        for _ in range(a.steps):
            opt.zero_grad()
            img = param.to_image()
            act = torch.stack(acts_of(net, view(img), targets))
            loss = -act.sum()
            if variants:
                hinge, chroma = penalties(param, img)
                loss = loss + (act.detach().abs() * (wh * hinge + wc * chroma)).sum()
            loss.backward()
            if mask is not None:
                param.spectrum.grad.mul_(mask)
            opt.step()
        img = param.to_image().detach()
        over, nat = measure(net, img, targets, tf)
        for b, (l, c) in enumerate(targets):
            key = f"{a.net}_{l.replace('.', '-')}_{c}"
            if variants:
                key += "_h{:g}c{:g}".format(*variants[b])
            save_image(img[b : b + 1], f"{a.out}/{key}_{a.mode}_{size}.png")
            stats.setdefault(key, []).append(dict(size=size, overview=over[b], native=nat[b]))
        print(
            f"{a.net} {a.mode} {size}px {time.time() - t0:.0f}s "
            + " ".join(f"{o:.3f}/{n:.3f}" for o, n in zip(over, nat)),
            flush=True,
        )
        del param, opt
        if dev == "mps":
            torch.mps.empty_cache()
    json.dump(stats, open(f"{a.out}/{a.net}_{a.mode}_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
