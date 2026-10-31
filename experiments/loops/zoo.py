"""Multi-scale crop renderer for the 512 / 1024 / 4K backbone studies (any striate.model.NETS name).

Usage:
  zoo.py sheet  --net googlenet --layers inception3a,... --per 6 --size 512 --out DIR
  zoo.py pick   --net resnet50_v1 --targets layer2:12,layer3:7 --size 512 --out DIR
  zoo.py big    --net googlenet --targets inception4b:200 --h 2160 --w 3840 --steps 2048 --out DIR
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, math, random, sys, time
import torch
import torch.nn.functional as F
from striate.model import Backbone
from striate.param import FourierImage
from striate.transforms import TransformRobustness
from striate.util import pick_device, save_image


class MultiScaleCrop:
    """Pick a scale s log-uniform in [min_scale, 1]; take a window of the canvas that
    becomes at most crop x crop pixels at that scale; resize it; then the usual
    robustness transforms. Small s shows the whole composition, s = 1 shows native
    detail of one region."""

    def __init__(self, min_scale, crop):
        self.m, self.crop, self.tf = min_scale, crop, TransformRobustness()

    def __call__(self, x):
        _, _, h, w = x.shape
        s = math.exp(random.uniform(math.log(self.m), 0.0))
        wh, ww = min(h, round(self.crop / s)), min(w, round(self.crop / s))
        top, left = random.randint(0, h - wh), random.randint(0, w - ww)
        x = x[:, :, top : top + wh, left : left + ww]
        oh, ow = max(1, round(wh * s)), max(1, round(ww * s))
        if (oh, ow) != (wh, ww):
            x = F.interpolate(
                x, size=(oh, ow), mode="bilinear", antialias=True, align_corners=False
            )
        return self.tf(x)


def render(net, targets, h, w, steps, crop, dev, lr=0.05, seed=0, label=""):
    """Optimize one image per (layer, channel) target, as a batch."""
    torch.manual_seed(seed)
    random.seed(seed)
    net.clear_hooks()
    for layer in {l for l, _ in targets}:
        net.hook(layer)
    param = FourierImage(h, w, batch=len(targets), device=dev)
    opt = torch.optim.Adam([param.spectrum], lr=lr)
    tf = MultiScaleCrop(224 / min(h, w), crop)
    t0 = time.time()
    for i in range(steps):
        opt.zero_grad()
        acts = net(tf(param.to_image()))
        loss = -sum(acts[l][b, c].mean() for b, (l, c) in enumerate(targets))
        loss.backward()
        opt.step()
        if (i + 1) % max(1, steps // 8) == 0:
            print(f"  {label} step {i + 1}/{steps} {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        return param.to_image().detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sheet", "pick", "big"])
    ap.add_argument("--net", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--targets", default="")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--h", type=int, default=0)
    ap.add_argument("--w", type=int, default=0)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = pick_device()
    net = Backbone(a.net, dev)
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "sheet":
        rng = random.Random(a.seed)
        targets = []
        for layer in a.layers.split(","):
            n = net.channels(layer)
            targets += [(layer, c) for c in sorted(rng.sample(range(n), a.per))]
    else:
        targets = [(t.split(":")[0], int(t.split(":")[1])) for t in a.targets.split(",")]
    h, w = (a.h, a.w) if a.h else (a.size, a.size)
    for i in range(0, len(targets), a.batch):
        chunk = targets[i : i + a.batch]
        imgs = render(
            net, chunk, h, w, a.steps, a.crop, dev, seed=a.seed, label=f"{a.net} {chunk[0][0]}"
        )
        for b, (l, c) in enumerate(chunk):
            save_image(imgs[b : b + 1], f"{a.out}/{a.net}_{l.replace('.', '-')}_{c}.png")
        print(f"{a.net} saved {[(l, c) for l, c in chunk]}", flush=True)
    json.dump(targets, open(f"{a.out}/{a.net}_targets.json", "w"))


if __name__ == "__main__":
    main()
