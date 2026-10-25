"""Interpolate between two saved VectorImages: does geometry morph where pixels crossfade?

Three paths from render A to render B, scored by both endpoints' channel activations:

  * index   -- shape i of A becomes shape i of B (arbitrary pairing: shapes fly)
  * matched -- an optimal assignment pairs shapes by position, size and color first
  * fade    -- pixel crossfade of the two renders, the raster baseline

Painting order is discrete. Each pair interpolates its depth rank from A's order to
B's, and every frame paints in the current rank order, so both endpoints are exact
and two overlapping shapes swap where their ranks cross.

    uv run python experiments/vector_interp.py \
        --a outputs/vint/3-191_s0.pt --a-target features.3:191 \
        --b outputs/vint/3-191_s1.pt --b-target features.3:191 --out outputs/vint/same
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import math
import random

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from striate.model import Backbone
from striate.objectives import ChannelObjective
from striate.util import pick_device
from striate.vector import _LOG_RADIUS_SCALE, VectorImage, random_camera

PARAMS = ("center", "log_radius", "harm", "color", "grad_angle", "alpha")


def descriptors(img):
    with torch.no_grad():
        return img.centers(), img.log_radius * _LOG_RADIUS_SCALE, img.fills().mean(1)


def match(a, b, pos_scale=0.05, rad_scale=0.5, col_scale=0.2):
    """perm[i] = index in B paired with shape i of A (Hungarian on a mixed distance)."""
    ca, ra, ka = descriptors(a)
    cb, rb, kb = descriptors(b)
    cost = (
        (torch.cdist(ca, cb) / pos_scale) ** 2
        + ((ra[:, None] - rb[None, :]) / rad_scale) ** 2
        + (torch.cdist(ka, kb) / col_scale) ** 2
    )
    _, perm = linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(perm, device=a.center.device)


class Path:
    """Shape-wise interpolation from A to B under a pairing ``perm``."""

    def __init__(self, a, b, perm):
        self.perm = perm
        self.bg = (a.background.detach(), b.background.detach())
        self.src = {n: getattr(a, n).detach() for n in PARAMS}
        self.dst = {n: getattr(b, n).detach()[perm] for n in PARAMS}
        # gradient direction: interpolate along the short arc
        d = self.dst["grad_angle"] - self.src["grad_angle"]
        self.dst["grad_angle"] = self.src["grad_angle"] + (d + math.pi) % (2 * math.pi) - math.pi
        k = perm.numel()
        self.rank_a = torch.arange(k, device=perm.device, dtype=torch.float32)
        self.rank_b = perm.float()  # B paints in index order, so its rank is its index
        self.frame = VectorImage(**a.config, device=a.center.device)

    @torch.no_grad()
    def at(self, t: float) -> VectorImage:
        order = torch.argsort((1 - t) * self.rank_a + t * self.rank_b)
        for n in PARAMS:
            v = (1 - t) * self.src[n] + t * self.dst[n]
            getattr(self.frame, n).copy_(v[order])
        self.frame.background.copy_((1 - t) * self.bg[0] + t * self.bg[1])
        return self.frame

    def mean_travel(self) -> float:
        """Average center displacement, in image heights."""
        a = self.src["center"] * 0.1
        b = self.dst["center"] * 0.1
        return float((b - a).norm(dim=-1).mean())


def render(img, size, camera=None):
    return img.render(size, size, 0.5, camera=camera)


@torch.no_grad()
def score_curve(backbone, objectives, frame_fn, ts, cameras, size=224):
    out = {name: [] for name in objectives}
    for t in ts:
        acc = {name: 0.0 for name in objectives}
        for cam in cameras:
            acts = backbone(frame_fn(t, cam, size))
            for name, obj in objectives.items():
                acc[name] += -obj(acts).item()
        for name in objectives:
            out[name].append(acc[name] / len(cameras))
    return out


def to_pil(x):
    arr = (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def ease(u):
    return u * u * (3 - 2 * u)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--a-target", required=True, help="layer:channel, e.g. features.3:191")
    p.add_argument("--b", required=True)
    p.add_argument("--b-target", required=True)
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--points", type=int, default=21, help="t samples for the activation curve")
    p.add_argument("--cameras", type=int, default=8)
    p.add_argument("--frames", type=int, default=36, help="frames per direction")
    p.add_argument("--anim-size", type=int, default=320)
    args = p.parse_args()

    device = pick_device()
    a = VectorImage.load(args.a, device)
    b = VectorImage.load(args.b, device)
    backbone = Backbone("tiny", device=device)
    objectives = {}
    for key, spec in (("A", args.a_target), ("B", args.b_target)):
        layer, ch = spec.split(":")
        objectives[f"{key} {spec}"] = ChannelObjective(layer, int(ch))
    for layer in {o.layer for o in objectives.values()}:
        backbone.hook(layer)

    k = a.center.shape[0]
    paths = {
        "index": Path(a, b, torch.arange(k, device=device)),
        "matched": Path(a, b, match(a, b)),
    }
    random.seed(0)
    cameras = [random_camera(device=device) for _ in range(args.cameras)]
    ts = np.linspace(0, 1, args.points).tolist()

    def path_frame(path):
        return lambda t, cam, size: render(path.at(t), size, cam)

    def fade_frame(t, cam, size):
        return (1 - t) * render(a, size, cam) + t * render(b, size, cam)

    results = {"t": ts}
    for name, path in paths.items():
        results[name] = score_curve(backbone, objectives, path_frame(path), ts, cameras)
        results[name]["travel"] = path.mean_travel()
    results["fade"] = score_curve(backbone, objectives, fade_frame, ts, cameras)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(results, f, indent=1)

    for name, curve in results.items():
        if name == "t":
            continue
        mids = {o: f"{v[len(ts) // 2]:.3f}" for o, v in curve.items() if isinstance(v, list)}
        extra = f"  travel {curve['travel']:.3f}" if "travel" in curve else ""
        print(f"{name:8s} midpoint {mids}{extra}")

    # animations: eased there and back, so the loop is seamless
    us = [ease(i / args.frames) for i in range(args.frames)]
    loop = us + [1 - u for u in us]
    s = args.anim_size
    with torch.no_grad():
        for name, path in paths.items():
            frames = [to_pil(render(path.at(t), s)) for t in loop]
            frames[0].save(
                f"{args.out}_{name}.webp",
                save_all=True,
                append_images=frames[1:],
                duration=50,
                loop=0,
                quality=88,
            )
        frames = [to_pil(fade_frame(t, None, s)) for t in loop]
        frames[0].save(
            f"{args.out}_fade.webp",
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
            quality=88,
        )
        # contact strip at t = 0, 1/4, 1/2, 3/4, 1 for each path
        strip = Image.new("RGB", (5 * s, 3 * s))
        for r, name in enumerate(["index", "matched", "fade"]):
            for c, t in enumerate([0, 0.25, 0.5, 0.75, 1]):
                x = render(paths[name].at(t), s) if name in paths else fade_frame(t, None, s)
                strip.paste(to_pil(x), (c * s, r * s))
        strip.save(args.out + "_strip.png")
    print(f"saved: {args.out}.json, _index/_matched/_fade.webp, _strip.png")


if __name__ == "__main__":
    main()
