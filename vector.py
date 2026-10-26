"""Feature visualization as vector art: optimize a stack of Fourier-outlined shapes.

Same objective as the stills (one channel's mean activation), different image
parameterization (``striate.vector.VectorImage``). Edge softness anneals from soft
(so gradients reach shapes from a distance) to sub-pixel (crisp), and the result is
written as an SVG plus a PNG re-rendered at ``--out-size``.

    uv run python vector.py --layer features.3 --channel 191
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time

import torch
from tqdm import trange

from striate.model import Backbone
from striate.objectives import ChannelObjective
from striate.util import pick_device, save_image
from striate.vector import _SOFT_MAX, VectorImage, random_camera


def optimize(
    backbone,
    objective,
    image: VectorImage,
    *,
    size: int,
    steps: int,
    lr: float,
    soft_start: float,
    soft_end: float,
    device: str,
    edge_cost: float = 0.0,
    soft_cost: float = 0.0,
    progress: bool = True,
):
    params = [p for p in image.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    iterator = trange(steps, desc=f"vector {size}px") if progress else range(steps)
    for i in iterator:
        # geometric anneal: soft edges early for long-range gradients, crisp at the end
        softness = soft_start * (soft_end / soft_start) ** (i / max(steps - 1, 1))
        optimizer.zero_grad()
        x = image.render(size, size, softness, camera=random_camera(device=device))
        loss = objective(backbone(x))
        if edge_cost or soft_cost:
            # costs are fractions of the current activation, so one setting means
            # the same trade on channels whose activations differ in scale
            cost = edge_cost * image.offset_norm().mean()
            cost = cost + soft_cost * image.edge_width().mean() / _SOFT_MAX
            loss = loss + loss.detach().abs() * cost
        loss.backward()
        optimizer.step()
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="tiny", choices=["tiny", "small", "base"])
    p.add_argument("--layer", default="features.3")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--shapes", type=int, default=128)
    p.add_argument("--harmonics", type=int, default=8)
    p.add_argument(
        "--radius", default="0.03,0.35", help="initial shape radius range, in image heights"
    )
    p.add_argument("--decay", type=float, default=1.0, help="outline preconditioning 1/k**decay")
    p.add_argument("--fill", default="linear", choices=["flat", "linear"])
    p.add_argument("--no-decorrelate", action="store_true", help="optimize fills in raw RGB")
    p.add_argument("--alpha", action="store_true", help="learn per-shape opacity")
    p.add_argument(
        "--field",
        type=int,
        default=0,
        help="smooth base field up to N cycles; shapes paint offsets",
    )
    p.add_argument("--learn-softness", action="store_true", help="learn per-shape edge width")
    p.add_argument(
        "--edge-cost", type=float, default=0.0, help="penalty on shape offsets from the field"
    )
    p.add_argument("--soft-cost", type=float, default=0.0, help="penalty on learned edge width")
    p.add_argument("--size", type=int, default=224, help="optimization resolution")
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--soft-start", type=float, default=4.0, help="edge width in px, first step")
    p.add_argument("--soft-end", type=float, default=0.5, help="edge width in px, last step")
    p.add_argument("--out-size", type=int, default=1024, help="PNG re-render resolution")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output path prefix (.svg and .png)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    backbone = Backbone(args.model, device=device)
    backbone.hook(args.layer)
    image = VectorImage(
        shapes=args.shapes,
        harmonics=args.harmonics,
        decay=args.decay,
        fill=args.fill,
        learn_alpha=args.alpha,
        decorrelate=not args.no_decorrelate,
        radius_range=tuple(float(r) for r in args.radius.split(",")),
        field=args.field,
        learn_softness=args.learn_softness,
        device=device,
    )
    t0 = time.time()
    optimize(
        backbone,
        ChannelObjective(args.layer, args.channel),
        image,
        size=args.size,
        steps=args.steps,
        lr=args.lr,
        soft_start=args.soft_start,
        soft_end=args.soft_end,
        device=device,
        edge_cost=args.edge_cost,
        soft_cost=args.soft_cost,
    )
    objective = ChannelObjective(args.layer, args.channel)
    with torch.no_grad():
        acts = [
            -objective(
                backbone(
                    image.render(args.size, args.size, args.soft_end, random_camera(device=device))
                )
            ).item()
            for _ in range(16)
        ]
    print(f"activation {sum(acts) / len(acts):.3f} (mean over 16 random cameras)")
    if args.field:
        with torch.no_grad():
            norm = image.offset_norm()
            width = image.edge_width() * args.out_size
        print(
            f"offsets: mean {norm.mean():.3f}, visible (>0.1) {(norm > 0.1).float().mean():.0%}"
            f"; edge width at {args.out_size}px: median {width.median():.1f}px"
        )
    backbone.clear_hooks()

    prefix = args.out or f"outputs/vector_{args.layer.replace('.', '-')}_ch{args.channel}"
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    image.save(prefix + ".pt")
    if not args.field:
        with open(prefix + ".svg", "w") as f:
            f.write(image.to_svg(args.out_size, args.out_size))
    with torch.no_grad():
        # half-pixel edges at the output size: antialiased, not blurred
        save_image(image.render(args.out_size, args.out_size, 0.5), prefix + ".png")
    kinds = ".png, .pt" if args.field else ".svg, .png, .pt"
    print(f"saved: {prefix}{kinds}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
