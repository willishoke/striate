"""Command-line entry point.

Examples:
    python -m striate.cli --layer features.5 --channel 42 --size 512
    python -m striate.cli --layer features.7 --channel 100 --progressive
    python -m striate.cli --list-layers
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

from .model import Backbone
from .objectives import ChannelObjective, LayerObjective, NeuronObjective
from .render import render, render_progressive
from .util import pick_device, save_image


def _build_objective(args):
    if args.objective == "channel":
        return ChannelObjective(args.layer, args.channel)
    if args.objective == "layer":
        return LayerObjective(args.layer)
    return NeuronObjective(args.layer, args.channel)


def main():
    p = argparse.ArgumentParser(description="Modern hi-res feature visualization (Distill 2017 method).")
    p.add_argument("--model", default="tiny", choices=["tiny", "small", "base"])
    p.add_argument("--layer", default="features.5", help="module path, e.g. features.5")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--objective", default="channel", choices=["channel", "layer", "neuron"])
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--no-decorrelate", action="store_true")
    p.add_argument("--progressive", action="store_true", help="octave render toward hi-res")
    p.add_argument("--sizes", default="256,384,512,768")
    p.add_argument("--width", type=int, default=None, help="target width for a 16:9 progressive render")
    p.add_argument("--height", type=int, default=None, help="target height for a 16:9 progressive render")
    p.add_argument("--out", default=None)
    p.add_argument("--list-layers", action="store_true")
    args = p.parse_args()

    device = pick_device()
    print(f"device: {device}  model: convnext_{args.model}")
    backbone = Backbone(args.model, device=device)

    if args.list_layers:
        for name in backbone.list_layers():
            print(name)
        return

    objective = _build_objective(args)
    layers = [args.layer]
    decorrelate = not args.no_decorrelate

    if args.progressive:
        if args.width and args.height:
            # octave schedule scaled to the target aspect ratio
            tw, th = args.width, args.height
            sizes = [(round(th * f), round(tw * f)) for f in (0.25, 0.5, 0.75, 1.0)]
        else:
            sizes = tuple(int(s) for s in args.sizes.split(","))
        image = render_progressive(
            backbone, objective, layers, sizes=sizes, steps=args.steps,
            lr=args.lr, decorrelate=decorrelate, device=device,
        )
    else:
        image = render(
            backbone, objective, layers, size=args.size, steps=args.steps,
            lr=args.lr, decorrelate=decorrelate, device=device,
        )

    out = args.out or f"outputs/{args.model}_{args.layer.replace('.', '-')}_ch{args.channel}.png"
    save_image(image, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
