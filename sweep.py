"""Lo-res channel sweep + contact sheet, for browsing before committing compute.

Loads the backbone once and renders a list of channels at low resolution, then
tiles them into a single labeled contact sheet so you can pick favorites.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time

from PIL import Image, ImageDraw

from striate.model import Backbone
from striate.objectives import ChannelObjective
from striate.render import render_progressive
from striate.util import pick_device, save_image


def contact_sheet(entries, cols, out, thumb_w=384):
    th = int(thumb_w * 9 / 16)
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * th), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(entries):
        thumb = Image.open(path).resize((thumb_w, th))
        r, c = divmod(i, cols)
        x, y = c * thumb_w, r * th
        sheet.paste(thumb, (x, y))
        draw.rectangle([x + 4, y + 4, x + 64, y + 22], fill=(0, 0, 0))
        draw.text((x + 8, y + 8), label, fill=(255, 255, 0))
    sheet.save(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tiny")
    p.add_argument("--layer", default="features.3")
    p.add_argument("--channels", default="0,20,40,60,80,100,120,140,160,180")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--sheet", default="outputs/sweep_contact.png")
    args = p.parse_args()

    device = pick_device()
    print(f"device: {device}  sweeping {args.layer} at {args.width}x{args.height}")
    backbone = Backbone(args.model, device=device)
    channels = [int(c) for c in args.channels.split(",")]

    os.makedirs("outputs/sweep", exist_ok=True)
    entries = []
    for ch in channels:
        t0 = time.time()
        objective = ChannelObjective(args.layer, ch)
        image = render_progressive(
            backbone,
            objective,
            [args.layer],
            sizes=[(args.height, args.width)],
            steps=args.steps,
            device=device,
            progress=False,
        )
        path = f"outputs/sweep/{args.layer.replace('.', '-')}_ch{ch}.png"
        save_image(image, path)
        entries.append((path, f"ch{ch}"))
        print(f"  ch{ch:>3} -> {path}  ({time.time() - t0:.0f}s)")

    out = contact_sheet(entries, args.cols, args.sheet)
    print(f"contact sheet -> {out}")


if __name__ == "__main__":
    main()
