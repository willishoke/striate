"""Labeled contact sheet of a corpus (channel# labeled), for picking morph pairs.
Instant -- renders from saved spectra, no optimization."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw

from striate.param import FourierImage
from striate.util import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="outputs/corpus.pt")
    ap.add_argument("--out", default="outputs/endpoint_sheet.png")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--tile", type=int, default=192)
    args = ap.parse_args()

    dev = pick_device()
    blob = torch.load(args.corpus, weights_only=False)
    spectra, channels, size = blob["spectra"].to(dev), blob["channels"], blob["size"]
    layer = blob.get("layer", "?")

    fp = FourierImage(size, size, batch=len(channels), device=dev)
    with torch.no_grad():
        fp.spectrum.copy_(spectra)
        imgs = fp.to_image().cpu()

    cols, T = args.cols, args.tile
    rows = (len(channels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * T, rows * T), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for i, ch in enumerate(channels):
        arr = (imgs[i].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        x, y = (i % cols) * T, (i // cols) * T
        sheet.paste(Image.fromarray(arr).resize((T, T)), (x, y))
        d.rectangle([x + 3, y + 3, x + 46, y + 20], fill=(0, 0, 0))
        d.text((x + 6, y + 5), f"{ch}", fill=(255, 255, 0))
    sheet.save(args.out)
    print(f"{len(channels)} {layer} endpoints -> {args.out} ({cols}x{rows} @ {T}px, channel# labeled)")


if __name__ == "__main__":
    main()
