"""Off-the-shelf super-res upscaling of rendered frames (Real-ESRGAN via spandrel).

The feature scale is fixed in pixels, so the texture only reads at its native render
size -- rendering bigger just adds repetitions and loses it. So: render the swim at
native scale, then let an SR model hallucinate the high-frequency detail up to delivery
resolution. Upscales a directory of PNG frames and re-encodes a gif.

With --down N the frames go back down to N px (Lanczos) after the upscale. That round
trip keeps the size but trades the render's finest stipple for crisper outlines and
more contrast; it is the last step of the README loop.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import glob

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader

from striate.util import pick_device


def upscale_image(model, im, device):
    x = (
        torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1)[None].float().to(device)
        / 255.0
    )
    with torch.no_grad():
        y = model(x)
    arr = (y[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="directory of PNG frames")
    ap.add_argument("--model", default="models/RealESRGAN_x4plus.pth")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--down", type=int, default=0, help="resize back to this many px (square)")
    ap.add_argument("--gif", default=None)
    ap.add_argument(
        "--no-gif", action="store_true", help="only write frames (don't hold them all in memory)"
    )
    args = ap.parse_args()

    dev = pick_device()
    model = ModelLoader().load_from_file(args.model).to(dev).eval()
    paths = sorted(glob.glob(f"{args.frames}/*.png"))
    tag = f"_x{model.scale}" + (f"_down{args.down}" if args.down else "")
    outdir = args.frames.rstrip("/") + tag
    os.makedirs(outdir, exist_ok=True)

    ups = []
    for p in paths:
        out = upscale_image(model, Image.open(p), dev)
        if args.down:
            out = out.resize((args.down, args.down), Image.LANCZOS)
        if dev == "mps":
            torch.mps.empty_cache()
        out.save(f"{outdir}/{os.path.basename(p)}")
        if not args.no_gif:
            ups.append(out)
    if not paths:
        print(f"no frames in {args.frames}")
        return
    if args.no_gif:
        print(f"upscaled {len(paths)} frames {tag} -> {out.size}  dir {outdir}/")
        return

    gif = args.gif or (args.frames.rstrip("/") + f"{tag}.gif")
    loop = ups + ups[-2:0:-1]
    loop[0].save(gif, save_all=True, append_images=loop[1:], duration=1000 // args.fps, loop=0)
    print(f"upscaled {len(ups)} frames {tag} -> {ups[0].size}  dir {outdir}/  gif {gif}")


if __name__ == "__main__":
    main()
