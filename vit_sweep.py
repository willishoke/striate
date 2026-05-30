"""Feature-viz channel sweep on a ViT backbone (SigLIP @512), for comparing
feature *scale* against the 224px-trained ConvNeXt.

ViT adaptation: hook a transformer block, reshape its patch tokens (B, N, D)
into a spatial grid (B, D, g, g), and treat the embedding dim D as channels.
SigLIP has no CLS/register tokens, so the reshape is clean.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import timm
import torch
from PIL import Image, ImageDraw
from tqdm import trange

from striate.param import FourierImage
from striate.transforms import TransformRobustness
from striate.util import pick_device, save_image


def tokens_to_grid(tokens):
    B, N, D = tokens.shape
    g = int(round(N ** 0.5))
    if g * g != N:                 # strip any leading CLS/register tokens
        tokens = tokens[:, N - g * g:, :]
        B, N, D = tokens.shape
        g = int(round(N ** 0.5))
    return tokens.reshape(B, g, g, D).permute(0, 3, 1, 2)  # (B, D, g, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_siglip_512.webli")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--channels", default="0,64,128,192,256,320,384,448,512,576,640,704")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=110)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--sheet", default="outputs/siglip_sheet.png")
    args = ap.parse_args()

    dev = pick_device()
    model = timm.create_model(args.model, pretrained=True, num_classes=0).eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(dev)
    print(f"device {dev}  {args.model}  block {args.block}  norm {cfg['mean']}  input {cfg['input_size']}")

    cap = {}
    handle = model.blocks[args.block].register_forward_hook(lambda m, i, o: cap.__setitem__("t", o))
    transforms = TransformRobustness(degrees=[])  # no rotation (grid_sample CPU fallback)

    channels = [int(c) for c in args.channels.split(",")]
    os.makedirs("outputs/siglip", exist_ok=True)
    entries = []
    for ch in channels:
        param = FourierImage(args.size, args.size, device=dev)
        opt = torch.optim.Adam([param.spectrum], lr=args.lr)
        for _ in trange(args.steps, leave=False, desc=f"ch{ch}"):
            opt.zero_grad()
            img = param.to_image()
            model.forward_features((transforms(img) - mean) / std)
            act = tokens_to_grid(cap["t"])
            (-act[:, ch].mean()).backward()
            opt.step()
        p = f"outputs/siglip/ch{ch}.png"
        save_image(param.to_image().detach(), p)
        entries.append((p, f"ch{ch}"))
        print(f"  ch{ch} done")
    handle.remove()

    cols = args.cols
    tw = 256
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * tw), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for i, (p, lab) in enumerate(entries):
        t = Image.open(p).resize((tw, tw))
        r, c = divmod(i, cols)
        sheet.paste(t, (c * tw, r * tw))
        d.rectangle([c * tw + 4, r * tw + 4, c * tw + 70, r * tw + 22], fill=(0, 0, 0))
        d.text((c * tw + 8, r * tw + 8), lab, fill=(255, 255, 0))
    sheet.save(args.sheet)
    print(f"sheet -> {args.sheet}")


if __name__ == "__main__":
    main()
