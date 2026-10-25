"""Single high-res feature-viz still on a ViT backbone (SigLIP @512).

ViT attention is O(tokens^2), so true 1080p (~8k tokens) OOMs the M5; 1280x720
(~3.6k tokens) is the largest that fits cleanly without gradient checkpointing.
Render there, then upscale to 4K. dynamic_img_size lets SigLIP accept non-512
input by interpolating its positional embedding.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import timm
import torch
from tqdm import trange

from striate.param import FourierImage
from striate.transforms import TransformRobustness
from striate.util import pick_device, save_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch16_siglip_512.webli")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--channel", type=int, default=320)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="outputs/siglip_still.png")
    args = ap.parse_args()

    dev = pick_device()
    model = (
        timm.create_model(args.model, pretrained=True, num_classes=0, dynamic_img_size=True)
        .eval()
        .to(dev)
    )
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(dev)
    gh, gw = args.height // 16, args.width // 16
    print(
        f"device {dev}  {args.model}  block {args.block}  ch {args.channel}  "
        f"@ {args.width}x{args.height}  token grid {gh}x{gw}"
    )

    cap = {}
    handle = model.blocks[args.block].register_forward_hook(lambda m, i, o: cap.__setitem__("t", o))
    transforms = TransformRobustness(degrees=[])
    param = FourierImage(args.height, args.width, device=dev)
    opt = torch.optim.Adam([param.spectrum], lr=args.lr)

    for _ in trange(args.steps):
        opt.zero_grad()
        img = param.to_image()
        model.forward_features((transforms(img) - mean) / std)
        tok = cap["t"]
        if tok.shape[1] != gh * gw:  # strip any leading CLS/registers
            tok = tok[:, tok.shape[1] - gh * gw :, :]
        act = tok.reshape(tok.shape[0], gh, gw, tok.shape[-1]).permute(0, 3, 1, 2)
        (-act[:, args.channel].mean()).backward()
        opt.step()
    handle.remove()

    save_image(param.to_image().detach(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
