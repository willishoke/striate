"""Raise objtube.py's texture resolution: each body's albedo (wrapped-still prior + learned residual)
is upscaled by Real-ESRGAN (x4, the same model as upscale.py), brought back down to --scale x the prior's
size (area resampling), and becomes the new prior; the coarse residual restarts at zero. --mode bilinear
is the control: the same albedo upsampled plainly. Either way, train the new prior afterwards (objphys.py
--train --lr-prior): the upscale is a starting point at the finer grid, not the detail itself.

The texture wraps around the body, so it is padded with its own other side before the upscale and
cropped after, so no seam appears where the turn closes. Along the body it is padded by reflection.

usage: texupres.py --load tube.pt --out tube_hr.pt [--scale 2]
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from spandrel import ModelLoader
from striate.util import pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "../../models/RealESRGAN_x4plus.pth"))
    ap.add_argument("--scale", type=int, default=2, help="new prior = this x the old one")
    ap.add_argument("--mode", default="esrgan", choices=["esrgan", "bilinear"])
    ap.add_argument("--pad", type=int, default=8, help="texels of wrap / reflection padding before the upscale")
    a = ap.parse_args()
    dev = pick_device()
    ck = torch.load(a.load, map_location="cpu", weights_only=False)
    st = ck["tubes"]
    tex0, tex = st["tex0"].float(), st["tex"].float()  # (K, 3, Ha, Wl) logits; (K, 3, ha, wl)
    K, _, Ha, Wl = tex0.shape
    # the residual on the prior's grid: wrap around, clamp along (as tex_sample does)
    p = torch.cat([tex[:, :, -1:], tex, tex[:, :, :1]], 2)
    up = F.interpolate(p, size=(Ha + 2 * Ha // tex.shape[2], Wl), mode="bilinear", align_corners=False)
    k = Ha // tex.shape[2]
    alb = torch.sigmoid(tex0 + up[:, :, k:k + Ha])  # (K, 3, Ha, Wl) in [0, 1]
    P = a.pad
    out = []
    if a.mode == "bilinear":  # control: the same albedo, upsampled plainly
        x = torch.cat([alb[:, :, -P:], alb, alb[:, :, :P]], 2)
        y = F.interpolate(x, scale_factor=a.scale, mode="bilinear", align_corners=False)
        out = [y[:, :, a.scale * P:a.scale * (P + Ha)]]
    else:
        model = ModelLoader().load_from_file(a.model).model.to(dev).eval()
    with torch.no_grad():
        for i in range(K if a.mode == "esrgan" else 0):
            x = alb[i:i + 1]
            x = torch.cat([x[:, :, -P:], x, x[:, :, :P]], 2)  # wrap around the body
            x = F.pad(x, (P, P, 0, 0), mode="reflect")  # reflect along it
            y = model(x.to(dev)).clamp(0, 1).cpu()  # x4
            y = y[:, :, 4 * P:4 * (P + Ha), 4 * P:4 * (P + Wl)]
            y = F.interpolate(y, size=(a.scale * Ha, a.scale * Wl), mode="area")
            out.append(y)
    hr = torch.cat(out).clamp(1 / 255, 1 - 1 / 255)
    st = dict(st)
    st["tex0"] = torch.log(hr / (1 - hr))
    st["tex"] = torch.zeros_like(tex)
    ck = dict(ck)
    ck["tubes"] = st
    ck["texupres"] = vars(a)
    torch.save(ck, a.out)
    d = (F.interpolate(hr, size=(Ha, Wl), mode="area") - alb).abs().mean()
    print(f"{K} bodies: prior {Ha}x{Wl} -> {a.scale * Ha}x{a.scale * Wl}; mean change at the old size {float(d):.4f}")


if __name__ == "__main__":
    main()
