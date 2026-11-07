"""Move the fine detail of objtube.py's paint into relief: textured bodies instead of painted ones.

The textures carry the pattern's shading in the paint (a pit is a painted dark dot), so when a body
swells or turns the light never moves across it and it reads as a decal. This starts a fine relief
(objtube.py --bump-amp) from the paint's luminance finer than --r texels: bright up, dark down (a
cavity map's convention, and what the light would have drawn), scaled to --gain x the relief's cap
at one standard deviation; and takes --flat of that detail out of the paint, keeping its hue.
Then train paint and relief together (objphys.py --train --lr-bump --lr-prior --whf), and the
channel decides how much of the pattern is shape.

usage: bumpinit.py --load tube.pt --out tube_bump.pt [--amp 1.5 --gain 0.6 --flat 0.6]
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def blur(x, r):
    """Box blur (K, 1, around, along), wrapping around and clamped along."""
    xp = F.pad(torch.cat([x[:, :, -r:], x, x[:, :, :r]], 2), (r, r, 0, 0), mode="replicate")
    return F.avg_pool2d(xp, 2 * r + 1, stride=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--amp", type=float, default=1.5, help="relief cap, px (objtube.py --bump-amp)")
    ap.add_argument("--around", type=int, default=64)
    ap.add_argument("--along", type=int, default=144)
    ap.add_argument("--r", type=int, default=4, help="detail: finer than this many paint texels")
    ap.add_argument("--gain", type=float, default=0.6, help="relief at one standard deviation of the detail, x --amp")
    ap.add_argument("--flat", type=float, default=0.6, help="fraction of the detail taken out of the paint")
    ap.add_argument("--sign", type=float, default=1.0, help="+1: bright up, dark down; -1 the reverse")
    ap.add_argument("--cavity", type=float, default=0.3)
    a = ap.parse_args()
    ck = torch.load(a.load, map_location="cpu", weights_only=False)
    st = dict(ck["tubes"])
    tex0, tex = st["tex0"].float(), st["tex"].float()
    K, _, Ha, Wl = tex0.shape
    k = Ha // tex.shape[2]
    p = torch.cat([tex[:, :, -1:], tex, tex[:, :, :1]], 2)
    up = F.interpolate(p, size=(Ha + 2 * k, Wl), mode="bilinear", align_corners=False)
    alb = torch.sigmoid(tex0 + up[:, :, k:k + Ha])
    Y = (alb * torch.tensor([0.299, 0.587, 0.114])[:, None, None]).sum(1, keepdim=True)
    hp = Y - blur(Y, a.r)
    h = F.interpolate(torch.cat([hp[:, :, -4:], hp, hp[:, :, :4]], 2), size=(a.around + 8 * a.around // Ha, a.along), mode="area")
    h = h[:, :, 4 * a.around // Ha:4 * a.around // Ha + a.around]
    h = a.sign * a.gain * a.amp * h / h.std()
    st["bump"] = torch.atanh((h[:, 0] / a.amp).clamp(-0.95, 0.95))
    Yn = Y - a.flat * hp
    albn = (alb * (Yn / Y.clamp(min=1e-3))).clamp(1 / 255, 1 - 1 / 255)
    st["tex0"] = torch.log(albn / (1 - albn))
    st["tex"] = torch.zeros_like(tex)
    ck = dict(ck)
    ck["tubes"] = st
    ck["args"] = {**ck["args"], "bump_amp": a.amp, "bump_around": a.around, "bump_along": a.along, "cavity": a.cavity, "cav_r": 3}
    ck["bumpinit"] = vars(a)
    torch.save(ck, a.out)
    b = a.amp * torch.tanh(st["bump"])
    print(f"{K} bodies: relief {a.around}x{a.along}, rms {float(b.pow(2).mean().sqrt()):.2f} px (cap {a.amp}); paint detail x{1 - a.flat:.2f}")
    pick = list(range(0, K, max(1, K // 8)))[:8]
    hb = F.interpolate(b[:, None], size=(Ha, Wl), mode="bilinear", align_corners=False)
    rows = []
    for X in (alb, albn, (hb / (2 * a.amp) + 0.5).expand(-1, 3, -1, -1)):
        rows.append(np.concatenate([np.pad(X[i].permute(1, 2, 0).numpy(), ((0, 4), (0, 4), (0, 0)), constant_values=1) for i in pick], 1))
    Image.fromarray((np.concatenate(rows, 0) * 255).round().clip(0, 255).astype(np.uint8)).save(a.out.replace(".pt", "_atlas.png"))


if __name__ == "__main__":
    main()
