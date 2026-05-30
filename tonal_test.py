"""Tonal test: does the anti-clip penalty restore highlight/shadow detail on a
deep-layer still? Sweep clip_weight and report the histogram."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image, ImageDraw

from striate.model import Backbone
from striate.objectives import ChannelObjective
from striate.render import render
from striate.util import pick_device, save_image


def hist(t):
    a = t[0].permute(1, 2, 0).cpu().numpy()
    return ((a < 0.02).mean() * 100, (a > 0.98).mean() * 100,
            ((a > 0.2) & (a < 0.8)).mean() * 100, float(a.std()))


dev = pick_device()
bb = Backbone("tiny", device=dev)
configs = [("baseline", 0.0), ("clip5", 5.0), ("clip20", 20.0), ("clip60", 60.0)]
entries = []
for name, cw in configs:
    obj = ChannelObjective("features.7", 100)
    img = render(bb, obj, ["features.7"], size=512, steps=300, lr=0.05,
                 device=dev, clip_weight=cw, progress=False)
    s = hist(img)
    print(f"  {name:9s} clip_w={cw:5.1f}:  shadows={s[0]:4.1f}%  highlights={s[1]:4.1f}%  "
          f"midtones={s[2]:4.1f}%  std={s[3]:.2f}")
    p = f"outputs/tonal_{name}.png"
    save_image(img, p)
    entries.append((name, p))

tw = th = 384
sheet = Image.new("RGB", (2 * tw, 2 * th), (12, 12, 12))
d = ImageDraw.Draw(sheet)
for i, (name, p) in enumerate(entries):
    t = Image.open(p).resize((tw, th))
    r, c = divmod(i, 2)
    sheet.paste(t, (c * tw, r * th))
    d.rectangle([c * tw + 4, r * th + 4, c * tw + 110, r * th + 22], fill=(0, 0, 0))
    d.text((c * tw + 8, r * th + 8), name, fill=(255, 255, 0))
sheet.save("outputs/tonal_compare.png")
print("comparison -> outputs/tonal_compare.png")
