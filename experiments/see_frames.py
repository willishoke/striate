"""Render 4 *consecutive* de-smeared transition frames at full size, to judge
temporal coherence by eye (coherent evolution vs strobing)."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image

from striate.model import Backbone
from striate.transforms import TransformRobustness
from striate.util import pick_device
from keyframe import render_spectra
from make_corpus import render_group
from morph_critic import Critic, desmear_sequence, to_pil

dev = pick_device()
critic = Critic().to(dev)
critic.load_state_dict(torch.load("outputs/critic.pt"))
critic.eval()
for p in critic.parameters():
    p.requires_grad_(False)
transforms = TransformRobustness(degrees=[])
bb = Backbone("tiny", device=dev)
sub = bb.model.features[:6]
SIZE, FRAMES = 320, 20

ep = render_group(
    bb, sub, [42, 90], size=SIZE, steps=160, lr=0.05, transforms=transforms, device=dev
).to(dev)
sA, sB = ep[0:1], ep[1:2]
ts = torch.linspace(0, 1, FRAMES)
dense = torch.cat([((1 - t) * sA + t * sB).cpu() for t in ts], dim=0)

des_spec = desmear_sequence(
    dense,
    critic,
    size=SIZE,
    lam=2,
    anchor=18,
    temporal=140,
    steps=40,
    lr=0.03,
    warmstart=True,
    transforms=transforms,
    device=dev,
)
des = render_spectra(des_spec, SIZE, SIZE, dev)
base = render_spectra(dense, SIZE, SIZE, dev)

idxs = [9, 10, 11, 12]  # consecutive transition frames
for name, imgs in [("des", des), ("base", base)]:
    sheet = Image.new("RGB", (2 * SIZE, 2 * SIZE), (12, 12, 12))
    for n, i in enumerate(idxs):
        sheet.paste(to_pil(imgs[i]), ((n % 2) * SIZE, (n // 2) * SIZE))
    sheet.save(f"outputs/consec_{name}.png")
print("saved outputs/consec_des.png and outputs/consec_base.png  (frames 9,10,11,12)")
