"""The pile seen from 0, 40, 140 and 180 degrees off straight down (180: from below, through the floor),
lights turning with the camera, side by side. usage: objbelow.py ckpt out.png"""
import os, sys, math, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import numpy as np, torch
from PIL import Image
from objtube import Mixed, Rig, Cam, build_pairs, render, rot_x, rot_z
from striate.util import pick_device
dev = pick_device()
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ta = argparse.Namespace(**ck["args"]); ta.mip = 1; ta.chunk = 40000
tb = Mixed.from_state(ck["tubes"], ta, dev); rig = Rig(0., .8, ta, dev); rig.load_state_dict(ck["rig"])
S = 512
ims = []
with torch.no_grad():
    sp = tb.spheres(); zm = float(sp["C"][0][..., 2].mean())
    for th in (0, 40, 140, 180):
        cam = Cam(rot_x(math.radians(th)) @ rot_z(0.5), S, S, (S / 2, S / 2, zm), S); cam.env = True
        ims.append(render(tb, sp, cam, build_pairs(sp, cam, 4.0), [rig()], ta, ground="none")[0])
img = torch.cat(ims, 2)
Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(sys.argv[2])
print("ok", img.shape)
