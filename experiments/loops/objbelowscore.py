"""Scores of the pile seen from 0, 40, 140 and 180 degrees off straight down (3 azimuths each, dark
backdrop, the light turning with the camera): how the undersides read against the tops.
usage: objbelowscore.py ckpt..."""
import os, sys, math, random, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import numpy as np, torch
from objtube import Mixed, Rig, Cam, build_pairs, render, rot_x, rot_z
from striate.util import pick_device
from striate.model import Backbone
from zoo import MultiScaleCrop
from ring import Ring
dev = pick_device()
net = Backbone("googlenet", dev); net.hook("inception4c"); rg_ = Ring(net, "inception4c", 7, 512, dev)
def score(x):
    t = 0
    for k in range(4):
        random.seed(1000 + k); t += rg_.act(MultiScaleCrop(224 / 512, 512)(x))
    return float(t / 4)
for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ta = argparse.Namespace(**ck["args"]); ta.mip = 1; ta.chunk = 40000
    tb = Mixed.from_state(ck["tubes"], ta, dev); rig = Rig(0., .8, ta, dev); rig.load_state_dict(ck["rig"])
    out = []
    with torch.no_grad():
        sp = tb.spheres(); zm = float(sp["C"][0][..., 2].mean())
        for th in (0, 40, 140, 180):
            sc = []
            for az in (0.5, 2.6, 4.7):
                cam = Cam(rot_x(math.radians(th)) @ rot_z(az), 512, 512, (256, 256, zm), 512); cam.env = True
                sc.append(score(render(tb, sp, cam, build_pairs(sp, cam, 4.0), [rig()], ta, ground="none")))
            out.append(f"{th}°:{np.mean(sc):.2f}")
    print(path.split("/")[-2], " ".join(out), flush=True)
