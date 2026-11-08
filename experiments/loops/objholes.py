"""Visible floor over a loop (top view, zoom 1, clay): share of pixels where the ground shows through,
mean and max over every 4th frame, for trajectories driving one checkpoint. usage: holes.py ckpt traj..."""
import os, sys, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import numpy as np, torch
from objtube import Mixed, Rig, Cam, build_pairs, render, tex_span
from striate.util import pick_device
dev = pick_device()
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ta = argparse.Namespace(**ck["args"]); ta.chunk = 60000
tb = Mixed.from_state(ck["tubes"], ta, dev); rig = Rig(0., .8, ta, dev); rig.load_state_dict(ck["rig"])
S = 512
cam = Cam(np.eye(3), S, S, (S / 2, S / 2, 0.), S)
with torch.no_grad():
    C0 = tb.spheres()["C"][0]
    sp0 = tb.spheres()
    g0 = float((render(tb, sp0, cam, build_pairs(sp0, cam, 4.0), [rig()], ta, clay=0.7, aux=True)["gown"][0] > 0.5).float().mean())
    print("rest: floor %.4f" % g0)
    for f in sys.argv[2:]:
        tj = torch.load(f, weights_only=False)
        tr, rd = tj["traj"].to(dev), tj["rads"].to(dev)
        fl = []
        for t in range(0, tr.shape[0], 4):
            sp = tb.spheres(wob=(tr[t] - C0)[None]); sp["r"] = rd[t]
            o = render(tb, sp, cam, build_pairs(sp, cam, 4.0), [rig()], ta, clay=0.7, aux=True)
            fl.append(float((o["gown"][0] > 0.5).float().mean()))
        print(f.split("/")[-2] + "/" + f.split("/")[-1][-26:], "floor mean %.4f max %.4f" % (np.mean(fl), np.max(fl)), flush=True)
