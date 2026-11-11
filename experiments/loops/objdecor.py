"""How much the paint follows the geometry's shading cues (objtube.decor_loss): per body, the correlation
of its paint with the key light's n.l, the sky it sees and its distance from the outline, in the rest
pose from straight above and from 40 deg off vertical (3 azimuths). Prints the signed mean correlation of
the luminance with each cue (painted-in shading and rim darkening: positive) and the term objphys
--wdecor trains on (mean squared correlation, luminance and chroma).
usage: objdecor.py ckpt[:key=val,...] ..."""
import os, sys, json, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from objtube import Mixed, Rig, Cam, build_pairs, render, rot_x, rot_z, decor_loss
from striate.util import pick_device
dev = pick_device()
res = {}
for spec in sys.argv[1:]:
    path, *ov = spec.split(":")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ta = argparse.Namespace(**ck["args"]); ta.chunk = 60000
    for kv in (ov[0].split(",") if ov else []):
        k, v = kv.split("="); setattr(ta, k, float(v))
    tb = Mixed.from_state(ck["tubes"], ta, dev); rig = Rig(0., .8, ta, dev); rig.load_state_dict(ck["rig"])
    r = {}
    with torch.no_grad():
        sp = tb.spheres()
        for nm, R in [("top", np.eye(3))] + [(f"tilt{i}", rot_x(np.radians(40)) @ rot_z(az)) for i, az in enumerate((0.3, 2.4, 4.5))]:
            cam = Cam(R, 512, 512, (256., 256., 0.), 512)
            o = render(tb, sp, cam, build_pairs(sp, cam, 4.0), [rig()], ta, aux=True)
            d, sig = decor_loss(o, tb.K)
            r[nm] = dict(term=float(d), **sig)
            del o
    agg = {k: float(np.mean([v[k] for v in r.values()])) for k in r["top"]}
    res[path] = dict(views=r, mean=agg)
    print(path.split("/")[-2], "top:", " ".join(f"{k} {v:+.3f}" for k, v in r["top"].items()), "| mean of 4 views:", " ".join(f"{k} {v:+.3f}" for k, v in agg.items()), flush=True)
    del tb
