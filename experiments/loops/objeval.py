"""Rest and turned scores, edge noise, and crops for objtube.py checkpoints.
turned t: every body's paint (and relief) turned t of the way round on its body, so what was underneath
faces the camera. sym: every body rolled a random amount about its spine, and at random mirrored and/or
reversed end for end (3 draws, mean; objphys --train-sym shows bodies this way). tilt: the camera 40 deg
off vertical, 3 azimuths, mean. alias: mean |one sample - 4x4 supersampled| over the bodies at 1024 px.
usage: objeval.py outdir ckpt[:key=val,...] ...   (key=val overrides the checkpoint's render args, e.g. mip=1)
"""
import os, sys, json, random, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
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
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
top = Cam(np.eye(3), 512, 512, (256, 256, 0.), 512)
cam2 = Cam(np.eye(3), 1024, 1024, (256, 256, 0.), 512, zoom=2.0)
res = {}
u8 = lambda x: Image.fromarray((x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
for spec in sys.argv[2:]:
    path, *ov = spec.split(":")
    name = path.split("/")[-2] + ("_" + ov[0].replace(",", "_").replace("=", "") if ov else "")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ta = argparse.Namespace(**ck["args"])
    for kv in (ov[0].split(",") if ov else []):
        k, v = kv.split("="); setattr(ta, k, int(v) if v.isdigit() else float(v))
    tb = Mixed.from_state(ck["tubes"], ta, dev); rig = Rig(0., .8, ta, dev); rig.load_state_dict(ck["rig"])
    r = {}
    with torch.no_grad():
        sp = tb.spheres(); pr = build_pairs(sp, top, 4.0)
        for t in (0.0, 0.25, 0.5):
            s_ = dict(sp)
            if t:
                s_["tex_shift"] = torch.tensor([[t, 0.0]], device=dev).expand(tb.K, 2).contiguous()
            acc = 0
            for oy in (-0.25, 0.25):
                for ox in (-0.25, 0.25):
                    acc = acc + render(tb, s_, top, pr, [rig()], ta, off=(oy, ox))
            im = acc / 4
            r[f"turn{t}"] = score(im)
            if t in (0.0, 0.5):
                u8(im[0]).save(f"{out}/{name}_t{t}.png")
        ss = lambda s_, cm, pr_: sum(render(tb, s_, cm, pr_, [rig()], ta, off=(oy, ox)) for oy in (-0.25, 0.25) for ox in (-0.25, 0.25)) / 4
        sy = []
        for d in range(3):
            g = torch.Generator().manual_seed(d)
            s_ = dict(sp)
            s_["tex_shift"] = torch.stack([torch.rand(tb.K, generator=g), torch.zeros(tb.K)], -1).to(dev)
            s_["tex_sign"] = torch.where(torch.rand(tb.K, 2, generator=g) < 0.5, -1.0, 1.0).to(dev)
            im = ss(s_, top, pr)
            sy.append(score(im))
            if d == 0:
                u8(im[0]).save(f"{out}/{name}_sym.png")
        r["sym"] = float(np.mean(sy))
        tl = []
        for d, az in enumerate((0.3, 2.4, 4.5)):
            cm = Cam(rot_x(np.radians(40)) @ rot_z(az), 512, 512, (256, 256, 0.), 512)
            im = ss(sp, cm, build_pairs(sp, cm, 4.0))
            tl.append(score(im))
            if d == 0:
                u8(im[0]).save(f"{out}/{name}_tilt.png")
        r["tilt"] = float(np.mean(tl))
        pr2 = build_pairs(sp, cam2, 4.0)
        one = render(tb, sp, cam2, pr2, [rig()], ta, aux=True)
        acc = 0
        for iy in range(4):
            for ix in range(4):
                acc = acc + render(tb, sp, cam2, pr2, [rig()], ta, off=((iy - 1.5) / 4, (ix - 1.5) / 4))
        body = (one["gown"][0] < 0.5).reshape(1024, 1024)
        r["alias"] = float(((one["img"][0] - acc[0] / 16).abs().mean(0))[body].mean())
        u8(one["img"][0][:, 300:660, 300:660]).resize((720, 720), Image.NEAREST).save(f"{out}/{name}_crop.png")
        cl = render(tb, sp, cam2, pr2, [rig()], ta, clay=0.72)
        u8(cl[0][:, 300:660, 300:660]).resize((720, 720), Image.NEAREST).save(f"{out}/{name}_clay.png")
    res[name] = r
    print(name, json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)
    del tb
json.dump(res, open(f"{out}/eval_{len(os.listdir(out))}.json", "w"), indent=1)
