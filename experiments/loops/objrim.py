"""Brightness toward the outline: mean luminance per band of facing (|n.v| of the body that owns the pixel,
from our rest geometry, which was fitted to the still), for the still and for renders of checkpoints.
Also by distance (px) from the nearest body boundary. usage: rimdiag.py still.png out ckpt[:key=val]..."""
import os, sys, json, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import numpy as np, torch
from PIL import Image
from scipy import ndimage as ndi
from objtube import Mixed, Rig, Cam, build_pairs, render
from striate.util import pick_device
dev = pick_device()
still = np.asarray(Image.open(sys.argv[1]).convert("RGB"), np.float32) / 255
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
S = 512
lum = lambda x: x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
bands = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
res = {}
geo = None
for spec in sys.argv[3:]:
    path, *ov = spec.split(":")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ta = argparse.Namespace(**ck["args"])
    for kv in (ov[0].split(",") if ov else []):
        k, v = kv.split("="); setattr(ta, k, int(v) if v.isdigit() else float(v))
    tb = Mixed.from_state(ck["tubes"], ta, dev); rig = Rig(0., .8, ta, dev); rig.load_state_dict(ck["rig"])
    cam = Cam(np.eye(3), S, S, (S / 2, S / 2, 0.), S)
    with torch.no_grad():
        sp = tb.spheres(); pr = build_pairs(sp, cam, 4.0)
        img = sum(render(tb, sp, cam, pr, [rig()], ta, off=(oy, ox)) for oy in (-0.25, 0.25) for ox in (-0.25, 0.25))[0] / 4
        o = render(tb, sp, cam, pr, [rig()], ta, aux=True)
        share = o["ep"][0] / o["den"][0][o["pix"]]
        top = torch.zeros(S * S, device=dev).scatter_reduce(0, o["pix"], share, "amax", include_self=True)
        dom = share >= top[o["pix"]] - 1e-9
        fz = torch.zeros(S * S, device=dev); fz[o["pix"][dom]] = o["nk"][0][dom][:, 2].abs()
        own = torch.full((S * S,), -1, device=dev, dtype=torch.long); own[o["pix"][dom]] = o["obj"][dom]
        body = (o["gown"][0] < 0.5)
    fz, own, body = fz.cpu().numpy().reshape(S, S), own.cpu().numpy().reshape(S, S), body.cpu().numpy().reshape(S, S)
    if geo is None:  # the still is scored against the first checkpoint's geometry
        edge = np.zeros((S, S), bool)
        edge[:, :-1] |= own[:, :-1] != own[:, 1:]; edge[:-1] |= own[:-1] != own[1:]
        dist = ndi.distance_transform_edt(~edge)
        geo = (fz, body, dist)
        Ls = lum(still)
        res["still"] = dict(facing={f"{lo}-{hi}": float(Ls[body & (fz >= lo) & (fz < hi)].mean()) for lo, hi in zip(bands[:-1], bands[1:])},
                            dist={f"{d}": float(Ls[body & (np.round(dist) == d)].mean()) for d in range(0, 7)})
    Li = lum(img.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
    fz0, body0, dist0 = geo
    name = path.split("/")[-2] + ("_" + ov[0] if ov else "")
    res[name] = dict(facing={f"{lo}-{hi}": float(Li[body0 & (fz0 >= lo) & (fz0 < hi)].mean()) for lo, hi in zip(bands[:-1], bands[1:])},
                     dist={f"{d}": float(Li[body0 & (np.round(dist0) == d)].mean()) for d in range(0, 7)})
    del tb
for k, v in res.items():
    print(k.ljust(14), "facing", " ".join(f"{b}:{x:.3f}" for b, x in v["facing"].items()), "| dist", " ".join(f"{b}:{x:.3f}" for b, x in v["dist"].items()))
json.dump(res, open(f"{out}/rim.json", "w"), indent=1)
