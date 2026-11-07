"""Where objtube.py's edges go wrong, at rest (top view, zoom Z, one sample per pixel):
  mix    1 - the top body's share of the pixel (soft z-buffer and coverage blending between bodies)
  fp     paint texels per output pixel at the dominant pair (largest singular value of d texel / d pixel)
  alias  |one sample - 4x4 supersampled|, binned by fp
Writes the maps, a crop of four (one sample, 16 samples, fp, mix) and stats.json.
usage: objedge.py ckpt outdir [zoom]
"""
import os, sys, math, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from objtube import Mixed, Rig, Cam, build_pairs, render
from striate.util import pick_device

_A = np.array([[0, 0, 0.02], [0.3, 0.05, 0.5], [0.85, 0.2, 0.3], [1, 0.75, 0.2], [1, 1, 0.85]])
class cm:  # a magma-like ramp for every map
    @staticmethod
    def ramp(x):
        x = np.clip(x, 0, 1) * (len(_A) - 1)
        i = np.minimum(x.astype(int), len(_A) - 2); f = (x - i)[..., None]
        return np.concatenate([(1 - f) * _A[i] + f * _A[i + 1], np.ones_like(f)], -1)
    magma = viridis = inferno = ramp

dev = pick_device()
path, out = sys.argv[1], sys.argv[2]
Z = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
os.makedirs(out, exist_ok=True)
ck = torch.load(path, map_location="cpu", weights_only=False)
ta = argparse.Namespace(**ck["args"])
tb = Mixed.from_state(ck["tubes"], ta, dev)
rig = Rig(0.0, 0.8, ta, dev); rig.load_state_dict(ck["rig"])
S = 512


def maps(zoom):
    N = int(S * zoom)
    cam = Cam(np.eye(3), N, N, (S / 2, S / 2, 0.0), S, zoom=zoom)
    sp = tb.spheres()
    pr = build_pairs(sp, cam, 4.0)
    o = render(tb, sp, cam, pr, [rig()], ta, aux=True)
    h = 0.5
    ox = render(tb, sp, cam, pr, [rig()], ta, aux=True, off=(0.0, h))
    oy = render(tb, sp, cam, pr, [rig()], ta, aux=True, off=(h, 0.0))
    Ha, Wl = tb.tex0.shape[2:]
    du = lambda a, b: ((b - a + 0.5) % 1.0 - 0.5)
    J = torch.stack([du(o["tu"][0], ox["tu"][0]) * Ha / h, du(o["tu"][0], oy["tu"][0]) * Ha / h,
                     (ox["tv"][0] - o["tv"][0]) * (Wl - 1) / h, (oy["tv"][0] - o["tv"][0]) * (Wl - 1) / h], -1)
    a_, b_, c_, d_ = J.unbind(-1)
    s1 = torch.sqrt(0.5 * (a_**2 + b_**2 + c_**2 + d_**2) + 0.5 * torch.sqrt(((a_**2 + b_**2 - c_**2 - d_**2)) ** 2 + 4 * (a_ * c_ + b_ * d_) ** 2))
    pix, obj = o["pix"], o["obj"]
    share = o["ep"][0] / o["den"][0][pix]
    P = N * N
    cover = torch.zeros(P, device=dev).index_add_(0, pix, share)
    top = torch.zeros(P, device=dev).scatter_reduce(0, pix, share, "amax", include_self=True)
    mix = (1 - top / cover.clamp(min=1e-6)) * (cover > 0.5)
    w = share * (share >= top[pix] - 1e-9)
    fp = torch.zeros(P, device=dev).index_add_(0, pix, w * s1) / torch.zeros(P, device=dev).index_add_(0, pix, w).clamp(min=1e-6)
    return dict(img=o["img"][0], mix=mix.reshape(N, N), fp=fp.reshape(N, N), cover=cover.reshape(N, N),
                tu=o["tu"][0], tv=o["tv"][0], obj=obj, share=share, top=top[pix], s1=s1, cam=cam, sp=sp, pr=pr)


with torch.no_grad():
    m = maps(Z)
    N = m["img"].shape[-1]
    acc = 0
    for iy in range(4):
        for ix in range(4):
            acc = acc + render(tb, m["sp"], m["cam"], m["pr"], [rig()], ta, off=((iy - 1.5) / 4, (ix - 1.5) / 4))
    ref = acc[0] / 16
    alias = (m["img"] - ref).abs().mean(0)
    # provenance: per paint texel, the finest footprint at which the still's resolution (zoom 1) sees it
    m1 = maps(1.0)
    K, _, Ha, Wl = tb.tex0.shape
    dom = m1["share"] >= m1["top"] - 1e-9
    idx = (m1["obj"] * Ha + (m1["tu"] * Ha).long() % Ha) * Wl + (m1["tv"].clamp(0, 1) * (Wl - 1)).round().long()
    prov = torch.full((K * Ha * Wl,), 1e3, device=dev).scatter_reduce(0, idx[dom], m1["s1"][dom], "amin", include_self=True)
    prov = prov.reshape(K, Ha, Wl)

fp, mix = m["fp"].cpu().numpy(), m["mix"].cpu().numpy()
al = alias.cpu().numpy()
body = m["cover"].cpu().numpy() > 0.5
st = {}
for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]:
    sel = body & (fp >= lo) & (fp < hi)
    st[f"fp {lo}-{hi}"] = dict(frac=float(sel.mean()), alias=float(al[sel].mean()) if sel.any() else None, mix=float(mix[sel].mean()) if sel.any() else None)
st["mix>0.1"] = float((mix > 0.1).mean())
st["mix>0.3"] = float((mix > 0.3).mean())
pv = prov.cpu().numpy()
st["texels seen at <=1 texel/px (still res)"] = float((pv <= 1).mean())
st["texels seen at 1-2"] = float(((pv > 1) & (pv <= 2)).mean())
st["texels seen at 2-4"] = float(((pv > 2) & (pv <= 4)).mean())
st["texels seen only coarser than 4 or never"] = float((pv > 4).mean())
st["texels never seen"] = float((pv >= 1e3).mean())
json.dump(st, open(f"{out}/stats.json", "w"), indent=1)
print(json.dumps(st, indent=1))

u8 = lambda x: (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
img = m["img"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
fpv = cm.magma(np.clip(np.log2(np.maximum(fp, 0.25)) / 4 + 0.5, 0, 1))[..., :3] * body[..., None]  # 0.25..4+ texels/px
mixv = cm.viridis(np.clip(mix / 0.5, 0, 1))[..., :3] * body[..., None]
alv = cm.inferno(np.clip(al / 0.15, 0, 1))[..., :3]
for nm, x in dict(img=img, fp=fpv, mix=mixv, alias=alv, ref=ref.clamp(0, 1).permute(1, 2, 0).cpu().numpy()).items():
    Image.fromarray(u8(x)).save(f"{out}/{nm}.png")
pvv = cm.magma(np.clip(np.log2(np.minimum(pv, 64)) / 6 + 0.25, 0, 1))[..., :3]
pick = list(range(0, K, max(1, K // 8)))[:8]
Image.fromarray(u8(np.concatenate([np.pad(pvv[k], ((0, 4), (0, 4), (0, 0)), constant_values=1) for k in pick], 1))).save(f"{out}/prov_atlas.png")
c = slice(int(0.3 * N), int(0.65 * N))
Image.fromarray(u8(np.concatenate([np.concatenate([img[c, c], ref.clamp(0, 1).permute(1, 2, 0).cpu().numpy()[c, c]], 1), np.concatenate([fpv[c, c], mixv[c, c]], 1)], 0))).save(f"{out}/crop4.png")
