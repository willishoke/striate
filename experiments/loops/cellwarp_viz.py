"""Motion diagnostics for a cellwarp run: displacement magnitude map, walls at two loop phases, amplitude stats.
usage: cellwarp_viz.py RUN OUTPREFIX"""

import json, math, os, sys
import numpy as np
import torch
from PIL import Image
from cellwarp import Warp
from scipy import ndimage as ndi

run, out = sys.argv[1], sys.argv[2]
ck = torch.load(f"{run}/warp.pt", map_location="cpu")
a = ck["args"]
z = np.load(a["labels"])
dom, cells = z["dom"].astype(np.int64), z["cells"].astype(np.int64)
nD, nC = int(dom.max()), int(cells.max())
dom_of = np.array(ndi.maximum(dom, cells, range(1, nC + 1))).astype(np.int64) - 1
m = np.load(a["labels"].replace(".npz", f"_m{a['topk']}_{a['sig_wall']:g}_{a['sig_band']:g}.npz"))
Sz = cells.shape[0]
wp = Warp(m["idx"], m["W"], m["cen"], m["area"], dom_of, nD, Sz, torch.device("cpu"), a["K"])
wp.D.cc.data.copy_(ck["D"])
wp.d.cc.data.copy_(ck["d"])
wp.s.cc.data.copy_(ck["s"])
T = a["frames"]
with torch.no_grad():
    f0, _, _ = wp.field(torch.tensor([0.0]))
    f1, _, _ = wp.field(torch.tensor([0.5]))
    # peak speed per pixel over the loop (px per frame), on 48 phases
    us = torch.arange(48) / 48
    prev = None
    mx = torch.zeros(Sz, Sz)
    amp = torch.zeros(Sz, Sz)
    for u in us:
        f, _, _ = wp.field(u[None])
        amp = torch.maximum(amp, f[0].norm(dim=-1))
        if prev is not None:
            mx = torch.maximum(mx, ((f - prev)[0].norm(dim=-1)) / (T / 48))
        prev = f
f0, f1 = f0[0].numpy(), f1[0].numpy()
base = np.array(Image.open(a["base"].replace(".pt", ".png")).convert("RGB"))
# magnitude of the displacement at phase 0.5 relative to phase 0, as a heat map
mag = np.linalg.norm(f1 - f0, axis=-1)
heat = np.clip(mag / 8, 0, 1)
hm = (np.stack([heat, heat * 0.6, 1 - heat], -1) * 255).astype(np.uint8)
Image.fromarray(hm).save(f"{out}_mag.png")
# walls at the two phases: boundary pixels of the label map, advected by -field (base pixel y lands at y - u(y))
b = (cells != np.roll(cells, 1, 0)) | (cells != np.roll(cells, 1, 1))
ys, xs = np.nonzero(b)


def draw(img, f, col, scale, box):
    x0, y0, w = box
    px = xs - f[ys, xs, 0]
    py = ys - f[ys, xs, 1]
    sel = (px >= x0) & (px < x0 + w) & (py >= y0) & (py < y0 + w)
    for x, y in zip(((px[sel] - x0) * scale).astype(int), ((py[sel] - y0) * scale).astype(int)):
        img[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = col


for i, box in enumerate([(384, 384, 256), (0, 0, 256)]):
    x0, y0, w = box
    im = np.array(
        Image.fromarray(base[y0 : y0 + w, x0 : x0 + w]).resize((w * 3, w * 3), Image.LANCZOS)
    )
    im = (im * 0.55).astype(np.uint8)
    draw(im, f0, (0, 230, 255), 3, box)
    draw(im, f1, (255, 230, 0), 3, box)
    Image.fromarray(im).save(f"{out}_walls{i}.png")
rep = json.load(open(f"{run}/report.json"))
st = dict(
    peak_disp_px_median=float(np.median(amp.numpy())),
    peak_disp_px_p95=float(np.quantile(amp.numpy(), 0.95)),
    peak_speed_px_per_frame_median=float(np.median(mx.numpy())),
    peak_speed_px_per_frame_p95=float(np.quantile(mx.numpy(), 0.95)),
    rmsD=rep["rmsD"],
    rmsd=rep["rmsd"],
    rmss=rep["rmss"],
    act=rep["act_mean"],
    act_before=rep["act_before"],
    base_alone=rep["base_alone"],
)
json.dump(st, open(f"{out}_motion.json", "w"), indent=1)
print(json.dumps(st, indent=1))
