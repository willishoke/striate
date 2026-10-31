"""Pictures for a lucid.py run: 2x close-up of four frames; for warp runs, the field drawn as arrows
over frame 0 (every 16 px, arrows scaled x4) at two loop phases and the per-pixel peak displacement.
usage: lucid_viz.py RUN OUTPREFIX"""

import json, os, sys
import numpy as np
from PIL import Image, ImageDraw

run, out = sys.argv[1], sys.argv[2]
rep = json.load(open(f"{run}/report.json"))
N = rep["args"]["n"]
fr = lambda j: Image.open(f"{run}/f{j:03d}.png").convert("RGB")
S = fr(0).size[0]
four = [np.array(fr(j)) for j in (0, N // 4, N // 2, 3 * N // 4)]
Image.fromarray(np.concatenate(four, 1)).resize((8 * S, 2 * S), Image.LANCZOS).save(
    f"{out}_four.jpg", quality=90
)
src = f"{run}/field.npy" if os.path.exists(f"{run}/field.npy") else f"{run}/flow.npy"
if os.path.exists(src):
    u = np.load(src)  # (N, 2, S, S) px, x then y; a flow is per adjacent pair
    N = u.shape[0]
    mag = np.linalg.norm(u, axis=1)
    peak = mag.max(0)
    print(
        f"{os.path.basename(run)} rms {np.sqrt((mag**2).mean()):.2f} px, peak median {np.median(peak):.2f} max {peak.max():.2f}"
    )
    panels = []
    for ph in (0, N // 4):
        im = fr(ph).resize((2 * S, 2 * S), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        for y in range(8, S, 16):
            for x in range(8, S, 16):
                dx, dy = u[ph, 0, y, x] * 4, u[ph, 1, y, x] * 4
                d.line(
                    [(2 * x, 2 * y), (2 * (x + dx), 2 * (y + dy))], fill=(255, 255, 255), width=2
                )
                d.ellipse([2 * x - 2, 2 * y - 2, 2 * x + 2, 2 * y + 2], fill=(0, 0, 0))
        panels.append(np.array(im))
    pk = (np.clip(peak / max(peak.max(), 1e-6), 0, 1) * 255).astype(np.uint8)
    pk = np.array(Image.fromarray(pk).resize((2 * S, 2 * S), Image.BILINEAR).convert("RGB"))
    Image.fromarray(np.concatenate(panels + [pk], 1)).save(f"{out}_field.jpg", quality=90)
