"""Content-change metrics for a cellwarp run.
- core_change / wall_change: mean abs frame difference at gap 8 (levels of 255), inside cell cores vs on walls
- mean_vs_base: mean abs difference between the loop's time-average frame and the base
- hue_swing: median over cells of the peak-to-peak hue angle of the cell's mean colour over the loop (degrees)
usage: content.py RUN [RUN ...]   (reads assets/; the membership cache is written there by cellwarp.py)"""

import json, os, sys
import numpy as np
from PIL import Image

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

base = np.array(Image.open(f"{A}/base.png").convert("RGB")).astype(np.float32)
m = np.load(f"{A}/labels_4e198_m4_3_7.npz")
W0 = m["W"][..., 0]
core = W0 > 0.9
wall = W0 < 0.6
cells = np.load(f"{A}/labels_4e198.npz")["cells"]
nC = int(cells.max())
idx = cells.ravel() - 1
cnt = np.bincount(idx, minlength=nC).astype(np.float32)


def cell_means(f):
    return np.stack([np.bincount(idx, f[..., c].ravel(), minlength=nC) / cnt for c in range(3)], 1)


def hue(rgb):
    # angle in the plane orthogonal to grey
    a = rgb[:, 0] - 0.5 * (rgb[:, 1] + rgb[:, 2])
    b = (np.sqrt(3) / 2) * (rgb[:, 1] - rgb[:, 2])
    return np.degrees(np.arctan2(b, a))


for run in sys.argv[1:]:
    files = sorted(f for f in os.listdir(run) if f.startswith("f") and f.endswith(".png"))
    T = len(files)
    load = lambda t: np.array(Image.open(f"{run}/{files[t]}").convert("RGB")).astype(np.float32)
    acc = np.zeros_like(base)
    cc, wc, n = 0.0, 0.0, 0
    hues = []
    for t in range(0, T, 4):
        f = load(t)
        acc += f
        if t % 8 == 0:
            g = load((t + 8) % T)
            d = np.abs(f - g).mean(-1)
            cc += d[core].mean()
            wc += d[wall].mean()
            n += 1
        hues.append(hue(cell_means(f)))
    acc /= len(range(0, T, 4))
    drift = float(
        np.abs(cell_means(acc) - cell_means(base)).mean()
    )  # time-averaged cell colour vs base, levels
    hues = np.stack(hues)  # (n_t, nC)
    dh = np.degrees(
        np.angle(np.exp(1j * np.radians(hues - hues[0:1])))
    )  # unwrap relative to the first frame
    swing = np.median(dh.max(0) - dh.min(0))
    st = dict(
        core_change=round(float(cc / n), 2),
        wall_change=round(float(wc / n), 2),
        mean_vs_base=round(float(np.abs(acc - base).mean()), 2),
        hue_swing_deg=round(float(swing), 1),
        cell_mean_drift=round(drift, 2),
    )
    rep = json.load(open(f"{run}/report.json"))
    st.update(
        act=round(rep["act_mean"], 3),
        rmsh=round(rep.get("rmsh", 0), 3),
        rmso=round(rep.get("rmso", 0), 2),
        res_rms=round(rep.get("res_rms", 0), 3),
        hf_rms=round(rep.get("hf_rms", 0), 3),
    )
    json.dump(st, open(f"{run}/content.json", "w"), indent=1)
    print(os.path.basename(run), st)
