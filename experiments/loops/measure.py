"""Measurements for a finished loop, or for two versions of one loop.

  measure.py loop DIR [DIR ...] --out scores.json
      Scores from DIR/report.json (activation, plus the Gram and pixel numbers of a lucid run),
      contrast swing (CV over frames of the 1-4 px band energy) and that energy, and RAFT
      motion-explained (flowfade.py) at each of --gaps. Loops are scored as closed, wrapping
      around, unless report.json says `loop: false`; loops longer than --window frames are
      scored on an open window of their first --window frames. The defaults are the cell warp
      setting; the shared-field (lucid) study used --res 256 --gaps 1,4.

  measure.py compare ORIG NEW --target inception4e:198 --out cmp.json
      For an edit of a loop that keeps its frames in step (e.g. upscale.py --down): channel score
      of every 8th frame under 4 fixed crop draws, band energies at 1-4, 4-16 and 16-64 px
      relative to ORIG, colour values within 2% of the rails, mean change from ORIG, and mean
      change between adjacent frames. Changes are in levels of 255.

Both merge into --out if it exists. Speeds are in px per frame at 1024 scale.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse
import glob
import json
import random

import numpy as np
import torch
from flowfade import Meter, measure
from PIL import Image
from ring import Ring, bandpass
from zoo import MultiScaleCrop

from striate.model import Backbone
from striate.util import pick_device

REPORT_KEYS = (
    "act_mean",
    "act_min",
    "act_max",
    "act_before",
    "gram_cos_adjacent",
    "gram_cos_far",
    "pix_rms_adjacent",
    "warp_px_rms",
    "warp_px_max",
)


def frames_of(d):
    return sorted(
        glob.glob(f"{d}/f[0-9][0-9][0-9].png") + glob.glob(f"{d}/f[0-9][0-9][0-9][0-9].png")
    )


def load(p):
    x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2, 0, 1).float()
    return x[None] / 255


def loop_scores(d, meter, gaps, window):
    files = frames_of(d)
    n = len(files)
    e = np.array([float(bandpass(load(f), 1, 4).pow(2).mean()) for f in files[:: max(1, n // 120)]])
    r = {"frames": n, "swing": float(e.std() / e.mean()), "fine_energy": float(e.mean())}
    rep = json.load(open(f"{d}/report.json")) if os.path.exists(f"{d}/report.json") else {}
    r.update({k: rep[k] for k in REPORT_KEYS if k in rep})
    closed = rep.get("args", {}).get("loop", True)
    for gap in gaps:
        # flowfade.measure: max_frames 0 scores the closed loop; anything shorter, an open window
        if closed and n <= window:
            mf = 0
        elif closed:
            mf = window
        else:
            mf = min(n - gap, window)
        m = measure(meter, d, gap, meter.res, max_frames=mf)
        r[f"explained_g{gap}"] = m["explained_mean"]
        r[f"explained_g{gap}_p10"] = m["explained_p10"]
        r[f"explained_g{gap}_min"] = m["explained_min"]
        r[f"speed_g{gap}"] = m["speed_px_per_frame"]
    return r


def compare(orig, new, rg, dev):
    fo, fs = frames_of(orig), frames_of(new)
    assert len(fo) == len(fs), f"{len(fo)} vs {len(fs)} frames"
    size = load(fo[0]).shape[-1]

    def score(x):
        with torch.no_grad():
            a = 0
            for k in range(4):
                random.seed(1000 + k)
                a += float(rg.act(MultiScaleCrop(224 / size, 512)(x.to(dev)))) / 4
        return a

    bands = {"fine": (1, 4), "mid": (4, 16), "coarse": (16, 64)}
    R = {k: [] for k in ("act_o", "act_s", "change", "clip_o", "clip_s")}
    R.update({f"{b}_{t}": [] for b in bands for t in "os"})
    for t in range(0, len(fo), 8):
        a, b = load(fo[t]), load(fs[t])
        R["act_o"].append(score(a))
        R["act_s"].append(score(b))
        for tag, x in (("o", a), ("s", b)):
            for name, (lo, hi) in bands.items():
                R[f"{name}_{tag}"].append(float(bandpass(x, lo, hi).pow(2).mean()))
            R[f"clip_{tag}"].append(float(((x < 0.02) | (x > 0.98)).float().mean()))
        R["change"].append(float((a - b).abs().mean() * 255))
    adj = {"o": [], "s": []}  # adjacent-frame change, on a stride so it stays cheap
    for t in range(0, len(fo) - 1, 4):
        for tag, fl in (("o", fo), ("s", fs)):
            adj[tag].append(float((load(fl[t]) - load(fl[t + 1])).abs().mean() * 255))
    m = lambda k: float(np.mean(R[k]))
    out = dict(
        act_orig=m("act_o"),
        act_new=m("act_s"),
        act_orig_min=float(np.min(R["act_o"])),
        act_new_min=float(np.min(R["act_s"])),
        change_mean=m("change"),
        clipped_orig=m("clip_o"),
        clipped_new=m("clip_s"),
        adjacent_orig=float(np.mean(adj["o"])),
        adjacent_new=float(np.mean(adj["s"])),
    )
    out.update({f"{b}_ratio": m(f"{b}_s") / m(f"{b}_o") for b in bands})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["loop", "compare"])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512, help="loop: working resolution for RAFT")
    ap.add_argument("--gaps", default="4,16", help="loop: frame gaps for RAFT")
    ap.add_argument("--window", type=int, default=240, help="loop: longest loop scored whole")
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--target", default="inception4e:198", help="compare: layer:channel")
    a = ap.parse_args()
    dev = pick_device()
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    if a.mode == "loop":
        meter = Meter(dev, a.res)
        gaps = [int(g) for g in a.gaps.split(",")]
        for d in a.dirs:
            name = os.path.basename(d.rstrip("/"))
            out[name] = r = loop_scores(d, meter, gaps, a.window)
            print(
                f"{name:24s} act {r.get('act_mean', float('nan')):6.3f} swing {r['swing']:.3f} "
                + " ".join(f"g{g} {r[f'explained_g{g}']:.3f}" for g in gaps)
                + f" speed {r[f'speed_g{gaps[0]}']:.2f}px/f",
                flush=True,
            )
            json.dump(out, open(a.out, "w"), indent=1)
    else:
        assert len(a.dirs) == 2, "compare takes ORIG NEW"
        layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
        net = Backbone(a.net, dev)
        net.hook(layer)
        size = load(frames_of(a.dirs[0])[0]).shape[-1]
        rg = Ring(net, layer, ch, size, dev)
        key = f"{os.path.basename(a.dirs[1].rstrip('/'))}_vs_{os.path.basename(a.dirs[0].rstrip('/'))}"
        out[key] = r = compare(*a.dirs, rg, dev)
        print(key, json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
