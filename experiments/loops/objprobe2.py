"""objprobe tier 2: does the channel care whether two neighbouring regions move as one object or as
two, once the textures may adapt? objprobe.py asked the still with forward passes only; here the
case is looped with objfield.py's sprites and the textures retrained under the motion.

Scene. One case, regions A and B (objprobe's own masks: neighbour pair, forest parts or fake cut).
Everything else is a static back layer: the still itself, 0 / -1 pixels included, with objprobe's
crevice colour (normalised Gaussian of the label-0 pixels, sigma 6) under A u B for what motion reveals.
  merged  A u B is one sprite.
  split   A and B are two sprites. A lies over B and B extends --ext px under A (objfield's amodal
          extension with mirror fill), so opening shows B's surface carrying on, not a hole.
          --reveal gap drops the extension: opening shows the back layer's crevice colour, as in objprobe.
Shapes are padded to 2 sprites, 2 candidates per pixel and one window size, so both conditions run
the same MPS graphs.
Motion is fixed, not learned: a translation along the edge normal n (A -> B), a sin(t), and a rotation
th cos(t) about the midpoint of the shared edge (objfield's linearised rotation). merged moves A u B
with A's motion; split moves B with the mirror one (-a, -th): every pixel moves by the same amount in
both conditions, only the relative motion across the shared edge differs.
Training (same seed, steps, frames): objfield's loss without the motion terms, -activation +
|activation| (wfid rest pose vs still over the case's footprint + wopq outline offsets), on the
colour residual and outline offsets.
Readout, at step 0 and trained: loop score (objfield's 4 fixed crops; mean and min over frames) and
edge band (objprobe's inception4c map of the circularly padded canvas, 32 x 32 cells, mean over the
cells holding the shared edge dilated one cell; mean and min over frames). split_cost = merged - split.

usage: objprobe2.py prep OUT                                      (CPU: cases.npz, cases.png)
       objprobe2.py run OUT --case NAME [--steps --frames --seed --reveal gap --no-video]
                                    (GPU: both conditions, side-by-side mp4; seed k > 0 and gap go below OUT/NAME)
       objprobe2.py report OUT                                    (results.json, summary.png)
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import sys, argparse, json, math, random, time, subprocess
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import objprobe
from objprobe import S, CELL, PAD

STILL = "../../outputs/sheet-4c-2026-10-31/googlenet_inception4c_7.png"
SEG = "../../outputs/obj3d-2026-11-01/seg8b"
PROBE1 = "../../outputs/obj3d-2026-11-01/probe1/pairs.json"
FFMPEG = ("/private/tmp/claude-501/-Users-lawvere-striate/39cfca31-2c02-4b02-a696-d14df31b10bf/scratchpad/videnv/"
          "lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")
# name: (kind, page ids, note in probe1 / how B is made, what tier 1 or the design predicts)
CASES = {
    "59-71": ("neighbour", (59, 71), "", "bound"),
    "croissant57": ("part", (57, 57), "A = forest part 0, B = parts 1 u 2", "free"),
    "43-49": ("neighbour", (43, 49), "", "free"),
    "same82": ("fake", (82, 82), "cut 175 deg", "costly (control)"),
    "sep55-63": ("neighbour", (55, 63), "", "free (control)"),
    "rim82": ("rim", (82, 82), "curved cut near the rim (lens cap, <= 12 px deep)", "?"),
}


# ---------------------------------------------------------------- prep (CPU)

def rim_cut(m, depth=12.0, rr=1.0):
    """Lens cap of mask m: the part inside a disk of radius rr * (object radius) whose centre lies
    outside m, placed so the cap is ``depth`` px thick at its middle. The cut is one arc that meets the
    outline at both ends and stays within ``depth`` of it. Direction: the one of 8 with the longest arc."""
    c = objprobe.centroid(m)  # (x, y)
    ys, xs = np.nonzero(m)
    rel = np.stack([objprobe.wrapd(xs - c[0]), objprobe.wrapd(ys - c[1])], 1)
    R = rr * np.sqrt((rel**2).sum(1)).max()
    best = None
    for k in range(8):
        u = np.array([np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)])
        h = (rel @ u).max()
        q = (h - depth + R) * u
        A = np.zeros_like(m)
        A[ys, xs] = ((rel - q) ** 2).sum(1) <= R * R
        A = ndi.binary_opening(A) & m
        lab, n = ndi.label(A)
        if n == 0:
            continue
        A = lab == 1 + np.argmax(ndi.sum(A, lab, range(1, n + 1)))
        e = (A & objprobe.dil4(m & ~A)).sum()
        if best is None or e > best[0]:
            best = (e, A)
    return best[1], m & ~best[1]


def prep(out):
    still8 = np.asarray(Image.open(STILL).convert("RGB"))
    still = still8.astype(np.float32) / 255
    z = np.load(f"{SEG}/labels.npz")
    lab, scores = z["labels"], z["scores"]
    a = SimpleNamespace(seg=SEG, gap=4, min_edge=10, n_same=24, extra="")
    t0 = time.time()
    pairs, _ = objprobe.build(a, still, lab, scores)
    print(f"{len(pairs)} pairs in {time.time() - t0:.0f}s", flush=True)
    p1 = json.load(open(PROBE1))["pairs"]
    lum = still @ np.array([0.299, 0.587, 0.114])
    save, meta = {}, {}
    for name, (kind, pg, note, pred) in CASES.items():
        ids = [v + 1 for v in pg]
        if kind == "rim":
            A, B = rim_cut(lab == ids[0])
            t1 = None
        elif kind == "part":
            sel = [p for p in pairs if p["kind"] == "part" and p["ids"][0] == ids[0]]
            P0 = next(p for p in sel if p["note"].startswith("forest 0/1"))
            P2 = next(p for p in sel if p["note"].startswith("forest 0/2"))
            A, B = P0["A"], P0["B"] | P2["B"]
            t1 = [r["strength"] for r in p1 if r["kind"] == "part" and r["page_ids"][0] == pg[0] and r["note"].startswith("forest 0/")]
        else:
            p = next(p for p in pairs if p["kind"] == kind and p["ids"] == ids and (not note or p["note"] == note))
            A, B = p["A"], p["B"]
            t1 = [next(r["strength"] for r in p1 if r["kind"] == kind and r["page_ids"] == list(pg) and (not note or r["note"] == note))]
        g = objprobe.edge_geom(A, B, lum)
        ey, ex = g["px"]
        piv = objprobe.centroid(np.isin(np.arange(S * S), ey * S + ex).reshape(S, S))  # (x, y) on the torus
        save[f"{name}_A"], save[f"{name}_B"], save[f"{name}_cells"] = A, B, g["cells"]
        meta[name] = dict(kind=kind, page_ids=list(pg), note=note, predicted=pred, tier1_strength=t1,
                          n_xy=g["n"].tolist(), t_xy=g["t"].tolist(), pivot_xy=piv.tolist(), edge_length=g["length"],
                          crevice_depth=round(g["depth"], 4), area_A=int(A.sum()), area_B=int(B.sum()))
        print(name, meta[name], flush=True)
    np.savez_compressed(f"{out}/cases.npz", **save)
    json.dump(meta, open(f"{out}/cases.json", "w"), indent=1)
    # cases.png: each case's A (red) and B (blue) over the still, edge cells outlined
    tiles = []
    for name in CASES:
        im = still8.astype(np.float32).copy()
        for msk, col in ((save[f"{name}_A"], (255, 60, 60)), (save[f"{name}_B"], (60, 120, 255))):
            im[msk] = 0.55 * im[msk] + 0.45 * np.array(col)
        cl = np.kron(save[f"{name}_cells"], np.ones((CELL, CELL), bool))
        im[cl & ~ndi.binary_erosion(cl)] = (255, 255, 0)
        t = Image.fromarray(im.astype(np.uint8))
        ImageDraw.Draw(t).text((6, 4), name, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        tiles.append(t)
    sheet = Image.new("RGB", (3 * S, 2 * S))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 3) * S, (i // 3) * S))
    sheet.save(f"{out}/cases.png")


# ---------------------------------------------------------------- scene (objfield's, cut down to one case)

def scene(regs, logits, pivot, H, a):
    """Sprites 1.. from masks regs (sprite 1 over sprite 2, sprite 2 extends a.ext px under sprite 1),
    padded to 2 sprites and 2 candidates per pixel. pivot (y, x): every sprite's centre (rotation pivot).
    Returns objfield.build_scene's dict."""
    from objfield import inpaint, mirror_fill, circ_centroid

    K, M = 2, 2
    sdf = np.full((K + 1, H, H), 100.0, np.float32)
    depth = np.full((K + 1, H, H), -1.0, np.float32)
    color = np.zeros((K + 1, 3, H, H), np.float32)
    origin = np.zeros((K + 1, 2), np.int64)
    centre = np.zeros((K + 1, 2), np.float32)
    seam = np.zeros((S, S), bool)
    cand = {}
    for k, m in enumerate(regs, 1):
        ys, xs = np.nonzero(m)
        cy, cx = circ_centroid(ys, xs, S)
        oy, ox = int(round(cy)) - H // 2, int(round(cx)) - H // 2
        origin[k] = oy, ox
        centre[k] = (pivot[0] - oy + S / 2) % S - S / 2, (pivot[1] - ox + S / 2) % S - S / 2
        iy, ix = (oy + np.arange(H)) % S, (ox + np.arange(H)) % S
        mask = m[np.ix_(iy, ix)]
        assert mask.sum() == m.sum(), "window too small"
        ext = mask.copy()
        if k == 2 and a.reveal == "surface":  # B carries on under A
            ext |= (ndi.distance_transform_edt(~mask) <= a.ext) & regs[0][np.ix_(iy, ix)]
        sd = ndi.distance_transform_edt(~ext) - ndi.distance_transform_edt(ext) + 0.5 * (2 * ext - 1)
        sdf[k] = ndi.gaussian_filter(sd.astype(np.float32), a.smooth)
        depth[k] = np.where(mask, 0.0 if k == 1 else -0.5, -1.0)
        wl = logits[:, iy][:, :, ix]
        color[k] = inpaint(wl, mask)
        color[k], rep = mirror_fill(wl, mask, ext & ~mask, color[k], a.smooth, a.inset)
        seam[np.ix_(iy, ix)] |= rep
        reach = ndi.distance_transform_edt(~ext) <= a.reach
        assert not (reach[0].any() or reach[-1].any() or reach[:, 0].any() or reach[:, -1].any()), "window too small"
        ry, rx = np.nonzero(reach)
        for y, x in zip(ry, rx):
            cand.setdefault(int(iy[y] * S + ix[x]), []).append((k, y, x))
    ck = np.zeros((S * S, M), np.int64)
    cq = np.zeros((S * S, M, 2), np.float32)
    for p, lst in cand.items():
        for j, (k, y, x) in enumerate(lst):
            ck[p, j] = k
            cq[p, j] = y, x
    wall = np.zeros((1, 4))  # Field wants walls; no wall terms here
    return dict(H=H, K=K, M=M, sdf=sdf, depth=depth, color=color, origin=origin, centre=centre, ck=ck, cq=cq,
                wall=wall, wall_c=np.zeros(1, np.float32), wall_n=np.zeros((1, 2), np.float32), seam=seam)


def window(regs, a):
    """Window size that holds every sprite of every condition, its extension and its reach."""
    from objfield import circ_centroid
    r = 0
    for m in regs:
        ys, xs = np.nonzero(m)
        cy, cx = circ_centroid(ys, xs, S)
        r = max(r, np.sqrt(objprobe.wrapd(ys - cy) ** 2 + objprobe.wrapd(xs - cx) ** 2).max())
    H = int(2 * math.ceil(r + a.ext + a.reach + 3))
    return -(-H // 16) * 16


# ---------------------------------------------------------------- run (GPU)

def run(a):
    import torch
    import torch.nn.functional as F
    from striate.model import Backbone
    from striate.util import pick_device
    from zoo import MultiScaleCrop
    from ring import Ring
    from objfield import Field

    dev = pick_device()
    var = "/gap" if a.reveal == "gap" else ""
    od = f"{a.out}/{a.case}{var}" + (f"/seed{a.seed}" if a.seed else "")  # variants and replicate seeds below the case
    os.makedirs(od, exist_ok=True)
    fdir = f"{a.frames_dir}/{a.case}" + var.replace("/", "_") + (f"_seed{a.seed}" if a.seed else "")
    os.makedirs(fdir, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)
    meta = json.load(open(f"{a.out}/cases.json"))[a.case]
    z = np.load(f"{a.out}/cases.npz")
    A, B, cells = z[f"{a.case}_A"], z[f"{a.case}_B"], z[f"{a.case}_cells"]
    still8 = np.asarray(Image.open(STILL).convert("RGB"))
    img = still8.astype(np.float32) / 255
    lab = np.load(f"{SEG}/labels.npz")["labels"]
    p = np.clip(img, 1 / 512, 1 - 1 / 512)
    logits = np.log(p / (1 - p)).transpose(2, 0, 1).astype(np.float32)
    # back layer: the still, crevice colour under A u B (objprobe's fill)
    bgm = (lab == 0).astype(float)
    wgt = ndi.gaussian_filter(bgm, 6, mode="wrap")
    fill = np.stack([ndi.gaussian_filter(img[..., c] * bgm, 6, mode="wrap") for c in range(3)], -1) / np.maximum(wgt, 1e-6)[..., None]
    fill[wgt < 0.05] = np.median(img[lab == 0], 0)
    U = A | B
    bg = torch.from_numpy(np.where(U[..., None], fill, img).transpose(2, 0, 1).reshape(3, -1).astype(np.float32))
    foot = torch.from_numpy(ndi.binary_dilation(U, iterations=int(a.reach)).astype(np.float32)).to(dev)
    piv = np.array(meta["pivot_xy"])[::-1]  # (y, x)
    n = torch.tensor(meta["n_xy"][::-1], dtype=torch.float32, device=dev)  # (y, x), A -> B
    H = window([U, A, B], a)
    still = torch.from_numpy(img.transpose(2, 0, 1))[None].to(dev)
    cells_t = torch.from_numpy(cells).to(dev)
    # cells at least a quarter covered by A u B: the whole case region, for reference
    ucells = torch.from_numpy(U.reshape(S // CELL, CELL, S // CELL, CELL).mean((1, 3)) > 0.25).to(dev)

    net = Backbone("googlenet", dev)
    net.hook("inception4c")
    rg = Ring(net, "inception4c", a.ch, S, dev)
    c0 = PAD // CELL

    def score(fr):  # objfield's loop score
        state = random.getstate()
        tot = 0
        for k in range(4):
            random.seed(1000 + k)
            tot = tot + rg.act(MultiScaleCrop(224 / S, 512)(fr))
        random.setstate(state)
        return tot / 4

    def amap(x):  # objprobe's map: (N, 3, S, S) -> (N, 32, 32)
        m = net(F.pad(x, (PAD,) * 4, mode="circular"))["inception4c"][:, a.ch]
        return m[:, c0 : c0 + S // CELL, c0 : c0 + S // CELL]

    with torch.no_grad():
        m0 = amap(still)[0]
        base = dict(loop=float(score(still)), band=float(m0[cells_t].mean()), region=float(m0[ucells].mean()))
    log(f"{a.case}: window {H} px, edge {meta['edge_length']} px; still loop {base['loop']:.3f} band {base['band']:.3f}")

    class Fixed(Field):
        """Field with a prescribed motion: sg[k] = +1 / -1 per sprite (0 for the padding sprite)."""

        def set_motion(self, sg):
            self.sg = torch.tensor(sg, dtype=torch.float32, device=self.dev)

        def at(self, u):
            B_ = u.shape[0]
            w = 2 * math.pi * u
            sg = self.sg[None, :, None]
            z = lambda d: torch.zeros(B_, self.K + 1, d, device=self.dev)
            return dict(d=sg * a.amp * n * torch.sin(w)[:, None, None],
                        r=sg * math.radians(a.rot) * torch.cos(w)[:, None, None],
                        s=z(1), m2=z(2), m3=z(2))

    T = a.frames
    res = dict(case=a.case, meta=meta, still=base, steps=a.steps, frames=T, seed=a.seed, amp_px=a.amp, rot_deg=a.rot, window=H)
    frames = {}
    for cond in ("merged", "split"):
        random.seed(a.seed)
        torch.manual_seed(a.seed)
        regs = [U] if cond == "merged" else [A, B]
        sc = scene(regs, logits, piv, H, a)
        fld = Fixed(sc, logits, dev, 1, bg, 0.0, a.edge_w)
        fld.set_motion([0, 1, 1] if cond == "merged" else [0, 1, -1])
        nk = len(regs)
        fid_w = torch.from_numpy(np.where(sc["seam"], 0.0, 1.0).astype(np.float32))[None, None].to(dev) * foot
        fid_n = float(fid_w.sum()) * 3

        def evaluate(write):
            loop, band, region = [], [], []
            with torch.no_grad():
                for i in range(0, T, a.mb):
                    u = torch.arange(i, min(T, i + a.mb), device=dev, dtype=torch.float32) / T
                    fr = fld.render(u)
                    loop.append(score(fr).cpu())
                    mp = amap(fr)
                    band.append(mp[:, cells_t].mean(1).cpu())
                    region.append(mp[:, ucells].mean(1).cpu())
                    if write:
                        for t, f in zip(range(i, i + len(u)), fr.cpu()):
                            np.save(f"{fdir}/{cond}_{t:04d}.npy", (f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
            st = lambda v: dict(mean=float(v.mean()), min=float(v.min()), per_frame=[round(float(x), 4) for x in v])
            return dict(loop=st(torch.cat(loop)), band=st(torch.cat(band)), region=st(torch.cat(region)))

        def rest_err():
            with torch.no_grad():
                r = fld.render(torch.zeros(1, device=dev), still=True)
                return float(((r - still).abs() * foot).sum() / foot.sum() / 3)

        r0 = dict(evaluate(False), rest_err=rest_err())
        log(f"{cond} step 0: loop {r0['loop']['mean']:.3f} [{r0['loop']['min']:.3f}] band {r0['band']['mean']:.3f} [{r0['band']['min']:.3f}] rest err {r0['rest_err']:.4f}")
        opt = torch.optim.Adam([{"params": [fld.res.spectrum], "lr": a.lr_res}, {"params": [fld.alpha], "lr": a.lr_alpha}])
        base_lr = [g["lr"] for g in opt.param_groups]
        curve = []
        ts = time.time()
        for it in range(a.steps):
            for g, b in zip(opt.param_groups, base_lr):
                g["lr"] = b * 0.5 * (1 + math.cos(math.pi * it / a.steps))
            opt.zero_grad()
            u = torch.tensor(random.sample(range(T), a.mb), device=dev, dtype=torch.float32) / T
            fr = fld.render(u)
            act = rg.act(MultiScaleCrop(224 / S, 512)(fr))
            opq = (fld.alpha[1 : 1 + nk] ** 2).mean() / 4.0
            rest = fld.render(torch.zeros(1, device=dev), still=True)
            fid = (((rest - still) ** 2) * fid_w).sum() / fid_n / 0.01
            pen = a.wopq * opq + a.wfid * fid
            loss = (-act + act.detach().abs() * pen).sum()
            loss.backward()
            opt.step()
            if it % 50 == 0 or it == a.steps - 1:
                curve.append((it, float(act.detach().mean())))
                log(f"{cond} step {it + 1}/{a.steps} act {act.mean():.3f} fid {float(fid):.3f} opq {float(opq):.4f}")
        secs = time.time() - ts
        r1 = dict(evaluate(True), rest_err=rest_err())
        log(f"{cond} trained: loop {r1['loop']['mean']:.3f} [{r1['loop']['min']:.3f}] band {r1['band']['mean']:.3f} [{r1['band']['min']:.3f}] rest err {r1['rest_err']:.4f}")
        res[cond] = dict(untrained=r0, trained=r1, curve=curve, train_secs=round(secs))
        torch.save({"alpha": fld.alpha.detach().cpu(), "res": fld.res.spectrum.detach().cpu(), "args": vars(a)}, f"{od}/{cond}.pt")
        del fld, opt
        if str(dev) == "mps":
            torch.mps.empty_cache()
    for st in ("untrained", "trained"):
        res[f"split_cost_{st}"] = {f"{r}_{s}": res["merged"][st][r][s] - res["split"][st][r][s]
                                   for r in ("loop", "band", "region") for s in ("mean", "min")}
    res["secs"] = round(time.time() - t0)
    json.dump(res, open(f"{od}/result.json", "w"), indent=1)
    log(f"split cost untrained {res['split_cost_untrained']}")
    log(f"split cost trained   {res['split_cost_trained']}")
    if not a.no_video:
        video(a, fdir, od, T)


def video(a, fdir, od, T):
    """merged | split side by side at full frame, labelled, as an mp4 loop."""
    try:
        fnt = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
    except Exception:
        fnt = ImageFont.load_default()
    for t in range(T):
        im = Image.new("RGB", (2 * S + 8, S), (0, 0, 0))
        for i, cond in enumerate(("merged", "split")):
            im.paste(Image.fromarray(np.load(f"{fdir}/{cond}_{t:04d}.npy")), (i * (S + 8), 0))
            ImageDraw.Draw(im).text((i * (S + 8) + 8, 6), f"{a.case} {cond}" + (" (gap reveal)" if a.reveal == "gap" and cond == "split" else ""), font=fnt, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        im.save(f"{fdir}/sbs_{t:04d}.png")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", "24", "-i", f"{fdir}/sbs_%04d.png", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", f"{od}/{a.case}" + ("_gap" if a.reveal == "gap" else "") + "_merged_vs_split.mp4"], check=True)


# ---------------------------------------------------------------- report

def report(out):
    import glob
    summ = {}
    keys = [f"{r}_{s}" for r in ("loop", "band", "region") for s in ("mean", "min")]
    for name in [c + v for v in ("", "/gap") for c in CASES]:
        f = f"{out}/{name}/result.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        m = dict(r["meta"])
        if name.endswith("/gap"):
            m["predicted"] = m["predicted"].split(" (")[0] + ", gap reveal"  # no extension: crevice colour shows
        reps = [r] + [json.load(open(g)) for g in sorted(glob.glob(f"{out}/{name}/seed*/result.json"))]
        tr = {k: [x["split_cost_trained"][k] for x in reps] for k in keys}
        summ[name] = dict(
            kind=m["kind"], page_ids=m["page_ids"], note=m["note"], predicted=m["predicted"], tier1_strength=m["tier1_strength"],
            edge_length=m["edge_length"], area_A=m["area_A"], area_B=m["area_B"], still=r["still"],
            **{cond: {st: dict(loop_mean=r[cond][st]["loop"]["mean"], loop_min=r[cond][st]["loop"]["min"],
                               band_mean=r[cond][st]["band"]["mean"], band_min=r[cond][st]["band"]["min"],
                               region_mean=r[cond][st]["region"]["mean"], rest_err=r[cond][st]["rest_err"])
                      for st in ("untrained", "trained")} for cond in ("merged", "split")},
            split_cost_untrained=r["split_cost_untrained"], split_cost_trained=r["split_cost_trained"],
            # seed 0 above; every seed (training noise only: step 0 is deterministic) below
            seeds=[x["seed"] for x in reps], split_cost_trained_by_seed=tr,
            split_cost_trained_mean={k: float(np.mean(v)) for k, v in tr.items()},
            split_cost_trained_spread={k: float(np.ptp(v)) for k, v in tr.items()},
            recovered_band=1 - float(np.mean(tr["band_mean"])) / r["split_cost_untrained"]["band_mean"],
            steps=r["steps"], frames=r["frames"], secs=[x["secs"] for x in reps])
    json.dump(summ, open(f"{out}/results.json", "w"), indent=1)
    plot(summ, f"{out}/summary.png")


def plot(summ, path):
    """Cases ranked by trained edge-band split cost: step-0 and trained bars per readout, seeds as dots,
    and the tier-1 strength alongside. PIL, as objprobe's figures."""
    names = sorted(summ, key=lambda k: -summ[k]["split_cost_trained_mean"]["band_mean"])
    try:
        fnt = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
        bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 14)
    except Exception:
        fnt = bold = ImageFont.load_default()
    INK, MUTED, GRID = (30, 30, 30), (110, 110, 110), (225, 225, 225)
    LIGHT, DARK, DOT, RED = (170, 190, 230), (42, 92, 200), (15, 35, 90), (200, 60, 50)
    LW, PW, TW, RH, TOP = 250, 330, 170, 46, 70
    W, H = LW + 2 * PW + TW + 40, TOP + RH * len(names) + 70
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    dr.text((12, 10), "inception4c:7: does moving A and B apart cost? (split cost = merged - split; > 0 costs)", font=bold, fill=INK)
    panels = [("band_mean", "edge band (cells on the shared edge)"), ("loop_mean", "loop score (whole frame)")]
    for k, (key, title) in enumerate(panels):
        x0 = LW + k * PW
        vals = [summ[n]["split_cost_untrained"][key] for n in names] + [v for n in names for v in summ[n]["split_cost_trained_by_seed"][key]]
        lo, hi = min(0, min(vals)), max(0, max(vals))
        pad = 0.08 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        X = lambda v: x0 + 10 + (v - lo) / (hi - lo) * (PW - 30)
        dr.text((x0 + 10, TOP - 28), title, font=bold, fill=INK)
        step = 10 ** math.floor(math.log10((hi - lo) / 3))
        step *= 2 if (hi - lo) / step > 8 else 1
        for g in np.arange(math.ceil(lo / step) * step, hi, step):
            dr.line([X(g), TOP, X(g), TOP + RH * len(names)], fill=GRID)
            dr.text((X(g), TOP + RH * len(names) + 6), f"{g:.{max(0, -math.floor(math.log10(step)))}f}", font=fnt, fill=MUTED, anchor="mt")
        dr.line([X(0), TOP - 4, X(0), TOP + RH * len(names)], fill=MUTED, width=1)
        for i, n in enumerate(names):
            yc = TOP + i * RH + RH / 2
            u = summ[n]["split_cost_untrained"][key]
            t = summ[n]["split_cost_trained_mean"][key]
            for v, dy, col in ((u, -9, LIGHT), (t, 9, DARK)):
                a_, b_ = sorted((X(0), X(v)))
                dr.rounded_rectangle([a_, yc + dy - 7, max(b_, a_ + 1), yc + dy + 7], radius=3, fill=col)
            for v in summ[n]["split_cost_trained_by_seed"][key]:
                dr.ellipse([X(v) - 3.5, yc + 9 - 3.5, X(v) + 3.5, yc + 9 + 3.5], fill=DOT, outline="white")
    for i, n in enumerate(names):
        yc = TOP + i * RH + RH / 2
        dr.text((12, yc - 8), n, font=bold, fill=INK, anchor="lm")
        dr.text((12, yc + 9), summ[n]["predicted"], font=fnt, fill=MUTED, anchor="lm")
    x0 = LW + 2 * PW + 20
    X = lambda v: x0 + 10 + v * (TW - 30)
    dr.text((x0 + 10, TOP - 28), "tier-1 strength", font=bold, fill=INK)
    for g in (0, 0.5, 1):
        dr.line([X(g), TOP, X(g), TOP + RH * len(names)], fill=GRID)
        dr.text((X(g), TOP + RH * len(names) + 6), f"{g:g}", font=fnt, fill=MUTED, anchor="mt")
    for i, n in enumerate(names):
        yc = TOP + i * RH + RH / 2
        s1 = summ[n]["tier1_strength"]
        if s1:
            for v in s1:
                dr.ellipse([X(v) - 5, yc - 5, X(v) + 5, yc + 5], fill=RED, outline="white")
        else:
            dr.text((X(0.5), yc), "not probed", font=fnt, fill=MUTED, anchor="mm")
    ly = H - 30
    for k, (col, lab) in enumerate(((LIGHT, "step 0 (untrained)"), (DARK, "trained, mean over seeds"))):
        dr.rounded_rectangle([LW + 10 + k * 220, ly, LW + 30 + k * 220, ly + 12], radius=3, fill=col)
        dr.text((LW + 36 + k * 220, ly + 6), lab, font=fnt, fill=INK, anchor="lm")
    dr.ellipse([LW + 450, ly + 2, LW + 458, ly + 10], fill=DOT)
    dr.text((LW + 464, ly + 6), "each seed", font=fnt, fill=INK, anchor="lm")
    dr.ellipse([LW + 560, ly + 1, LW + 570, ly + 11], fill=RED)
    dr.text((LW + 576, ly + 6), "tier-1 strength (one per pair edge)", font=fnt, fill=INK, anchor="lm")
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "run", "report"])
    ap.add_argument("out")
    ap.add_argument("--case", default="")
    ap.add_argument("--frames-dir", default="", help="per-frame arrays and side-by-side pngs (scratch)")
    ap.add_argument("--ch", type=int, default=7)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--amp", type=float, default=2.5, help="translation along the edge normal, px")
    ap.add_argument("--rot", type=float, default=3.0, help="rotation about the edge midpoint, deg")
    ap.add_argument("--ext", type=float, default=10, help="B's extension under A, px")
    ap.add_argument("--reveal", default="surface", choices=["surface", "gap"],
                    help="what split's opening shows: B's amodal extension, or (objprobe's) the crevice colour of the back layer")
    ap.add_argument("--reach", type=float, default=10, help="how far a sprite can move, px")
    ap.add_argument("--smooth", type=float, default=2.5)
    ap.add_argument("--edge-w", type=float, default=0.5)
    ap.add_argument("--inset", type=float, default=1.5)
    ap.add_argument("--wfid", type=float, default=2.0)
    ap.add_argument("--wopq", type=float, default=0.5)
    ap.add_argument("--lr-res", type=float, default=0.01)
    ap.add_argument("--lr-alpha", type=float, default=0.02)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage == "prep":
        prep(a.out)
    elif a.stage == "run":
        run(a)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
