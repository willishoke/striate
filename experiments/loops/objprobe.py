"""What does a channel treat as one object? Forward passes only: move one region of the still a little
while its neighbour stays, and compare with moving both together by the same motion. If the channel's
activation at their shared edge drops more when they move apart, the channel binds them.

Pairs. neighbour: two segments that touch or are parted only by a band of 0 / -1 pixels <= --gap px
(the band is split between them by alternate growth, so the pair tiles its union). part: inside an
accepted object, SAM2 part masks from segsam's containment forest (step 3), taken greedily by area
while they overlap the chosen ones by < 20%, if >= 2 cover >= 70% of it (rest grown into the nearest);
each touching pair of them. --extra adds a fixed split from other labels (e.g. seg7's 72+73 / 75).
Calibration (no ground truth): same = fake cuts, 3 lines through the centroid of the largest confident
(view confidence >= 0.9) objects; separate = neighbour pairs whose crevice is in the top quartile of
depth (1 - min / ends of the luminance profile across the edge, median along it). Named objects are
left out of both.

Motion. Rigid, applied to an image and its 1 px feathered alpha, sampled bilinearly on the torus.
Kinds: open / close (translate along the edge normal, 1-4 px), slide (along the tangent), rot (2-5 deg
about the moved region's centroid: the mover's alone, A u B's jointly; same angle). The joint condition
otherwise applies the identical transform to A u B. (Rotating A u B about the mover's centroid instead
swings the partner's far side by several px and made joint cost > independent cost for clear crevices.) Revealed
pixels get the local crevice colour (normalized Gaussian of the still's label-0 pixels, sigma 6), the
same in both conditions. 4 kinds x 2 movers x 2 conditions = one fixed batch of 16.
Readout. inception4c channel map of the circularly padded (128 px) canvas, cropped to the canvas's
32 x 32 cells (as layers.act_map, without its high-pass). drop = baseline - perturbed, mean over the
cells holding the shared edge, dilated by one cell. signal = cost_ind - cost_joint.
Strength = the signal's position between the separate and same reference medians, clipped to [0, 1].

usage: objprobe.py still.png segdir outdir [--draws 3] [--page dir] [--extra lab.npy:page:a+b/c]
       objprobe.py ... --replot   (stage 3 only, from outdir/raw.npz)
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import sys, argparse, json, time, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segsam

S, CELL, PAD = 512, 16, 128
KINDS = ("open", "close", "slide", "rot")
NAMED = {"59-49-43": (59, 49, 43), "74 parts": (74,), "57 parts": (57,), "37 lobe": (37,), "34 lobe": (34,), "71 (seg7 72/73 + 75)": (71,)}


def wrapd(d):
    return (d + S / 2) % S - S / 2


def disk(r):
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r + 0.5


def dil4(m):
    return m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)


def centroid(m):
    """Circular mean (x, y) of a mask on the torus."""
    ys, xs = np.nonzero(m)
    c = []
    for v in (xs, ys):
        a = 2 * np.pi * v / S
        c.append((np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * np.pi) * S) % S)
    return np.array(c)


def grow(parts, allow):
    """Hand pixels of `allow` not in any part to the parts by alternate 4-neighbour growth."""
    parts = [p.copy() for p in parts]
    free = allow & ~np.any(parts, 0)
    while free.any():
        moved = False
        for p in parts:
            n = dil4(p) & free
            if n.any():
                p |= n
                free &= ~n
                moved = True
        if not moved:
            break
    return parts


def edge_geom(A, B, lum):
    """Shared boundary pixels (in A, 4-adjacent to B), tangent, normal (A -> B), band cells, crevice depth."""
    e = A & dil4(B)
    ys, xs = np.nonzero(e)
    if len(xs) < 2:
        return None
    p = np.stack([xs, ys], 1).astype(float)
    d = wrapd(p - p[0])
    d -= d.mean(0)
    w, v = np.linalg.eigh(d.T @ d)
    t = v[:, 1]
    n = np.array([-t[1], t[0]])
    if n @ wrapd(centroid(B) - centroid(A)) < 0:
        n = -n
    cells = np.zeros((S // CELL,) * 2, bool)
    cells[ys // CELL, xs // CELL] = True
    cells = ndi.maximum_filter(cells, size=3, mode="wrap")
    k = np.arange(-6, 7)
    sel = np.linspace(0, len(xs) - 1, min(len(xs), 200)).astype(int)
    q = p[sel, None] + 0.5 * n + k[None, :, None] * n  # the edge runs between A's pixel and B's
    prof = ndi.map_coordinates(lum, [q[..., 1].ravel(), q[..., 0].ravel()], order=1, mode="grid-wrap").reshape(len(sel), -1)
    ends = prof[:, np.abs(k) >= 4].mean(1)
    depth = float(np.median(1 - prof.min(1) / np.maximum(ends, 1e-3)))
    return dict(px=(ys, xs), t=t, n=n, cells=cells, depth=depth, length=int(len(xs)), proj=d @ t)


def forest(z, pix, per, R):
    acc, G = segsam.select(z, pix, per, R, log=lambda m: None)
    i, j, inter, area = segsam.overlaps([g["px"][g["f"] >= 2 / 3] for g in G], R * R)
    parent = -np.ones(len(G), int)
    for a, b, x in zip(i, j, inter):  # as segsam.select step 3
        if a != b and area[b] < area[a] and x >= 0.8 * area[b]:
            if parent[b] < 0 or area[a] < area[parent[b]]:
                parent[b] = a
    kids = [[] for _ in G]
    for b, a in enumerate(parent):
        if a >= 0:
            kids[a].append(b)
    core = []
    for g in G:  # still-size 2/3 core, the label rule
        u, v = segsam.to_still(g["px"], g["f"], R, S)
        m = np.zeros(S * S, bool)
        m[u[v >= 2 / 3]] = True
        core.append(m.reshape(S, S))
    return acc, G, kids, core


def build(a, still, lab, scores):
    lum = still @ np.array([0.299, 0.587, 0.114])
    ids = [o for o in range(1, lab.max() + 1) if (lab == o).sum() >= 20]
    area = {o: int((lab == o).sum()) for o in ids}
    pairs = []

    def add(kind, A, B, ia, ib, note=""):
        g = edge_geom(A, B, lum)
        if g is None or g["length"] < a.min_edge:
            return
        pairs.append(dict(kind=kind, ids=[int(ia), int(ib)], A=A, B=B, note=note, **g))

    # (a) neighbours
    dl = {o: ndi.maximum_filter(lab == o, footprint=disk(a.gap), mode="wrap") for o in ids}
    for ia in ids:
        for ib in ids:
            if ib <= ia or (dl[ia] & (lab == ib)).sum() < a.min_edge:
                continue
            band = dl[ia] & dl[ib] & (lab <= 0)
            A, B = grow([lab == ia, lab == ib], band)
            add("neighbour", A, B, ia, ib)
    nn = len(pairs)
    # (b) parts from the containment forest
    z, pix, per = segsam.load_cands(f"{a.seg}/cands.npz")
    acc, G, kids, core = forest(z, pix, per, int(z["R"]))
    gi = {id(g): k for k, g in enumerate(G)}
    for o in ids:
        m = lab == o
        best, bi = 0, -1
        for g in acc:
            k = gi[id(g)]
            inter = (core[k] & m).sum()
            iou = inter / ((core[k] | m).sum() + 1e-9)
            if iou > best:
                best, bi = iou, k
        desc, st = [], list(kids[bi]) if bi >= 0 else []
        while st:
            k = st.pop()
            desc.append(k)
            st += kids[k]
        cand = sorted(((core[k] & m).sum(), k) for k in desc)[::-1]
        chosen, cov = [], np.zeros_like(m)
        for ar, k in cand:
            pm = core[k] & m
            if ar < 0.1 * area[o] or (pm & cov).sum() > 0.2 * ar:
                continue
            chosen.append(pm & ~cov)
            cov |= pm
        if len(chosen) < 2 or cov.sum() < 0.7 * area[o]:
            continue
        P = grow(chosen, m)
        for x in range(len(P)):
            for y in range(x + 1, len(P)):
                add("part", P[x], P[y], o, o, f"forest {x}/{y} of {len(P)}")
    if a.extra:  # fixed split from other labels: path:page:a+b/c (page ids)
        path, pg, sp = a.extra.split(":")
        L2 = np.load(path)
        o = int(pg) + 1
        m = lab == o
        sa, sb = [[int(v) + 1 for v in s.split("+")] for s in sp.split("/")]
        P = grow([m & np.isin(L2, sa), m & np.isin(L2, sb)], m)
        add("part", P[0], P[1], o, o, f"extra {sp}")
    # calibration: same = fake cuts through large confident objects; separate = deepest crevices
    named = {v + 1 for t in NAMED.values() for v in t}
    big = sorted((o for o in ids if scores[o - 1] >= 0.9 and o not in named), key=lambda o: -area[o])[: a.n_same]
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:S, :S]
    for o in big:
        m = lab == o
        c = centroid(m)
        ph = rng.uniform(0, np.pi)
        for k in range(3):
            th = ph + k * np.pi / 3
            side = wrapd(xx - c[0]) * np.cos(th) + wrapd(yy - c[1]) * np.sin(th) < 0
            A, B = m & side, m & ~side
            if min(A.sum(), B.sum()) >= 0.25 * area[o]:
                add("fake", A, B, o, o, f"cut {np.degrees(th):.0f} deg")
    dep = np.array([p["depth"] for p in pairs[:nn]])
    thr = np.percentile(dep, 75)
    for p in pairs[:nn]:
        p["sep_ref"] = bool(p["depth"] >= thr and p["length"] >= 16 and not (set(p["ids"]) & named))
    return pairs, thr


def run_gpu(a, still, lab, pairs):
    import torch
    import torch.nn.functional as F
    from striate.util import pick_device
    from striate.model import Backbone

    dev = pick_device()
    net = Backbone("googlenet", dev)
    net.hook("inception4c")
    c0 = PAD // CELL

    def amap(x):  # (N, 3, S, S) -> (N, 32, 32)
        with torch.no_grad():
            m = net(F.pad(x, (PAD,) * 4, mode="circular"))["inception4c"][:, a.ch]
        return m[:, c0 : c0 + S // CELL, c0 : c0 + S // CELL]

    I = torch.tensor(still, dtype=torch.float32, device=dev).permute(2, 0, 1)
    bg = (lab == 0).astype(float)
    wgt = ndi.gaussian_filter(bg, 6, mode="wrap")
    fill = np.stack([ndi.gaussian_filter(still[..., c] * bg, 6, mode="wrap") for c in range(3)], -1) / np.maximum(wgt, 1e-6)[..., None]
    fill[wgt < 0.05] = np.median(still[lab == 0], 0)
    Fl = torch.tensor(fill, dtype=torch.float32, device=dev).permute(2, 0, 1)
    gk = torch.exp(-torch.arange(-2, 3, device=dev, dtype=torch.float32) ** 2 / (2 * 0.7**2))
    gk = gk / gk.sum()

    def soft(m):  # 1 px feather, periodic
        x = torch.tensor(m, dtype=torch.float32, device=dev)[None, None]
        x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="circular"), gk.view(1, 1, 1, 5))
        return F.conv2d(F.pad(x, (0, 0, 2, 2), mode="circular"), gk.view(1, 1, 5, 1))[0]

    yy, xx = torch.meshgrid(torch.arange(S, device=dev, dtype=torch.float32), torch.arange(S, device=dev, dtype=torch.float32), indexing="ij")

    def warp(x, c, th, t):
        """x (N, C, S, S) moved by rotation th about c then translation t (per item), bilinear on the torus."""
        N = x.shape[0]
        c, t, th = (torch.tensor(np.asarray(v), dtype=torch.float32, device=dev) for v in (c, t, th))
        dx = (xx[None] - t[:, 0, None, None] - c[:, 0, None, None] + S / 2) % S - S / 2
        dy = (yy[None] - t[:, 1, None, None] - c[:, 1, None, None] + S / 2) % S - S / 2
        co, si = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
        sx = (c[:, 0, None, None] + co * dx + si * dy) % S
        sy = (c[:, 1, None, None] - si * dx + co * dy) % S
        x0, y0 = sx.floor(), sy.floor()
        fx, fy = (sx - x0)[:, None], (sy - y0)[:, None]
        x0, y0 = x0.long() % S, y0.long() % S  # float % can land on S
        x1, y1 = (x0 + 1) % S, (y0 + 1) % S
        b = torch.arange(N, device=dev)[:, None, None]
        g = lambda yi, xi: x[b, :, yi, xi].permute(0, 3, 1, 2)
        return (g(y0, x0) * (1 - fx) + g(y0, x1) * fx) * (1 - fy) + (g(y1, x0) * (1 - fx) + g(y1, x1) * fx) * fy

    M0 = amap(I[None])[0]
    out = np.zeros((len(pairs), a.draws, len(KINDS), 2, 2), np.float32)  # pair, draw, kind, mover, (ind, joint)
    base = np.zeros(len(pairs), np.float32)
    t0 = time.time()
    for pi, p in enumerate(pairs):
        cells = torch.tensor(p["cells"], device=dev)
        base[pi] = float(M0[cells].mean())
        al = {"A": soft(p["A"]), "B": soft(p["B"]), "AB": soft(p["A"] | p["B"])}
        cen = {"A": centroid(p["A"]), "B": centroid(p["B"]), "AB": centroid(p["A"] | p["B"])}
        n, tg = p["n"], p["t"]
        for d in range(a.draws):
            rng = np.random.default_rng([pi, d, 7])
            regs, cs, ths, ts = [], [], [], []
            for ki, kind in enumerate(KINDS):
                mag, sg = rng.uniform(1, 4), rng.choice([-1, 1])
                ang = np.radians(rng.uniform(2, 5)) * sg
                for mi, mv in enumerate("AB"):
                    s_ = 1 if mv == "A" else -1  # B moves the mirror way, so open / close keep their meaning
                    if kind == "open":
                        t, th = -s_ * mag * n, 0.0
                    elif kind == "close":
                        t, th = s_ * mag * n, 0.0
                    elif kind == "slide":
                        t, th = s_ * sg * mag * tg, 0.0
                    else:
                        t, th = np.zeros(2), s_ * ang
                    for r in (mv, "AB"):
                        regs.append(r)
                        cs.append(cen[r])
                        ths.append(th)
                        ts.append(t)
            alpha = torch.stack([al[r] for r in regs])  # (16, 1, S, S)
            bgs = (1 - alpha) * I[None] + alpha * Fl[None]
            mov = warp(torch.cat([I[None].expand(len(regs), -1, -1, -1), alpha], 1), np.array(cs), np.array(ths), np.array(ts))
            img = mov[:, 3:] * mov[:, :3] + (1 - mov[:, 3:]) * bgs
            drop = (M0[None] - amap(img))[:, cells].mean(1).cpu().numpy()
            out[pi, d] = drop.reshape(len(KINDS), 2, 2)
        if pi % 20 == 0:
            print(f"pair {pi}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
        if pi == 0:
            Image.fromarray((img[:, :3].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(f"{a.out}/example_pair0.png")
    return out, base


def auc(x, y):
    """P(x > y) for x the same reference, y the separate one."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None]
    return float((x > y).mean() + 0.5 * (x == y).mean())


def report(a, still, lab, pairs, out, base, thr):

    ind, jnt = out[..., 0], out[..., 1]  # (P, D, K, mover)
    sig = (ind - jnt).mean(-1)  # (P, D, K)
    kind = np.array([p["kind"] for p in pairs])
    same = kind == "fake"
    sep = np.array([p.get("sep_ref", False) for p in pairs])
    sk = sig.mean(1)  # (P, K) mean over draws
    comb = sk.mean(1)
    stats = {}
    for k, name in enumerate(KINDS + ("all",)):
        v = sk[:, k] if k < len(KINDS) else comb
        su, se = v[same], v[sep]
        stats[name] = dict(auc=auc(su, se), same_median=float(np.median(su)), sep_median=float(np.median(se)),
                           dprime=float((su.mean() - se.mean()) / np.sqrt(0.5 * (su.var() + se.var()) + 1e-12)))
    pick = max(stats, key=lambda k: stats[k]["auc"])
    ki = KINDS.index(pick) if pick in KINDS else None
    val = lambda s: s[..., ki] if ki is not None else s.mean(-1)
    lo, hi = stats[pick]["sep_median"], stats[pick]["same_median"]
    strength = lambda s: np.clip((s - lo) / (hi - lo), 0, 1)
    st_draw = strength(val(sig))  # (P, D)
    st = strength(val(sk))
    # mover agreement: A-alone vs B-alone signal, real pairs, picked kind (or all)
    mvs = (ind - jnt).mean(1)  # (P, K, mover)
    mA, mB = val(np.moveaxis(mvs, -1, 0))
    real = ~same
    agree = float(np.corrcoef(mA[real], mB[real])[0, 1])
    rec = []
    for i, p in enumerate(pairs):
        rec.append(dict(kind=p["kind"], ids=p["ids"], page_ids=[v - 1 for v in p["ids"]], note=p["note"], edge_length=p["length"],
                        crevice_depth=round(p["depth"], 4), sep_ref=bool(sep[i]), base_act=round(float(base[i]), 4),
                        cost_ind=round(float(val(ind.mean((1, 3)))[i]), 5), cost_joint=round(float(val(jnt.mean((1, 3)))[i]), 5),
                        signal_by_kind={k: round(float(sk[i, j]), 5) for j, k in enumerate(KINDS)},
                        cost_ind_A=round(float(mA[i] + val(jnt[:, :, :, 0].mean(1))[i]), 5), cost_ind_B=round(float(mB[i] + val(jnt[:, :, :, 1].mean(1))[i]), 5),
                        strength=round(float(st[i]), 3), strength_draws=[round(float(v), 3) for v in st_draw[i]],
                        strength_sd=round(float(st_draw[i].std()), 3)))
    sd = st_draw[real].std(1)
    summ = dict(kind_used=pick, separation=stats, crevice_depth_threshold=float(thr), n_same_ref=int(same.sum()), n_sep_ref=int(sep.sum()),
                n_neighbour=int((kind == "neighbour").sum()), n_part=int((kind == "part").sum()), draws=a.draws,
                strength_sd_median=float(np.median(sd)), strength_sd_p90=float(np.percentile(sd, 90)), mover_agreement_r=agree)
    # named cases
    nm = {}
    for name, pg in NAMED.items():
        o = {v + 1 for v in pg}
        if len(o) > 1:
            sel = [r for r in rec if r["kind"] == "neighbour" and set(r["ids"]) <= o]
        else:
            sel = [r for r in rec if r["kind"] == "part" and r["ids"][0] in o]
        nm[name] = [dict(page_ids=r["page_ids"], note=r["note"], edge=r["edge_length"], strength=r["strength"], sd=r["strength_sd"],
                         cost_ind=r["cost_ind"], cost_joint=r["cost_joint"]) for r in sel]
    summ["named"] = nm
    json.dump(dict(summary=summ, pairs=rec), open(f"{a.out}/pairs.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    # calib.png: per kind, the two references (histograms) and the real pairs (ticks below)
    rows = [(KINDS[k], sk[:, k]) for k in range(len(KINDS))] + [("all", comb)]
    W, H, L, T = 900, 190, 60, 30
    fig = Image.new("RGB", (W, H * len(rows) + 40), "white")
    dr = ImageDraw.Draw(fig, "RGBA")
    try:
        fnt = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    except Exception:
        fnt = ImageFont.load_default()
    BLUE, RED = (59, 111, 214), (214, 69, 59)
    for r_, (name, v) in enumerate(rows):
        y0 = r_ * H
        lo_, hi_ = np.percentile(v, [1, 99])
        lo_, hi_ = min(lo_, 0), max(hi_, 0)
        X = lambda u: L + (np.clip(u, lo_, hi_) - lo_) / (hi_ - lo_ + 1e-12) * (W - L - 20)
        bins = np.linspace(lo_, hi_, 41)
        hs, hf = np.histogram(v[sep], bins)[0], np.histogram(v[same], bins)[0]
        hs, hf = hs / max(hs.sum(), 1), hf / max(hf.sum(), 1)
        top = max(hs.max(), hf.max(), 1e-9)
        base_y = y0 + H - 70
        for h_, col in ((hs, BLUE), (hf, RED)):
            for k in range(40):
                if h_[k] > 0:
                    dr.rectangle([X(bins[k]) + 1, base_y - h_[k] / top * (H - 110), X(bins[k + 1]) - 1, base_y], fill=(*col, 120))
        dr.line([L, base_y, W - 20, base_y], fill=(150, 150, 150))
        dr.line([X(0), y0 + T, X(0), base_y], fill=(190, 190, 190))
        for vv, yy_, col in ((v[(kind == "neighbour") & ~sep], base_y + 12, (90, 90, 90)), (v[kind == "part"], base_y + 28, (176, 122, 0))):
            for u in vv:
                dr.line([X(u), yy_ - 6, X(u), yy_ + 6], fill=col)
        for u in np.linspace(lo_, hi_, 6):
            dr.text((X(u), base_y + 40), f"{u:.3f}", fill=(80, 80, 80), font=fnt, anchor="mt")
        dr.text((L, y0 + 8), f"{name}: AUC {stats[name]['auc']:.2f}, d' {stats[name]['dprime']:.2f}" + ("   (used for strength)" if name == pick else ""), fill=(0, 0, 0), font=fnt)
        dr.text((8, base_y + 12), "neighbours", fill=(90, 90, 90), font=fnt, anchor="lm")
        dr.text((8, base_y + 28), "parts", fill=(176, 122, 0), font=fnt, anchor="lm")
    yl = H * len(rows) + 10
    dr.rectangle([L, yl, L + 14, yl + 12], fill=(*BLUE, 120))
    dr.text((L + 20, yl), f"separate ref: deep crevice neighbours (n={sep.sum()})", fill=(0, 0, 0), font=fnt)
    dr.rectangle([L + 330, yl, L + 344, yl + 12], fill=(*RED, 120))
    dr.text((L + 350, yl), f"same ref: fake cuts (n={same.sum()})   x: signal = cost_ind - cost_joint (share of each ref)", fill=(0, 0, 0), font=fnt)
    fig.save(f"{a.out}/calib.png")
    # strength lines: full frame at 2x, page overlay at 4x
    cmap = lambda u: np.array([np.interp(u, [0, 0.5, 1], c) for c in ((59, 200, 214), (111, 200, 69), (214, 200, 59))]) / 255  # blue, grey, red
    real_i = [i for i in range(len(pairs)) if kind[i] != "fake"]

    def lines(Z, e):  # each edge pixel as its Z x Z block grown by e: 4 px lines at Z = 2, e = 1 and Z = 4, e = 0
        ov = np.zeros((S * Z, S * Z, 4), np.uint8)
        yy, xx = np.mgrid[-e : Z + e, -e : Z + e]
        for i in sorted(real_i, key=lambda i: pairs[i]["length"])[::-1]:
            p = pairs[i]
            ys, xs = p["px"]
            if p["kind"] == "part":
                on = (np.floor(p["proj"] / 3) % 2 == 0)  # dashed along the edge
                ys, xs = ys[on], xs[on]
            col = (cmap(st[i]) * 255).astype(np.uint8)
            cy, cx = ys * Z, xs * Z
            for dy, dx in zip(yy.ravel(), xx.ravel()):
                ov[(cy + dy) % (S * Z), (cx + dx) % (S * Z)] = (*col, 255)
        return ov

    big = np.asarray(Image.fromarray(still8).resize((2 * S, 2 * S), Image.LANCZOS)).astype(np.float32)
    ov = lines(2, 1).astype(np.float32)
    comp = big * (1 - ov[..., 3:] / 255) + ov[..., :3] * (ov[..., 3:] / 255)
    im = Image.fromarray(comp.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    for o in np.unique(lab[lab > 0]):
        m = lab == o
        if m.sum() < 20:
            continue
        dt = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(dt.argmax(), dt.shape)
        dr.text((2 * x, 2 * y), str(o - 1), font=font, fill=(255, 255, 255), anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
    im.save(f"{a.out}/strength_full.png")
    if a.page:
        os.makedirs(a.page, exist_ok=True)
        for f in ("still.png", "sam.png", "ids.png"):  # the segmentation's own page layers
            shutil.copy(f"{a.page_src}/{f}", f"{a.page}/{f}")
        Image.fromarray(lines(4, 0), "RGBA").save(f"{a.page}/probe.png")


def main():
    global still8
    ap = argparse.ArgumentParser()
    ap.add_argument("still")
    ap.add_argument("seg")
    ap.add_argument("out")
    ap.add_argument("--ch", type=int, default=7)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--gap", type=int, default=4, help="max 0 / -1 band between neighbours, px")
    ap.add_argument("--min-edge", type=int, default=10)
    ap.add_argument("--n-same", type=int, default=24, help="objects given fake cuts")
    ap.add_argument("--extra", default="")
    ap.add_argument("--page", default="", help="write review-page layers here (probe.png + copies from --page-src)")
    ap.add_argument("--page-src", default="", help="segsam --page dir of the segmentation")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N pairs of each kind")
    ap.add_argument("--replot", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    still8 = np.asarray(Image.open(a.still).convert("RGB"))
    still = still8.astype(np.float32) / 255
    z = np.load(f"{a.seg}/labels.npz")
    lab, scores = z["labels"], z["scores"]
    t0 = time.time()
    pairs, thr = build(a, still, lab, scores)
    if a.limit:
        pairs = [p for k in ("neighbour", "part", "fake") for p in [q for q in pairs if q["kind"] == k][: a.limit]] + \
                [p for p in pairs if p.get("sep_ref")][: a.limit]
    print(f"{len(pairs)} pairs ({', '.join(f'{k} {sum(p['kind'] == k for p in pairs)}' for k in ('neighbour', 'part', 'fake'))}, "
          f"sep ref {sum(p.get('sep_ref', False) for p in pairs)}) in {time.time() - t0:.0f}s", flush=True)
    if a.replot:
        r = np.load(f"{a.out}/raw.npz")
        out, base = r["out"], r["base"]
    else:
        out, base = run_gpu(a, still, lab, pairs)
        np.savez(f"{a.out}/raw.npz", out=out, base=base)
    report(a, still, lab, pairs, out, base, thr)


if __name__ == "__main__":
    main()
