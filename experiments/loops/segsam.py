"""Object segmentation of a periodic render from SAM2's own masks, by fixed rules (replaces segobj.py's
post-processing; segobj.py stays for seg7). Nothing is fitted to a channel: the constants are SAM2's
own (256 px logit grid, 32x32 point grid, its score thresholds and box NMS), majorities, or fixed
structure (pad, zoom levels, the 0.8 containment, a 25% sanity cap on object area).

1. Views. SAM2 automatic masks on the wrap-padded canvas (pad S/4) at three zooms: the whole padded
   canvas, then 3x3 and 7x7 half-overlapping crops (each level halves the crop), on two renderings of
   the still: the still itself and a periodic low-pass of it (lowpass(); takes out the network's pixel
   grid). A network re-render at 4x (stillup.py, --hires) is optional: it is clean, but it redraws
   internal structure and SAM2 then reads wholes as parts (ch7's coil 54+37 is never found whole on it).
2. Masks. Each mask is reduced to its largest connected piece with its holes filled (SAM2 decodes a
   256 px logit grid, so pinholes and stray islands are decoding noise, not shape). Masks touching a
   view border are truncated and dropped; the rest go onto the torus at 2x the still's resolution.
3. Groups. Two masks are one object when they differ only by boundary jitter (same_object: their
   symmetric difference, spread along the outline, is at most one logit cell of the coarser view).
   Leaders are taken in order of how many masks they match and take their unclaimed matches. A group's
   shape is the per-pixel vote of the views that found it; its confidence is the share of the views
   that could have seen it whole (it fits, and spans at least two of SAM2's point spacings) that did.
4. Wholes and parts. A group contains another when it holds most of it (>= 0.8 of the smaller). On
   that forest, bottom up, a group seen whole in at least half its eligible views is kept and its
   parts are dropped (they tile it, or are detail such as spots); otherwise its parts stand in for it.
   Accepted objects that still overlap without nesting: one mostly inside larger ones is their part.
5. Labels. A pixel is inside an object where >= 2/3 of the views that found it agree, outside where
   <= 1/3 do. Inside exactly one object: its id; inside two, or in dispute for an object that no
   other claims: -1 (unknown); otherwise 0 (background / crevice). No forced partition.

Writes labels.npz (labels at the still's size, labels_hr at 2x, scores = view confidence, sources =
views detected / eligible / masks / views per zoom), overlay.png, fill.png, metrics.json (generic
quality numbers, see metrics()). Raw SAM2 masks are cached in <outdir>/cands.npz; with it present
the GPU stage is skipped.

usage: segsam.py still.png outdir [--sources still,lowpass] [--hires x4.png] [--ref old_labels.npy] [--page dir]
Needs `transformers` (SAM2), which is not a project dependency.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, gc, json, time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

VIEW = 1024  # SAM2's input size
LEVELS = 3  # zooms: whole padded canvas, 3x3, 7x7


def views(L, levels):
    """(level, y0, x0, size) squares over an L x L canvas: level l is (2^(l+1)-1)^2 crops of L / 2^l."""
    out = []
    for l in range(levels):
        n, c = 2 ** (l + 1) - 1, L / 2**l
        step = (L - c) / max(n - 1, 1)
        out += [(l, i * step, j * step, c) for i in range(n) for j in range(n)]
    return out


def lowpass(img, keep=0.5):
    """Periodic low-pass of a still: frequencies above `keep` x Nyquist (max-norm) are removed with a
    raised-cosine roll-off over 0.8..1.2 x keep. At 0.5 this takes out the pixel checkerboard and most of
    the period-4 grid a strided network writes, and keeps the forms."""
    S = img.shape[0]
    f = np.maximum(np.abs(np.fft.fftfreq(S))[:, None], np.abs(np.fft.rfftfreq(S))[None, :]) * 2
    m = 0.5 + 0.5 * np.cos(np.pi * np.clip((f - 0.8 * keep) / (0.4 * keep), 0, 1))
    x = np.fft.irfft2(np.fft.rfft2(img.astype(np.float64), axes=(0, 1)) * m[..., None], s=(S, S), axes=(0, 1))
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def same_object(sym, pa, pb, cell):
    """Two masks are one object when they differ only by boundary jitter: their symmetric difference,
    spread along their mean outline, is no wider than one SAM2 logit cell (4 view px)."""
    return sym <= cell * (pa + pb) / 2


def clean(m):
    """Largest connected piece, holes filled."""
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1
    return ndi.binary_fill_holes(m)


def sam_stage(sources, S, R, out):
    """Run SAM2 on every view of every source image (renderings of one scene at integer multiples of
    the still's size); store the candidates (torus masks at R px, as bbox crops)."""
    import torch
    from transformers import pipeline

    P = S // 4
    L = S + 2 * P  # padded canvas, in still px
    gen = pipeline("mask-generation", model="facebook/sam2.1-hiera-large", device="mps" if torch.backends.mps.is_available() else "cpu")
    k = R / S  # torus px per still px
    V = [(si, *v) for si in range(len(sources)) for v in views(L, LEVELS)]
    cand = dict(view=[], score=[], box=[], bits=[])
    t0 = time.time()
    for vi, (si, lvl, y0, x0, c) in enumerate(V):
        f = sources[si].shape[0] // S
        if vi == 0 or V[vi - 1][0] != si:
            big = Image.fromarray(np.pad(sources[si], ((P * f, P * f), (P * f, P * f), (0, 0)), mode="wrap"))
        box = tuple(round(v * f) for v in (x0, y0, x0 + c, y0 + c))
        crop = big.crop(box).resize((VIEW, VIEW), Image.LANCZOS)
        # SAM2's own box NMS (0.7) stays; turning it off cost 2.5x the time and changed no whole on ch7.
        # Masks it lets through that are one object up to decoding jitter are reduced to the best-scored one.
        res = gen(crop, points_per_batch=16, pred_iou_thresh=0.8, stability_score_thresh=0.85)
        ms, sc = res["masks"], res["scores"].float().cpu().numpy()
        del res
        nraw, nk, g = len(ms), 0, VIEW // 4
        K, Kp, nlo = np.zeros((nraw, g * g), np.float32), np.zeros(nraw), 0
        for o in np.argsort(-sc):
            m, s = np.asarray(ms[o], bool), float(sc[o])
            ms[o] = None
            if not m.any():
                continue
            lo = m.reshape(g, 4, g, 4).mean((1, 3)) > 0.5  # SAM2's logit grid
            lp = crack(lo)
            lo = lo.ravel().astype(np.float32)
            if nlo:
                it = K[:nlo] @ lo
                if same_object(K[:nlo].sum(1) + lo.sum() - 2 * it, Kp[:nlo], lp, 1.0).any():
                    continue
            m = clean(m)
            if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
                continue  # truncated by the view
            K[nlo], Kp[nlo] = lo, lp
            nlo += 1
            ys, xs = np.nonzero(m)
            a0, a1, b0, b1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            # torus px r has its centre at (r + 0.5) / k still px, i.e. view px (u + P - y0) * VIEW / c - 0.5
            q = VIEW / c / k  # view px per torus px
            r0, r1 = int(np.floor((y0 - P + a0 * c / VIEW) * k)) - 1, int(np.ceil((y0 - P + a1 * c / VIEW) * k)) + 1
            s0, s1 = int(np.floor((x0 - P + b0 * c / VIEW) * k)) - 1, int(np.ceil((x0 - P + b1 * c / VIEW) * k)) + 1
            if r1 - r0 > R or s1 - s0 > R:
                continue
            fm = m.astype(np.float32)
            if q > 1.5:  # torus px coarser than view px: box-filter first (antialiasing)
                fm = ndi.uniform_filter(fm, size=int(round(q)))
            gy = ((np.arange(r0, r1) + 0.5) / k + P - y0) * VIEW / c - 0.5
            gx = ((np.arange(s0, s1) + 0.5) / k + P - x0) * VIEW / c - 0.5
            t = ndi.map_coordinates(fm, np.meshgrid(gy, gx, indexing="ij"), order=1, mode="constant")
            t = t > 0.5
            if t.sum() < 4:
                continue
            ys, xs = np.nonzero(t)
            t = t[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            cand["view"].append(vi)
            cand["score"].append(s)
            cand["box"].append((r0 + ys.min(), s0 + xs.min(), t.shape[0], t.shape[1]))
            cand["bits"].append(np.packbits(t.ravel()))
            nk += 1
        del K, ms
        print(f"[{time.time() - t0:.0f}s] view {vi + 1}/{len(V)} level {lvl}: {nraw} masks, {nk} kept", flush=True)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    del gen
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    np.savez_compressed(
        out,
        view=np.array(cand["view"]),
        score=np.array(cand["score"], np.float32),
        box=np.array(cand["box"], np.int32).reshape(-1, 4),
        bits=np.concatenate(cand["bits"]) if cand["bits"] else np.zeros(0, np.uint8),
        off=np.cumsum([0] + [len(b) for b in cand["bits"]]),
        views=np.array(V, np.float32),  # source, level, y0, x0, size (padded still px)
        S=S,
        R=R,
        P=P,
    )


def plabel(m):
    """Connected components of a bool mask on the torus (4-connected)."""
    lab, n = ndi.label(m)
    if n == 0:
        return lab, 0
    par = np.arange(n + 1)

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for a, b in ((lab[0], lab[-1]), (lab[:, 0], lab[:, -1])):
        for x, y in zip(a[(a > 0) & (b > 0)], b[(a > 0) & (b > 0)]):
            ra, rb = find(x), find(y)
            if ra != rb:
                par[max(ra, rb)] = min(ra, rb)
    root = np.array([find(i) for i in range(n + 1)])
    u, inv = np.unique(root, return_inverse=True)
    return inv.reshape(-1)[lab], len(u) - 1


def load_cands(path):
    z = np.load(path)
    R = int(z["R"])
    bits, off, box = z["bits"], z["off"], z["box"]
    pix, per = [], []
    for i, (r0, s0, h, w) in enumerate(box):
        t = np.unpackbits(bits[off[i] : off[i + 1]])[: h * w].reshape(h, w).astype(bool)
        ys, xs = np.nonzero(t)
        pix.append((((ys + r0) % R) * R + (xs + s0) % R).astype(np.int64))
        per.append(crack(np.pad(t, 1)))
    return z, pix, np.array(per, np.float64)


def overlaps(pix, n_px):
    """Sparse pairwise intersection counts and areas of pixel sets."""
    from scipy import sparse

    ind = np.concatenate(pix)
    ptr = np.cumsum([0] + [len(p) for p in pix])
    M = sparse.csr_matrix((np.ones(len(ind), np.float32), ind, ptr), shape=(len(pix), n_px))
    inter = (M @ M.T).tocoo()
    area = np.diff(ptr).astype(np.float64)
    return inter.row, inter.col, inter.data.astype(np.float64), area


def select(z, pix, per, R, log=print):
    """Groups, confidences, the whole/part choice and per-object votes. Returns accepted objects."""
    S, P = int(z["S"]), int(z["P"])
    k = R / S
    V = z["views"]
    view, score, box = z["view"], z["score"], z["box"]
    N = len(pix)
    # 1. group duplicates across views and zooms
    i, j, inter, area = overlaps(pix, R * R)
    cell = 4 * V[:, 4] / VIEW * k  # one logit cell of each view, in torus px
    cv = cell[view]
    d = same_object(area[i] + area[j] - 2 * inter, per[i], per[j], np.maximum(cv[i], cv[j]))
    i, j = i[d], j[d]
    nb = [[] for _ in range(N)]
    for a, b in zip(i, j):
        nb[a].append(b)
    deg = np.array([len(x) for x in nb])
    grp = -np.ones(N, int)
    groups = []
    for a in np.lexsort((-score, -deg)):
        if grp[a] >= 0:
            continue
        mem = [a] + [b for b in nb[a] if grp[b] < 0 and b != a]
        grp[mem] = len(groups)
        groups.append(mem)
    log(f"{N} masks -> {len(groups)} groups")
    # 2. per group: vote per pixel (share of the views that found it, each view counted once), eligibility
    G = []
    spacing = VIEW / 32  # SAM2's automatic point grid
    for g, mem in enumerate(groups):
        vs = np.unique(view[mem])
        pv = [np.unique(np.concatenate([pix[m] for m in mem if view[m] == v])) for v in vs]
        px, cnt = np.unique(np.concatenate(pv), return_counts=True)
        f = cnt / len(vs)
        maj = px[f > 0.5]
        if len(maj) == 0:
            continue
        # extent (still px, unwrapped) from the leader's box; a view is eligible if some torus copy of
        # it fits inside, and the object spans at least two point spacings there
        r0, s0, h, w = box[mem[0]]
        by, bx, bh, bw = r0 / k, s0 / k, h / k, w / k
        elig = []
        for vi, (_, l, y0, x0, c) in enumerate(V):
            vy, vx = y0 - P, x0 - P  # view origin on the torus, still px
            m = c / VIEW
            if len(maj) / k**2 * (VIEW / c) ** 2 < np.pi * spacing**2:
                continue
            fy = [n for n in range(-2, 3) if vy + m <= by + n * S and by + n * S + bh <= vy + c - m]
            fx = [n for n in range(-2, 3) if vx + m <= bx + n * S and bx + n * S + bw <= vx + c - m]
            if fy and fx:
                elig.append(vi)
        elig = np.array(elig, int)
        det = np.intersect1d(vs, elig)
        conf = len(det) / len(elig) if len(elig) else 0.0
        G.append(dict(mem=mem, views=vs, px=px, f=f, maj=maj, elig=len(elig), det=len(det), conf=conf,
                      score=float(np.mean(score[mem])), levels=np.bincount(V[vs, 1].astype(int), minlength=LEVELS)))
    # 3. containment forest over the majority masks; bottom-up whole-vs-parts choice
    i, j, inter, area = overlaps([g["maj"] for g in G], R * R)
    parent = -np.ones(len(G), int)
    for a, b, x in zip(i, j, inter):  # b inside a?
        if a != b and area[b] < area[a] and x >= 0.8 * area[b]:
            if parent[b] < 0 or area[a] < area[parent[b]]:
                parent[b] = a
    kids = [[] for _ in G]
    for b, a in enumerate(parent):
        if a >= 0:
            kids[a].append(b)
    best = [None] * len(G)
    for a in np.argsort(area):
        own = G[a]["conf"] >= 0.5 and area[a] <= 0.25 * R * R
        best[a] = [a] if own else [u for c in kids[a] for u in best[c]]
    sel = [u for a in range(len(G)) if parent[a] < 0 for u in best[a]]
    # 4. selected objects that overlap without nesting: one mostly (> 1/2) inside larger selected ones is
    # taken as their part (wholes first, as in step 3)
    sel.sort(key=lambda u: -len(G[u]["maj"]))
    taken = np.zeros(R * R, bool)
    acc = []
    for u in sel:
        m = G[u]["maj"]
        if taken[m].mean() > 0.5:
            continue
        taken[m] = True
        acc.append(G[u])
    log(f"{len(G)} groups with a majority mask, {len(sel)} selected, {len(acc)} accepted")
    return acc, G


def label_votes(acc, n_px):
    """Per-pixel rule: inside an object where >= 2/3 of its views agree, outside where <= 1/3 do.
    Inside exactly one object: that object; inside none and disputed by none: 0; otherwise -1."""
    nin = np.zeros(n_px, np.int16)
    ndis = np.zeros(n_px, np.int16)
    who = np.zeros(n_px, np.int32)
    for o, (px, f) in enumerate(acc, 1):
        ins = f >= 2 / 3
        nin[px[ins]] += 1
        who[px[ins]] = o
        ndis[px[(f > 1 / 3) & ~ins]] += 1
    lab = np.where(nin == 1, who, 0)
    lab[(nin > 1) | ((nin == 0) & (ndis > 0))] = -1
    return lab


def keep_largest(lab):
    """Pieces of a label cut off from its largest piece become -1."""
    out = lab.copy()
    for o in np.unique(lab[lab > 0]):
        c, n = plabel(lab == o)
        if n > 1:
            big = np.argmax(np.bincount(c.ravel())[1:]) + 1
            out[(c > 0) & (c != big)] = -1
    return out


def edge_strength(img):
    """Colour gradient magnitude of the image blurred at 1 px (periodic), so pixel checkerboard doesn't count."""
    col = img.astype(np.float32) / 255
    e = 0
    for c in range(3):
        b = ndi.gaussian_filter(col[..., c], 1.0, mode="wrap")
        e = e + ndi.sobel(b, 0, mode="wrap") ** 2 + ndi.sobel(b, 1, mode="wrap") ** 2
    return np.sqrt(e)


def crack(m):
    """Outline length of a periodic bool mask, in pixel edges."""
    return int((m != np.roll(m, 1, 0)).sum() + (m != np.roll(m, 1, 1)).sum())


def align(lab, img):
    """Mean edge strength on object outlines over its image mean (1 = outlines ignore the image's edges)."""
    e = edge_strength(img)
    sh = [np.roll(lab, d, ax) for ax in (0, 1) for d in (1, -1)]
    bnd = (lab > 0) & np.any([lab != x for x in sh], 0)
    return round(float(e[bnd].mean() / e.mean()), 3) if bnd.any() else None


def metrics(lab, img, ref=None, lab_hr=None, img_hr=None):
    """Generic quality numbers for a label image (ids > 0 objects, 0 background, -1 unknown).
    With lab_hr / img_hr, edge alignment is also measured at 2x against the clean re-render."""
    S = lab.shape[0]
    ids = np.unique(lab[lab > 0])
    jag, wts, tiny, frag = [], [], 0, 0
    for o in ids:
        m = lab == o
        sm = ndi.gaussian_filter(m.astype(np.float32), 2.0, mode="wrap") > 0.5
        if sm.any():
            jag.append(crack(m) / max(crack(sm), 1))
            wts.append(m.sum())
        tiny += m.sum() < 0.0005 * S * S
        frag += plabel(m)[1] > 1
    obj = lab > 0
    out = {
        "objects": int(len(ids)),
        "cover_objects": round(float(obj.mean()), 3),
        "background": round(float((lab == 0).mean()), 3),
        "unknown": round(float((lab < 0).mean()), 3),
        "jaggedness_median": round(float(np.median(jag)), 3) if jag else None,
        "jaggedness_area_weighted": round(float(np.average(jag, weights=wts)), 3) if jag else None,
        "edge_alignment": align(lab, img),
        "tiny_segments": int(tiny),
        "fragmented_segments": int(frag),
        "area_px_median": int(np.median([np.sum(lab == o) for o in ids])) if len(ids) else 0,
    }
    if lab_hr is not None:
        R = lab_hr.shape[0]
        out["edge_alignment_hr"] = align(lab_hr, np.asarray(Image.fromarray(img_hr).resize((R, R), Image.LANCZOS)))
    if ref is not None:
        rid = np.unique(ref[ref > 0])

        def best(a, b, bid):
            r = []
            for o in np.unique(a[a > 0]):
                m = a == o
                hit = np.unique(b[m & (b > 0)])
                r.append(max([(m & (b == h)).sum() / (m | (b == h)).sum() for h in hit], default=0))
            return np.array(r)

        f, g = best(lab, ref, rid), best(ref, lab, ids)
        out["vs_ref"] = {
            "new_best_iou_median": round(float(np.median(f)), 3),
            "ref_best_iou_median": round(float(np.median(g)), 3),
            "ref_matched_iou_0.5": int((g >= 0.5).sum()),
            "ref_segments": int(len(rid)),
            "new_matched_iou_0.5": int((f >= 0.5).sum()),
        }
    return out


def to_still(px, f, R, S):
    """Votes at R px averaged onto the still's grid."""
    k = R // S
    y, x = np.divmod(px, R)
    idx = (y // k) * S + x // k
    u, inv = np.unique(idx, return_inverse=True)
    return u, np.bincount(inv, weights=f) / k**2


def edges(lab):
    e = np.zeros(lab.shape, bool)
    for ax in (0, 1):
        d = lab != np.roll(lab, 1, ax)
        e |= d | np.roll(d, -1, ax)
    return e


def overlays(lab_hr, still, outdir):
    """overlay.png: the still at 2x with white outlines, background dimmed, unknown magenta;
    fill.png: flat colours per object."""
    R = lab_hr.shape[0]
    img = np.asarray(Image.fromarray(still).resize((R, R), Image.LANCZOS)).astype(np.float32)
    ov = img.copy()
    ov[lab_hr == 0] *= 0.35
    ov[lab_hr < 0] = 0.5 * ov[lab_hr < 0] + 0.5 * np.array([255, 0, 200])
    obj = np.where(lab_hr > 0, lab_hr, 0)
    ov[edges(obj) & (obj > 0)] = 255
    Image.fromarray(ov.astype(np.uint8)).save(f"{outdir}/overlay.png")
    pal = np.random.default_rng(1).integers(40, 255, (max(int(lab_hr.max()), 0) + 1, 3)).astype(np.float32)
    fill = np.where((lab_hr > 0)[..., None], 0.45 * img + 0.55 * pal[obj], 0.25 * img)
    fill[lab_hr < 0] = (255, 0, 200)
    Image.fromarray(fill.astype(np.uint8)).save(f"{outdir}/fill.png")


def page_layers(lab, lab_hr, still, d, Z=4):
    """Review-page layers at Z x the still: still.png (nearest), sam.png (white outlines + ids at each
    segment's deepest point; ids shown 0-based, as on the seg7 page), unk.png (tint: background grey,
    unknown magenta), ids.png (still-size segment id - 1 in red, 255 for none; green 255)."""
    from PIL import ImageDraw, ImageFont

    os.makedirs(d, exist_ok=True)
    S = lab.shape[0]
    R = lab_hr.shape[0]
    Image.fromarray(still).resize((S * Z, S * Z), Image.NEAREST).save(f"{d}/still.png")
    big = np.kron(lab_hr, np.ones((S * Z // R,) * 2, np.int32))
    obj = np.where(big > 0, big, 0)
    a = np.zeros(big.shape + (4,), np.uint8)
    a[edges(obj) & (obj > 0)] = (255, 255, 255, 255)
    sam = Image.fromarray(a, "RGBA")
    dr = ImageDraw.Draw(sam)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for o in np.unique(lab[lab > 0]):
        m = lab == o
        if m.sum() < 20:
            continue
        dt = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(dt.argmax(), dt.shape)
        dr.text((x * Z, y * Z), str(o - 1), font=font, fill=(255, 255, 255, 255), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 255))
    sam.save(f"{d}/sam.png")
    u = np.zeros(big.shape + (4,), np.uint8)
    u[big == 0] = (20, 20, 20, 150)
    u[big < 0] = (255, 0, 200, 140)
    Image.fromarray(u, "RGBA").save(f"{d}/unk.png")
    assert lab.max() <= 255
    r = np.where(lab > 0, lab - 1, 255).astype(np.uint8)
    Image.fromarray(np.stack([r, np.full_like(r, 255), np.zeros_like(r)], -1)).save(f"{d}/ids.png")  # green 255: no shell id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("still")
    ap.add_argument("outdir")
    ap.add_argument("--hires", default="", help="clean re-render of the still at an integer multiple (stillup.py)")
    ap.add_argument("--sources", default="still,lowpass", help="renderings SAM2 sees: still, lowpass, hires")
    ap.add_argument("--ref", default="", help="older labels (S x S, ids >= 1) to report agreement with")
    ap.add_argument("--page", default="", help="also write review-page layers (still, sam, unk, ids) here")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    still = np.asarray(Image.open(a.still).convert("RGB"))
    S = still.shape[0]
    hires = np.asarray(Image.open(a.hires).convert("RGB")) if a.hires else still
    named = {"still": still, "lowpass": lowpass(still)}
    if a.hires:
        named["hires"] = hires
    srcs = [named[n] for n in a.sources.split(",")]
    R = 2 * S
    cpath = f"{a.outdir}/cands.npz"
    if not os.path.exists(cpath):
        sam_stage(srcs, S, R, cpath)
    LOG = []

    def log(m):
        print(m, flush=True)
        LOG.append(m)

    z, pix, per = load_cands(cpath)
    acc, G = select(z, pix, per, R, log)
    # deterministic ids: by the deepest point of each object's majority mask, in raster order
    lab_hr = keep_largest(label_votes([(o["px"], o["f"]) for o in acc], R * R).reshape(R, R))
    lab = keep_largest(label_votes([to_still(o["px"], o["f"], R, S) for o in acc], S * S).reshape(S, S))
    order = []
    for o in range(1, len(acc) + 1):
        m = lab == o
        if not m.any():
            order.append((1 << 40, o))
            continue
        dt = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]
        order.append((int(dt.argmax()), o))
    order.sort()
    remap = -np.ones(len(acc) + 2, np.int32)  # index -1 is unknown; objects gone at the still's size too
    remap[0] = 0
    keep = [o for p_, o in order if p_ < 1 << 40]
    for n_, o in enumerate(keep, 1):
        remap[o] = n_
    lab, lab_hr = remap[lab], remap[lab_hr]
    accd = [acc[o - 1] for o in keep]
    ref = None
    if a.ref:
        ref = np.load(a.ref).astype(int)
    M = metrics(lab, still, ref, *((lab_hr, hires) if a.hires else ()))
    M["groups"] = len(G)
    M["masks"] = len(pix)
    log(json.dumps(M))
    np.savez_compressed(
        f"{a.outdir}/labels.npz",
        labels=lab.astype(np.int32),
        labels_hr=lab_hr.astype(np.int32),
        scores=np.array([o["conf"] for o in accd], np.float32),
        sam_scores=np.array([o["score"] for o in accd], np.float32),
        sources=np.array([[o["det"], o["elig"], len(o["mem"]), *o["levels"]] for o in accd], np.int32).reshape(-1, 3 + LEVELS),
        sources_cols=np.array(["views_detected", "views_eligible", "masks", *[f"views_level{l}" for l in range(LEVELS)]]),
    )
    M["log"] = LOG
    json.dump(M, open(f"{a.outdir}/metrics.json", "w"), indent=1)
    overlays(lab_hr, still, a.outdir)
    if a.page:
        page_layers(lab, lab_hr, still, a.page)


if __name__ == "__main__":
    main()
