"""Rim refill: repaint the skin a checkpoint's rest view shows only at its bodies' outlines, from the skin
just inside them, so the paint holds colour only and not the still's crevice shadow or slivers of the
neighbours (the init wrap assigned paint by render ownership and divided out only a one-light fit; the
outline band kept the still's dark seams, and training, which scores only activation, never cleaned it).

Untrusted texels (paint tex0 with the residual tex folded in, the caps' discs capt, and the relief bump /
capb registered to them): those the rest view (top down, 2x, footprint-filtered) shows only within --band
still px of their body's outline, plus (with --sam 1) those shown mostly where the still's SAM segment is a
different body. Trusted: shown further in, and not by another body's segment. Skin the rest view never
shows is kept as it is (and is not a source).

Refill, body by body, by local continuation, with no colour prior: the low frequencies are the trusted
paint's low-pass carried out from the nearest trusted texel (wrapping around) and softened, so a rim
texel takes the colour of the skin next to it at the same place along and around the body (a rainbow
patch running into the rim keeps running; no texel takes a body-wide mean); the detail comes from
patch_fill on the trusted detail, and the relief's detail is carried with it.

usage: objrefill.py ckpt out_dir [--band 3] [--sam 0] [--relief 1] [--fig 59,56,88,32,70]
       writes out_dir/tube.pt, refill.json, masks.png (unwraps: before | trust | after), crops.png"""
import os, sys, json, math, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from objtube import Mixed, Rig, Cam, build_pairs, render, tex_sample, cap_weight, patch_fill, carry
from striate.util import pick_device

LABELS = "/Users/lawvere/striate/outputs/objfield-2026-10-31/seg7/labels_display.npy"


def blur(x, s, wrap_rows):
    """Separable Gaussian (sigma s texels) of x (N, C, H, W): rows (around) wrap or replicate, columns replicate."""
    r = int(math.ceil(3 * s))
    g = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) / s) ** 2)
    g = g / g.sum()
    N, C, H, W = x.shape
    y = x.reshape(N * C, 1, H, W)
    if wrap_rows:
        y = torch.cat([y[:, :, -r:] if r <= H else y.repeat(1, 1, r // H + 1, 1)[:, :, -r:], y, y[:, :, :r] if r <= H else y.repeat(1, 1, r // H + 1, 1)[:, :, :r]], 2)
    else:
        y = F.pad(y, (0, 0, r, r), mode="replicate")
    y = F.conv2d(y, g.reshape(1, 1, -1, 1))
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="replicate"), g.reshape(1, 1, 1, -1))
    return y.reshape(N, C, H, W)


def continue_low(x, t, wrap_rows, s=1.5):
    """x (C, H, W) known where t (H, W) is 1: its trusted low-pass (normalized convolution, sigma s
    texels) carried out to every other texel from the nearest trusted one (rows wrap), then softened by
    the same sigma: strictly local, so no texel takes a colour from across the body."""
    C, H, W = x.shape
    tf = t.float()
    lp = blur((x * tf)[None], s, wrap_rows)[0] / blur(tf[None, None], s, wrap_rows)[0].clamp(min=1e-6)
    T = t.bool().cpu().numpy()
    if wrap_rows:
        _, (iy, ix) = ndi.distance_transform_edt(~np.concatenate([T, T, T], 0), return_indices=True)
        iy, ix = iy[H:2 * H] % H, ix[H:2 * H]
    else:
        _, (iy, ix) = ndi.distance_transform_edt(~T, return_indices=True)
    ext = lp[:, torch.from_numpy(iy).to(x.device), torch.from_numpy(ix).to(x.device)]
    return blur(ext[None], s, wrap_rows)[0]


def high(x, wrap_rows, s=1.5):
    """Detail finer than ~2 s texels."""
    return x - blur(x[None], s, wrap_rows)[0]


def refill(x, U, T, wrap_rows, seed, regs=()):
    """x (C, H, W) in [0, 1]; U untrusted (refilled), T trusted (the sources) (H, W) bool. regs: maps
    (Cy, h, w) registered to x, refilled the same way (low by continuation, detail carried). -> x, regs."""
    Tf = T.float()
    low = continue_low(x, Tf, wrap_rows)
    det = high(x, wrap_rows)
    keep = (~U).float()[None]
    dfill, src = patch_fill(det, keep, det, wrap_rows=wrap_rows, seed=seed, msrc=Tf[None])
    xn = torch.where(U[None], (low + dfill).clamp(0.002, 0.998), x)
    H, W = U.shape
    outs = []
    for y in regs:
        h, w = y.shape[1:]
        Ub = F.adaptive_avg_pool2d(U.float()[None, None], (h, w))[0, 0] > 0.5
        Tb = F.adaptive_avg_pool2d(Tf[None, None], (h, w))[0, 0] > 0.5
        if Tb.sum() < 4 or Ub.sum() == 0:
            outs.append(y)
            continue
        lw = continue_low(y, Tb.float(), wrap_rows)
        dy = high(y, wrap_rows, 1.5 * h / H)
        outs.append(torch.where(Ub[None], lw + carry(dy, src, H, W), y))
    return xn, outs


def rest_view(tb, ta, rig, dev, Z):
    cam = Cam(np.eye(3), Z, Z, (256.0, 256.0, 0.0), 512, zoom=Z / 512)
    with torch.no_grad():
        sp = tb.spheres()
        return render(tb, sp, cam, build_pairs(sp, cam, 4.0), [rig()], ta, aux=True), sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--band", type=float, default=3.0, help="still px from a body's rest outline: shown only within it = untrusted")
    ap.add_argument("--sam", type=int, default=0, help="also untrust skin shown mostly where the still's SAM segment is another body (costs twice the activation for little)")
    ap.add_argument("--relief", type=int, default=1, help="refill the relief (bump, capb) with the paint")
    ap.add_argument("--fig", default="59,56,88,32,70,52", help="bodies for the unwrap and crop figures")
    ap.add_argument("--chunk", type=int, default=60000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = pick_device()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ta = argparse.Namespace(**ck["args"])
    ta.chunk, ta.mip = a.chunk, 1
    disc = getattr(ta, "caps", "proj") == "disc"
    tb = Mixed.from_state(ck["tubes"], ta, dev)
    rig = Rig(0.0, 0.8, ta, dev)
    rig.load_state_dict(ck["rig"])
    Z = 1024
    zm = Z / 512
    o, sp = rest_view(tb, ta, rig, dev, Z)
    img0 = o["img"][0].cpu()
    K = tb.K
    Kt, _, Ha, Wl = tb.tex0.shape
    pix, obj = o["pix"], o["obj"]
    share = o["ep"][0] / o["den"][0][pix]
    top = torch.zeros(Z * Z, device=dev).scatter_reduce(0, pix, share, "amax", include_self=True)
    dom = (share >= top[pix] - 1e-9) & (share > 0.3)
    own = torch.full((Z * Z,), -1, dtype=torch.long, device=dev)
    own[pix[dom]] = obj[dom]
    own[o["gown"][0] > 0.5] = -1
    own_np = own.reshape(Z, Z).cpu().numpy()
    # still px from the outline of the body that owns each pixel
    p = np.pad(own_np, 1, mode="wrap")
    E = (p[1:-1, 1:-1] != p[:-2, 1:-1]) | (p[1:-1, 1:-1] != p[2:, 1:-1]) | (p[1:-1, 1:-1] != p[1:-1, :-2]) | (p[1:-1, 1:-1] != p[1:-1, 2:])
    dpx = (ndi.distance_transform_edt(~E) + 0.5) / zm
    L = np.load(LABELS).astype(np.int64) - 1
    L2 = np.kron(L, np.ones((2, 2), np.int64)) if Z == 1024 else L
    sam_bad = (L2 >= 0) & (L2 != own_np) & (own_np >= 0)
    # per dominant pair: its pixel's distance and segment
    pd = pix[dom]
    dd = torch.from_numpy(dpx.reshape(-1)).float().to(dev)[pd]
    sb = torch.from_numpy(sam_bad.reshape(-1)).to(dev)[pd]
    ob, tu, tv = obj[dom], o["tu"][0][dom], o["tv"][0][dom]
    # each pixel marks the texels its footprint filters (a rip-map cell: around and along apart, since a
    # flank seen edge-on is squeezed around only), not just the one under its centre
    la = torch.log2((o["fpa"][0][dom] * Ha).clamp(min=1)).ceil().long().clamp(0, 4)
    ll = torch.log2((o["fpl"][0][dom] * (Wl - 1)).clamp(min=1)).ceil().long().clamp(0, 4)
    lv = torch.minimum(ll, torch.full_like(ll, 2))
    if disc:
        wd = cap_weight(o["rho"][0][dom], o["cap"][0][dom])
        cap, rho = o["cap"][0][dom].clamp(min=0), o["rho"][0][dom]
    else:
        wd = torch.zeros_like(tu)
    rim = dd <= a.band
    inner = ~rim
    if a.sam:
        inner = inner & ~sb

    def mark_tube(sel, count=False):
        out = torch.zeros(Kt, Ha, Wl, device=dev)
        for i_ in range(5):
            for j_ in range(5):
                s_ = sel & (la == i_) & (ll == j_) & (wd < 0.75)
                if not bool(s_.any()):
                    continue
                Hq, Wq = Ha >> i_, Wl >> j_
                gi = (ob[s_] * Hq + (tu[s_] * Hq).long() % Hq) * Wq + (tv[s_].clamp(0, 1) * (Wq - 1)).round().long()
                cell = torch.zeros(Kt * Hq * Wq, device=dev)
                cell = cell.index_add_(0, gi, torch.ones_like(gi, dtype=torch.float32)) if count else cell.index_fill_(0, gi, 1.0)
                up = F.interpolate(cell.reshape(Kt, 1, Hq, Wq), size=(Ha, Wl), mode="nearest")[:, 0]
                out = out + up if count else torch.maximum(out, up)
        return out

    def mark_disc(sel, D, count=False):
        out = torch.zeros(Kt * 2, D, D, device=dev)
        xd = 0.5 + 0.5 * rho * torch.cos(2 * math.pi * tu)
        yd = 0.5 + 0.5 * rho * torch.sin(2 * math.pi * tu)
        ci = ob * 2 + cap
        for l_ in range(3):
            s_ = sel & (lv == l_) & (wd > 0.25)
            Dq = max(D >> l_, 1)
            gi = (ci[s_] * Dq + (yd[s_] * Dq).long().clamp(0, Dq - 1)) * Dq + (xd[s_] * Dq).long().clamp(0, Dq - 1)
            cell = torch.zeros(Kt * 2 * Dq * Dq, device=dev)
            cell = cell.index_add_(0, gi, torch.ones_like(gi, dtype=torch.float32)) if count else cell.index_fill_(0, gi, 1.0)
            up = F.interpolate(cell.reshape(Kt * 2, 1, Dq, Dq), size=(D, D), mode="nearest")[:, 0]
            out = out + up if count else torch.maximum(out, up)
        return out.reshape(Kt, 2, D, D)

    with torch.no_grad():
        tin, trim = mark_tube(inner), mark_tube(rim)
        tin = F.max_pool2d(tin[:, None], (1, 3), stride=1, padding=(0, 1))[:, 0]  # close the gaps between samples along
        U = (trim > 0) & (tin == 0)
        T = tin > 0
        if a.sam:
            nb, ng = mark_tube(sb & ~rim, count=True), mark_tube(inner, count=True)
            Us = nb > ng
            U, T = U | Us, T & ~Us
        rep = dict(args=vars(a), seen=float((T | U).float().mean()), trusted=float(T.float().mean()), untrusted=float(U.float().mean()),
                   sam_px=float(sam_bad.sum() / max((own_np >= 0).sum(), 1)), per_body={})
        # the paint: prior and residual folded, as the renderer adds them
        uu = (torch.arange(Ha, device=dev).float() / Ha)[:, None].expand(Ha, Wl)
        vv = (torch.arange(Wl, device=dev).float() / (Wl - 1))[None].expand(Ha, Wl)
        before = {}
        figk = [int(k) for k in a.fig.split(",") if k]
        skipped = []
        for k in range(Kt):
            res = tex_sample(tb.tex, torch.full_like(uu, k, dtype=torch.long), uu, vv).permute(2, 0, 1)  # (3, Ha, Wl)
            x = torch.sigmoid(tb.tex0[k] + res)
            if k in figk:
                before[k] = x.clone()
            nU, nT = int(U[k].sum()), int(T[k].sum())
            rep["per_body"][k] = dict(untrusted=nU, trusted=nT)
            if nU == 0:
                continue
            if nT < 200:
                skipped.append(k)
                continue
            regs = [tb.bump[k][None]] if (a.relief and hasattr(tb, "bump")) else []
            xn, regs = refill(x, U[k], T[k], True, seed=k, regs=regs)
            tb.tex0[k] = torch.where(U[k][None], torch.logit(xn) - res, tb.tex0[k])  # (a trusted texel's sigmoid can round to 1)
            if regs:
                tb.bump[k] = regs[0][0]
        rep["skipped_tube"] = skipped
        if disc and hasattr(tb, "capt"):
            D = tb.capt.shape[-1]
            din, drim = mark_disc(inner, D), mark_disc(rim, D)
            Ud, Td = (drim > 0) & (din == 0), din > 0
            if a.sam:
                nb, ng = mark_disc(sb & ~rim, D, count=True), mark_disc(inner, D, count=True)
                Ud, Td = Ud | (nb > ng), Td & ~(nb > ng)
            yy, xx = torch.meshgrid((torch.arange(D, device=dev) + 0.5) / D * 2 - 1, (torch.arange(D, device=dev) + 0.5) / D * 2 - 1, indexing="ij")
            inside = (yy ** 2 + xx ** 2) <= 1.0
            Ud, Td = Ud & inside, Td & inside
            dsk = 0
            for k in range(Kt):
                for c in range(2):
                    if int(Ud[k, c].sum()) == 0:
                        continue
                    if int(Td[k, c].sum()) < 30:  # nothing to continue from on the disc: the rim's own tube colour near that end
                        dsk += 1
                        continue
                    x = torch.sigmoid(tb.capt[k, c])
                    regs = [tb.capb[k, c][None]] if (a.relief and hasattr(tb, "capb")) else []
                    xn, regs = refill(x, Ud[k, c], Td[k, c], False, seed=1000 + 2 * k + c, regs=regs)
                    tb.capt[k, c] = torch.where(Ud[k, c][None], torch.logit(xn), tb.capt[k, c])
                    if regs:
                        tb.capb[k, c] = regs[0][0]
            rep["disc"] = dict(untrusted=float(Ud.float().sum() / inside.sum() / Kt / 2), trusted=float(Td.float().sum() / inside.sum() / Kt / 2), skipped=dsk)
        # the unwraps: before | trust (green trusted, red untrusted, dim unseen) | after
        rows = []
        for k in figk:
            res = tex_sample(tb.tex, torch.full_like(uu, k, dtype=torch.long), uu, vv).permute(2, 0, 1)
            after = torch.sigmoid(tb.tex0[k] + res)
            tr = before[k] * 0.25
            tr = torch.where(T[k][None], 0.6 * before[k] + 0.4 * torch.tensor([0.0, 1.0, 0.0], device=dev)[:, None, None], tr)
            tr = torch.where(U[k][None], 0.5 * before[k] + 0.5 * torch.tensor([1.0, 0.0, 0.0], device=dev)[:, None, None], tr)
            rows.append(torch.cat([F.pad(before[k], (0, 6)), F.pad(tr, (0, 6)), after], 2))
        if rows:
            m = torch.cat([F.pad(r_, (0, 0, 0, 6), value=1.0) for r_ in rows], 1)
            Image.fromarray((m.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(f"{a.out}/masks.png")
    o2, _ = rest_view(tb, ta, rig, dev, Z)
    img1 = o2["img"][0].cpu()
    del o, o2
    ck2 = dict(ck)
    ck2["tubes"] = {**ck["tubes"], **{k: v.detach().cpu() for k, v in tb.state_dict().items()}}
    ck2["refill"] = {k: v for k, v in rep.items() if k != "per_body"}
    torch.save(ck2, f"{a.out}/tube.pt")
    # crops of the figure bodies at rest, lit, before and after (still px x2)
    tiles = []
    for k in figk:
        ys, xs = np.nonzero(own_np == k)
        if len(ys) == 0:
            continue
        cy, cx = int(np.median(ys)), int(np.median(xs))
        h = 110
        iy, ix = np.arange(cy - h, cy + h) % Z, np.arange(cx - h, cx + h) % Z
        tiles.append(torch.cat([img0[:, iy][:, :, ix], torch.ones(3, 2 * h, 4), img1[:, iy][:, :, ix]], 2))
    if tiles:
        g = torch.cat([F.pad(t_, (0, 0, 0, 16), value=0.0) for t_ in tiles], 1)
        im = Image.fromarray((g.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        dr = ImageDraw.Draw(im)
        for i, k in enumerate([k for k in figk if (own_np == k).any()]):
            dr.text((4, i * (220 + 16) + 221), f"body {k}: rest, lit, 2x: before | after", fill=(255, 255, 255))
        im.save(f"{a.out}/crops.png")
    json.dump(rep, open(f"{a.out}/refill.json", "w"), indent=1)
    print(f"seen at rest {rep['seen']:.1%} of the paint texels: trusted {rep['trusted']:.1%}, refilled {rep['untrusted']:.1%}; "
          f"rest px under another body's segment {rep['sam_px']:.1%}; bodies skipped (too little trusted skin): {skipped}"
          + (f"; discs refilled {rep['disc']['untrusted']:.1%} of their texels (skipped {rep['disc']['skipped']})" if "disc" in rep else ""), flush=True)


if __name__ == "__main__":
    main()
