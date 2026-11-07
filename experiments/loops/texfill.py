"""Fill the parts of objtube.py's textures that no camera ever sees.

The textures train only where the still (and the tilted training views) show them: the undersides
and hidden flanks keep whatever they started with. Turning the paint about a body (objphys.py
--tex-turn) brings those parts into view as smears. This renders the rest pose from a few views,
records how much of each texel is seen, and fills every ring around the body (one texel column
along it) outside its seen arc with that arc mirrored back and forth, so a turn shows more of the
same paint. Seen texels are kept; a soft edge blends the two.

The albedo (prior + upsampled residual) becomes the new prior and the residual restarts at zero,
as texupres.py does. Also writes vis.png and atlas_before/after.png (a few bodies' textures, flat).

usage: texfill.py --load tube.pt --out tube_fill.pt [--tilts 25 --az 6 --zoom 2]
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from objtube import Mixed, Rig, Cam, build_pairs, render, rot_x, rot_z
from striate.util import pick_device


def albedo(st):
    tex0, tex = st["tex0"].float(), st["tex"].float()
    K, _, Ha, Wl = tex0.shape
    k = Ha // tex.shape[2]
    p = torch.cat([tex[:, :, -1:], tex, tex[:, :, :1]], 2)  # wrap around, as tex_sample does
    up = F.interpolate(p, size=(Ha + 2 * k, Wl), mode="bilinear", align_corners=False)
    return torch.sigmoid(tex0 + up[:, :, k:k + Ha])


def refl(x, L):
    """Index into an arc of length L, mirrored back and forth past both ends."""
    x = x % (2 * L)
    return np.where(x >= L, 2 * L - 1 - x, x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tilts", default="25", help="extra views this many degrees off vertical (comma list; 0 is always included)")
    ap.add_argument("--az", type=int, default=6, help="azimuths per tilt")
    ap.add_argument("--zoom", type=float, default=2.0, help="render the views at this zoom (more texels hit)")
    ap.add_argument("--blur", type=int, default=2, help="visibility box blur radius, texels")
    ap.add_argument("--thresh", type=float, default=0.2, help="seen: blurred visibility above this fraction of the body's 80th percentile")
    ap.add_argument("--mode", default="arc", choices=["arc", "ring"], help="arc: one smoothed seen arc per body; ring: each ring's own longest arc")
    ap.add_argument("--smooth", type=float, default=12.0, help="arc: smoothing of its centre and width along the body, texels")
    ap.add_argument("--min-half", type=float, default=16.0, help="arc: least half width, texels (a narrower arc mirrors into stripes)")
    a = ap.parse_args()
    dev = pick_device()
    ck = torch.load(a.load, map_location="cpu", weights_only=False)
    ta = argparse.Namespace(**ck["args"])
    tb = Mixed.from_state(ck["tubes"], ta, dev)
    rig = Rig(0.0, 0.8, ta, dev)
    rig.load_state_dict(ck["rig"])
    S = 512  # objphys.py canvas
    Z = int(round(S * a.zoom))
    alb = albedo(ck["tubes"])
    K, _, Ha, Wl = alb.shape
    vis = torch.zeros(K * Ha * Wl)
    views = [np.eye(3)] + [rot_x(math.radians(float(t))) @ rot_z(2 * math.pi * j / a.az) for t in a.tilts.split(",") if float(t) > 0 for j in range(a.az)]
    with torch.no_grad():
        sp = tb.spheres()
        for R in views:
            cam = Cam(R, Z, Z, (S / 2, S / 2, 0.0), S, zoom=a.zoom)
            pr = build_pairs(sp, cam, 4.0)
            o = render(tb, sp, cam, pr, [rig()], ta, aux=True)
            w = (o["ep"][0] / o["den"][0][o["pix"]]).cpu()
            x = (o["tu"][0].cpu() * Ha).long() % Ha
            y = (o["tv"][0].cpu().clamp(0, 1) * (Wl - 1)).round().long()
            vis.index_add_(0, (o["obj"].cpu() * Ha + x) * Wl + y, w)
    vis = vis.reshape(K, 1, Ha, Wl)
    b = a.blur
    vb = F.avg_pool2d(F.pad(torch.cat([vis[:, :, -b:], vis, vis[:, :, :b]], 2), (b, b, 0, 0), mode="replicate"), 2 * b + 1, stride=1)
    ref = torch.quantile(vb.reshape(K, -1), 0.8, dim=1).clamp(min=1e-6)[:, None, None, None]
    keep = (vb / (a.thresh * ref)).clamp(0, 1)  # 1 seen, 0 never seen, soft between
    seen = (keep > 0.5)[:, 0].numpy()
    A = alb.numpy()
    fill = A.copy()
    if a.mode == "arc":  # one seen arc per body, its centre and width smoothed along it: a coherent mirror
        ang = 2 * np.pi * np.arange(Ha) / Ha
        g = np.exp(-0.5 * (np.arange(-3 * a.smooth, 3 * a.smooth + 1) / a.smooth) ** 2)
        sm = lambda v: np.convolve(np.pad(v, len(g) // 2, mode="edge"), g / g.sum(), mode="valid")
        for k in range(K):
            m = seen[k].astype(np.float64)  # (Ha, Wl)
            n = m.sum(0)
            cs, sn = sm((m * np.cos(ang)[:, None]).sum(0)), sm((m * np.sin(ang)[:, None]).sum(0))
            c = np.arctan2(sn, cs) / (2 * np.pi) * Ha  # arc centre, texels
            h = np.maximum(sm(n) / 2, a.min_half)  # half width
            if n.sum() == 0:
                continue
            d = (np.arange(Ha)[:, None] - c[None] + Ha / 2) % Ha - Ha / 2  # offset from the centre
            q = (d + h[None]) % (4 * h[None])
            src = np.where(q > 2 * h[None], 4 * h[None] - q, q) - h[None]  # mirrored back and forth into [-h, h]
            src = np.where(np.abs(d) <= h[None], d, src) + c[None]
            x0 = np.floor(src)
            fx = src - x0
            i0, i1 = x0.astype(int) % Ha, (x0.astype(int) + 1) % Ha
            yy = np.broadcast_to(np.arange(Wl)[None], (Ha, Wl))
            fill[k] = (1 - fx) * A[k][:, i0, yy] + fx * A[k][:, i1, yy]
    for k in range(K if a.mode == "ring" else 0):
        for yv in range(Wl):
            m = seen[k, :, yv]
            if m.all() or not m.any():
                continue
            # the longest seen arc around the ring (it may wrap)
            mm = np.concatenate([m, m])
            best, s0, run = 0, 0, 0
            for i in range(2 * Ha):
                run = run + 1 if mm[i] else 0
                if run > best:
                    best, s0 = run, i - run + 1
            L = min(best, Ha)
            s0 %= Ha
            arc = (s0 + np.arange(L)) % Ha
            idx = np.nonzero(~m)[0]
            d_end = (idx - (s0 + L - 1)) % Ha  # steps past the arc's end
            d_start = (s0 - idx) % Ha  # steps before its start
            pos = np.where(d_end <= d_start, L - 1 + d_end, -d_start)
            fill[k, :, idx, yv] = A[k, :, arc[refl(pos, L)], yv]
    keep_ = keep.numpy()
    new = keep_ * A + (1 - keep_) * fill
    hr = torch.from_numpy(new).clamp(1 / 255, 1 - 1 / 255)
    st = dict(ck["tubes"])
    st["tex0"] = torch.log(hr / (1 - hr)).float()
    st["tex"] = torch.zeros_like(st["tex"])
    ck = dict(ck)
    ck["tubes"] = st
    ck["texfill"] = vars(a)
    torch.save(ck, a.out)
    print(f"{K} bodies, {len(views)} views: seen {float((keep > 0.5).float().mean()):.1%} of texels; filled the rest", flush=True)
    d = os.path.dirname(os.path.abspath(a.out))
    pick = list(range(0, K, max(1, K // 8)))[:8]
    def atlas(X):  # around down, along across; bodies side by side
        t = np.concatenate([np.pad(X[k].transpose(1, 2, 0), ((0, 0), (0, 4), (0, 0)), constant_values=1) for k in pick], 1)
        return Image.fromarray((t * 255).round().clip(0, 255).astype(np.uint8))
    atlas(A).save(f"{d}/atlas_before.png")
    atlas(new).save(f"{d}/atlas_after.png")
    atlas(np.repeat(keep_, 3, 1)).save(f"{d}/vis.png")
    np.save(f"{d}/keep.npy", keep_[:, 0].astype(np.float16))  # (K, around, along): 1 seen, 0 never


if __name__ == "__main__":
    main()
