"""Filler bodies (caulk): small shells tucked under the pile, seen only where motion opens a gap.

objtube.py's bodies tile the frame at rest (visible ground 0.05%), but once they breathe, wobble or
push each other (objphys.py) the dark ground shows between them. Here extra bodies of the same
families are added beneath the pile so that a gap reveals more shell, not grout:

  gaps      the loaded scene is posed --map-poses times (every body jittered and turned as in
            training, and in half the poses each body's radii shrunk by up to --breathe), and the
            visible ground of each frame is averaged: a map of where the pile tends to open.
  seeds     greedy on the blurred map: take its peak, cut the connected part of the map near it,
            run objtube's seed_body on that patch (a principal curve along the crevice), give it a
            radius drawn from the existing bodies' size distribution (--size-q quantiles) and a
            length clamped to their aspect ratios; an elongated patch becomes a tube laid along the
            crevice, a round one a coil copied from the nearest coil body (turned at random). The
            pixels it covers leave the map; repeat until --cover of the map's mass is taken.
  hidden    the visible height of the pile at rest (closed-form ray hits, top view) bounds each
            filler: every one of its spheres must stay --zmargin px under the surface over its
            footprint (its own displacement included), so the filler sinks as a whole to the
            lowest such level (below the ground plane if need be: the ground is a backdrop, not an
            occluder, in the top view). Fillers still seen at rest are sunk further.
  texture   copied from a neighbour that shows near the gap (wrapped-still prior + residual + relief),
            turned around the body and maybe reversed along it, with low-frequency noise: a new seed of
            the same material. Then only the fillers' textures (and relief) are trained on the channel
            objective in perturbed poses (rig light + a random light; motion widened by --train-amp so
            more of each filler is seen), every other parameter frozen (gradients masked to fillers).
  scores    the rest pose (2x2 supersampled, rig light, objtube's 4 fixed crops) and held-out poses
            (jitter-only, breathing, and with --traj frames of an objphys.py loop, fillers at rest),
            with and without fillers: channel score, visible ground, the fillers' share of the frame.

The saved tube.pt is objtube.py's format (--load with --body mixed) with K + F bodies, the fillers
last, and a boolean buffer is_filler.

usage: objfill.py --load m1b/tube.pt --out DIR [--traj phys/traj.pt] [--init DIR/tube_init.pt]
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, math, random, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from striate.model import Backbone
from striate.util import pick_device
from zoo import MultiScaleCrop
from ring import Ring
from objsurf import Light
from objtube import Mixed, Cam, Rig, build_pairs, render, render_ss, body_hits, seed_body, cr_matrix, lin_matrix

FILL_RGB = (0.95, 0.5, 0.2)  # clay tint of the fillers


# ---------------------------------------------------------------- scene

def load_scene(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    tb = Mixed.from_state(ck["tubes"], a, dev)
    isf = ck["tubes"].get("is_filler", torch.zeros(tb.K, dtype=torch.bool))
    tb.register_buffer("is_filler", isf.to(dev))
    az, el = map(float, a.train_light.split(","))
    rig = Rig(math.radians(az), math.radians(el), a, dev)
    rig.load_state_dict(ck["rig"])
    for p in list(tb.parameters()) + list(rig.parameters()):
        p.requires_grad_(False)
    return ck, a, tb, rig


def make_pose(K, F_, seed, o, breathe=True, B=1):
    """Per-body jitter (B, K+F, 2) px, yaw (B, K+F), radius scale (K+F,). The K originals draw from
    `seed` alone, so a scene with and without fillers is posed identically; fillers move less
    (--fill-jitter, --fill-yaw) and keep their size."""
    g = torch.Generator().manual_seed(seed)
    jit = (torch.randn(B, K, 2, generator=g) * o.jitter).clamp(-2.2 * o.jitter, 2.2 * o.jitter)
    yaw = torch.randn(B, K, generator=g) * o.yaw
    rs = 1 - torch.rand(K, generator=g) * o.breathe if breathe else torch.ones(K)
    gf = torch.Generator().manual_seed(seed + 7919)
    jf = (torch.randn(B, F_, 2, generator=gf) * o.fill_jitter).clamp(-2.2 * o.fill_jitter, 2.2 * o.fill_jitter)
    yf = torch.randn(B, F_, generator=gf) * o.fill_yaw
    return torch.cat([jit, jf], 1), torch.cat([yaw, yf], 1), torch.cat([rs, torch.ones(F_)])


def posed(tb, pose, dev):
    """Spheres in a pose: None (rest), (jit, yaw, rs) from make_pose, or a physics frame dict(C, r) of
    the K originals (objphys.py traj.pt), the fillers left at rest."""
    if pose is None:
        return tb.spheres()
    if isinstance(pose, dict):
        K0 = pose["r"].shape[0]
        C0 = tb.spheres()["C"][0]
        wob = torch.zeros_like(C0)
        wob[:K0] = pose["C"].to(dev) - C0[:K0]
        sph = dict(tb.spheres(wob=wob))
        r = sph["r"].clone()
        r[:K0] = pose["r"].to(dev)
        sph["r"] = r
        return sph
    jit, yaw, rs = pose
    K = tb.K
    sph = dict(tb.spheres(jit=jit[:, :K].to(dev), yaw=yaw[:, :K].to(dev)))
    sph["r"] = sph["r"] * rs[:K].to(dev)[:, None]
    return sph


@torch.no_grad()
def surface(tb, sph, cam, pairs, a):
    """Top view: the visible body height (w = z) per pixel where some body covers it (else -1e4),
    and the body that shows there (-1 where none does)."""
    dev = tb.dev
    H, W = cam.H, cam.W
    R = torch.tensor(cam.R, dtype=torch.float32, device=dev)
    ctr = torch.tensor(cam.centre, dtype=torch.float32, device=dev)
    pix = torch.from_numpy(pairs["pix"]).to(dev)
    obj = torch.from_numpy(pairs["obj"]).to(dev)
    shf = F.pad(torch.from_numpy(pairs["shift"]).to(dev), (0, 1))
    u = (pix % W).float() - W / 2
    v = torch.div(pix, W, rounding_mode="floor").float() - H / 2
    X0 = ctr + u[:, None] * R[0] + v[:, None] * R[1] - shf
    Nv = torch.cross(sph["T"], sph["Bv"], dim=-1)
    args = (sph["C"], sph["r"], sph["T"], sph["Bv"], Nv, sph["f"], sph["s"], R[2], a.sig, a.tau_s, sph.get("disp"), sph["s_lo"], sph["s_len"])
    hb = torch.cat([body_hits(X0[i:i + a.chunk], obj[i:i + a.chunk], *args) for i in range(0, len(pix), a.chunk)], 1)[0]
    wk, ck = hb[:, 0], hb[:, 1]
    hgt = torch.full((H * W,), -1e4, device=dev).scatter_reduce(0, pix, torch.where(ck > 0.5, wk, torch.full_like(wk, -1e4)), "amax")
    key = wk + a.tau_z * torch.log(ck + 1e-6)
    zmax = torch.full((H * W,), -1e9, device=dev).scatter_reduce(0, pix, key, "amax")
    top = (key >= zmax[pix]) & (ck > 0.5)
    owner = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    owner[pix[top]] = obj[top]
    return hgt.reshape(H, W).cpu().numpy(), owner.reshape(H, W).cpu().numpy()


def shares(o, isf, HW):
    """Per pixel, the fillers' share of what shows (B, HW)."""
    ep, den, pix, obj = o["ep"], o["den"], o["pix"], o["obj"]
    w = ep * isf[obj].float()[None]
    return torch.zeros(ep.shape[0], HW, device=ep.device).index_add_(1, pix, w) / den


# ---------------------------------------------------------------- seeding (numpy)

def body_stats(tb):
    """Mean sphere radius per body, spine length, and the median tube radius profile (N,)."""
    with torch.no_grad():
        sph = tb.spheres()
        r = sph["r"].cpu().numpy()
        L = sph["s"][:, -1].cpu().numpy()
    rm = r.mean(1)
    tube = ~tb.is_coil.cpu().numpy()
    prof = np.median(r[tube] / rm[tube, None], 0)
    return dict(rmean=rm, length=L, aspect=L[tube] / rm[tube], prof=prof, C=sph["C"][0].cpu().numpy())


def torus_d2(yy, xx, py, px, S):
    dy = (yy - py + S / 2) % S - S / 2
    dx = (xx - px + S / 2) % S - S / 2
    return dy * dy + dx * dx


def seed_fillers(G, owner, st, is_coil, o, S, rng, log):
    """Greedy filler placement on the gap map G (S, S). Returns a list of specs."""
    Gs = ndi.gaussian_filter(G, o.map_blur, mode="wrap")
    rem = Gs.copy()
    tot = float(Gs.sum())
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
    rq = np.quantile(st["rmean"], [o.size_q[0], o.size_q[1]])
    asp_lo, asp_hi = np.quantile(st["aspect"], [0.25, 0.6])
    coil_ids = np.nonzero(is_coil)[0]
    cen = st["C"][:, :, :2].mean(1)
    specs = []
    while len(specs) < o.max_fill and rem.sum() > (1 - o.cover) * tot:
        p = int(rem.argmax())
        py, px = divmod(p, S)
        if rem[py, px] < o.min_peak:
            break
        rf = float(rng.uniform(*rq))
        reach = 0.5 * asp_hi * rf + rf
        # the crevice near the peak, in a frame rolled so the peak sits at the centre (no wrap there)
        sh = (S // 2 - py, S // 2 - px)
        loc = np.roll(rem, sh, (0, 1))
        cy = cx = S // 2
        near = (yy - cy) ** 2 + (xx - cx) ** 2 < reach**2
        patch = near & (loc > o.patch_rel * loc[cy, cx])
        lab_, _ = ndi.label(patch)
        patch = lab_ == lab_[cy, cx]
        ctl2, _ = seed_body(patch, S, o.nctl, o.nrad)
        ctl2 = ctl2 - np.array([sh[1], sh[0]])  # back to canvas coordinates (x, y)
        pts = cr_matrix(o.nctl, 64) @ ctl2
        ell = float(np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum())
        kind = "tube" if ell >= o.coil_below * rf else "coil"
        if kind == "tube":
            want = float(np.clip(ell, asp_lo * rf, asp_hi * rf))
            mu = ctl2.mean(0)
            ctl2 = mu + (ctl2 - mu) * want / max(ell, 1e-3)
            pts = cr_matrix(o.nctl, 64) @ ctl2
            d2 = np.min([torus_d2(yy, xx, q[1], q[0], S) for q in pts[::4]], 0)
            foot = d2 < (0.9 * rf) ** 2
        else:
            foot = torus_d2(yy, xx, py, px, S) < (1.4 * rf) ** 2
        # donors: the bodies that show around the peak at rest (texture), the nearest coil (shape)
        ring_ = torus_d2(yy, xx, py, px, S) < (rf + 4) ** 2
        ids, cnt = np.unique(owner[ring_ & (owner >= 0)], return_counts=True)
        if len(ids) == 0:
            ids, cnt = np.array([int(np.argmin(((cen - [px, py] + S / 2) % S - S / 2) ** 2 @ [1, 1]))]), np.array([1])
        top = np.argsort(-cnt)[:2]
        donor = int(rng.choice(ids[top], p=cnt[top] / cnt[top].sum()))
        dc = ((cen[coil_ids] - [px, py] + S / 2) % S - S / 2)
        cdonor = int(coil_ids[np.argmin((dc**2).sum(1))])
        specs.append(dict(kind=kind, p=(float(px), float(py)), ctl2=ctl2.tolist(), rf=rf, donor=donor, cdonor=cdonor,
                          roll=int(rng.integers(0, 8)), flip=bool(rng.random() < 0.5), turn=float(rng.uniform(0, 2 * math.pi)),
                          seed=int(rng.integers(0, 2**31)), mass=float((rem * foot).sum() / tot)))
        rem[foot] = 0
    log(f"seeded {len(specs)} fillers ({sum(s['kind'] == 'tube' for s in specs)} tubes, {sum(s['kind'] == 'coil' for s in specs)} coils); "
        f"radius range {rq[0]:.1f}-{rq[1]:.1f} px; map mass taken {1 - rem.sum() / tot:.2f}")
    return specs


def extend_state(st, specs, stats, o):
    """The Mixed state dict with one more body per spec: every per-body tensor grows by F rows."""
    K = st["ctl"].shape[0]
    keys = [k for k, v in st.items() if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == K and k != "ground"]
    rows = {k: [] for k in keys}
    is_coil = st["is_coil"].numpy()
    tubes = np.nonzero(~is_coil)[0]
    med = lambda k: st[k][torch.from_numpy(tubes)].median(0).values
    prof = np.interp(np.linspace(0, 1, o.nrad), np.linspace(0, 1, len(stats["prof"])), stats["prof"])
    for sp in specs:
        d, cd = sp["donor"], sp["cdonor"]
        g = torch.Generator().manual_seed(sp["seed"])
        r_ = {k: st[k][d].clone() for k in keys}
        # texture: the donor's, turned around the body by roll/8 of a turn, maybe reversed along it, plus
        # low-frequency noise in the residual
        for k in ("tex0", "tex", "disp"):
            if k not in r_:
                continue
            t = r_[k]
            sft = sp["roll"] * t.shape[-2] // 8
            t = torch.roll(t, sft, -2)
            if sp["flip"]:
                t = torch.flip(t, (-1,))
            r_[k] = t
        nz = torch.randn(1, 3, 6, 8, generator=g) * o.tex_noise
        r_["tex"] = r_["tex"] + F.interpolate(nz, size=r_["tex"].shape[-2:], mode="bicubic", align_corners=False)[0]
        if sp["kind"] == "tube":
            r_["is_coil"] = torch.tensor(False)
            ctl = torch.zeros(o.nctl, 3)
            ctl[:, :2] = torch.tensor(sp["ctl2"], dtype=torch.float32)
            r_["ctl"] = ctl
            r_["logr"] = torch.log(torch.tensor(sp["rf"] * prof, dtype=torch.float32))
            r_["logr0"] = r_["logr"].clone()
            if is_coil[d]:  # squash and ripple from the tube family when the texture donor is a coil
                for k in ("fq", "rip", "ripf", "ripp"):
                    r_[k] = med(k)
        else:
            r_["is_coil"] = torch.tensor(True)
            for k in keys:
                if k.startswith("coil."):
                    r_[k] = st[k][cd].clone()
            for k in ("fq", "rip", "ripf", "ripp"):
                r_[k] = st[k][cd].clone()
            r_["coil.x"] = torch.tensor(sp["p"][0])
            r_["coil.y"] = torch.tensor(sp["p"][1])
            r_["coil.beta"] = r_["coil.beta"] + sp["turn"]
            r_["coil.lrho"] = r_["coil.lrho"] + math.log(sp["rf"] / stats["rmean"][cd])
            r_["coil.lrho0"] = r_["coil.lrho"].clone()
            r_["coil.lift"] = torch.tensor(0.0)
        if "capc" in r_:
            r_["capc"] = torch.tensor(sp["p"])
            r_["capr"] = torch.tensor(3.0 * sp["rf"])
        for k in keys:
            rows[k].append(r_[k].to(st[k].dtype))
    new = dict(st)
    for k in keys:
        new[k] = torch.cat([st[k], torch.stack(rows[k])], 0)
    new["is_filler"] = torch.cat([st.get("is_filler", torch.zeros(K, dtype=torch.bool)), torch.ones(len(specs), dtype=torch.bool)])
    return new


@torch.no_grad()
def sink(tb, fidx, hgt, o):
    """Lower each filler as a whole until every sphere sits --zmargin px under the pile's surface over
    its footprint (hgt: visible height of the other bodies at rest, -1e4 = uncovered, ignored)."""
    S = hgt.shape[0]
    sph = tb.spheres()
    C = sph["C"][0].cpu().numpy()
    r = sph["r"].cpu().numpy()
    f = sph["f"].cpu().numpy()
    da = getattr(tb.a, "disp_amp", 0.0)
    dz = np.zeros(len(fidx))
    for i, k in enumerate(fidx):
        need = np.inf
        for j in range(C.shape[1]):
            x, y, z = C[k, j]
            R = r[k, j] + da
            n = int(math.ceil(R)) + 1
            ys = np.arange(int(math.floor(y)) - n, int(math.floor(y)) + n + 2)
            xs = np.arange(int(math.floor(x)) - n, int(math.floor(x)) + n + 2)
            d2 = (ys[:, None] - y) ** 2 + (xs[None] - x) ** 2
            h = hgt[np.ix_(ys % S, xs % S)]
            ok = (d2 < R * R) & (h > -1e3)
            if ok.any():
                topz = f[k] * np.sqrt(np.clip(R * R - d2, 0, None))
                need = min(need, float((h - o.zmargin - topz)[ok].min()) - z)
        dz[i] = need if np.isfinite(need) else 0.0
    if not getattr(o, "raise_fill", False):
        dz = np.minimum(dz, 0.0)
    # with --raise-fill a filler also rises to that level, so one under a hole sits just below the
    # hole's rim (a shell in the gap) instead of on the floor, where occlusion makes it a dark pit
    for i, k in enumerate(fidx):
        if bool(tb.is_coil[k]):
            tb.coil.lift[k] += float(dz[i])
        else:
            tb.ctl[k, :, 2] += float(dz[i])
    return dz


# ---------------------------------------------------------------- images

def to_img(x):
    return Image.fromarray((x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))


def heat(m, lo=0.0, hi=1.0):
    """(H, W) in [lo, hi] -> RGB image, dark blue -> teal -> yellow -> white."""
    stops = np.array([[0.03, 0.03, 0.08], [0.15, 0.25, 0.55], [0.2, 0.65, 0.55], [0.95, 0.85, 0.25], [1.0, 1.0, 1.0]])
    t = np.clip((m - lo) / max(hi - lo, 1e-9), 0, 1) * (len(stops) - 1)
    k = np.minimum(t.astype(int), len(stops) - 2)
    c = stops[k] + (stops[k + 1] - stops[k]) * (t - k)[..., None]
    return Image.fromarray((c * 255).round().astype(np.uint8))


def side(ims, labels=None, pad=6):
    W = sum(i.width for i in ims) + pad * (len(ims) - 1)
    Hh = max(i.height for i in ims) + (18 if labels else 0)
    out = Image.new("RGB", (W, Hh), (18, 18, 18))
    x = 0
    dr = ImageDraw.Draw(out)
    for i, im in enumerate(ims):
        out.paste(im, (x, 18 if labels else 0))
        if labels:
            dr.text((x + 4, 3), labels[i], fill=(220, 220, 220))
        x += im.width + pad
    return out


@torch.no_grad()
def clay_tinted(tb, sph, cam, pairs, light, a, sub=None):
    """Clay render, the fillers in FILL_RGB: textures swapped for flat per-body albedo."""
    isf = tb.is_filler
    col = torch.where(isf[:, None], torch.tensor(FILL_RGB, device=tb.dev)[None], torch.full((1, 3), 0.72, device=tb.dev))
    lt = torch.log(col / (1 - col))[:, :, None, None].expand(-1, -1, 2, 2).contiguous()
    t0 = tb.tex0
    tb.tex0 = torch.zeros(tb.K, 3, 2, 2, device=tb.dev)
    try:
        img = render(tb, sph, cam, pairs, [light], a, ground="flat", tex=lt if sub is None else lt[sub], sub=sub)
    finally:
        tb.tex0 = t0
    return img[0]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True, help="objtube.py tube.pt (mixed bodies) to add fillers to")
    ap.add_argument("--init", help="a tube.pt with fillers already placed (from this script): skip seeding")
    ap.add_argument("--out", required=True)
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--target", default="inception4c:7")
    ap.add_argument("--seed", type=int, default=0)
    # perturbed poses (the gap map, training and evaluation)
    ap.add_argument("--jitter", type=float, default=3.5, help="px rms per body (as objtube training)")
    ap.add_argument("--yaw", type=float, default=0.08, help="rad rms per body")
    ap.add_argument("--breathe", type=float, default=0.15, help="radii shrink by up to this fraction in a breathing pose")
    ap.add_argument("--fill-jitter", type=float, default=1.0, help="fillers: px rms")
    ap.add_argument("--fill-yaw", type=float, default=0.03, help="fillers: rad rms")
    ap.add_argument("--map-poses", type=int, default=16, help="poses averaged into the gap map (half breathing)")
    ap.add_argument("--eval-poses", type=int, default=8, help="held-out poses for the scores (half breathing)")
    ap.add_argument("--traj", help="objphys.py traj.pt (simulated from --load's bodies): also score its frames, and train on them")
    ap.add_argument("--eval-frames", type=int, default=12, help="physics frames scored, evenly spaced")
    ap.add_argument("--traj-train", type=float, default=0.5, help="fraction of training items drawn from the physics loop")
    # seeding
    ap.add_argument("--map-blur", type=float, default=1.5, help="px, before picking peaks")
    ap.add_argument("--max-fill", type=int, default=60)
    ap.add_argument("--cover", type=float, default=0.85, help="stop when this fraction of the map's mass is under fillers")
    ap.add_argument("--min-peak", type=float, default=0.02, help="stop when the blurred map's peak falls below this")
    ap.add_argument("--size-q", type=float, nargs=2, default=(0.1, 0.4), help="filler radius: uniform between these quantiles of the bodies' mean radii")
    ap.add_argument("--patch-rel", type=float, default=0.15, help="crevice patch: map above this fraction of its peak")
    ap.add_argument("--coil-below", type=float, default=2.0, help="a patch shorter than this many radii becomes a coil")
    ap.add_argument("--tex-noise", type=float, default=0.3, help="low-frequency logit noise added to the copied residual texture")
    ap.add_argument("--zmargin", type=float, default=3.0, help="px a filler stays under the pile's surface at rest")
    ap.add_argument("--raise-fill", action="store_true", help="also raise fillers up to --zmargin under the surface around them (not only lower)")
    ap.add_argument("--sink-rounds", type=int, default=3, help="rounds of sinking fillers still seen at rest")
    ap.add_argument("--sink-px", type=float, default=2.0)
    # training (fillers' textures only)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--mb", type=int, default=2, help="items per step: the rig light, then random lights")
    ap.add_argument("--lr-tex", type=float, default=0.03)
    ap.add_argument("--lr-disp", type=float, default=0.02)
    ap.add_argument("--train-disp", action="store_true", help="also train the fillers' relief (about 3x slower: the ray hits are recomputed in backward)")
    ap.add_argument("--train-amp", type=float, default=1.5, help="training poses: jitter, yaw and breathing scaled by this (wider gaps show more of each filler)")
    ap.add_argument("--skip-init-eval", action="store_true", help="don't score the fillers before training")
    ap.add_argument("--wkeep", type=float, default=0.01, help="pull of the fillers' residual texture toward its copied init")
    ap.add_argument("--rebuild", type=int, default=20, help="steps between pixel-pair rebuilds")
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)

    ck0, ta, base, rig = load_scene(a.load, dev)
    K = base.K
    S = 512
    top = Cam(np.eye(3), S, S, (S / 2, S / 2, 0.0), S)
    a.nctl, a.nrad = ta.nctl, ta.nrad
    layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
    net = Backbone(a.net, dev)
    net.hook(layer)
    rg = Ring(net, layer, ch, S, dev)

    def score(fr):
        state = random.getstate()
        tot = 0
        for k in range(4):
            random.seed(1000 + k)
            tot = tot + rg.act(MultiScaleCrop(224 / S, 512)(fr))
        random.setstate(state)
        return float(tot / 4)

    with torch.no_grad():
        light = rig()
    clayL = Light(math.radians(135), math.radians(50))
    traj = torch.load(a.traj, map_location="cpu", weights_only=False) if a.traj else None
    # evaluation poses: ("jit" | "breathe", seed) synthetic, ("phys", frame) from the physics loop
    eval_poses = [("breathe" if i % 2 else "jit", 500 + i) for i in range(a.eval_poses)]
    if traj is not None:
        assert traj["traj"].shape[1] == K, "the physics loop must be simulated from --load's bodies"
        T_ = traj["traj"].shape[0]
        eval_poses += [("phys", int(t)) for t in np.linspace(0, T_, a.eval_frames, endpoint=False).round()]

    at = argparse.Namespace(**vars(a))  # training poses: the motion scaled by --train-amp
    at.jitter, at.yaw, at.breathe = a.jitter * a.train_amp, a.yaw * a.train_amp, min(0.5, a.breathe * a.train_amp)

    def pose_of(spec, F_, B=1, o=a):
        if spec[0] == "phys":
            return dict(C=traj["traj"][spec[1]], r=traj["rads"][spec[1]])
        return make_pose(K, F_, spec[1], o, breathe=spec[0] == "breathe", B=B)

    @torch.no_grad()
    def evaluate(tb, tag):
        F_ = tb.K - K
        sph = tb.spheres()
        pr = build_pairs(sph, top, 4.0)
        rest = score(render_ss(tb, sph, top, pr, light, ta))
        o = render(tb, sph, top, pr, [light], ta, aux=True, shadows=False, clay=0.5)
        res = dict(rest_score=rest, rest_ground=float(o["gown"].mean()))
        if F_ > 0:
            sf = shares(o, tb.is_filler, S * S)[0]
            res["rest_filler_share"] = float(sf.mean())
            res["rest_filler_px"] = int((sf > 0.5).sum())
        per = []
        for spec in eval_poses:
            sp = posed(tb, pose_of(spec, F_), dev)
            pp = build_pairs(sp, top, 4.0)
            s_ = score(render_ss(tb, sp, top, pp, light, ta))
            o = render(tb, sp, top, pp, [light], ta, aux=True, shadows=False, clay=0.5)
            q = dict(kind=spec[0], id=spec[1], score=s_, ground=float(o["gown"].mean()))
            if F_ > 0:
                q["filler_share"] = float(shares(o, tb.is_filler, S * S)[0].mean())
            per.append(q)
        res["poses"] = per
        msg = []
        for nm in ("jit", "breathe", "phys"):
            qs = [q for q in per if q["kind"] == nm]
            if qs:
                res[f"{nm}_score"] = float(np.mean([q["score"] for q in qs]))
                res[f"{nm}_ground"] = float(np.mean([q["ground"] for q in qs]))
                if F_ > 0:
                    res[f"{nm}_filler_share"] = float(np.mean([q["filler_share"] for q in qs]))
                msg.append(f"{nm} {res[f'{nm}_score']:.3f} ground {res[f'{nm}_ground']:.4f}")
        log(f"{tag}: rest {rest:.3f} ground {res['rest_ground']:.4f}" + (f" filler share {res['rest_filler_share']:.4f} ({res['rest_filler_px']} px)" if F_ > 0 else "")
            + "; " + "; ".join(msg))
        return res

    rep = dict(load=a.load, args=vars(a))
    # ---- the gap map and the pile's surface, from the scene without fillers
    with torch.no_grad():
        sph0 = base.spheres()
        pr0 = build_pairs(sph0, top, 4.0)
        hgt, owner = surface(base, sph0, top, pr0, ta)
        o = render(base, sph0, top, pr0, [light], ta, aux=True, shadows=False, clay=0.5)
        g_rest = o["gown"][0].reshape(S, S).cpu().numpy()
        G = np.zeros((S, S))
        for i in range(a.map_poses):
            sp = posed(base, make_pose(K, 0, 100 + i, a, breathe=i % 2 == 1), dev)
            o = render(base, sp, top, build_pairs(sp, top, 4.0), [light], ta, aux=True, shadows=False, clay=0.5)
            G += o["gown"][0].reshape(S, S).cpu().numpy() / a.map_poses
    log(f"gap map over {a.map_poses} poses: mean visible ground {G.mean():.4f} (rest {g_rest.mean():.4f}); "
        f"surface at rest {np.percentile(hgt[hgt > -1e3], [5, 50, 95]).round(1).tolist()} px")
    np.save(f"{a.out}/gapmap.npy", G.astype(np.float32))

    # ---- fillers
    if a.init:
        ck1, _, tb, _ = load_scene(a.init, dev)
        specs = json.load(open(os.path.join(os.path.dirname(a.init), "fillers.json")))
        log(f"fillers from {a.init}")
        json.dump(specs, open(f"{a.out}/fillers.json", "w"), indent=1)
    else:
        stats = body_stats(base)
        specs = seed_fillers(G, owner, stats, base.is_coil.cpu().numpy(), a, S, rng, log)
        st = extend_state({k: v.cpu() for k, v in base.state_dict().items()}, specs, stats, a)
        tb = Mixed.from_state(st, ta, dev)
        tb.register_buffer("is_filler", st["is_filler"].to(dev))
        for p in tb.parameters():
            p.requires_grad_(False)
        fidx = list(range(K, tb.K))
        dz = sink(tb, fidx, hgt, a)
        log(f"sunk fillers: dz median {np.median(dz):.1f} px, range [{dz.min():.1f}, {dz.max():.1f}]")
        # fillers still seen at rest (outside where the ground showed anyway) go further down
        for rd in range(a.sink_rounds):
            with torch.no_grad():
                sph = tb.spheres()
                o = render(tb, sph, top, build_pairs(sph, top, 4.0), [light], ta, aux=True, shadows=False, clay=0.5)
                sf = shares(o, tb.is_filler, S * S)[0].reshape(S, S).cpu().numpy()
                _, own1 = surface(tb, sph, top, build_pairs(sph, top, 4.0), ta)
            seen = (sf > 0.3) & (g_rest < 0.5)
            bad = [k for k in fidx if ((own1 == k) & seen).sum() > 4]
            log(f"sink round {rd}: {int(seen.sum())} px of fillers seen at rest over the pile; {len(bad)} fillers to lower")
            if not bad:
                break
            with torch.no_grad():
                for k in bad:
                    if bool(tb.is_coil[k]):
                        tb.coil.lift[k] -= a.sink_px
                    else:
                        tb.ctl[k, :, 2] -= a.sink_px
        for i, k in enumerate(fidx):
            with torch.no_grad():
                specs[i]["z_bottom"] = float((tb.base()[0][k, :, 2] - tb.f()[k] * tb.base()[1][k]).min())
        json.dump(specs, open(f"{a.out}/fillers.json", "w"), indent=1)
    F_ = tb.K - K
    fidx = list(range(K, tb.K))
    rep["fillers"] = dict(count=F_, indices=fidx, tubes=sum(s["kind"] == "tube" for s in specs), coils=sum(s["kind"] == "coil" for s in specs),
                          radius_px=[s["rf"] for s in specs], z_bottom=[s.get("z_bottom") for s in specs])

    def save_ck(path):
        torch.save({"tubes": tb.state_dict(), "rig": ck0["rig"], "args": ck0["args"]}, path)

    save_ck(f"{a.out}/tube_init.pt")
    rep["base"] = evaluate(base, "without fillers")
    if not a.skip_init_eval:
        rep["filled_init"] = evaluate(tb, "with fillers, copied textures")

    # ---- train the fillers' textures in perturbed poses
    if a.steps > 0:
        mask = tb.is_filler.float()
        train = [(tb.tex, a.lr_tex)] + ([(tb.disp, a.lr_disp)] if a.train_disp else [])
        for p, _ in train:
            p.requires_grad_(True)
            p.register_hook(lambda g, m=mask: g * m.view(-1, *([1] * (g.dim() - 1))))
        opt = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in train])
        base_lr = [lr for _, lr in train]
        tex_init = tb.tex.detach().clone()
        lrng = random.Random(a.seed + 1)
        curve = []
        margin = 2.5 * at.jitter + 4
        for it in range(a.steps):
            for g_, b_ in zip(opt.param_groups, base_lr):
                g_["lr"] = b_ * 0.5 * (1 + math.cos(math.pi * it / a.steps))
            if it % a.rebuild == 0:
                with torch.no_grad():
                    pairs = build_pairs(tb.spheres(), top, margin)
            # one pose per step, batched over the lights (rig, then random): jitter and yaw differ per
            # item, the radii (breathing) are shared; a physics frame (--traj-train) is one item
            if traj is not None and lrng.random() < a.traj_train:
                pose, lights = pose_of(("phys", lrng.randrange(traj["traj"].shape[0])), F_), [light]
            else:
                pose = pose_of(("breathe" if lrng.random() < 0.5 else "jit", 10000 + it), F_, B=a.mb, o=at)
                lights = [light] + [Light.random(lrng) for _ in range(a.mb - 1)]
            sph = posed(tb, pose, dev)
            img = render(tb, sph, top, pairs, lights, ta)
            act = rg.act(MultiScaleCrop(224 / S, 512)(img))
            keep = ((tb.tex - tex_init) ** 2)[K:].mean()
            opt.zero_grad()
            (-act + act.detach().abs() * a.wkeep * keep).mean().backward()
            opt.step()
            if it % 10 == 0 or it == a.steps - 1:
                curve.append((it, float(act.detach().mean())))
                log(f"step {it + 1}/{a.steps} act {act.detach().mean():.3f} [{', '.join(f'{float(x):.2f}' for x in act.detach())}] keep {float(keep):.3f}")
        for p, _ in train:
            p.requires_grad_(False)
        rep["curve"] = curve
    save_ck(f"{a.out}/tube.pt")
    rep["filled"] = evaluate(tb, "with fillers, trained")

    # ---- images
    with torch.no_grad():
        sph = tb.spheres()
        pr = build_pairs(sph, top, 4.0)
        to_img(render_ss(tb, sph, top, pr, light, ta)[0]).save(f"{a.out}/rest.png")
        sph0 = base.spheres()
        to_img(render_ss(base, sph0, top, build_pairs(sph0, top, 4.0), light, ta)[0]).save(f"{a.out}/rest_base.png")
        pose = pose_of(eval_poses[-1] if traj is not None else eval_poses[1], F_)
        spb, spf = posed(base, pose, dev), posed(tb, pose, dev)
        ppb, ppf = build_pairs(spb, top, 4.0), build_pairs(spf, top, 4.0)
        imb = to_img(render_ss(base, spb, top, ppb, light, ta)[0])
        imf = to_img(render_ss(tb, spf, top, ppf, light, ta)[0])
        pname = "physics frame" if traj is not None else "breathing pose"
        side([imb, imf], [f"{pname}, without fillers", "with fillers"]).save(f"{a.out}/perturbed_compare.png")
        crop = (128, 128, 384, 384)
        side([imb.crop(crop).resize((512, 512), Image.NEAREST), imf.crop(crop).resize((512, 512), Image.NEAREST)],
             ["without (centre, 2x)", "with (centre, 2x)"]).save(f"{a.out}/perturbed_compare_zoom.png")
        c_rest = to_img(clay_tinted(tb, sph, top, pr, clayL, ta))
        c_pert = to_img(clay_tinted(tb, spf, top, ppf, clayL, ta))
        ss = tb.spheres(sub=fidx)
        c_only = to_img(clay_tinted(tb, ss, top, build_pairs(ss, top, 4.0), clayL, ta, sub=fidx))
        side([c_rest, c_pert, c_only], ["clay at rest (fillers orange)", pname, "fillers alone"]).save(f"{a.out}/clay_fillers.png")
        # ground visibility: mean over the evaluation poses, without and with fillers
        Gb, Gf = np.zeros((S, S)), np.zeros((S, S))
        for spec in eval_poses:
            pose = pose_of(spec, F_)
            for m_, acc in ((base, Gb), (tb, Gf)):
                sp = posed(m_, pose, dev)
                o = render(m_, sp, top, build_pairs(sp, top, 4.0), [light], ta, aux=True, shadows=False, clay=0.5)
                acc += o["gown"][0].reshape(S, S).cpu().numpy() / len(eval_poses)
    hi = max(float(np.quantile(G, 0.999)), 1e-3)
    hm = [heat(np.sqrt(m / hi)) for m in (G, Gb, Gf)]
    dr = ImageDraw.Draw(hm[0])
    for s_ in specs:
        x, y = s_["p"]
        dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 80, 40) if s_["kind"] == "tube" else (255, 0, 255), width=2)
    side(hm, ["gap map (seeding poses) + seeds", f"held-out poses, without: {Gb.mean():.4f}", f"with fillers: {Gf.mean():.4f}"]).save(f"{a.out}/ground_vis.png")
    rep["ground_vis_mean"] = dict(gap_map=float(G.mean()), without=float(Gb.mean()), with_fillers=float(Gf.mean()))
    rep["secs"] = round(time.time() - t0)
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
