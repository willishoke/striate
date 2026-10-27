"""Measure a channel's ridge at 256px (handoff steps 1-4).

  landscape.py corpus   --net resnet50_v1 --target layer3:530 --n 64 --out DIR
      N attractors (joint octave renders with the color penalties of ring.py), their robust
      activation, and three descriptors per attractor; a shift-invariance check of each.
  landscape.py coverage --corpus DIR/corpus.pt --frames LOOPDIR --out DIR
      how much of the corpus's descriptor variance a loop's frames span, how close each
      attractor comes to the loop, and the effective dimension of both sets.
  landscape.py width    --corpus DIR/corpus.pt --out DIR
      from several attractors, the step along random spectrum directions (per frequency
      band) at which activation falls to 0.9 of the attractor's.
  landscape.py graph    --corpus DIR/corpus.pt --out DIR
      strings along kNN edges of the corpus; the lowest activation on each string is the
      barrier; single linkage over barriers gives the disconnectivity tree.

PYTHONPATH must include the repo root and this folder (see ../ridge-landscape-handoff.md).
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
from striate.param import FourierImage, seed_spectrum_from_image, _rfft2d_freqs
from striate.util import pick_device
from zoo import Net, MultiScaleCrop
from octave import truncate, Stop
from ring import Ring, ChunkedAdam, bandpass, anchors

# an early layer whose Gram matrix (texture statistics) describes an attractor
DESC_LAYER = {
    "resnet50_v1": "layer1",
    "resnet50_v2": "layer1",
    "googlenet": "inception3a",
    "convnext": "features.1",
}
CROP = 512
BANDS = [
    ("low>32px", 0.0, 1 / 32),
    ("mid8-32px", 1 / 32, 1 / 8),
    ("high<8px", 1 / 8, 10.0),
    ("all", 0.0, 10.0),
]


# ----------------------------------------------------------------------------- shared


def setup(net_name, target, size, dev):
    layer, ch = target.split(":")[0], int(target.split(":")[1])
    net = Net(net_name, dev)
    net.hook(layer)
    desc = DESC_LAYER[net_name]
    if desc != layer:
        net.hook(desc)
    truncate(net, {layer, desc})
    rg = Ring(net, layer, ch, size, dev)
    rg.desc_layer = desc
    return rg


def view(size):
    return MultiScaleCrop(224 / size, CROP)


@torch.no_grad()
def robust_act(rg, specs, mb=8, draws=4):
    """Channel mean over fixed view draws shared by every item (comparable across calls)."""
    tot = 0
    for d in range(draws):
        r = []
        for i in range(0, len(specs), mb):
            random.seed(1000 + d)
            r.append(rg.act(view(rg.size)(rg.image(specs[i : i + mb].to(rg.dev)))).cpu())
        tot = tot + torch.cat(r)
    return tot / draws


@torch.no_grad()
def images(rg, specs, mb=8):
    return torch.cat(
        [rg.image(specs[i : i + mb].to(rg.dev)).cpu() for i in range(0, len(specs), mb)]
    )


@torch.no_grad()
def spectra_of(rg, imgs, mb=16):
    out = []
    for i in range(0, len(imgs), mb):
        p = FourierImage(rg.size, rg.size, batch=len(imgs[i : i + mb]), device=rg.dev)
        seed_spectrum_from_image(p, imgs[i : i + mb].to(rg.dev))
        out.append(p.spectrum.detach())
    return torch.cat(out)


def gram(a):
    """(B, C, h, w) -> unit Gram vectors (B, C*C): texture statistics, shift-invariant up to borders."""
    B, C, h, w = a.shape
    A = a.reshape(B, C, h * w)
    g = (torch.bmm(A, A.transpose(1, 2)) / (h * w)).reshape(B, -1)
    return g / (g.norm(dim=1, keepdim=True) + 1e-6)


def radial_power(img, bins=64):
    """Per color channel, mean log power in radial-frequency bins: exactly shift-invariant."""
    B, C, H, W = img.shape
    p = torch.fft.rfft2(img - img.mean((2, 3), keepdim=True)).abs() ** 2
    k = torch.tensor(_rfft2d_freqs(H, W), dtype=torch.float32, device=img.device)
    idx = (k / 0.7072 * bins).long().clamp(0, bins - 1).flatten()
    cnt = torch.zeros(bins, device=img.device).index_add_(
        0, idx, torch.ones_like(idx, dtype=torch.float32)
    )
    out = torch.zeros(B, C, bins, device=img.device).index_add_(
        2, idx, torch.log(p + 1e-8).flatten(2)
    )
    return (out / cnt).flatten(1)


PAD = 32  # circular padding before the descriptor pass, so the Gram sees the image as periodic


@torch.no_grad()
def gram_periodic(rg, x):
    """Gram of the descriptor layer on the circularly padded image, border of the feature map
    dropped, averaged over the layer's sub-stride sampling phases: exactly shift-invariant
    (a roll by a multiple of the stride leaves it unchanged; the phase average covers the rest)."""
    xp = F.pad(x, [PAD] * 4, mode="circular")
    tot, S, ph = 0, 1, 0
    while ph < S * S:
        rg.net.acts = {}
        try:
            rg.net(torch.roll(xp, shifts=(ph // S, ph % S), dims=(2, 3)))
        except Stop:
            pass
        f = rg.net.acts[rg.desc_layer]
        S = round(xp.shape[-1] / f.shape[-1])
        c = round(PAD / S)
        tot = tot + gram(f[:, :, c:-c, c:-c])
        ph += 1
    return tot / (S * S)


@torch.no_grad()
def describe_images(rg, imgs, mb=8):
    """Three descriptors of each image: Gram of an early layer (on the circularly padded image,
    border of the feature map dropped, so it is shift-invariant), radial log power, 16x16 squint."""
    out = dict(gram=[], radial=[], squint=[])
    for i in range(0, len(imgs), mb):
        x = imgs[i : i + mb].to(rg.dev)
        out["gram"].append(gram_periodic(rg, x).cpu())
        out["radial"].append(radial_power(x).cpu())
        out["squint"].append(F.adaptive_avg_pool2d(x, 16).flatten(1).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def nn_dist(X):
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    return D.min(1).values


def load_frames(d):
    files = sorted(f for f in os.listdir(d) if f.startswith("f") and f.endswith(".png"))
    load = lambda f: (
        torch.from_numpy(np.asarray(Image.open(f"{d}/{f}").convert("RGB"))).permute(2, 0, 1).float()
        / 255
    )
    return torch.stack([load(f) for f in files])


def sheet(imgs, labels, path, cols=8, T=128):
    rows = (len(imgs) + cols - 1) // cols
    out = Image.new("RGB", (cols * T, rows * T), (12, 12, 12))
    for i, im in enumerate(imgs):
        a = Image.fromarray(
            (im.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ).resize((T, T), Image.LANCZOS)
        ImageDraw.Draw(a).text((3, 2), labels[i], fill=(255, 255, 255))
        out.paste(a, ((i % cols) * T, (i // cols) * T))
    out.save(path, quality=90)


def log_factory():
    t0 = time.time()
    return lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)


# ----------------------------------------------------------------------------- step 1


@torch.no_grad()
def shift_check(rg, imgs, desc, specs, m=16, seed=0):
    """Roll m attractors by random offsets: descriptor distance moved vs the corpus's median
    nearest-neighbour distance (should be far below 1 for a shift-invariant descriptor),
    and the activation ratio under the roll."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(imgs), generator=g)[:m]
    dy, dx = (
        torch.randint(0, rg.size, (m,), generator=g),
        torch.randint(0, rg.size, (m,), generator=g),
    )
    rolled = torch.stack(
        [torch.roll(imgs[i], shifts=(int(y), int(x)), dims=(1, 2)) for i, y, x in zip(idx, dy, dx)]
    )
    d2 = describe_images(rg, rolled)
    out = {}
    for k in desc:
        nn = nn_dist(desc[k]).median()
        moved = (d2[k] - desc[k][idx]).norm(dim=1) / nn
        out[k] = dict(median=float(moved.median()), max=float(moved.max()))
    a0 = robust_act(rg, specs[idx.to(specs.device)])
    a1 = robust_act(rg, spectra_of(rg, rolled))
    out["act_roll_ratio"] = dict(mean=float((a1 / a0).mean()), min=float((a1 / a0).min()))
    return out


def cmd_corpus(a):
    dev = pick_device()
    log = log_factory()
    rg = setup(a.net, a.target, a.size, dev)
    sizes = [s for s in (224, 320, 448, 640, 896) if s < a.size] + [a.size]
    specs = []
    for b in range(0, a.n, a.batch):
        k = min(a.batch, a.n - b)
        specs.append(anchors(rg, k, sizes, a.anchor_steps, a.seed + b, dev))
        log(f"rendered {b + k}/{a.n}")
        if dev == "mps":
            torch.mps.empty_cache()
    specs = torch.cat(specs)
    act = robust_act(rg, specs)
    imgs = images(rg, specs)
    desc = describe_images(rg, imgs)
    shift = shift_check(rg, imgs, desc, specs)
    log(
        f"act mean {act.mean():.3f} min {act.min():.3f} max {act.max():.3f}; shift check {json.dumps(shift)}"
    )
    os.makedirs(a.out, exist_ok=True)
    torch.save(
        dict(
            spectra=specs.cpu(),
            act=act,
            desc=desc,
            net=a.net,
            target=a.target,
            size=a.size,
            desc_layer=rg.desc_layer,
            seed=a.seed,
        ),
        f"{a.out}/corpus.pt",
    )
    sat = float(((imgs < 0.02) | (imgs > 0.98)).float().mean())
    sheet(imgs, [f"{i} {float(v):.2f}" for i, v in enumerate(act)], f"{a.out}/corpus_sheet.jpg")
    nn = {k: nn_dist(v) for k, v in desc.items()}
    json.dump(
        dict(
            act=act.tolist(),
            shift=shift,
            clipped=sat,
            nn_median={k: float(v.median()) for k, v in nn.items()},
            nn_cv={k: float(v.std() / v.mean()) for k, v in nn.items()},
        ),
        open(f"{a.out}/corpus.json", "w"),
        indent=1,
    )


# ----------------------------------------------------------------------------- step 2


def participation_ratio(X):
    s2 = torch.linalg.svdvals(X - X.mean(0)) ** 2
    return float(s2.sum() ** 2 / (s2**2).sum())


def coverage_stats(Xc, Xl, ks=(1, 2, 4, 8, 16, 32)):
    Xc0, Xl0 = Xc - Xc.mean(0), Xl - Xl.mean(0)
    _, S, Vh = torch.linalg.svd(Xl0, full_matrices=False)
    V = Vh[S > S[0] * 1e-6]
    total = float((Xc0**2).sum())
    share = lambda B: float(((Xc0 @ B.T) ** 2).sum()) / total
    Sc = torch.linalg.svdvals(Xc0)
    Q, _ = torch.linalg.qr(
        torch.randn(
            Xc.shape[1], V.shape[0], dtype=Xc.dtype, generator=torch.Generator().manual_seed(0)
        )
    )
    nn = nn_dist(Xc)
    ratio = torch.cdist(Xc, Xl).min(1).values / nn.median()
    loop_ratio = torch.cdist(Xl, Xc).min(1).values / nn.median()
    return dict(
        share_full=share(V),
        loop_rank=int(V.shape[0]),
        random_share_same_rank=share(Q.T),
        share_topk={k: share(V[:k]) for k in ks if k <= len(V)},
        corpus_own_topk={k: float((Sc[:k] ** 2).sum() / (Sc**2).sum()) for k in ks if k <= len(Sc)},
        ratio_median=float(ratio.median()),
        ratio_min=float(ratio.min()),
        ratio_max=float(ratio.max()),
        frac_within_1=float((ratio < 1).float().mean()),
        frac_within_2=float((ratio < 2).float().mean()),
        visited=[int(i) for i in torch.nonzero(ratio < 1).flatten()],
        frames_nearest_attractor_ratio_median=float(loop_ratio.median()),
        pr_loop=participation_ratio(Xl),
        pr_corpus=participation_ratio(Xc),
    )


def cmd_describe(a):
    """Recompute the descriptors and the shift check of an existing corpus in place."""
    dev = pick_device()
    C = torch.load(a.corpus, weights_only=False)
    rg = setup(C["net"], C["target"], C["size"], dev)
    specs = C["spectra"].to(dev)
    imgs = images(rg, specs)
    C["desc"] = describe_images(rg, imgs)
    shift = shift_check(rg, imgs, C["desc"], specs)
    torch.save(C, a.corpus)
    j = json.load(open(a.corpus.replace("corpus.pt", "corpus.json")))
    nn = {k: nn_dist(v) for k, v in C["desc"].items()}
    j.update(
        shift=shift,
        nn_median={k: float(v.median()) for k, v in nn.items()},
        nn_cv={k: float(v.std() / v.mean()) for k, v in nn.items()},
    )
    json.dump(j, open(a.corpus.replace("corpus.pt", "corpus.json"), "w"), indent=1)
    print(json.dumps(dict(shift=shift, nn_median=j["nn_median"], nn_cv=j["nn_cv"]), indent=1))


def cmd_coverage(a):
    dev = pick_device()
    C = torch.load(a.corpus, weights_only=False)
    rg = setup(C["net"], C["target"], C["size"], dev)
    frames = load_frames(a.frames)
    note = ""
    if frames.shape[-1] != C["size"]:  # texture scale is fixed in pixels: crop, never downsample
        s, o = C["size"], (frames.shape[-1] - C["size"]) // 2
        frames = frames[:, :, o : o + s, o : o + s]
        note = f"center-cropped {s}px from {a.frames}"
    ld = describe_images(rg, frames)
    lact = robust_act(rg, spectra_of(rg, frames))
    rep = dict(
        frames=len(frames),
        note=note,
        loop_act_mean=float(lact.mean()),
        loop_act_min=float(lact.min()),
        corpus_act_max=float(C["act"].max()),
        corpus_act_mean=float(C["act"].mean()),
    )
    for k in ("gram", "radial", "squint"):
        rep[k] = coverage_stats(C["desc"][k].double(), ld[k].double())
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(f"{a.out}/coverage_{a.tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "squint"}, indent=1))


# ----------------------------------------------------------------------------- step 3


def band_masks(size, dev):
    k = torch.tensor(_rfft2d_freqs(size, size), dtype=torch.float32, device=dev)[None, :, :, None]
    return {name: ((k >= lo) & (k < hi)).float() for name, lo, hi in BANDS}


@torch.no_grad()
def unit_dirs(rg, d):
    """Scale each direction so a step of 1 changes the pre-sigmoid image by RMS 1 (logit units)."""
    rms = rg.pre(d).pow(2).mean((1, 2, 3)).sqrt()
    return d / rms.view(-1, 1, 1, 1, 1)


@torch.no_grad()
def radius(rg, A, dirs, a0, target=0.9, t0=0.05, grow=8, bisect=6):
    """Step along each direction at which robust activation / a0 first falls below target."""
    K = len(dirs)
    f = lambda t: robust_act(rg, A[None] + t.view(-1, 1, 1, 1, 1) * dirs) / a0
    lo, hi = torch.zeros(K), torch.full((K,), float("nan"))
    t = torch.full((K,), t0)
    curve = []
    for _ in range(grow):
        v = f(t.to(rg.dev))
        curve.append((float(t[0]), v.tolist()))
        new = (v < target) & hi.isnan()
        hi[new] = t[new]
        still = hi.isnan()
        lo[still] = t[still]
        if not still.any():
            break
        t = torch.where(still, t * 2, t)
    censored = hi.isnan()
    hi[censored] = lo[censored] * 2
    for _ in range(bisect):
        mid = (lo + hi) / 2
        v = f(mid.to(rg.dev))
        hi, lo = torch.where(v < target, mid, hi), torch.where(v < target, lo, mid)
    r = (lo + hi) / 2
    img0 = rg.image(A[None])
    pix = (
        (rg.image(A[None] + r.to(rg.dev).view(-1, 1, 1, 1, 1) * dirs) - img0)
        .pow(2)
        .mean((1, 2, 3))
        .sqrt()
    )
    return r, censored, pix.cpu(), curve


def cmd_width(a):
    dev = pick_device()
    log = log_factory()
    C = torch.load(a.corpus, weights_only=False)
    rg = setup(C["net"], C["target"], C["size"], dev)
    specs, act = C["spectra"].to(dev), C["act"]
    N = len(specs)
    order = act.argsort(descending=True)
    chosen = [int(i) for i in order[:: max(1, N // a.m)][: a.m]]
    masks = band_masks(C["size"], dev)
    D = torch.cdist(C["desc"]["gram"], C["desc"]["gram"])
    D.fill_diagonal_(float("inf"))
    g = torch.Generator(device="cpu").manual_seed(0)
    out = dict(chosen=chosen, target=0.9, unit="RMS pre-sigmoid (logit) change", per=[])
    for i in chosen:
        A, a0 = specs[i], float(act[i])
        rec = dict(attractor=i, act=a0, bands={})
        for name, mask in masks.items():
            d = torch.randn(a.k, *A.shape, generator=g).to(dev) * mask
            r, cen, pix, curve = radius(rg, A, unit_dirs(rg, d), a0)
            rec["bands"][name] = dict(
                radius=r.tolist(), censored=cen.tolist(), pixel_rms=pix.tolist(), curve=curve
            )
        # the straight line toward the nearest and a random other attractor, same units
        j, jr = int(D[i].argmin()), int(torch.randint(0, N, (1,), generator=g))
        jr = jr if jr != i else (jr + 1) % N
        d = torch.stack([specs[j] - A, specs[jr] - A])
        r, cen, pix, _ = radius(rg, A, unit_dirs(rg, d), a0)
        full = unit_dirs(rg, d)
        rec["toward"] = dict(
            nearest=j,
            random=jr,
            radius=r.tolist(),
            censored=cen.tolist(),
            pixel_rms=pix.tolist(),
            line_length=[float(v) for v in d.flatten(1).norm(dim=1) / full.flatten(1).norm(dim=1)],
        )
        out["per"].append(rec)
        log(
            f"attractor {i} act {a0:.3f} "
            + " ".join(f"{b}: {np.median(v['radius']):.3f}" for b, v in rec["bands"].items())
            + f" | toward nn {r[0]:.3f} rand {r[1]:.3f} (line {rec['toward']['line_length'][0]:.2f}, {rec['toward']['line_length'][1]:.2f})"
        )
    summary = {
        b: dict(
            median=float(np.median([r for p in out["per"] for r in p["bands"][b]["radius"]])),
            q10=float(np.quantile([r for p in out["per"] for r in p["bands"][b]["radius"]], 0.1)),
            q90=float(np.quantile([r for p in out["per"] for r in p["bands"][b]["radius"]], 0.9)),
            pixel_rms_median=float(
                np.median([r for p in out["per"] for r in p["bands"][b]["pixel_rms"]])
            ),
            censored=int(sum(sum(p["bands"][b]["censored"]) for p in out["per"])),
        )
        for b in masks
    }
    summary["toward_nearest"] = float(np.median([p["toward"]["radius"][0] for p in out["per"]]))
    summary["toward_random"] = float(np.median([p["toward"]["radius"][1] for p in out["per"]]))
    summary["line_length_nearest"] = float(
        np.median([p["toward"]["line_length"][0] for p in out["per"]])
    )
    out["summary"] = summary
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/width.json", "w"), indent=1)
    log("summary " + json.dumps(summary))


# ----------------------------------------------------------------------------- step 4


@torch.no_grad()
def resample(rg, chain, n, mb=8):
    """n interior points at equal band-passed image distance along an open chain."""
    b = torch.cat([bandpass(rg.image(chain[i : i + mb]), 2, 32) for i in range(0, len(chain), mb)])
    s = torch.cat([chain.new_zeros(1), (b[1:] - b[:-1]).flatten(1).norm(dim=1).cumsum(0)])
    s = s / s[-1]
    out = chain.new_empty(n, *chain.shape[1:])
    for i, u in enumerate(torch.linspace(0, 1, n + 2, device=chain.device)[1:-1]):
        j = int(torch.searchsorted(s, u).clamp(1, len(s) - 1))
        w = (u - s[j - 1]) / (s[j] - s[j - 1] + 1e-12)
        torch.lerp(chain[j - 1], chain[j], w, out=out[i])
    return out


def strings(rg, A, B, schedule, lr, mb, log=None):
    """E open strings A[e] -> B[e] at once, coarse to fine; returns a list of full chains."""
    E = len(A)
    levels = [tuple(map(int, l.split(":"))) for l in schedule.split(",")]
    X = None
    for li, (m, steps) in enumerate(levels):
        if X is None:
            t = torch.linspace(0, 1, m + 2, device=A.device)[1:-1].view(1, -1, 1, 1, 1, 1)
            X = A[:, None] + t * (B - A)[:, None]
        else:
            assert m == 2 * X.shape[1] + 1, (X.shape, m)
            new = X.new_empty(E, m, *X.shape[2:])
            for e in range(E):
                chain = [A[e]] + list(X[e]) + [B[e]]
                for i in range(m):
                    if i % 2 == 0:
                        torch.lerp(chain[i // 2], chain[i // 2 + 1], 0.5, out=new[e, i])
                    else:
                        new[e, i].copy_(chain[i // 2 + 1])
            X = new
        X = torch.nn.Parameter(X)
        opt = ChunkedAdam(X, lr=lr)
        last = li == len(levels) - 1
        for it in range(steps):
            if last:
                opt.param_groups[0]["lr"] = lr * 0.5 * (1 + math.cos(math.pi * it / steps))
            opt.zero_grad()
            seed = random.randrange(1 << 30)
            flat = X.view(E * m, *X.shape[2:])
            acts = []
            for i in range(0, E * m, mb):
                random.seed(seed)
                loss, act = rg.loss(flat[i : i + mb], view(rg.size))
                loss.backward()
                acts.append(act)
            with torch.no_grad():
                for e in range(E):
                    chain = torch.cat([A[e : e + 1], X[e], B[e : e + 1]])
                    tau = (chain[2:] - chain[:-2]).flatten(1)
                    tau = tau / tau.norm(dim=1, keepdim=True)
                    g = X.grad[e].flatten(1)
                    X.grad[e].copy_((g - (g * tau).sum(1, keepdim=True) * tau).view_as(X[e]))
            opt.step()
            if it % 10 == 9:
                with torch.no_grad():
                    for e in range(E):
                        X.data[e] = resample(
                            rg, torch.cat([A[e : e + 1], X.data[e], B[e : e + 1]]), m
                        )
            if log and (it == steps - 1):
                acts = torch.cat(acts)
                log(
                    f"  level {m}x{E} step {it + 1}/{steps} act mean {acts.mean():.3f} min {acts.min():.3f}"
                )
        X = X.detach()
        del opt
    return [torch.cat([A[e : e + 1], X[e], B[e : e + 1]]) for e in range(E)]


def disconnectivity(N, barriers):
    """Single linkage over barriers (highest first). Returns merges [(theta, members_a, members_b)]."""
    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    members = {i: [i] for i in range(N)}
    merges = []
    for (i, j), b in sorted(barriers.items(), key=lambda kv: -kv[1]):
        ri, rj = find(i), find(j)
        if ri != rj:
            merges.append(dict(theta=b, a=members[ri], b=members[rj]))
            parent[rj] = ri
            members[ri] = members[ri] + members.pop(rj)
    return merges


def clusters_at(N, barriers, theta):
    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (i, j), b in barriers.items():
        if b >= theta:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def draw_tree(imgs, leaf_theta, merges, path, T=48, H=520, theta_min=None):
    """Disconnectivity graph: leaves are attractor thumbnails at their own activation ratio,
    horizontal bars join groups at the barrier (theta) where they connect."""
    N = len(imgs)
    # leaf order: recursive layout of the merge tree
    node = {i: i for i in range(N)}  # cluster key -> tree
    tree = None
    for m in merges:
        ka, kb = tuple(sorted(m["a"]))[0], tuple(sorted(m["b"]))[0]
        tree = (m["theta"], node[ka], node[kb])
        node[min(ka, kb)] = tree
        node.pop(max(ka, kb), None)
    roots = list(node.values())

    order = []

    def leaves(t):
        if isinstance(t, int):
            order.append(t)
        else:
            leaves(t[1])
            leaves(t[2])

    for r in roots:
        leaves(r)
    thetas = [m["theta"] for m in merges] + leaf_theta
    tmin = theta_min if theta_min is not None else min(thetas) - 0.02
    top, bottom = 30, H - T - 20
    y = lambda th: bottom - (th - tmin) / (1 - tmin) * (bottom - top)
    W = N * T
    im = Image.new("RGB", (W + 60, H), (250, 250, 250))
    d = ImageDraw.Draw(im)
    for th in np.arange(math.ceil(tmin * 20) / 20, 1.0001, 0.05):
        d.line([(50, y(th)), (W + 55, y(th))], fill=(225, 225, 225))
        d.text((4, y(th) - 5), f"{th:.2f}", fill=(80, 80, 80))
    xs = {leaf: 50 + (k + 0.5) * T for k, leaf in enumerate(order)}
    for leaf in order:
        im.paste(
            Image.fromarray(
                (imgs[leaf].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            ).resize((T, T), Image.LANCZOS),
            (int(xs[leaf] - T / 2), H - T),
        )
        d.line([(xs[leaf], H - T), (xs[leaf], y(leaf_theta[leaf]))], fill=(40, 40, 40))

    def place(t):
        if isinstance(t, int):
            return xs[t], y(leaf_theta[t])
        th, l, r = t
        xl, yl = place(l)
        xr, yr = place(r)
        d.line([(xl, yl), (xl, y(th))], fill=(40, 40, 40))
        d.line([(xr, yr), (xr, y(th))], fill=(40, 40, 40))
        d.line([(xl, y(th)), (xr, y(th))], fill=(200, 60, 40), width=2)
        return (xl + xr) / 2, y(th)

    for r in roots:
        place(r)
    im.save(path)


def cmd_graph(a):
    dev = pick_device()
    log = log_factory()
    C = torch.load(a.corpus, weights_only=False)
    rg = setup(C["net"], C["target"], C["size"], dev)
    specs, act = C["spectra"].to(dev), C["act"]
    N = len(specs)
    amax = float(act.max())
    X = C["desc"]["gram"]
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    knn = D.topk(a.k, largest=False).indices
    edges = sorted({tuple(sorted((i, int(j)))) for i in range(N) for j in knn[i]})
    if a.max_edges:
        edges = edges[: a.max_edges]
    log(
        f"{N} attractors, k={a.k}: {len(edges)} edges, schedule {a.schedule}, {a.edges_per_batch} strings per batch"
    )
    os.makedirs(a.out, exist_ok=True)
    out_path = f"{a.out}/graph.json"
    done = json.load(open(out_path))["edges"] if a.resume and os.path.exists(out_path) else {}
    todo = [e for e in edges if f"{e[0]}-{e[1]}" not in done]
    t0 = time.time()
    for b in range(0, len(todo), a.edges_per_batch):
        batch = todo[b : b + a.edges_per_batch]
        A = specs[[i for i, _ in batch]]
        B = specs[[j for _, j in batch]]
        chains = strings(rg, A, B, a.schedule, a.lr, a.mb, log)
        for (i, j), chain in zip(batch, chains):
            prof = robust_act(rg, chain)
            straight = robust_act(rg, torch.lerp(specs[i], specs[j], 0.5)[None])[0]
            done[f"{i}-{j}"] = dict(
                min=float(prof.min()),
                profile=prof.tolist(),
                straight_mid=float(straight),
                ends=[float(act[i]), float(act[j])],
            )
        del chains
        if dev == "mps":
            torch.mps.empty_cache()
        per = (time.time() - t0) / (b + len(batch))
        log(
            f"{b + len(batch)}/{len(todo)} strings, {per:.1f}s each, ~{per * (len(todo) - b - len(batch)) / 60:.1f} min left"
        )
        json.dump(dict(edges=done, k=a.k, schedule=a.schedule, act_max=amax), open(out_path, "w"))
    barriers = {tuple(map(int, k.split("-"))): v["min"] / amax for k, v in done.items()}
    rel = {k: v["min"] / min(v["ends"]) for k, v in done.items()}
    merges = disconnectivity(N, barriers)
    sweep = {
        str(th): [len(c) for c in clusters_at(N, barriers, th)]
        for th in (0.98, 0.96, 0.94, 0.92, 0.9, 0.85, 0.8, 0.7)
    }
    sweep_members = {
        str(th): clusters_at(N, barriers, th)
        for th in (0.98, 0.96, 0.94, 0.92, 0.9, 0.85, 0.8, 0.7)
    }
    straight = [v["straight_mid"] / amax for v in done.values()]
    rep = dict(
        edges=done,
        k=a.k,
        schedule=a.schedule,
        act_max=amax,
        merges=merges,
        clusters_at=sweep,
        members_at=sweep_members,
        barrier_median=float(np.median(list(barriers.values()))),
        barrier_min=float(min(barriers.values())),
        barrier_rel_endpoints_median=float(np.median(list(rel.values()))),
        straight_midpoint_median=float(np.median(straight)),
        secs=round(time.time() - t0),
    )
    json.dump(rep, open(out_path, "w"), indent=1)
    imgs = images(rg, specs)
    draw_tree(imgs, (act / amax).tolist(), merges, f"{a.out}/tree.png")
    log(
        f"barriers median {rep['barrier_median']:.3f} min {rep['barrier_min']:.3f} (rel to endpoints {rep['barrier_rel_endpoints_median']:.3f}); "
        f"straight midpoints {rep['straight_midpoint_median']:.3f}; clusters {sweep}"
    )


# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("corpus")
    p.add_argument("--net", default="resnet50_v1")
    p.add_argument("--target", default="layer3:530")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--anchor-steps", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("describe")
    p.add_argument("--corpus", required=True)
    p = sub.add_parser("coverage")
    p.add_argument("--corpus", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--tag", default="loop")
    p.add_argument("--out", required=True)
    p = sub.add_parser("width")
    p.add_argument("--corpus", required=True)
    p.add_argument("--m", type=int, default=8, help="attractors probed")
    p.add_argument("--k", type=int, default=16, help="directions per band")
    p.add_argument("--out", required=True)
    p = sub.add_parser("graph")
    p.add_argument("--corpus", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--schedule", default="4:100,9:60,19:40")
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--mb", type=int, default=16)
    p.add_argument("--edges-per-batch", type=int, default=8)
    p.add_argument("--max-edges", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(
        corpus=cmd_corpus,
        describe=cmd_describe,
        coverage=cmd_coverage,
        width=cmd_width,
        graph=cmd_graph,
    )[a.cmd](a)


if __name__ == "__main__":
    main()
