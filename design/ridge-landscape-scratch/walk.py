"""Walks by design (handoff step 5): Langevin dynamics on the channel objective.

  walk.py langevin --corpus DIR/corpus.pt --graph DIR/graph.json --start 12 --T 1 --steps 3000 --out DIR

Preconditioned Langevin in spectrum space: x += eta * M * grad(act) + sqrt(2 eta T) * sqrt(M) * noise,
with M a fixed diagonal preconditioner (1 / per-coefficient gradient RMS at the start), so
the stationary measure is still proportional to exp(act / T). --T is given in units of
T_ref = eta * median(gradient RMS) / 2: the temperature at which the per-coefficient noise
step equals the typical gradient step. --T-lo and --period alternate T and T-lo every
period/2 steps (an annealing schedule). Frames are saved at equal step intervals so the
trajectory can be smoothed and closed by smooth.py; the step 4 clusters say which cluster
each frame is nearest to.

String tours are ring.py with --anchors-from corpus.pt --anchor-ids a,b,c, then smooth.py.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, math, random, time
import numpy as np
import torch
from PIL import Image
from striate.util import pick_device
from landscape import (
    setup,
    view,
    robust_act,
    describe_images,
    images,
    log_factory,
    clusters_at,
    unit_dirs,
)
from ring import temporal_report


def cluster_labels(graph, N, theta):
    labels = [-1] * N
    for c, members in enumerate(
        clusters_at(
            N,
            {
                tuple(map(int, k.split("-"))): v["min"] / graph["act_max"]
                for k, v in graph["edges"].items()
            },
            theta,
        )
    ):
        for i in members:
            labels[i] = c
    return labels


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("langevin")
    p.add_argument("--corpus", required=True)
    p.add_argument("--graph", default="")
    p.add_argument(
        "--theta", type=float, default=0.9, help="cluster threshold for labelling frames"
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--T", type=float, default=1.0, help="in units of T_ref")
    p.add_argument("--T-lo", type=float, default=-1.0)
    p.add_argument("--period", type=int, default=0)
    p.add_argument("--eta", type=float, default=0.01)
    p.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="if > 0: instead of the Langevin term, add white spectrum noise of this pre-sigmoid RMS "
        "per step (the width probes' unit), isotropic in image terms rather than in M",
    )
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--probe-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = pick_device()
    log = log_factory()
    t0 = time.time()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    C = torch.load(a.corpus, weights_only=False)
    rg = setup(C["net"], C["target"], C["size"], dev)
    N = len(C["spectra"])
    labels = cluster_labels(json.load(open(a.graph)), N, a.theta) if a.graph else [0] * N
    gram_c = C["desc"]["gram"]
    x = C["spectra"][a.start].to(dev).clone()
    a0 = float(C["act"][a.start])

    def grad(x):
        xp = x[None].clone().requires_grad_(True)
        loss, act = rg.loss(xp, view(rg.size))
        loss.backward()
        return -xp.grad[0], float(act[0])

    g2 = torch.zeros_like(x)
    for _ in range(32):
        g2 += grad(x)[0] ** 2
    rms = (g2 / 32).sqrt()
    M = 1 / (rms + 1e-2 * rms.median())
    T_ref = a.eta * float(rms.median()) / 2
    log(
        f"start {a.start} act {a0:.3f}; gradient RMS median {float(rms.median()):.2e}; T_ref {T_ref:.2e}; T {a.T} T_ref"
    )
    T_hi, T_lo = a.T * T_ref, (a.T_lo if a.T_lo >= 0 else a.T) * T_ref
    save_at = {int(round(s)): k for k, s in enumerate(np.linspace(0, a.steps - 1, a.frames))}
    os.makedirs(a.out, exist_ok=True)
    frames = torch.empty(a.frames, *x.shape, device=dev)
    probes = []  # (step, act_view, nearest attractor, cluster, gram distance from start)
    g_start = None
    for it in range(a.steps):
        T = T_hi if (a.period == 0 or (it // (a.period // 2)) % 2 == 0) else T_lo
        g, act = grad(x)
        with torch.no_grad():
            if a.sigma > 0:
                x += a.eta * M * g + a.sigma * unit_dirs(rg, torch.randn_like(x)[None])[0]
            else:
                x += a.eta * M * g + math.sqrt(2 * a.eta * T) * M.sqrt() * torch.randn_like(x)
        if it in save_at:
            frames[save_at[it]] = x
        if it % a.probe_every == 0 or it == a.steps - 1:
            with torch.no_grad():
                gd = describe_images(rg, rg.image(x[None]).cpu())["gram"][0]
            g_start = gd if g_start is None else g_start
            dist = (gram_c - gd).norm(dim=1)
            j = int(dist.argmin())
            probes.append(
                dict(
                    step=it,
                    act=act,
                    T=T / T_ref,
                    nearest=j,
                    cluster=labels[j],
                    nearest_dist=float(dist[j]),
                    from_start=float((gd - g_start).norm()),
                )
            )
        if it % 500 == 0 or it == a.steps - 1:
            pr = probes[-1]
            log(
                f"step {it} act {act:.3f} T {pr['T']:.2g} nearest {pr['nearest']} cluster {pr['cluster']} moved {pr['from_start']:.3f}"
            )
    with torch.no_grad():
        ract = robust_act(rg, frames)
        imgs = images(rg, frames)
    for i, f in enumerate(imgs):
        Image.fromarray((f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
            f"{a.out}/f{i:03d}.png"
        )
    nn_med = float(
        json.load(open(os.path.join(os.path.dirname(a.corpus), "corpus.json")))["nn_median"]["gram"]
    )
    seq = [p["cluster"] for p in probes]
    rep = dict(
        start=a.start,
        start_act=a0,
        sigma=a.sigma,
        T=a.T,
        T_lo=a.T_lo,
        period=a.period,
        eta=a.eta,
        T_ref=T_ref,
        steps=a.steps,
        act_mean=float(ract.mean()),
        act_min=float(ract.min()),
        act_max=float(ract.max()),
        act=ract.tolist(),
        clipped=float(((imgs < 0.02) | (imgs > 0.98)).float().mean()),
        cluster_switches=int(sum(1 for p, q in zip(seq, seq[1:]) if p != q)),
        attractors_visited=len({p["nearest"] for p in probes}),
        clusters_visited=sorted({p["cluster"] for p in probes}),
        nearest_dist_over_nn=float(np.median([p["nearest_dist"] for p in probes]) / nn_med),
        final_from_start_over_nn=probes[-1]["from_start"] / nn_med,
        probes=probes,
        secs=round(time.time() - t0),
    )
    rep.update(temporal_report(imgs))
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(
        f"done act {rep['act_mean']:.3f} [{rep['act_min']:.3f}, {rep['act_max']:.3f}] of {a0:.3f}; "
        f"{rep['attractors_visited']} attractors, clusters {rep['clusters_visited']}, {rep['cluster_switches']} switches; "
        f"nearest/nn {rep['nearest_dist_over_nn']:.2f}; moved {rep['final_from_start_over_nn']:.2f} nn; clipped {rep['clipped']:.1%}"
    )


if __name__ == "__main__":
    main()
