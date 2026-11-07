"""Soft-body physics loops for objtube.py's bodies: the shells breathe, push each other, wobble, return.

Each body is already a chain of --nsph squashed spheres, so the physics state is just their centres
(K x N particles) and the renderer takes them back as per-sphere displacements (spheres(wob=...)) with
per-frame radii. The texture lives in body coordinates, so it rides along: the motion is transport,
not a crossfade.

  rest pose  the trained bodies (a checkpoint from objtube.py) are the equilibrium. Every force below
             measures a departure from it, so with the drive off nothing moves and frame 0 is the still.
  drive      each body swells about its bottom by up to --amp of its size (only --zswell of that upward,
             so a swelling body pushes its neighbours aside rather than rising over them) on a periodic waveform: a
             quick swell and a slower release (--attack, --release, fractions of the cycle), --cycles
             per loop. Phases are random per body (--drive pulse) or a travelling wave across the torus
             (--drive wave, --wave kx,ky), or (--drive field) a smooth breathing, signed, from the sum of a few
             slow plane waves (--field) at each body's centre: everything moves all the time, neighbours
             together, with no sudden onsets. Optionally a travelling bend along each tube's spine (--bend px).
  fillers    bodies flagged is_filler (objfill.py: hidden shells under the pile) are not driven; they
             are only shoved, and show where the pile opens.
  pokers     optional (--pokers M): unseen spheres travelling closed paths under the pile, shoving
             the bodies above them up and aside, like something moving under a blanket.
  elastic    cluster shape matching (Mueller et al. 2005): overlapping windows of --cluster spheres
             along the spine each find their best rigid fit to the (driven) rest pose, and pull their
             spheres toward it (--k-shape). The windows overlap, so a body bends and wobbles, but every
             window keeps its shape. A weak spring holds every sphere toward its rest place (--k-anchor),
             so the pile is held where the still put it and a pushed body drifts back.
  contact    sphere pairs of different bodies (squashed spheres: support radius r * sqrt(nx^2 + ny^2 +
             f^2 nz^2)) repel when they overlap MORE than they did at rest (--k-contact, Hertz-like
             pen^1.5), so the interpenetration the still relies on stays, and a swelling body shoves
             its neighbours. --prestress shrinks that allowance by a fraction of the radii: the pile
             is packed under pressure, neighbours follow, gaps close. The ground pushes up likewise.
  volume     a sphere's radius scales with the local stretch of the spine as stretch^-nu (--nu 0.5 is
             incompressible): squash and stretch. Drawn only: contact uses the driven radius, since
             thickening where the spine is squeezed presses harder and squeezes it more (a runaway).
  effects    at render time, all closed over the loop: --hue-walk (each body's colour turns about the
             grey axis), --tex-turn / --tex-slide (each body's texture turns about its spine and slides
             along it), --funhouse (a smooth 3D warp of the whole pile; the light follows it).
  loop       semi-implicit Euler, --substeps per frame, damping --zeta of the shape mode. The drive is
             periodic, so after --warm periods the motion settles onto a cycle; one period is recorded
             and its closure error reported (and the remainder removed by a linear ramp in position).

Then the recorded period is rendered under the checkpoint's learned light (lit, and clay), scored per
frame with the 4 fixed crops, and its visible ground (gaps) measured. --train N retrains the texture
under random frames of the loop (positions fixed, grads only into textures and ground).

usage: objphys.py --load tube.pt --image still.png --target inception4c:7 --out DIR [--render] [--train N]
       Simulation runs on the CPU (no GPU lock needed with --sim-only); rendering and training need MPS.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, math, random, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from objtube import Mixed, Rig, Cam, build_pairs, render, render_ss, encode, rot_x, rot_z, with_dir


# ---------------------------------------------------------------- physics (CPU, float64)

class Pile:
    def __init__(self, C0, r0, f, a, S, log, still=None):
        self.a, self.S = a, S
        dd = torch.float64
        self.C0, self.r0, self.f = C0.to(dd), r0.to(dd), f.to(dd)
        K, N, _ = C0.shape
        self.K, self.N = K, N
        self.T = a.frames
        g = torch.Generator().manual_seed(a.seed)
        # pivot: the body swells about its centre in x, y and its bottom in z, so it grows off what it rests on
        bot = (self.C0[..., 2] - self.f[:, None] * self.r0).min(1)[0]
        self.piv = torch.stack([self.C0[..., 0].mean(1), self.C0[..., 1].mean(1), bot], -1)[:, None]
        self.zfloor = torch.minimum(self.C0[..., 2] - self.f[:, None] * self.r0, torch.zeros(()))
        if a.drive == "pulse":
            self.phase = torch.rand(K, generator=g, dtype=dd)
        elif a.drive == "field":
            # a few slow plane waves on the torus, each with its own direction, speed and phase; a body
            # breathes with their sum at its centre, so all of them move all the time, neighbours together
            W_ = torch.tensor([[float(x) for x in w.split(",")] for w in a.field.split(";")], dtype=dd)
            self.fk, self.fc = W_[:, :2], W_[:, 2]
            self.fph = torch.rand(len(W_), generator=g, dtype=dd) * 2 * math.pi
            self.fx = (self.piv[:, 0, :2] / S) @ self.fk.T  # (K, M) spatial phase, turns
            self.phase = torch.zeros(K, dtype=dd)
        else:
            kx, ky = map(float, a.wave.split(","))
            self.phase = -(kx * self.piv[:, 0, 0] / S + ky * self.piv[:, 0, 1] / S) % 1.0
        self.amp = a.amp * (1 + a.amp_spread * (torch.rand(K, generator=g, dtype=dd) * 2 - 1))
        self.moves = torch.ones(K, dtype=dd)  # 0: not driven (fillers keep their size and place)
        if still is not None:
            self.moves[still] = 0
            self.amp = self.amp * self.moves
        self.bph = torch.rand(K, generator=g, dtype=dd) * 2 * math.pi
        self.mass = ((self.r0.mean(1) / self.r0.mean(1).median()) ** 2).clamp(0.5, 4.0)
        # bend direction: horizontal, perpendicular to the spine at rest
        Tn = torch.cat([self.C0[:, 1:2] - self.C0[:, 0:1], self.C0[:, 2:] - self.C0[:, :-2], self.C0[:, -1:] - self.C0[:, -2:-1]], 1)
        pp = torch.stack([-Tn[..., 1], Tn[..., 0], torch.zeros_like(Tn[..., 0])], -1)
        self.perp = pp / (pp.norm(dim=-1, keepdim=True) + 1e-9)
        seg = (self.C0[:, 1:] - self.C0[:, :-1]).norm(dim=-1)
        s = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(-1)], -1)
        self.u = s / (s[:, -1:] + 1e-6)
        self.is_tube = torch.ones(K, dtype=torch.bool)
        # clusters: overlapping windows along the spine
        W, st = a.cluster, a.cstride
        starts = list(range(0, N - W + 1, st))
        if starts[-1] != N - W:
            starts.append(N - W)
        self.idx = torch.tensor([list(range(s0, s0 + W)) for s0 in starts])  # (nc, W)
        Qr = self.C0[:, self.idx] - self.C0[:, self.idx].mean(2, keepdim=True)
        self.qq = (Qr ** 2).sum((-1, -2))[..., None, None] / 3  # (K, nc, 1, 1): the window's spread
        self.cnt = torch.zeros(N, dtype=dd).index_add_(0, self.idx.reshape(-1), torch.ones(self.idx.numel(), dtype=dd))
        # contact pairs: spheres of different bodies close at rest (anything that could meet)
        self.pi, self.pj, self.pen0 = self.contact_pairs(log)

    def wrap(self, d):
        return torch.cat([d[..., :2] - self.S * torch.round(d[..., :2] / self.S), d[..., 2:]], -1)

    def support(self, r, f, n):
        return r * torch.sqrt(n[..., 0] ** 2 + n[..., 1] ** 2 + (f * n[..., 2]) ** 2)

    def contact_pairs(self, log):
        a, K, N = self.a, self.K, self.N
        P = self.C0.reshape(-1, 3)
        R = self.r0.reshape(-1)
        F_ = self.f[:, None].expand(K, N).reshape(-1)
        B = torch.arange(K)[:, None].expand(K, N).reshape(-1)
        mu = self.C0.mean(1)
        ext = ((self.C0 - mu[:, None]).norm(dim=-1) + self.r0).max(1)[0]
        dm = self.wrap(mu[None] - mu[:, None]).norm(dim=-1)
        bi, bj = torch.nonzero((dm < ext[:, None] + ext[None] + a.reach) & torch.triu(torch.ones(K, K, dtype=torch.bool), 1), as_tuple=True)
        I, J = [], []
        for k, l in zip(bi.tolist(), bj.tolist()):
            ii = torch.arange(k * N, (k + 1) * N)
            jj = torch.arange(l * N, (l + 1) * N)
            d = self.wrap(P[jj][None] - P[ii][:, None]).norm(dim=-1)
            ok = d < R[ii][:, None] + R[jj][None] + a.reach
            x, y = torch.nonzero(ok, as_tuple=True)
            I.append(ii[x]); J.append(jj[y])
        I, J = torch.cat(I), torch.cat(J)
        d = self.wrap(P[J] - P[I])
        dist = d.norm(dim=-1).clamp(min=1e-6)
        n = d / dist[:, None]
        pen = self.support(R[I], F_[I], n) + self.support(R[J], F_[J], n) - dist
        allow = pen.clamp(min=0) - a.prestress * (R[I] + R[J])
        log(f"contact pairs: {len(I)} between {len(bi)} body pairs; overlapping at rest {(pen > 0).float().mean():.2f}, "
            f"median overlap {pen[pen > 0].median():.2f}px")
        return I, J, allow

    def swell(self, t):
        a = self.a
        if a.drive == "field":  # smooth, signed: swell and shrink about the still's size, mean zero
            return torch.cos(2 * math.pi * (self.fc[None] * t / self.T - self.fx) + self.fph[None]).mean(1)
        ph = (a.cycles * t / self.T + self.phase) % 1.0
        at, rl = a.attack, a.release
        up = 0.5 * (1 - torch.cos(math.pi * ph / at))
        dn = 0.5 * (1 + torch.cos(math.pi * (ph - at) / rl))
        return torch.where(ph < at, up, torch.where(ph < at + rl, dn, torch.zeros_like(ph)))

    def rest(self, t):
        """Driven rest pose (K, N, 3) and radii (K, N) at time t (frames)."""
        sc = 1 + self.amp * self.swell(t)
        scz = 1 + self.a.zswell * self.amp * self.swell(t)  # mostly sideways: rising would lift it over its neighbours
        q = self.piv + torch.stack([sc, sc, scz], -1)[:, None] * (self.C0 - self.piv)
        r = self.r0 * sc[:, None]
        if self.a.bend > 0:
            w = torch.sin(math.pi * self.a.bend_waves * self.u - 2 * math.pi * self.a.cycles * t / self.T + self.bph[:, None])
            q = q + self.perp * (self.a.bend * w * self.u.clamp(0.15, 1) * self.moves[:, None])[..., None]
        return q, r

    def radii(self, p, q, rq):
        """Radius under local stretch along the spine: stretch^-nu."""
        lp = (p[:, 1:] - p[:, :-1]).norm(dim=-1)
        lq = (q[:, 1:] - q[:, :-1]).norm(dim=-1).clamp(min=1e-6)
        ls = lp / lq
        lam = torch.cat([ls[:, :1], 0.5 * (ls[:, 1:] + ls[:, :-1]), ls[:, -1:]], 1)
        return rq * lam.clamp(0.5, 2.0) ** (-self.a.nu)

    def forces(self, p, v, t):
        a, K, N = self.a, self.K, self.N
        q, rq = self.rest(t)
        r = self.radii(p, q, rq)  # what is drawn; contact uses the driven radius rq, since thickening
        # where the spine is squeezed would press harder and squeeze it more (a runaway)
        # shape matching per cluster
        P, Q = p[:, self.idx], q[:, self.idx]  # (K, nc, W, 3)
        pc, qc = P.mean(2, keepdim=True), Q.mean(2, keepdim=True)
        Apq = torch.einsum("kcwi,kcwj->kcij", P - pc, Q - qc)
        # a nearly straight window leaves its roll about the spine undetermined; a little of the
        # identity settles it (no roll) instead of letting the SVD pick a new one every substep
        Apq = Apq + self.a.sm_reg * self.qq * torch.eye(3, dtype=Apq.dtype)
        U, _, Vh = torch.linalg.svd(Apq)
        dsg = torch.sign(torch.det(U @ Vh))
        D = torch.diag_embed(torch.stack([torch.ones_like(dsg), torch.ones_like(dsg), dsg], -1))
        Rm = U @ D @ Vh
        G = torch.einsum("kcij,kcwj->kcwi", Rm, Q - qc) + pc
        pull = torch.zeros_like(p).index_add_(1, self.idx.reshape(-1), (G - P).reshape(K, -1, 3)) / self.cnt[None, :, None]
        Fr = a.k_shape * pull + a.k_anchor * (q - p)
        # contacts: by the driven radius, or (--contact-drawn) by the radius drawn, so a body thickened by
        # a squeeze pushes back instead of passing into its neighbour
        rc = r if getattr(a, "contact_drawn", 0) else rq
        pf, vf, rf = p.reshape(-1, 3), v.reshape(-1, 3), rc.reshape(-1)
        ff = self.f[:, None].expand(K, N).reshape(-1)
        d = self.wrap(pf[self.pj] - pf[self.pi])
        dist = d.norm(dim=-1).clamp(min=1e-6)
        n = d / dist[:, None]
        pen = self.support(rf[self.pi], ff[self.pi], n) + self.support(rf[self.pj], ff[self.pj], n) - dist - self.pen0
        on = pen > 0
        vn = ((vf[self.pj] - vf[self.pi]) * n).sum(-1)
        # the bodies are dense chains of overlapping spheres, so one contact is many sphere pairs: each
        # sphere feels the mean push of the spheres of each body it touches, not their sum
        bI, bJ = self.pi // N, self.pj // N
        oi = torch.zeros(K * N * K, dtype=pen.dtype).index_add_(0, self.pi * K + bJ, on.to(pen.dtype))
        oj = torch.zeros(K * N * K, dtype=pen.dtype).index_add_(0, self.pj * K + bI, on.to(pen.dtype))
        wgt = 1 / torch.sqrt(oi[self.pi * K + bJ].clamp(min=1) * oj[self.pj * K + bI].clamp(min=1))
        fm = (a.k_contact * pen.clamp(min=0) ** 1.5 - a.c_contact * vn * on) * on * wgt
        Fc = torch.zeros_like(pf).index_add_(0, self.pi, -fm[:, None] * n).index_add_(0, self.pj, fm[:, None] * n)
        Fc = Fc.reshape(K, N, 3) / self.mass[:, None, None]
        # ground: a sphere's bottom stays above where it was at rest (or 0)
        zb = p[..., 2] - self.f[:, None] * rc
        Fg = torch.zeros_like(p)
        Fg[..., 2] = a.k_contact * torch.relu(self.zfloor - zb)
        if a.pokers:
            Fg = Fg + self.poke(p, rq, t)
        omega = math.sqrt(a.k_shape)
        self.excess = float((pen.clamp(min=0) ** 2).mean().sqrt())
        return Fr + Fc + Fg - 2 * a.zeta * omega * v, r, int(on.sum())

    def poker_pos(self, t):
        """(M, 3): unseen spheres, each on its own closed path under the pile (a Lissajous loop of
        --poke-path px, once per cycle), at --poke-z above the ground."""
        a = self.a
        g = torch.Generator().manual_seed(a.seed + 3)
        M = a.pokers
        c = torch.rand(M, 2, generator=g, dtype=torch.float64) * self.S
        ph = torch.rand(M, 2, generator=g, dtype=torch.float64) * 2 * math.pi
        w = 2 * math.pi * a.cycles * t / self.T
        xy = c + a.poke_path * torch.stack([torch.cos(w + ph[:, 0]), torch.sin(2 * w + ph[:, 1]) * 0.6], -1)
        return torch.cat([xy, torch.full((M, 1), a.poke_z, dtype=torch.float64)], -1)

    def poke(self, p, r, t):
        a = self.a
        P = self.poker_pos(t)
        d = self.wrap(p[:, :, None] - P[None, None])  # (K, N, M, 3)
        dist = d.norm(dim=-1).clamp(min=1e-6)
        pen = torch.relu(a.poke_r + r[..., None] - dist)
        f = a.k_poke * pen ** 1.5 / self.mass[:, None, None]
        return (f[..., None] * d / dist[..., None]).sum(2)

    def run(self, log):
        a = self.a
        p, _ = self.rest(0.0)
        p = p.clone()
        v = torch.zeros_like(p)
        dt = 1.0 / a.substeps
        T = self.T
        traj, rads, nc, ex = [], [], [], []
        ends = []
        t0 = time.time()
        for fr in range((a.warm + 1) * T + 1):
            if fr >= a.warm * T:
                _, r = self.forces(p, v, float(fr))[:2]
                traj.append(p.clone()); rads.append(r.clone())
            if fr % T == 0:
                ends.append(p.clone())
                if len(ends) > 1:
                    e = (ends[-1] - ends[-2]).norm(dim=-1)
                    log(f"period {len(ends) - 1}: change since the last period max {e.max():.3f}px rms {e.pow(2).mean().sqrt():.3f}px ({time.time() - t0:.0f}s)")
            if fr == (a.warm + 1) * T:
                break
            for sub in range(a.substeps):
                t = fr + sub * dt
                Ft, _, ncon = self.forces(p, v, t)
                v = v + dt * Ft
                p = p + dt * v
            nc.append(ncon)
            ex.append(self.excess)
        traj, rads = torch.stack(traj), torch.stack(rads)  # (T + 1, K, N, 3)
        gap = traj[-1] - traj[0]
        close = dict(max=float(gap.norm(dim=-1).max()), rms=float(gap.norm(dim=-1).pow(2).mean().sqrt()))
        ramp = torch.linspace(0, 1, T + 1, dtype=traj.dtype)[:, None, None, None]
        traj = (traj - ramp * gap)[:T]  # the leftover of the cycle, spread over the period
        rads = (rads - ramp[..., 0] * (rads[-1] - rads[0]))[:T]
        q0 = torch.stack([self.rest(float(t))[0] for t in range(T)])
        dev_rest = (traj - self.C0[None]).norm(dim=-1)  # displacement from the still's pose
        dev_drive = (traj - q0).norm(dim=-1)  # the part not prescribed by the drive: pushes, lag, wobble
        stats = dict(closure=close, disp_mean=float(dev_rest.mean()), disp_p99=float(dev_rest.flatten().quantile(0.99)),
                     disp_max=float(dev_rest.max()), dyn_mean=float(dev_drive.mean()), dyn_p99=float(dev_drive.flatten().quantile(0.99)),
                     contacts_mean=float(np.mean(nc[-T:])), excess_rms=float(np.mean(ex[-T:])))
        log(f"closure max {close['max']:.3f}px rms {close['rms']:.4f}px; displacement from rest mean {stats['disp_mean']:.2f} p99 {stats['disp_p99']:.2f} "
            f"max {stats['disp_max']:.2f}px; beyond the drive mean {stats['dyn_mean']:.2f} p99 {stats['dyn_p99']:.2f}px; contacts {stats['contacts_mean']:.0f}, "
            f"overlap past rest rms {stats['excess_rms']:.3f}px")
        return traj.float(), rads.float(), stats


def padded(pr, bucket=1 << 16):
    """Pixel-body pairs padded up to a multiple of `bucket` with pairs whose ray passes far from
    every body (no coverage, no weight). Every frame has its own pair count, and MPS caches a
    graph per tensor shape: unpadded, a few hundred training steps run out of memory."""
    n = len(pr["pix"])
    m = -(-n // bucket) * bucket - n
    return dict(pix=np.concatenate([pr["pix"], np.zeros(m, pr["pix"].dtype)]),
                obj=np.concatenate([pr["obj"], np.zeros(m, pr["obj"].dtype)]),
                shift=np.concatenate([pr["shift"], np.full((m, 2), 1e5, np.float32)]))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", required=True, help="objtube.py checkpoint (tube.pt) with --body mixed")
    ap.add_argument("--image", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tex-from", help="copy the first bodies' textures and the ground from this checkpoint")
    ap.add_argument("--traj", help="traj.pt from an earlier run: skip the simulation")
    ap.add_argument("--sim-only", action="store_true", help="simulate on the CPU and stop (no GPU)")
    ap.add_argument("--render", action="store_true", help="render the loop, lit and clay, and score it")
    ap.add_argument("--frames", type=int, default=120, help="frames per loop (24 fps)")
    ap.add_argument("--warm", type=int, default=3, help="periods simulated before the recorded one")
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drive", default="pulse", choices=["pulse", "wave", "field"])
    ap.add_argument("--field", default="1,0,1;0,1,1;1,-1,1;2,1,2", help="field: plane waves kx,ky,cycles per loop (integers), ';'-separated")
    ap.add_argument("--wave", default="1,0", help="wave: phase advance per torus period in x, y (integers)")
    ap.add_argument("--cycles", type=int, default=1, help="drive cycles per loop")
    ap.add_argument("--amp", type=float, default=0.15, help="swell, fraction of the body's size")
    ap.add_argument("--amp-spread", type=float, default=0.3, help="per-body amp varies by +- this fraction")
    ap.add_argument("--zswell", type=float, default=0.3, help="fraction of the swell applied upward")
    ap.add_argument("--attack", type=float, default=0.12, help="swell time, fraction of a cycle")
    ap.add_argument("--release", type=float, default=0.4, help="release time, fraction of a cycle")
    ap.add_argument("--bend", type=float, default=0.0, help="travelling bend along the spine, px")
    ap.add_argument("--bend-waves", type=float, default=1.5, help="half-waves of the bend along the spine")
    ap.add_argument("--pokers", type=int, default=0, help="unseen spheres moving under the pile on closed paths")
    ap.add_argument("--poke-r", type=float, default=14.0, help="poker radius, px")
    ap.add_argument("--k-poke", type=float, default=0.3)
    ap.add_argument("--poke-z", type=float, default=0.0, help="poker centre height, px")
    ap.add_argument("--poke-path", type=float, default=60.0, help="poker path size, px")
    ap.add_argument("--cluster", type=int, default=10, help="spheres per shape-matching window")
    ap.add_argument("--cstride", type=int, default=5)
    ap.add_argument("--sm-reg", type=float, default=0.05, help="shape matching: identity bias, fraction of the window's spread")
    ap.add_argument("--k-shape", type=float, default=0.25, help="per frame^2: wobble period 2 pi / sqrt(k) frames")
    ap.add_argument("--k-anchor", type=float, default=0.08)
    ap.add_argument("--k-contact", type=float, default=0.2)
    ap.add_argument("--contact-drawn", type=int, default=0, help="contacts by the radius drawn (thickened where squeezed), not the driven one")
    ap.add_argument("--c-contact", type=float, default=0.3)
    ap.add_argument("--zeta", type=float, default=0.12, help="damping ratio of the shape mode")
    ap.add_argument("--nu", type=float, default=0.5, help="radius ~ stretch^-nu")
    ap.add_argument("--prestress", type=float, default=0.0, help="packing pressure: contact allowance cut by this fraction of the radii")
    ap.add_argument("--reach", type=float, default=6.0, help="px beyond touching at rest that a sphere pair is tracked")
    ap.add_argument("--zoom", type=float, default=1.0, help="render: output px per world px (2: the loop at 1024 px; scored downsampled)")
    ap.add_argument("--idmaps", type=int, default=0, help="render: also save which body owns each pixel (ids/fNNNN.png)")
    ap.add_argument("--hue-walk", type=float, default=0.0, help="render: each body's hue wanders this many radians rms, closing over the loop")
    ap.add_argument("--hue-drift", type=float, default=0.0, help="render: every hue turns this many full turns per loop")
    ap.add_argument("--tex-turn", type=float, default=0.0, help="render: each body's texture turns about its spine this many turns rms")
    ap.add_argument("--tex-slide", type=float, default=0.0, help="render: and slides along it this fraction of its length rms")
    ap.add_argument("--funhouse", type=float, default=0.0, help="render: px of smooth 3D warp bending the whole pile")
    ap.add_argument("--fh-waves", default="1,0,1;0,1,1;1,1,2;2,-1,1;1,-2,1", help="funhouse plane waves kx,ky,cycles per loop")
    ap.add_argument("--ss", type=int, default=1, help="render with 2x2 supersampling (1) or one pass (0)")
    ap.add_argument("--clay", type=int, default=1, help="also render the loop in clay")
    ap.add_argument("--tilt", type=float, default=0.0, help="also render the loop from this many degrees off vertical (clay and lit)")
    ap.add_argument("--train", type=int, default=0, help="steps of texture training under the loop")
    ap.add_argument("--render-after", type=int, default=1, help="training: also render the loop afterwards (0: just save tube.pt)")
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--train-turn", type=float, default=0.0, help="training: turn each body's texture by up to this many turns at random (and slide by half that)")
    ap.add_argument("--view-frac", type=float, default=0.0, help="training: fraction of frames seen through a tilted camera")
    ap.add_argument("--view-tilt", type=float, default=40.0, help="training: tilted camera, 10 to this many degrees off vertical, any azimuth")
    ap.add_argument("--also-trajs", help="training: comma-separated traj.pt files of other loops to train under as well")
    ap.add_argument("--lr-tex", type=float, default=0.02)
    ap.add_argument("--lr-ground", type=float, default=0.02)
    ap.add_argument("--wrim", type=float, default=1.0, help="training: cost of texture darkening toward a body's limb")
    ap.add_argument("--wdark", type=float, default=1.0, help="training: cost of texture darker than --dark-floor")
    ap.add_argument("--dark-floor", type=float, default=0.15)
    ap.add_argument("--ao-dif", type=float, default=None, help="share of the occlusion applied to the key light (default: the checkpoint's)")
    ap.add_argument("--amb-scale", type=float, default=None, help="scale the rig's ambient (default: the checkpoint's)")
    ap.add_argument("--sky", type=float, default=None, help="objtube's 3D sky occlusion strength (default: the checkpoint's)")
    ap.add_argument("--sky-pen0", type=float, default=1.5)
    ap.add_argument("--sky-pen", type=float, default=0.15)
    ap.add_argument("--outline", type=float, default=None, help="objtube's drawn ring at every body boundary: strength (default: the checkpoint's)")
    ap.add_argument("--outline-w", type=float, default=2.0)
    ap.add_argument("--ao", type=float, default=None, help="ambient occlusion strength (default: the checkpoint's)")
    ap.add_argument("--lr-prior", type=float, default=0.0, help="also train the wrapped-still prior texture at this rate")
    ap.add_argument("--bump-amp", type=float, default=None, help="fine relief on the texture's grid, px (default: the checkpoint's; 0 off)")
    ap.add_argument("--bump-around", type=int, default=64)
    ap.add_argument("--bump-along", type=int, default=144)
    ap.add_argument("--cavity", type=float, default=None, help="pits in the fine relief darken by exp(-this * depth px) (default: the checkpoint's)")
    ap.add_argument("--cav-r", type=int, default=3)
    ap.add_argument("--train-light", type=float, default=0.0, help="training: each sample's key light turns up to this many degrees in azimuth and drops up to half of it in elevation, so shading painted for one light fails the rest")
    ap.add_argument("--light-swing", type=float, default=0.0, help="render: the key light swings this many degrees in azimuth (and a quarter of it in elevation) over the loop")
    ap.add_argument("--bump-rim", type=float, default=None, help="relief displacement fades toward the outline (default: the checkpoint's)")
    ap.add_argument("--mip", type=int, default=None, help="filter textures and relief to each pixel's footprint (objtube.py --mip; default: the checkpoint's)")
    ap.add_argument("--mip-bias", type=float, default=0.0)
    ap.add_argument("--wskin", type=float, default=0.0, help="training: the skin a camera never saw sharply (finer than --skin-fp) "
                    "must match the pattern statistics (Gram matrices of --skin-layers) of the skin it did, body by body")
    ap.add_argument("--skin-fp", type=float, default=2.0, help="--wskin: seen sharply = at most this many texels per still pixel at rest")
    ap.add_argument("--skin-layers", default="conv3,inception3a")
    ap.add_argument("--skin-bodies", type=int, default=24, help="--wskin: bodies per step")
    ap.add_argument("--lr-bump", type=float, default=0.0, help="training: the fine relief's rate")
    ap.add_argument("--whf", type=float, default=0.0, help="training: penalty on the paint's fine luminance detail (relative to its start), so the detail moves into the relief")
    ap.add_argument("--hf-r", type=int, default=3, help="--whf: fine = finer than this many texels")
    ap.add_argument("--grow", default=None, help="textures grown by a lipgrow.py rule (its .pt): with --train, the rule trains instead of the paint")
    ap.add_argument("--lr-grow", type=float, default=1e-3, help="training: the growth rule's rate")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)
    save = lambda x, p: Image.fromarray((x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(p)
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    ck = torch.load(a.load, map_location="cpu", weights_only=False)
    ta = argparse.Namespace(**ck["args"])
    if a.ao is not None:  # ambient occlusion: from the checkpoint's args unless given here
        ta.ao = a.ao
    if a.sky is not None:
        ta.sky, ta.sky_pen0, ta.sky_pen = a.sky, a.sky_pen0, a.sky_pen
    for n_ in ("ao_dif", "amb_scale"):  # lighting overrides
        if getattr(a, n_) is not None:
            setattr(ta, n_, getattr(a, n_))
    if a.outline is not None:
        ta.outline, ta.outline_w = a.outline, a.outline_w
    if a.bump_amp is not None:  # the relief's settings travel with the checkpoint, so it reloads with its relief
        ta.bump_amp, ta.bump_around, ta.bump_along = a.bump_amp, a.bump_around, a.bump_along
        ck["args"] = {**ck["args"], "bump_amp": a.bump_amp, "bump_around": a.bump_around, "bump_along": a.bump_along}
    if a.bump_rim is not None:
        ta.bump_rim = a.bump_rim
        ck["args"] = {**ck["args"], "bump_rim": a.bump_rim}
    if a.mip is not None:
        ta.mip, ta.mip_bias = a.mip, a.mip_bias
        ck["args"] = {**ck["args"], "mip": a.mip, "mip_bias": a.mip_bias}
    if a.cavity is not None:
        ta.cavity, ta.cav_r = a.cavity, a.cav_r
        ck["args"] = {**ck["args"], "cavity": a.cavity, "cav_r": a.cav_r}
    cpu = torch.device("cpu")
    tb0 = Mixed.from_state(ck["tubes"], ta, cpu)
    is_filler = ck["tubes"].get("is_filler")
    with torch.no_grad():
        s0 = tb0.spheres()
    S = 512
    rep = dict(args=vars(a))
    if a.traj:
        tj = torch.load(a.traj)
        traj, rads, rep["sim"] = tj["traj"], tj["rads"], tj["stats"]
        log(f"trajectory from {a.traj}")
    else:
        pile = Pile(s0["C"][0], s0["r"], s0["f"], a, S, log, still=is_filler.bool() if is_filler is not None else None)
        if is_filler is not None:
            log(f"{int(is_filler.sum())} fillers (objfill.py): pushed about, but not driven")
        traj, rads, stats = pile.run(log)
        rep["sim"] = stats
        torch.save(dict(traj=traj, rads=rads, stats=stats, args=vars(a)), f"{a.out}/traj.pt")
    if a.sim_only:
        json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
        log("done (sim only)")
        return

    from striate.model import Backbone
    from striate.util import pick_device
    from zoo import MultiScaleCrop
    from ring import Ring
    dev = pick_device()
    tb = Mixed.from_state(ck["tubes"], ta, dev)
    rig = Rig(0.0, 0.8, ta, dev)
    rig.load_state_dict(ck["rig"])
    grower = None
    if a.grow:  # the texture is the rule's growth: the paint residual is off, the prior is the grown pattern
        from lipgrow import load_grow
        grower = load_grow(a.grow, dev)
        tsz = tuple(tb.tex0.shape[2:])
        with torch.no_grad():
            tb.tex.zero_()
            tb.tex0 = grower(size=tsz)
    if a.tex_from:  # textures (and the ground) for the first bodies from another checkpoint, e.g. one trained under a loop
        st = torch.load(a.tex_from, map_location="cpu", weights_only=False)["tubes"]
        with torch.no_grad():
            for n in ("tex", "tex0"):
                src = st[n].to(dev)
                getattr(tb, n)[:src.shape[0]] = src
            tb.ground.copy_(st["ground"].to(dev))
        log(f"textures of the first {st['tex'].shape[0]} bodies and the ground from {a.tex_from}")
    layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
    net = Backbone(ta.net, dev)
    net.hook(layer)
    rg = Ring(net, layer, ch, S, dev)
    img = np.asarray(Image.open(a.image).convert("RGB")).astype(np.float32) / 255
    still = torch.from_numpy(img.transpose(2, 0, 1))[None].to(dev)
    top = Cam(np.eye(3), S, S, (S / 2, S / 2, 0.0), S)

    def score(fr):
        state = random.getstate()
        tot = 0
        for k in range(4):
            random.seed(1000 + k)
            tot = tot + rg.act(MultiScaleCrop(224 / S, 512)(fr))
        random.setstate(state)
        return float(tot / 4)

    with torch.no_grad():
        C0 = tb.spheres()["C"][0]
    traj, rads = traj.to(dev), rads.to(dev)

    pool = [(traj, rads)]  # loops the texture trains under: this one, and any --also-trajs
    for pth in (a.also_trajs.split(",") if a.also_trajs else []):
        tj = torch.load(pth)
        pool.append((tj["traj"].to(dev), tj["rads"].to(dev)))

    T_ = traj.shape[0]
    gen = torch.Generator().manual_seed(a.seed + 101)

    def walk(n, harm=3):
        """n smooth closed walks over the loop, unit rms: a few harmonics with random phases, 1/h."""
        ph = torch.rand(n, harm, generator=gen) * 2 * math.pi
        amp = torch.randn(n, harm, generator=gen) / torch.arange(1, harm + 1)
        t = torch.arange(T_)[:, None, None] / T_
        w = (amp[None] * torch.cos(2 * math.pi * torch.arange(1, harm + 1)[None, None] * t + ph[None])).sum(-1)
        return (w / w.pow(2).mean().sqrt().clamp(min=1e-6)).to(dev)  # (T, n)

    Kb = tb.K
    hue_w = a.hue_walk * walk(Kb) + a.hue_drift * 2 * math.pi * torch.arange(T_, device=dev)[:, None] / T_ if (a.hue_walk or a.hue_drift) else None
    tex_w = torch.stack([a.tex_turn * walk(Kb), a.tex_slide * walk(Kb)], -1) if (a.tex_turn or a.tex_slide) else None
    fh = None
    if a.funhouse:  # plane waves of displacement on the torus: k (cycles per canvas), c (cycles per loop), a random direction each
        W_ = torch.tensor([[float(x) for x in w.split(",")] for w in a.fh_waves.split(";")])
        dirs = torch.randn(len(W_), 3, generator=gen) * torch.tensor([1.0, 1.0, 0.4])
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        fph = torch.rand(len(W_), generator=gen) * 2 * math.pi
        fh = (W_.to(dev), dirs.to(dev), fph.to(dev))

    def frame_sph(t, j=0):
        wob = pool[j][0][t] - C0
        if fh is not None:  # the whole pile bends together: a funhouse mirror in 3D, so the light follows it
            Wk, dk, pk = fh
            P_ = pool[j][0][t]
            arg = 2 * math.pi * ((P_[..., 0:1] * Wk[:, 0] + P_[..., 1:2] * Wk[:, 1]) / S + Wk[:, 2] * t / T_) + pk  # (K, N, M)
            wob = wob + a.funhouse / math.sqrt(len(Wk) / 2) * (torch.sin(arg)[..., None] * dk).sum(-2)
        sp = tb.spheres(wob=wob[None])
        r = pool[j][1][t]
        sp["r"] = r
        # material coordinates: arclength along the moved spine, caps from the moved radii, so the
        # texture stretches with the body instead of sliding, and a swollen cap doesn't run past the
        # end of the texture (which clamps into streaks)
        C = sp["C"][0]
        seg = (C[:, 1:] - C[:, :-1]).norm(dim=-1)
        s = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(-1)], -1)
        sp["s"], sp["s_lo"], sp["s_len"] = s, -r[:, 0], s[:, -1] + r[:, 0] + r[:, -1]
        if hue_w is not None:
            sp["hue"] = hue_w[t % T_]
        if tex_w is not None:
            sp["tex_shift"] = tex_w[t % T_]
        return sp

    if a.train:
        groups = [{"params": [tb.tex], "lr": a.lr_tex}, {"params": [tb.ground], "lr": a.lr_ground}]
        if grower is not None:
            groups = [{"params": list(grower.parameters()), "lr": a.lr_grow}, {"params": [tb.ground], "lr": a.lr_ground}]
        elif a.lr_prior > 0 and hasattr(tb, "tex0"):
            # the wrapped-still prior (64 x 144) carries the fine texture, and the streaks the hidden
            # parts were filled with: let it learn too
            t0_ = tb.tex0.detach().clone()
            del tb.tex0
            tb.tex0 = torch.nn.Parameter(t0_)
            groups.append({"params": [tb.tex0], "lr": a.lr_prior})
        if a.lr_bump > 0 and hasattr(tb, "bump"):
            groups.append({"params": [tb.bump], "lr": a.lr_bump})

        def paint_hf():  # the prior paint's luminance finer than --hf-r texels (wrapping around)
            Y_ = (torch.sigmoid(tb.tex0) * torch.tensor([0.299, 0.587, 0.114], device=dev)[:, None, None]).sum(1, keepdim=True)
            r_ = a.hf_r
            Yp = F.pad(torch.cat([Y_[:, :, -r_:], Y_, Y_[:, :, :r_]], 2), (r_, r_, 0, 0), mode="replicate")
            return (Y_ - F.avg_pool2d(Yp, 2 * r_ + 1, stride=1)).pow(2).mean()

        hf0 = float(paint_hf()) if a.whf > 0 else 1.0

        def wrap_pad(x, p):  # around wraps, along repeats its ends
            return F.pad(torch.cat([x[:, :, -p:], x, x[:, :, :p]], 2), (p, p, 0, 0), mode="replicate")

        if a.wskin > 0:
            # which texels the rest pose shows sharply: footprints from a 2x render, the finest per texel
            ta2 = argparse.Namespace(**vars(ta))
            ta2.mip = 1
            Z2 = 2 * S
            cam2 = Cam(np.eye(3), Z2, Z2, (S / 2, S / 2, 0.0), S, zoom=2.0)
            Kt, _, Ha, Wl = tb.tex0.shape
            with torch.no_grad():
                sp_ = tb.spheres()
                o = render(tb, sp_, cam2, build_pairs(sp_, cam2, 4.0), [rig()], ta2, aux=True)
                pix_, obj_ = o["pix"], o["obj"]
                share = o["ep"][0] / o["den"][0][pix_]
                top_ = torch.zeros(Z2 * Z2, device=dev).scatter_reduce(0, pix_, share, "amax", include_self=True)
                dom = (share >= top_[pix_] - 1e-9) & (share > 0.3)
                fp_ = torch.maximum(o["fpa"][0] * Ha, o["fpl"][0] * (Wl - 1)) * 2.0  # texels per still pixel
                idx = (obj_ * Ha + (o["tu"][0] * Ha).long() % Ha) * Wl + (o["tv"][0].clamp(0, 1) * (Wl - 1)).round().long()
                best = torch.full((Kt * Ha * Wl,), 1e3, device=dev).scatter_reduce(0, idx[dom], fp_[dom], "amin", include_self=True)
                good = (best.reshape(Kt, 1, Ha, Wl) <= a.skin_fp).float()
                good = (F.avg_pool2d(wrap_pad(good, 1), 3, stride=1) > 0.4).float()
                del o
            gfrac = good.mean((1, 2, 3))
            skin_ok = ((gfrac > 0.05) & (gfrac < 0.95)).nonzero()[:, 0].tolist()
            log(f"skin: {float(good.mean()):.1%} of texels seen sharply at rest; {len(skin_ok)} bodies take the skin term")
            pk = list(range(0, Kt, max(1, Kt // 8)))[:8]
            save(torch.cat([F.pad(good[k], (0, 4, 0, 4), value=0.5).expand(3, -1, -1) for k in pk], 2), f"{a.out}/skin_good.png")
            snet = Backbone(ta.net, dev)
            for L_ in a.skin_layers.split(","):
                snet.hook(L_)
            skin_rng = random.Random(a.seed + 11)

            def skin_loss():
                sel = torch.tensor(skin_rng.sample(skin_ok, min(a.skin_bodies, len(skin_ok))), device=dev)
                t0_, tx_ = tb.tex0[sel], tb.tex[sel]
                k_ = Ha // tx_.shape[2]
                up = F.interpolate(torch.cat([tx_[:, :, -1:], tx_, tx_[:, :, :1]], 2), size=(Ha + 2 * k_, Wl), mode="bilinear", align_corners=False)
                alb_ = wrap_pad(torch.sigmoid(t0_ + up[:, :, k_:k_ + Ha]), 16)
                m_ = wrap_pad(good[sel], 16)
                tot = 0
                for L_, f_ in snet(alb_).items():
                    mm = F.adaptive_avg_pool2d(m_, f_.shape[-2:])[:, 0]
                    f_ = f_ / (f_.detach().pow(2).mean((0, 2, 3), keepdim=True).sqrt() + 1e-6)
                    ff = f_.flatten(2)
                    gram = lambda w_: torch.bmm(ff * w_.flatten(1)[:, None], ff.transpose(1, 2)) / w_.sum((1, 2))[:, None, None].clamp(min=1e-3)
                    mean = lambda w_: torch.einsum("kchw,khw->kc", f_, w_) / w_.sum((1, 2))[:, None].clamp(min=1e-3)
                    Gg, Gb = gram(mm).detach(), gram(1 - mm)
                    Mg, Mb = mean(mm).detach(), mean(1 - mm)
                    tot = tot + ((Gb - Gg) ** 2).sum((1, 2)).div((Gg ** 2).sum((1, 2))).mean() + ((Mb - Mg) ** 2).sum(1).div((Mg ** 2).sum(1)).mean()
                return tot
        opt = torch.optim.Adam(groups)
        base = [g["lr"] for g in opt.param_groups]
        rng = random.Random(a.seed + 7)
        log(f"training textures under the loop: {a.train} steps, {a.mb} frames each")
        for it in range(a.train):
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * 0.5 * (1 + math.cos(math.pi * it / a.train))
            js = [rng.randrange(len(pool)) for _ in range(a.mb)]
            ts = [rng.randrange(pool[j][0].shape[0]) for j in js]
            with torch.no_grad():
                sps = [frame_sph(t, j) for t, j in zip(ts, js)]
                # some frames through a tilted camera, so the flanks and undersides the still never
                # showed get trained too (a lifted or tilted body shows them)
                cams = [top if rng.random() >= a.view_frac else
                        Cam(rot_x(math.radians(rng.uniform(10, a.view_tilt))) @ rot_z(rng.uniform(0, 2 * math.pi)), S, S, (S / 2, S / 2, 0.0), S)
                        for _ in sps]
                prs = [padded(build_pairs(sp, cm, 4.0)) for sp, cm in zip(sps, cams)]
            ref = rig()
            if grower is not None:  # grow the textures afresh: the gradient reaches the rule
                tb.tex0 = grower(size=tsz)
            acts, pens = [], []
            for sp, pr, cm in zip(sps, prs, cams):
                lt_ = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in ref.items()}
                if a.train_light > 0:
                    az_ = math.atan2(float(lt_["l"][1]), float(lt_["l"][0])) + math.radians(rng.uniform(-a.train_light, a.train_light))
                    el_ = math.asin(max(-1.0, min(1.0, float(lt_["l"][2])))) - math.radians(rng.uniform(0, a.train_light / 2))
                    lt_ = with_dir(lt_, az_, max(el_, math.radians(15)))
                if a.train_turn > 0:  # textures that hold up turned and slid on their bodies: paint all the way round
                    sp = dict(sp)
                    sp["tex_shift"] = torch.stack([(torch.rand(tb.K, device=dev) * 2 - 1) * a.train_turn,
                                                   (torch.rand(tb.K, device=dev) * 2 - 1) * a.train_turn / 2], -1)
                o = render(tb, sp, cm, pr, [lt_], ta, aux=True)
                acts.append(rg.act(MultiScaleCrop(224 / S, 512)(o["img"])))
                # paint may not do the geometry's job: no texture darker than --dark-floor, and none darker
                # toward a body's limb than 75% of its front (objtube.py's rim term)
                own = o["ep"][0] / o["den"][0][o["pix"]]
                Y = (o["alb"][0] * torch.tensor([0.299, 0.587, 0.114], device=dev)).sum(-1)
                fz = o["nk"][0][:, 2]
                obj = o["obj"]
                fw = own * (fz > 0.7).float()
                refY = torch.zeros(tb.K, device=dev).index_add_(0, obj, fw * Y) / torch.zeros(tb.K, device=dev).index_add_(0, obj, fw).clamp(min=1)
                band = own * (fz < 0.4).float()
                rim = (band * torch.relu(0.75 * refY.detach()[obj] - Y) ** 2).sum() / band.sum().clamp(min=1) / 0.01
                dark = (own * torch.relu(a.dark_floor - Y) ** 2).sum() / own.sum().clamp(min=1) / 0.01
                pens.append(a.wrim * rim + a.wdark * dark)
            act = torch.stack(acts).mean()
            pen = torch.stack(pens).mean()
            if a.whf > 0:
                pen = pen + a.whf * paint_hf() / hf0
            if a.wskin > 0:
                sk = skin_loss()
                pen = pen + a.wskin * sk
            opt.zero_grad()
            (-act + act.detach().abs() * pen).backward()
            opt.step()
            if it % 25 == 0 or it == a.train - 1:
                log(f"step {it + 1}/{a.train} act {float(act):.3f} pen {float(pen):.3f}" + (f" skin {float(sk):.3f}" if a.wskin > 0 else "") + f" (frames {ts})")
                if str(dev).startswith("mps"):
                    torch.mps.empty_cache()
            if (it + 1) % 100 == 0 and it + 1 < a.train:  # a crash late in a run keeps most of it
                torch.save({**ck, "tubes": {**ck["tubes"], **{k: v.detach().cpu() for k, v in tb.state_dict().items()}}}, f"{a.out}/tube_partial.pt")
        if grower is not None:
            torch.save(dict(cfg=grower.cfg, state={k: v.detach().cpu() for k, v in grower.state_dict().items()}), f"{a.out}/grow.pt")
            with torch.no_grad():
                tb.tex0 = grower(size=tsz)
        ck2 = dict(ck)
        ck2["tubes"] = {**ck["tubes"], **{k: v.detach().cpu() for k, v in tb.state_dict().items()}}
        ck2["phys_train"] = vars(a)
        torch.save(ck2, f"{a.out}/tube.pt")
        log(f"saved {a.out}/tube.pt")

    if a.render or (a.train and a.render_after):
        with torch.no_grad():
            ref = rig()
            rep["still"] = score(still)
            sp = tb.spheres()
            pr = build_pairs(sp, top, 4.0)
            rest = render_ss(tb, sp, top, pr, ref, ta)
            rep["rest_ss"] = score(rest)
            save(rest[0], f"{a.out}/rest.png")
            log(f"still {rep['still']:.3f}; rest pose {rep['rest_ss']:.3f}")
            d, dc = f"{a.out}/frames", f"{a.out}/clay"
            os.makedirs(d, exist_ok=True)
            os.makedirs(dc, exist_ok=True)
            acts, gnd = [], []
            Z = int(round(a.zoom * S))  # output size: --zoom 2 renders the same view at 1024 px
            camR = Cam(np.eye(3), Z, Z, (S / 2, S / 2, 0.0), S, zoom=a.zoom)
            down = (lambda x: F.interpolate(x, size=(S, S), mode="area")) if Z != S else (lambda x: x)
            ref0 = ref
            az0 = math.atan2(float(ref0["l"][1]), float(ref0["l"][0]))
            el0 = math.asin(max(-1.0, min(1.0, float(ref0["l"][2]))))
            for t in range(a.frames):
                if a.light_swing:  # a moving key light: painted shading can't follow it, relief does
                    ph = 2 * math.pi * t / a.frames
                    sw = math.radians(a.light_swing)
                    ref = with_dir(ref0, az0 + sw * math.sin(ph), min(max(el0 + sw / 4 * math.sin(2 * ph), math.radians(10)), math.radians(85)))
                sp = frame_sph(t)
                pr = padded(build_pairs(sp, camR, 4.0))
                fr = render_ss(tb, sp, camR, pr, ref, ta) if a.ss else render(tb, sp, camR, pr, [ref], ta)
                acts.append(score(down(fr)))
                save(fr[0], f"{d}/f{t:04d}.png")
                o = render(tb, sp, camR, pr, [ref], ta, clay=0.72, aux=True)
                gnd.append(float(o["gown"].mean()))
                if a.idmaps:  # which body shows at each pixel (255: the ground), for recolouring in the browser
                    ep_, pix_, obj_ = o["ep"][0], o["pix"], o["obj"]
                    best = torch.full((Z * Z,), -1.0, device=dev).scatter_reduce(0, pix_, ep_, "amax", include_self=True)
                    win = ep_ >= best[pix_] - 1e-12
                    idm = torch.full((Z * Z,), 255, dtype=torch.long, device=dev)
                    idm[pix_[win]] = obj_[win]
                    idm[o["gown"][0] > 0.5] = 255
                    os.makedirs(f"{a.out}/ids", exist_ok=True)
                    Image.fromarray(idm.reshape(Z, Z).cpu().numpy().astype(np.uint8)).save(f"{a.out}/ids/f{t:04d}.png")
                if a.clay:
                    save(o["img"][0], f"{dc}/f{t:04d}.png")
                if a.tilt:  # the same frame seen from --tilt deg off vertical: the pile in depth
                    camt = Cam(rot_x(math.radians(a.tilt)), Z, Z, (S / 2, S / 2, 0.0), S, zoom=a.zoom)
                    prt = build_pairs(sp, camt, 4.0)
                    os.makedirs(f"{a.out}/tilt", exist_ok=True)
                    os.makedirs(f"{a.out}/tilt_lit", exist_ok=True)
                    save(render(tb, sp, camt, prt, [ref], ta, clay=0.72)[0], f"{a.out}/tilt/f{t:04d}.png")
                    save(render(tb, sp, camt, prt, [ref], ta)[0], f"{a.out}/tilt_lit/f{t:04d}.png")
                if t % 20 == 0:
                    log(f"frame {t}: score {acts[-1]:.3f}, visible ground {gnd[-1] * 100:.2f}%")
            encode(d, f"{a.out}/loop.mp4")
            if a.clay:
                encode(dc, f"{a.out}/clay.mp4")
            if a.tilt:
                encode(f"{a.out}/tilt", f"{a.out}/tilt.mp4")
                encode(f"{a.out}/tilt_lit", f"{a.out}/tilt_lit.mp4")
            rep["loop"] = dict(mean=float(np.mean(acts)), min=float(np.min(acts)), per_frame=acts)
            rep["ground"] = dict(mean=float(np.mean(gnd)), max=float(np.max(gnd)), per_frame=gnd)
            log(f"loop: mean {np.mean(acts):.3f} min {np.min(acts):.3f}; visible ground mean {np.mean(gnd) * 100:.2f}% max {np.max(gnd) * 100:.2f}%")
    rep["secs"] = round(time.time() - t0)
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
