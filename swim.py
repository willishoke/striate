"""The swim: structure-preserving traversal within one facet mode (homotopy class).

Order the facets into a smooth nearest-neighbor tour, interpolate along it as a loop
(the raw interp is off-manifold = facet-blend mush), and project each frame back onto
the mode with three forces:
  - discriminator  : reject facet-blends -> stay a valid facet of the mode (the swim)
  - channel objective: keep firing the channel -> stays the feature, blocks adversarial
    cheats (garbage doesn't activate the channel)
  - frequency-banded temporal coherence: low band tracks the smooth tour, high band
    tracks the previous frame -> smooth in time without muting.
Baseline (raw tour interp) vs swim, with jitter + on-manifold (disc) scores.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from striate.model import Backbone
from striate.param import FourierImage
from striate.transforms import TransformRobustness
from striate.util import pick_device
from keyframe import render_spectra
from critic import Critic, OrbitAugment
from morph_critic import _lowpass, render_one, prep, to_pil, save_gif, jitter, CRIT_SIZE


def tight_neighborhood(feat, K):
    """The K facets closest to the medoid: a tight sub-region of the diffuse mode,
    so consecutive waypoints are near -> small on-manifold hops -> a smooth swim."""
    D = torch.cdist(feat, feat)
    medoid = int(D.sum(1).argmin())
    return D[medoid].argsort()[:K].tolist()


def nn_tour(feat):
    N = feat.shape[0]
    D = torch.cdist(feat, feat)
    cur, order, seen = 0, [0], {0}
    while len(order) < N:
        D[:, cur] = 1e9
        nxt = int(D[cur].argmin())
        order.append(nxt)
        seen.add(nxt)
        cur = nxt
    return order


def loop_path(spectra_ordered, M):
    K = spectra_ordered.shape[0]
    seq = torch.cat([spectra_ordered, spectra_ordered[:1]], dim=0)       # close the loop
    out = []
    for m in range(M):
        u = (m / M) * K
        i = int(u)
        t = u - i
        out.append((1 - t) * seq[i] + t * seq[i + 1])
    return torch.stack(out)


def loop_path_polar(spectra_ordered, M):
    """Polar interpolation: blend amplitude linearly, interpolate PHASE along the
    shortest arc. Phase carries spatial position (shift theorem), so moving it makes
    features FLOW through space (fundamental shifting) instead of cross-fading in place."""
    K = spectra_ordered.shape[0]
    c = torch.view_as_complex(spectra_ordered.contiguous())             # (K,3,H,Wf) complex
    seq = torch.cat([c, c[:1]], dim=0)
    out = []
    for m in range(M):
        u = (m / M) * K
        i = int(u)
        t = u - i
        a = (1 - t) * seq[i].abs() + t * seq[i + 1].abs()
        dphi = seq[i + 1].angle() - seq[i].angle()
        dphi = (dphi + math.pi) % (2 * math.pi) - math.pi               # shortest arc
        phi = seq[i].angle() + t * dphi
        out.append(torch.view_as_real(a * torch.exp(1j * phi)))
    return torch.stack(out)


def add_phase_flow(dense, flow_mag, cutoff, cycles, height, width):
    """Inject structure-preserving flow: a loop-periodic, low-frequency phase ramp.
    A phase ramp 2*pi*(fx*dx + fy*dy) translates the image by (dx,dy) (shift theorem);
    weighting it toward low frequencies makes the LARGE-SCALE structure drift (fundamental
    shifting) while fine detail rides along. The drift traces a circle over the loop so it
    returns to start (seamless). Pure isometry-orbit motion -- decoupled from facet diversity."""
    M = dense.shape[0]
    fy = torch.fft.fftfreq(height)[:, None]
    fx = torch.fft.rfftfreq(width)[None, :]
    f = torch.sqrt(fx ** 2 + fy ** 2)
    loww = torch.exp(-(f / cutoff) ** 2)                       # emphasize low freq (large-scale flow)
    c = torch.view_as_complex(dense.contiguous())             # (M,3,H,Wf)
    out = []
    for m in range(M):
        t = m / M
        dx = flow_mag * (math.cos(2 * math.pi * cycles * t) - 1)
        dy = flow_mag * math.sin(2 * math.pi * cycles * t)
        ramp = 2 * math.pi * (fx * dx + fy * dy) * loww        # (H, Wf) phase offset
        out.append(torch.view_as_real(c[m] * torch.exp(1j * ramp)))
    return torch.stack(out)


def swim_sequence(dense, disc, bb, sub, channel, *, size, lam_disc, lam_chan, anchor_lo,
                  temporal_hi, k, steps, lr, transforms, disc_aug, device):
    out, prev_spec, prev_hi = [], None, None
    for fi in range(dense.shape[0]):
        interp = dense[fi:fi + 1].to(device)
        interp_lo = _lowpass(render_one(interp, size, device), k)
        param = FourierImage(size, size, device=device)
        with torch.no_grad():
            param.spectrum.copy_(prev_spec if prev_spec is not None else interp)
        opt = torch.optim.Adam([param.spectrum], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            img = param.to_image()
            act = sub(bb.normalize(transforms(img)))
            lo = _lowpass(img, k)
            loss = (lam_disc * torch.sigmoid(disc(prep(img, CRIT_SIZE, disc_aug)))
                    - lam_chan * act[:, channel].mean()
                    + anchor_lo * ((lo - interp_lo) ** 2).mean())
            if temporal_hi > 0 and prev_hi is not None:
                loss = loss + temporal_hi * (((img - lo) - prev_hi) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = param.to_image()
            prev_hi = (final - _lowpass(final, k)).detach()
        prev_spec = param.spectrum.detach().clone()
        out.append(prev_spec.cpu())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facets", default="outputs/facets_l3_84.pt")
    ap.add_argument("--disc", default="outputs/disc_l3_84.pt")
    ap.add_argument("--channel", type=int, default=84)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--waypoints", type=int, default=8, help="facets to tour (tight neighborhood)")
    ap.add_argument("--proj-steps", type=int, default=25)
    ap.add_argument("--lam-disc", type=float, default=5.0)
    ap.add_argument("--lam-chan", type=float, default=1.0)
    ap.add_argument("--anchor-lo", type=float, default=25.0)
    ap.add_argument("--temporal-hi", type=float, default=60.0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--proj-lr", type=float, default=0.03)
    ap.add_argument("--interp", default="linear", choices=["linear", "polar"])
    ap.add_argument("--flow-mag", type=float, default=0.0, help="phase-flow drift magnitude (px); 0=off")
    ap.add_argument("--flow-cutoff", type=float, default=0.12, help="low-freq cutoff for the flow band")
    ap.add_argument("--flow-cycles", type=float, default=1.0, help="flow orbits over the loop")
    ap.add_argument("--no-desmear", action="store_true", help="just render the interpolated path (no disc projection)")
    ap.add_argument("--keyframes-only", action="store_true", help="render just the sharp facet waypoints (for flow-interp)")
    ap.add_argument("--tag", default="swim_l3_84")
    args = ap.parse_args()

    dev = pick_device()
    blob = torch.load(args.facets, weights_only=False)
    spectra, size, layer = blob["spectra"].to(dev), blob["size"], blob["layer"]
    N = spectra.shape[0]
    bb = Backbone("tiny", device=dev)
    sub = bb.model.features[:int(layer.split(".")[1]) + 1]
    disc = Critic().to(dev)
    disc.load_state_dict(torch.load(args.disc))
    disc.eval()
    for p in disc.parameters():
        p.requires_grad_(False)
    transforms = TransformRobustness(degrees=[])
    disc_aug = OrbitAugment(crop=CRIT_SIZE, degrees=0, translate=0)   # disc-eval crop-matches training FOV

    fp = FourierImage(size, size, batch=N, device=dev)
    with torch.no_grad():
        fp.spectrum.copy_(spectra)
        squint = F.interpolate(fp.to_image(), size=(16, 16), mode="area").reshape(N, -1).cpu()
    sel = tight_neighborhood(squint, min(args.waypoints, N))
    tour = nn_tour(squint[sel])
    waypoints = [sel[t] for t in tour]
    if args.keyframes_only:
        seq = waypoints + [waypoints[0]]                      # loop closure
        kf = render_spectra(spectra[seq].cpu(), size, size, dev, chunk=6)
        fdir = f"outputs/{args.tag}_keyframes"
        os.makedirs(fdir, exist_ok=True)
        for i in range(kf.shape[0]):
            to_pil(kf[i]).save(f"{fdir}/k{i:03d}.png")
        print(f"{len(seq)} sharp keyframes @ {size}px -> {fdir}/  (flow-interp these)")
        return

    interp_fn = loop_path_polar if args.interp == "polar" else loop_path
    dense = interp_fn(spectra[waypoints].cpu(), args.frames)
    if args.flow_mag > 0:
        dense = add_phase_flow(dense, args.flow_mag, args.flow_cutoff, args.flow_cycles, size, size)
    print(f"swim {layer}:{args.channel}  {len(waypoints)} tight waypoints -> {args.frames}-frame {args.interp} loop "
          f"({args.frames // len(waypoints)}/seg) @ {size}px  idx={waypoints}")

    base = render_spectra(dense, size, size, dev, chunk=6)

    if args.no_desmear:
        save_gif(base, f"outputs/{args.tag}.gif", 20)
        fdir = f"outputs/{args.tag}_frames"
        os.makedirs(fdir, exist_ok=True)
        for i in range(base.shape[0]):
            to_pil(base[i]).save(f"{fdir}/f{i:04d}.png")
        idxs = [round(k * (args.frames - 1) / 5) for k in range(6)]   # spread across the loop
        sheet = Image.new("RGB", (3 * size, 2 * size), (12, 12, 12))
        for n, i in enumerate(idxs):
            sheet.paste(to_pil(base[i]), ((n % 3) * size, (n // 3) * size))
        sheet.save(f"outputs/{args.tag}_spread.png")
        print(f"{args.interp} path (no de-smear)  jitter={jitter(base):.3f}  "
              f"-> outputs/{args.tag}.gif, {fdir}/ ({base.shape[0]} frames), {args.tag}_spread.png")
        return
    swam_spec = swim_sequence(dense, disc, bb, sub, args.channel, size=size, lam_disc=args.lam_disc,
                              lam_chan=args.lam_chan, anchor_lo=args.anchor_lo, temporal_hi=args.temporal_hi,
                              k=args.k, steps=args.proj_steps, lr=args.proj_lr, transforms=transforms,
                              disc_aug=disc_aug, device=dev)
    swam = render_spectra(swam_spec, size, size, dev, chunk=6)

    def disc_score(imgs, n=8):
        idx = [round(i * (imgs.shape[0] - 1) / (n - 1)) for i in range(n)]
        with torch.no_grad():
            return float(np.mean([torch.sigmoid(disc(prep(imgs[i][None].to(dev), CRIT_SIZE, disc_aug))).item()
                                  for i in idx]))

    print(f"  baseline (raw tour): jitter={jitter(base):.3f}  on-manifold(disc)={disc_score(base):.2f}")
    print(f"  swim:                jitter={jitter(swam):.3f}  on-manifold(disc)={disc_score(swam):.2f}")

    save_gif(base, f"outputs/{args.tag}_base.gif", 20)
    save_gif(swam, f"outputs/{args.tag}.gif", 20)
    # consecutive transition frames, full size, to judge the flow
    for name, seq in [("", swam), ("_base", base)]:
        sheet = Image.new("RGB", (4 * size, size), (12, 12, 12))
        for n, i in enumerate([20, 22, 24, 26]):
            sheet.paste(to_pil(seq[i]), (n * size, 0))
        sheet.save(f"outputs/{args.tag}{name}_consec.png")
    print(f"-> outputs/{args.tag}.gif (swim), {args.tag}_base.gif (raw), {args.tag}_consec.png")


if __name__ == "__main__":
    main()
