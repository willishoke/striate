"""Roll-our-own interpolator: optical-flow morphing between facets (blur-free).

The blur came from blending UNCORRESPONDED textures. Here we optimize a smooth
correspondence FLOW between consecutive facets, then warp BOTH toward each
intermediate so features ALIGN before blending -- blending aligned features stays
sharp (classic morphing); blending misaligned ones ghosts. Per-pair flow
optimization exploits that consecutive facets are similar (tight neighborhood),
which off-the-shelf flow can't assume. The flow is a low-res control grid (so it's
inherently smooth = coherent flow, not distortion).
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import torch
import torch.nn.functional as F

from striate.param import FourierImage
from striate.util import pick_device
from keyframe import render_spectra
from morph_critic import to_pil, save_gif, jitter
from swim import tight_neighborhood, nn_tour


def base_grid(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)[None]  # (1,H,W,2)


def optimize_flow(A, B, base, *, ctrl, steps, lr, smooth):
    H, W = A.shape[-2:]
    flow = torch.zeros(1, 2, ctrl, ctrl, device=A.device, requires_grad=True)
    opt = torch.optim.Adam([flow], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        f = F.interpolate(flow, (H, W), mode="bilinear", align_corners=True).permute(0, 2, 3, 1)
        wA = F.grid_sample(A, base + f, align_corners=True, padding_mode="reflection")
        loss = ((wA - B) ** 2).mean()
        if smooth > 0:
            loss = (
                loss
                + smooth * ((flow[:, :, 1:] - flow[:, :, :-1]) ** 2).mean()
                + smooth * ((flow[:, :, :, 1:] - flow[:, :, :, :-1]) ** 2).mean()
            )
        loss.backward()
        opt.step()
    return flow.detach()


def morph_frame(A, B, flow, t, base):
    H, W = A.shape[-2:]
    f = F.interpolate(flow, (H, W), mode="bilinear", align_corners=True).permute(0, 2, 3, 1)
    wA = F.grid_sample(
        A, base + t * f, align_corners=True, padding_mode="reflection"
    )  # A -> midpoint
    wB = F.grid_sample(
        B, base - (1 - t) * f, align_corners=True, padding_mode="reflection"
    )  # B -> midpoint
    return (1 - t) * wA + t * wB  # blend ALIGNED features = sharp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facets", default="outputs/facets_l3_84.pt")
    ap.add_argument("--waypoints", type=int, default=12)
    ap.add_argument("--per-seg", type=int, default=8)
    ap.add_argument(
        "--ctrl", type=int, default=48, help="flow control-grid resolution (smoothness)"
    )
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--tag", default="flowmorph")
    args = ap.parse_args()

    dev = pick_device()
    blob = torch.load(args.facets, weights_only=False)
    spectra, size = blob["spectra"].to(dev), blob["size"]
    N = spectra.shape[0]

    fp = FourierImage(size, size, batch=N, device=dev)
    with torch.no_grad():
        fp.spectrum.copy_(spectra)
        imgs = fp.to_image()
        squint = F.interpolate(imgs, size=(16, 16), mode="area").reshape(N, -1).cpu()
    sel = tight_neighborhood(squint, min(args.waypoints, N))
    tour = nn_tour(squint[sel])
    wp = [sel[t] for t in tour]
    wp = wp + [wp[0]]  # loop closure
    print(f"flow-morph {len(wp) - 1} segments x {args.per_seg} @ {size}px  ctrl {args.ctrl}")

    base = base_grid(size, size, dev)
    frames = []
    mid_first = None
    for s in range(len(wp) - 1):
        A, B = imgs[wp[s] : wp[s] + 1], imgs[wp[s + 1] : wp[s + 1] + 1]
        flow = optimize_flow(
            A, B, base, ctrl=args.ctrl, steps=args.steps, lr=args.lr, smooth=args.smooth
        )
        for k in range(args.per_seg):
            t = k / args.per_seg
            fr = morph_frame(A, B, flow, t, base).detach()
            frames.append(fr.cpu())
            if s == 0 and k == args.per_seg // 2:
                mid_first = fr.cpu()

    fdir = f"outputs/{args.tag}_frames"
    os.makedirs(fdir, exist_ok=True)
    seq = torch.cat(frames, dim=0)
    for i in range(seq.shape[0]):
        to_pil(seq[i]).save(f"{fdir}/f{i:04d}.png")
    to_pil(mid_first[0]).save(f"outputs/{args.tag}_midcheck.png")
    save_gif(seq, f"outputs/{args.tag}.gif", 20)
    print(
        f"{seq.shape[0]} frames  jitter={jitter(seq):.3f}  -> {fdir}/, {args.tag}.gif, {args.tag}_midcheck.png"
    )


if __name__ == "__main__":
    main()
