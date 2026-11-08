"""Each body's underside against its own top: the paint's luminance (albedo, before lighting) where
the body owns the pixel, seen from straight below (through the floor) and from straight above, at
rest, lights turning with the camera. Prints the share of underside pixels darker than 0.15, how
many bodies are mostly that dark underneath, and how far each body's underside luminance stats
(mean, std, quantiles) sit from its top's. body_lum_stats is also objphys.py's --wmatch term.
usage: objunder.py [--cpu] [--png out.png] ckpt..."""
import os, sys, math, json, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import numpy as np, torch

LUM = (0.299, 0.587, 0.114)
QS = (0.1, 0.5, 0.9)


def body_lum_stats(Y, w, obj, K, qs=QS):
    """Per body, of the pair luminances Y weighted by w: (K, 2 + len(qs)) = mean, std, weighted
    quantiles, and (K,) the total weight. Fixed shapes throughout (one sort of every pair, keyed by
    body then luminance), so padded pair lists don't make MPS cache a graph per step; the quantiles
    pass gradient to the pairs they pick."""
    dev = Y.device
    z = lambda: torch.zeros(K, device=dev)
    W = z().index_add_(0, obj, w)
    Wc = W.clamp(min=1e-6)
    m = z().index_add_(0, obj, w * Y) / Wc
    sd = (z().index_add_(0, obj, w * (Y - m[obj]) ** 2) / Wc + 1e-6).sqrt()
    order = torch.argsort(obj.float() * 4 + Y.detach().clamp(0, 1))
    ws, Ys, os_ = w[order], Y[order], obj[order]
    cdf = (ws.cumsum(0) - (W.cumsum(0) - W)[os_]) / Wc[os_]  # within each body, inclusive
    n = z().index_add_(0, obj, torch.ones_like(w))
    start = n.cumsum(0) - n
    out = [m, sd]
    for q in qs:
        c = z().index_add_(0, os_, (cdf < q).float())
        idx = (start + torch.minimum(c, (n - 1).clamp(min=0))).long().clamp(0, len(Ys) - 1)
        out.append(Ys[idx])
    return torch.stack(out, -1), W


def lum(alb):
    return (alb * torch.tensor(LUM, device=alb.device)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--png", help="albedo from above | from below | lit from below, side by side (first ckpt)")
    ap.add_argument("--json")
    ap.add_argument("--floor", type=float, default=0.15)
    ap.add_argument("--minw", type=float, default=30.0, help="bodies owning fewer px than this in a view are left out")
    a = ap.parse_args()
    from objtube import Mixed, Rig, Cam, build_pairs, render, rot_x
    from striate.util import pick_device
    dev = torch.device("cpu") if a.cpu else pick_device()
    S = 512
    res = {}
    for path in a.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ta = argparse.Namespace(**ck["args"]); ta.mip = 1; ta.chunk = 40000
        tb = Mixed.from_state(ck["tubes"], ta, dev); rig = Rig(0., .8, ta, dev); rig.load_state_dict(ck["rig"])
        K = tb.K
        with torch.no_grad():
            sp = tb.spheres(); zm = float(sp["C"][0][..., 2].mean())
            st, ims = {}, []
            for nm, th in (("top", 0.0), ("below", math.pi)):
                cam = Cam(rot_x(th), S, S, (S / 2, S / 2, zm), S); cam.env = True
                o = render(tb, sp, cam, build_pairs(sp, cam, 4.0), [rig()], ta, ground="none", aux=True)
                own = o["ep"][0] / o["den"][0][o["pix"]]
                Y = lum(o["alb"][0])
                s_, W = body_lum_stats(Y, own, o["obj"], K)
                dk = torch.zeros(K, device=dev).index_add_(0, o["obj"], own * (Y < a.floor).float()) / W.clamp(min=1e-6)
                Yl = lum(o["img"][0].reshape(3, -1).T)[o["pix"]]  # lit: what the eye sees of the same pairs
                sl_, _ = body_lum_stats(Yl, own, o["obj"], K)
                dl = torch.zeros(K, device=dev).index_add_(0, o["obj"], own * (Yl < a.floor).float()) / W.clamp(min=1e-6)
                st[nm] = dict(stats=s_.cpu(), W=W.cpu(), dark=dk.cpu(), dark_all=float((own * (Y < a.floor).float()).sum() / own.sum()),
                              lit=sl_.cpu(), litdark=dl.cpu(), litdark_all=float((own * (Yl < a.floor).float()).sum() / own.sum()))
                if a.png and path == a.ckpts[0]:
                    am = torch.zeros(S * S, 3, device=dev).index_add_(0, o["pix"], own[:, None] * o["alb"][0])
                    ims.append(am.T.reshape(3, S, S))
                    if nm == "below":
                        ims.append(o["img"][0])
        ok = (st["top"]["W"] > a.minw) & (st["below"]["W"] > a.minw)
        d = (st["below"]["stats"] - st["top"]["stats"])[ok]
        r = dict(below_dark=st["below"]["dark_all"], top_dark=st["top"]["dark_all"],
                 bodies=int(ok.sum()), bodies_half_dark_below=int((st["below"]["dark"][ok] > 0.5).sum()),
                 bodies_half_dark_top=int((st["top"]["dark"][ok] > 0.5).sum()),
                 mean_below=float(st["below"]["stats"][ok, 0].mean()), mean_top=float(st["top"]["stats"][ok, 0].mean()),
                 rms_gap=[float(x) for x in d.pow(2).mean(0).sqrt()],
                 lit_below_dark=st["below"]["litdark_all"], lit_top_dark=st["top"]["litdark_all"],
                 lit_mean_below=float(st["below"]["lit"][ok, 0].mean()), lit_mean_top=float(st["top"]["lit"][ok, 0].mean()),
                 bodies_half_litdark_below=int((st["below"]["litdark"][ok] > 0.5).sum()))
        worst = torch.argsort(st["below"]["dark"] * ok.float(), descending=True)[:8].tolist()
        r["worst"] = [(k, round(float(st["below"]["dark"][k]), 3), round(float(st["top"]["stats"][k, 0]), 3), round(float(st["below"]["stats"][k, 0]), 3)) for k in worst]
        res[path] = r
        print(path.split("/")[-2], f"dark<{a.floor}: below {r['below_dark']:.1%} top {r['top_dark']:.1%}; bodies >50% dark below "
              f"{r['bodies_half_dark_below']}/{r['bodies']} (top {r['bodies_half_dark_top']}); mean lum below {r['mean_below']:.3f} top "
              f"{r['mean_top']:.3f}; rms gap (mean, std, q10, q50, q90) " + " ".join(f"{x:.3f}" for x in r["rms_gap"]), flush=True)
        print(f"  lit: dark<{a.floor} below {r['lit_below_dark']:.1%} top {r['lit_top_dark']:.1%}; mean below {r['lit_mean_below']:.3f} "
              f"top {r['lit_mean_top']:.3f}; bodies >50% dark lit below {r['bodies_half_litdark_below']}", flush=True)
        print("  darkest undersides (body, dark share, top mean, below mean):", r["worst"], flush=True)
        if ims:
            from PIL import Image
            img = torch.cat(ims, 2)
            Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(a.png)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
