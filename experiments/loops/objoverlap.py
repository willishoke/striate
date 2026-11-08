"""Overlap past rest, by the radii drawn, over an objphys.py loop: how far bodies pass into each other
beyond what the still already has them overlap (rms, quantiles, max, share of contacts over 1 px).
usage: objoverlap.py traj.pt                 measure a saved trajectory
       objoverlap.py traj.pt key=val ...     re-simulate with those objphys.py args, save next to it, and measure
"""
import sys, os, argparse, json, time
src = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # checkpoints name their sources relative to here
import torch
from objtube import Mixed
from objphys import Pile
tj = torch.load(src, weights_only=False)
a = argparse.Namespace(**tj["args"])
def parse(v):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


for kv in sys.argv[2:]:
    k, v = kv.split("=")
    old = getattr(a, k, None)
    setattr(a, k, type(old)(v) if old is not None else parse(v))
ck = torch.load(a.load, map_location="cpu", weights_only=False)
ta = argparse.Namespace(**ck["args"])
tb0 = Mixed.from_state(ck["tubes"], ta, torch.device("cpu"))
with torch.no_grad():
    s0 = tb0.spheres()
log = lambda s: print(s, flush=True)
pile = Pile(s0["C"][0], s0["r"], s0["f"], a, 512, log)
if len(sys.argv) > 2:
    t0 = time.time()
    traj, rads, st = pile.run(log)
    log(f"sim {time.time() - t0:.0f}s")
    out = src.replace(".pt", "_" + "_".join(sys.argv[2:]) + ".pt")
    torch.save(dict(traj=traj, rads=rads, stats=st, args=vars(a)), out)
    log(f"saved {out}")
else:
    traj, rads = tj["traj"], tj["rads"]
K, N = pile.K, pile.N
ex, exq = [], []
for t in range(traj.shape[0]):
    p = traj[t].double().reshape(-1, 3)
    r = rads[t].double().reshape(-1)
    _, rq = pile.rest(float(t))
    rq = rq.reshape(-1)
    ff = pile.f[:, None].expand(K, N).reshape(-1)
    e = (pile.overlap(p, r, ff, pile.pi, pile.pj)[0] - pile.pen0).clamp(min=0)
    eq = (pile.overlap(p, rq, ff, pile.pi, pile.pj)[0] - pile.pen0).clamp(min=0)
    ex.append(e); exq.append(eq)
ex, exq = torch.stack(ex), torch.stack(exq)
res = dict(drawn=dict(rms=float(ex.pow(2).mean().sqrt()), p99=float(ex.flatten().quantile(0.99)), p999=float(ex.flatten().quantile(0.999)), max=float(ex.max()),
                      frac_over_1px=float((ex > 1).float().mean())),
           contact=dict(rms=float(exq.pow(2).mean().sqrt()), p99=float(exq.flatten().quantile(0.99)), max=float(exq.max())))
dev_rest = (traj - pile.C0[None].float()).norm(dim=-1)
res["disp"] = dict(mean=float(dev_rest.mean()), p99=float(dev_rest.flatten().quantile(0.99)))
print(json.dumps(res))
