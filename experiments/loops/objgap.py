"""Do pressed pairs come apart? Over a loop, for sphere pairs overlapping at rest, the signed change of
their overlap (negative: separating, a gap opening at their wall), mean and 1st percentile, plus the share
of rest-overlapping pairs that lose > 0.5 px."""
import sys, os, argparse, torch
from objtube import Mixed
from objphys import Pile
files = [os.path.abspath(f) for f in sys.argv[1:]]
pile = None
os.chdir("/Users/lawvere/striate/experiments/loops")
for f in files:
    tj = torch.load(f, weights_only=False)
    a = argparse.Namespace(**tj["args"]); a.contact_metric = "scaled"
    if pile is None:
        ck = torch.load(a.load, map_location="cpu", weights_only=False)
        tb = Mixed.from_state(ck["tubes"], argparse.Namespace(**ck["args"]), torch.device("cpu"))
        with torch.no_grad(): s0 = tb.spheres()
        pile = Pile(s0["C"][0], s0["r"], s0["f"], a, 512, lambda s: None)
    K, N = pile.K, pile.N
    ff = pile.f[:, None].expand(K, N).reshape(-1)
    sel = pile.pen0 > 0.5
    ch = []
    for t in range(tj["traj"].shape[0]):
        ov = pile.overlap(tj["traj"][t].double().reshape(-1, 3), tj["rads"][t].double().reshape(-1), ff, pile.pi, pile.pj)[0]
        _, rq = pile.rest(float(t))
        ov0 = pile.overlap(pile.rest(float(t))[0].reshape(-1, 3), rq.reshape(-1), ff, pile.pi, pile.pj)[0]  # what the drive alone would give
        ch.append(ov[sel])
    ov = torch.stack(ch); ch = ov - pile.pen0[sel]
    print(f.split("/")[-1][-28:].ljust(28), "rest-pressed pairs: change mean %.3f | apart (overlap<0) %.4f, gap>0.5px %.4f" % (ch.mean(), (ov < 0).float().mean(), (ov < -0.5).float().mean()))
