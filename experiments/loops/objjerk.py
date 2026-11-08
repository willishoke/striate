"""Motion smoothness of objphys trajectories: speed, acceleration, jerk (px/frame^k), p99 and max; frames with pops."""
import sys, torch
for f in sys.argv[1:]:
    x = torch.load(f, weights_only=False)["traj"].double()
    v = (torch.roll(x, -1, 0) - x).norm(dim=-1)
    a = torch.roll(x, -1, 0) - 2 * x + torch.roll(x, 1, 0)
    j = (torch.roll(a, -1, 0) - a).norm(dim=-1); a = a.norm(dim=-1)
    pops = (a.amax((1, 2)) > 0.6).nonzero().flatten().tolist()
    print(f.split("/")[-1][-60:], "| speed p99 %.3f max %.3f | acc p99 %.3f max %.3f | jerk p99 %.4f max %.3f | pop frames %s" % (
        v.quantile(.99), v.max(), a.flatten().quantile(.99), a.max(), j.flatten().quantile(.99), j.max(), pops))
