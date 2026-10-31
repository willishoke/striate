"""Hold cell areas without jitter: solve the weights at every frame with a long Newton solve, then fit each slot's
correction (solved minus path weights) with K Fourier harmonics over the frames where it is alive, and save it into
the tissue. Rendering with solve 0 then uses smooth weights that hold the target areas.
Usage: tissue_wfit.py RUNDIR OUTDIR SOLVE K   (RUNDIR from cells.py)"""

import os, sys, json, time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch, numpy as np
from cells import *

dev = torch.device("mps")
run, out, n, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(out, exist_ok=True)
d = torch.load(run + "/tissue.pt")
S = T = 1024
tis = Tissue(d["N"], d["p"], S, dev, L=d["life"], tau=d["tau"])
tis.init_paths(torch.zeros(d["N"], 2), torch.zeros(d["N"], 2), torch.zeros(d["N"]))
for P_, k in ((tis.s, "s"), (tis.w, "w"), (tis.o, "o")):
    P_.c0.data.copy_(d[k][0])
    P_.cc.data.copy_(d[k][1])
tis.damp = 1.0
tex = torch.zeros(1, 3, S, S, device=dev)
t0 = time.time()
D, Al = [], []
with torch.no_grad():
    for i in range(0, T, 4):
        ts = torch.arange(i, i + 4, device=dev, dtype=torch.float32)
        tis.solve = 0
        w0 = tis.state(ts, T)["w"]  # path weights incl. the extrusion sink
        tis.solve = n
        _, st, _, _ = tis.render(ts, T, tex, need_stats=False)
        D.append(((st["w"] - w0) / tis.W0).cpu())
        Al.append(st["alive"].cpu())
D = torch.cat(D).double()
Al = torch.cat(Al).double()  # (T, N)
# weights are only defined up to a constant per frame (the gauge): remove each frame's mean correction
D = D - (D * Al).sum(1, keepdim=True) / Al.sum(1, keepdim=True)
torch.save({"D": D, "Al": Al}, out + "/solved.pt")
u = torch.arange(T, dtype=torch.float64) / T
k = torch.arange(1, K + 1, dtype=torch.float64)
Bm = torch.cat(
    [
        torch.ones(T, 1, dtype=torch.float64),
        torch.cos(2 * np.pi * u[:, None] * k),
        torch.sin(2 * np.pi * u[:, None] * k),
    ],
    1,
)  # (T, 2K+1)
coef = torch.zeros(d["N"], 2 * K + 1, dtype=torch.float64)
for j in range(d["N"]):
    wj = Al[:, j]
    A_ = Bm * wj[:, None]
    lam = 1e-3 * torch.eye(2 * K + 1, dtype=torch.float64)
    coef[j] = torch.linalg.solve(A_.T @ Bm + lam, A_.T @ D[:, j])
fit = Bm @ coef.T
res = ((fit - D) * Al).pow(2).sum() / Al.sum()
c0 = coef[:, :1].float()
cc = torch.stack([coef[:, 1 : K + 1], coef[:, K + 1 :]], 2).float()[..., None]  # (N, K, 2, 1)
d["wcorr"] = [c0, cc]
torch.save(d, out + "/tissue.pt")
print(
    f"fit residual rms {float(res.sqrt()):.3f} cell areas (weights in A); correction rms {float((D * Al).pow(2).sum().div(Al.sum()).sqrt()):.3f} ({time.time() - t0:.0f}s)"
)
import shutil

shutil.copy(run + "/report.json", out + "/report.json")
