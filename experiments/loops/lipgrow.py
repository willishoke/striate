"""Grow objtube.py's textures instead of painting them: a learned reaction-diffusion rule on the lip.

A shell's pattern is laid down at its growing lip and recorded as the shell grows (Meinhardt, The
Algorithmic Beauty of Sea Shells). objtube.py's coordinates already are (around the lip, along the
growth), so the texture here is a ring of cells around each body's lip, carrying C chemicals, stepped
row by row along the body; each row's state, read out, is that row's colour. One rule is shared by every
body: each chemical diffuses at its own learned rate (the RD part), and a small network reacts them
locally, told also which way is up (around the ring) and how far along the body it is (positional
information, so the pattern can know its top from its flank). Each body has only a latent vector and
its own starting noise. The rule fills the whole surface, so there is no unpainted underside.

Stages:
  distill: fit the rule to a checkpoint's textures, blurred (the colour layout, not the dot positions,
           which a growth rule can't and needn't copy), over the texels a camera sees (texfill.py's keep.npy)
  then     objphys.py --grow grow.pt --train N: the rule trains through the renderer against the channel

usage: lipgrow.py --load tube.pt --keep keep.npy --out grow.pt [--steps 1500]
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LipGrow(nn.Module):
    def __init__(self, K, Ha=64, Wl=144, C=12, Z=8, hid=64, sub=2, warm=24, seed=0):
        super().__init__()
        self.cfg = dict(K=K, Ha=Ha, Wl=Wl, C=C, Z=Z, hid=hid, sub=sub, warm=warm, seed=seed)
        self.Ha, self.Wl, self.C, self.sub, self.warm = Ha, Wl, C, sub, warm
        g = torch.Generator().manual_seed(seed)
        self.z = nn.Parameter(torch.randn(K, Z, generator=g) * 0.5)
        self.register_buffer("n0", torch.randn(K, C, Ha, generator=g))
        self.register_buffer("nr", torch.randn(K, 1, Ha, Wl + warm, generator=g))
        self.noise = nn.Parameter(torch.tensor(0.05))
        ang = 2 * math.pi * torch.arange(Ha) / Ha  # 0: the body's transported "up" (objtube.py's tu = 0)
        self.register_buffer("pos", torch.stack([torch.cos(ang), torch.sin(ang)]))
        self.logD = nn.Parameter(torch.linspace(-3.0, 1.0, C))  # per chemical: slow activators, fast inhibitors
        self.zin = nn.Linear(Z, C)
        self.f1 = nn.Conv1d(3 * C + Z + 3, hid, 1)
        self.f2 = nn.Conv1d(hid, C, 1)
        nn.init.normal_(self.f2.weight, std=0.01)
        nn.init.zeros_(self.f2.bias)
        self.read = nn.Conv1d(C + Z + 2, 3, 1)

    def step(self, s, zk, tv, nz):
        lap = s.roll(1, -1) + s.roll(-1, -1) - 2 * s
        grd = (s.roll(-1, -1) - s.roll(1, -1)) / 2
        K, _, Ha = s.shape
        x = torch.cat([s, lap, grd, zk, self.pos[None].expand(K, -1, -1), tv.expand(K, 1, Ha)], 1)
        D = torch.sigmoid(self.logD)[None, :, None] * 0.45  # explicit diffusion stays stable below 0.5
        s = s + D * lap + 0.2 * self.f2(F.relu(self.f1(x)))
        s = torch.cat([s[:, :1] + nz, s[:, 1:]], 1)
        return 3 * torch.tanh(s / 3)

    def rows(self, s, zk, j0, j1):
        out = []
        for j in range(j0, j1):
            tv = torch.tensor(max(j - self.warm, 0) / (self.Wl - 1), device=s.device).reshape(1, 1, 1)
            for _ in range(self.sub):
                s = self.step(s, zk, tv, self.noise * self.nr[:, :, :, j])
            if j >= self.warm:
                out.append(self.read(torch.cat([s, zk, self.pos[None].expand(s.shape[0], -1, -1)], 1)))
        return s, out

    def forward(self, size=None, chunk=12):
        """Texture logits (K, 3, Ha, Wl), or resampled to size=(around, along) (wrapping around)."""
        K = self.z.shape[0]
        zk = self.z[:, :, None].expand(-1, -1, self.Ha)
        s = self.zin(self.z)[:, :, None] + 0.5 * self.n0
        cols = []
        J = self.Wl + self.warm
        for j0 in range(0, J, chunk):
            j1 = min(j0 + chunk, J)
            if torch.is_grad_enabled():  # recompute each chunk of rows in the backward pass
                s, c = checkpoint(self._chunk, s, zk, j0, j1, use_reentrant=False)
            else:
                s, c = self._chunk(s, zk, j0, j1)
            if c is not None:
                cols.append(c)
        tex = torch.cat(cols, -1)  # (K, 3, Ha, Wl)
        if size is not None and tuple(size) != (self.Ha, self.Wl):
            k = max(1, size[0] // self.Ha)
            p = torch.cat([tex[:, :, -1:], tex, tex[:, :, :1]], 2)
            up = F.interpolate(p, size=(size[0] + 2 * k, size[1]), mode="bilinear", align_corners=False)
            tex = up[:, :, k:k + size[0]]
        return tex

    def _chunk(self, s, zk, j0, j1):
        s, out = self.rows(s, zk, j0, j1)
        return s, (torch.stack(out, -1) if out else None)


def load_grow(path, dev):
    d = torch.load(path, map_location="cpu", weights_only=False)
    g = LipGrow(**d["cfg"]).to(dev)
    g.load_state_dict(d["state"])
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", required=True, help="objtube checkpoint whose textures to distill")
    ap.add_argument("--keep", required=True, help="texfill.py's keep.npy (which texels a camera sees)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--blur", type=int, default=4, help="compare the colour layout at 1/this resolution")
    ap.add_argument("--around", type=int, default=64)
    ap.add_argument("--along", type=int, default=144)
    ap.add_argument("--chem", type=int, default=12)
    a = ap.parse_args()
    from striate.util import pick_device
    dev = pick_device()
    ck = torch.load(a.load, map_location="cpu", weights_only=False)
    st = ck["tubes"]
    tex0, tex = st["tex0"].float(), st["tex"].float()
    K, _, Ha, Wl = tex0.shape
    k = Ha // tex.shape[2]
    p = torch.cat([tex[:, :, -1:], tex, tex[:, :, :1]], 2)
    up = F.interpolate(p, size=(Ha + 2 * k, Wl), mode="bilinear", align_corners=False)
    alb = torch.sigmoid(tex0 + up[:, :, k:k + Ha])
    keep = torch.from_numpy(np.load(a.keep).astype(np.float32))[:, None]
    tgt = F.adaptive_avg_pool2d(alb * keep, (a.around // a.blur, a.along // a.blur)).to(dev)
    wt = F.adaptive_avg_pool2d(keep, (a.around // a.blur, a.along // a.blur)).to(dev)
    tgt = tgt / wt.clamp(min=1e-3)
    g = LipGrow(K, a.around, a.along, a.chem).to(dev)
    opt = torch.optim.Adam(g.parameters(), lr=a.lr)
    t0 = time.time()
    for it in range(a.steps):
        for pg in opt.param_groups:
            pg["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * it / a.steps))
        y = torch.sigmoid(g())
        yb = F.avg_pool2d(y, a.blur)
        loss = (wt * (yb - tgt) ** 2).sum() / wt.sum()
        # keep some contrast at the fine scale (the blurred fit alone is happy with flat colour)
        hp = y - F.interpolate(yb, size=y.shape[-2:], mode="nearest")
        loss = loss - 0.02 * hp.pow(2).mean().clamp(max=0.01)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == a.steps - 1:
            print(f"[{time.time() - t0:.0f}s] step {it + 1}/{a.steps} loss {float(loss):.5f}", flush=True)
    torch.save(dict(cfg=g.cfg, state={k_: v.cpu() for k_, v in g.state_dict().items()}, distill=vars(a)), a.out)
    with torch.no_grad():
        y = torch.sigmoid(g()).cpu()
    from PIL import Image
    pick = list(range(0, K, max(1, K // 8)))[:8]
    alb_s = F.interpolate(alb, size=y.shape[-2:], mode="area")
    rows = []
    for X in (alb_s, y):
        rows.append(np.concatenate([np.pad(X[i].permute(1, 2, 0).numpy(), ((0, 4), (0, 4), (0, 0)), constant_values=1) for i in pick], 1))
    Image.fromarray((np.concatenate(rows, 0) * 255).round().clip(0, 255).astype(np.uint8)).save(a.out.replace(".pt", "_atlas.png"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
