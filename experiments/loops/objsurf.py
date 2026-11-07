"""Surface field (gate test, static): the shells as lit 2.5D surfaces instead of cut-out pixels.

objfield.py cuts the still into sprites, and the still's lighting comes along: rim darkening and
contact shadows are painted into each sprite, grow and shrink with it, and the cut lands mid-shadow.
Here a shell is geometry plus an unlit texture, and the light is computed per frame:

  sprite k  (same windows, outlines and depth order as objfield.py, from segobj/objgraph/objdepth)
            thickness  rounded edge (radius --rho) + dome slope (--dome) * inner distance + a coarse
                    learned residual. A shell extends under a neighbour only where the depth vote is
                    sure (--min-vote); elsewhere the two sit side by side and a gap shows the floor.
            albedo  unlit colour logits: the still with our own shading divided out (--init deshade)
                    or each object's mean colour (--init flat), plus a learned Fourier residual.
  frame     every pixel gathers its candidate sprites, sorts them by depth surface and composites
            albedo front to back (the same "over" as objfield.py). Height is the pile: a dark floor
            plus the thickness of every shell stacked there, so each shell rests on what is beneath.
  light     normals from the composited height gradient; Lambert + Blinn specular; soft cast shadows
            by marching the composited height field toward the light (penumbra widens with distance);
            ambient occlusion from how far the surface sits below its blurred surroundings. The canvas
            is a torus, so shadows and blurs wrap.

Training jitters every sprite by a few px each step, so shading that depends on the neighbours
(contact shadows, occlusion) can't be painted into the albedo: it would land in the wrong place. The
light is the loop's own (--train-light), so the albedo can be coordinated with it as the still's is;
--train-light random draws a new light every step instead. A rim term stops the albedo darkening
toward a shell's own outline, which jitter can't catch.

Gate question: under fixed held-out lights, how do the channel score and the look compare with the
still (8.735 at inception4c:7)?

usage: objsurf.py --image still.png --seg segdir --target inception4c:7 --out DIR
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
from scipy import ndimage as ndi
from striate.model import Backbone
from striate.param import FourierImage
from striate.util import pick_device
from zoo import MultiScaleCrop
from ring import Ring
from objfield import build_scene, gather_bilinear


def shift(x, sy, sx):
    """x (S, S) sampled at (y + sy, x + sx) on the torus, bilinear; sy, sx python floats."""
    iy, ix = math.floor(sy), math.floor(sx)
    fy, fx = sy - iy, sx - ix
    r = lambda dy, dx: torch.roll(x, (-(iy + dy), -(ix + dx)), (0, 1))
    return (1 - fy) * ((1 - fx) * r(0, 0) + fx * r(0, 1)) + fy * ((1 - fx) * r(1, 0) + fx * r(1, 1))


def gauss_fft(x, sigma):
    """Periodic gaussian blur of x (..., S, S)."""
    S = x.shape[-1]
    f = torch.fft.fftfreq(S, device=x.device)
    g = torch.exp(-2 * (math.pi * sigma) ** 2 * (f[:, None] ** 2 + f[None, : S // 2 + 1] ** 2))
    return torch.fft.irfft2(torch.fft.rfft2(x) * g, s=(S, S))


class Light:
    """az (rad, image plane, 0 = light from +x), el (rad above the plane), ambient, diffuse, specular."""

    def __init__(self, az, el, amb=0.32, dif=0.78, spec=0.18, shin=24.0):
        self.az, self.el, self.amb, self.dif, self.spec, self.shin = az, el, amb, dif, spec, shin

    def vec(self):
        c = math.cos(self.el)
        return (c * math.cos(self.az), c * math.sin(self.az), math.sin(self.el))  # (x, y, z)

    @staticmethod
    def random(rng):
        return Light(rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(35, 75)), amb=rng.uniform(0.25, 0.42))


class Surf:
    def __init__(self, sc, z, still_bg, dev, a):
        self.dev, self.a = dev, a
        self.K, self.H = sc["K"], sc["H"]
        K1 = self.K + 1
        self.sdf0 = torch.from_numpy(sc["sdf"]).to(dev)
        self.depth = torch.from_numpy(sc["depth"]).to(dev)
        self.color0 = torch.from_numpy(sc["color"]).to(dev)
        self.ck = torch.from_numpy(sc["ck"]).to(dev)
        self.cq = torch.from_numpy(sc["cq"]).to(dev)
        self.valid = (self.ck > 0).float()
        self.S = int(round(math.sqrt(self.ck.shape[0])))
        self.bg = still_bg.to(dev)  # (3, P) unlit colour of the farthest layer
        g = max(4, self.H // 4)
        self.alpha = torch.nn.Parameter(torch.zeros(K1, 1, g, g, device=dev))  # outline offset, px
        self.hres = torch.nn.Parameter(torch.zeros(K1, 1, g, g, device=dev))  # height residual, px
        self.res = FourierImage(self.H, self.H, batch=K1, device=dev)
        with torch.no_grad():
            self.res.spectrum.zero_()

    def up(self, p):
        return F.interpolate(p, size=(self.H, self.H), mode="bicubic", align_corners=False)

    def sprites(self):
        """(K+1, 6, H, H): coverage, depth surface, thickness, albedo logits (3).

        A shell is a thickness, not an absolute height: it rests on whatever is beneath it (see
        composite), so its edge steps up from the surface below and casts a shadow onto it. With
        absolute heights a front shell's rounded edge fell to its own base at the outline, below the
        shell behind, and the outline became a ditch: dark, with no drop shadow."""
        a = self.a
        sdf = self.sdf0[:, None] + self.up(self.alpha)
        cov = torch.sigmoid(-sdf / a.edge_w)
        d = torch.relu(-sdf)  # inner distance
        edge = a.rho * torch.sqrt((1 - (1 - (d / a.rho).clamp(max=1)) ** 2).clamp(min=1e-4))
        t = torch.relu(edge + a.dome * d + self.up(self.hres)) * (d > 0)
        col = self.color0 + self.res.preact()
        self._rim = (d, col)
        return torch.cat([cov, self.depth[:, None], t, col], 1)

    def rim_loss(self):
        """A shell's colour doesn't darken toward its own outline: near the edge (inside 3 px) its
        luminance may not fall below 75% of its interior's. Darkness at a contact is the light's job;
        left free, the network paints a black outline round every shell, since that helps under any
        light and moves with the shell."""
        d, col = self._rim
        y = (torch.sigmoid(col) * torch.tensor([0.299, 0.587, 0.114], device=self.dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
        inner = (d > 6).float()
        ref = (y * inner).sum((2, 3), keepdim=True) / inner.sum((2, 3), keepdim=True).clamp(min=1)
        band = ((d > 0) & (d < 3)).float()
        return ((torch.relu(0.75 * ref.detach() - y) ** 2) * band)[1:].sum() / band[1:].sum().clamp(min=1) / 0.01

    def composite(self, jit=None):
        """Front-to-back over. jit (B, K+1, 2) px per sprite or None. Returns dict of (B, C, S, S)."""
        B = 1 if jit is None else jit.shape[0]
        spr = self.sprites()
        src = self.cq.expand(B, *self.cq.shape)
        if jit is not None:
            src = src - jit[:, self.ck]  # (B, P, M, 2)
        v = gather_bilinear(spr, self.ck.expand(B, *self.ck.shape), src)  # (B, P, M, 6)
        al = v[..., 0] * self.valid
        pile = (al * v[..., 2]).sum(-1)  # floor plus the thickness of every shell stacked at this pixel
        z = torch.where(self.valid > 0, v[..., 1], torch.full_like(v[..., 1], -9.0))
        order = torch.argsort(z, -1, descending=True)
        al = torch.gather(al, -1, order)
        q = torch.sigmoid(torch.gather(v[..., 3:], -2, order[..., None].expand(*order.shape, 3)))
        T = torch.cumprod(torch.cat([torch.ones_like(al[..., :1]), 1 - al], -1), -1)
        w = T[..., :-1] * al
        alb = (w[..., None] * q).sum(-2) + T[..., -1:] * self.bg.T[None]
        alb = alb.permute(0, 2, 1).reshape(B, 3, self.S, self.S)
        # shells are rigid: the top one rests on a smoothed version of what is beneath it, rather than
        # draping over the ends of hidden extensions (which raised ridges inside every shell)
        tg = torch.gather(v[..., 2], -1, order)
        top = (w * tg).sum(-1)
        beneath = (pile - top).reshape(B, self.S, self.S)
        h = top.reshape(B, self.S, self.S) + gauss_fft(beneath, self.a.rest_blur)
        gy = (torch.roll(h, -1, 1) - torch.roll(h, 1, 1)) / 2
        gx = (torch.roll(h, -1, 2) - torch.roll(h, 1, 2)) / 2
        k = (self.a.gmax / torch.sqrt(gy**2 + gx**2 + 1e-8)).clamp(max=1.0)  # cap the slope at outlines
        return dict(h=h, gy=gy * k, gx=gx * k, alb=alb)

    def shade(self, c, lights, parts=False):
        """Lit frames (B, 3, S, S) from a composite and one Light per batch item."""
        a = self.a
        h, gy, gx, alb = c["h"], c["gy"], c["gx"], c["alb"]
        n = torch.stack([-gx, -gy, torch.ones_like(gx)], 1)
        n = n / n.norm(dim=1, keepdim=True)
        # ambient occlusion: how far the surface sits below its surroundings, at three radii
        occ = sum(torch.relu(gauss_fft(h, r) - h) / r for r in (2.0, 5.0, 12.0))
        ao = torch.exp(-a.ao * occ)
        outs, sh_all = [], []
        for b, L in enumerate(lights):
            lx, ly, lz = L.vec()
            ndl = torch.relu(n[b, 0] * lx + n[b, 1] * ly + n[b, 2] * lz)
            # soft cast shadow: march toward the light over the height field
            vis = torch.ones_like(h[b])
            if L.el < math.radians(89):
                ux, uy = math.cos(L.az), math.sin(L.az)
                tan_e = math.tan(L.el)
                for t in np.arange(1.0, a.march + 1e-6, a.march_step):
                    occl = shift(h[b], t * uy, t * ux) - h[b] - t * tan_e
                    vis = vis * torch.sigmoid(-occl / (a.pen0 + a.pen * t))
            hx, hy, hz = lx, ly, lz + 1.0
            hn = math.sqrt(hx * hx + hy * hy + hz * hz)
            ndh = torch.relu((n[b, 0] * hx + n[b, 1] * hy + n[b, 2] * hz) / hn)
            expo = (L.amb + L.dif) / (L.amb + L.dif * lz)  # exposure: a flat top reads the same under any sun
            shade = (L.amb * ao[b] + L.dif * ndl * vis) * expo
            spec = L.spec * ndh**L.shin * vis
            outs.append(alb[b] * shade + spec)
            sh_all.append(torch.stack([shade, spec, vis, ao[b]]))
        img = torch.stack(outs).clamp(0, 1)
        return (img, torch.stack(sh_all)) if parts else img

    def render(self, lights, jit=None):
        return self.shade(self.composite(jit), lights)

    def render_ss(self, light, jit):
        """One frame (1, 3, S, S), 2x2 supersampled: the sprites are sampled at four quarter-pixel
        offsets, each sample lit on its own, and the four averaged, so silhouettes are antialiased."""
        acc = 0
        for oy in (-0.25, 0.25):
            for ox in (-0.25, 0.25):
                o = torch.tensor([oy, ox], device=self.dev)
                j = jit + o * (torch.arange(self.K + 1, device=self.dev) > 0).float()[None, :, None]
                acc = acc + self.render([light], j)
        return acc / 4


def deshade(sf, still, light, lo=0.5):
    """Albedo canvas: the still with our own shading (geometry as initialized, one light) divided out."""
    with torch.no_grad():
        c = sf.composite()
        c["alb"] = torch.ones_like(c["alb"])
        _, parts = sf.shade(c, [light], parts=True)
        shade, spec = parts[0, 0], parts[0, 1]
        alb = ((still[0] - spec) / shade.clamp(min=lo)).clamp(1 / 512, 1 - 1 / 512)
    return alb, shade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--labels", default="labels_display.npy")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--net", default="googlenet")
    ap.add_argument("--pin", default="38-55,73-74")
    ap.add_argument("--soft", default="64,65,75,77:0.5")
    ap.add_argument("--ext", type=float, default=14)
    ap.add_argument("--reach", type=float, default=16)
    ap.add_argument("--smooth", type=float, default=2.5)
    ap.add_argument("--edge-w", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--wall-sigma", type=float, default=6.0)
    ap.add_argument("--fill", default="mirror")
    ap.add_argument("--inset", type=float, default=1.5)
    # geometry and light
    ap.add_argument("--rho", type=float, default=7.0, help="radius of the rounded edge, px")
    ap.add_argument("--dome", type=float, default=0.25, help="height gained per px inward past the edge")
    ap.add_argument("--gmax", type=float, default=4.0, help="slope cap")
    ap.add_argument("--ao", type=float, default=0.6)
    ap.add_argument("--march", type=float, default=24.0, help="shadow march length, px")
    ap.add_argument("--march-step", type=float, default=1.5)
    ap.add_argument("--pen0", type=float, default=0.4, help="shadow softness at contact, px of height")
    ap.add_argument("--pen", type=float, default=0.12, help="penumbra growth per px of distance")
    ap.add_argument("--init", default="deshade", choices=["deshade", "flat", "still"])
    ap.add_argument("--jitter", type=float, default=3.5, help="per-sprite position jitter in training, px rms")
    ap.add_argument("--min-vote", type=float, default=0.4, help="a shell extends under a neighbour only if the depth vote is this sure")
    ap.add_argument("--train-light", default="135,50", help="az,el in deg for every training step, or 'random'")
    ap.add_argument("--floor-dark", type=float, default=2.5, help="logit shift that darkens the floor under the pile")
    ap.add_argument("--rest-blur", type=float, default=10.0, help="smoothing of the surface a shell rests on, px")
    ap.add_argument("--wrim", type=float, default=1.0, help="no darkening toward a shell's own outline")
    # training
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--lr-res", type=float, default=0.01)
    ap.add_argument("--lr-h", type=float, default=0.03)
    ap.add_argument("--lr-alpha", type=float, default=0.02)
    ap.add_argument("--wh", type=float, default=0.2, help="height residual stays small (per (3 px)^2)")
    ap.add_argument("--wopq", type=float, default=0.5)
    ap.add_argument("--wfid", type=float, default=0.0, help="overhead-lit render against the still")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", help="surf.pt of a trained run: skip training")
    ap.add_argument("--anim", type=int, default=0, help="also render a loop of this many frames: every shell drifts")
    ap.add_argument("--anim-amp", type=float, default=2.5, help="drift per shell in the loop, px rms")
    ap.add_argument("--anim-light", type=float, default=135.0, help="light azimuth for the loop, deg (elevation 50)")
    ap.add_argument("--anim-seed", type=int, default=11)
    a = ap.parse_args()
    dev = pick_device()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed + 1)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:.0f}s] {s}", flush=True)
    save = lambda x, p: Image.fromarray((x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(p)

    L = np.load(f"{a.seg}/{a.labels}").astype(np.int64)
    S = L.shape[0]
    img = np.asarray(Image.open(a.image).convert("RGB")).astype(np.float32) / 255
    lg = lambda im: np.log(np.clip(im, 1 / 512, 1 - 1 / 512) / (1 - np.clip(im, 1 / 512, 1 - 1 / 512))).transpose(2, 0, 1).astype(np.float32)
    dj = json.load(open(f"{a.seg}/depth.json"))
    votes = [(v["i"], v["j"], v["vote"]) for v in dj["votes"]]
    aff = {(q["i"], q["j"]): q["affinity"] for q in json.load(open(f"{a.seg}/graph.json"))["pairs"]}
    pins = {tuple(sorted(map(int, s.split("-")))) for s in a.pin.split(",") if s}
    soft = {}
    for grp in a.soft.split(";"):
        if grp:
            ids, c = grp.split(":")
            for i in ids.split(","):
                soft[int(i)] = float(c)
    still = torch.from_numpy(img.transpose(2, 0, 1))[None].to(dev)
    overhead = Light(0.0, math.radians(90))

    def scene(im):
        return build_scene(L, lg(im), votes, aff, pins, soft, a)

    def bg_of(im):
        l = lg(im)
        b = np.stack([ndi.gaussian_filter(l[c], 8, mode="wrap") for c in range(3)]) - a.floor_dark
        return torch.from_numpy(1 / (1 + np.exp(-b))).reshape(3, -1)

    sc = scene(img)
    sf = Surf(sc, dj["z"], bg_of(img), dev, a)
    log(f"{sc['K']} sprites of {sc['H']} px, up to {sc['M']} per pixel")
    if a.init == "deshade":
        alb, shade0 = deshade(sf, still, overhead)
        alb_np = alb.permute(1, 2, 0).cpu().numpy()
        save(shade0[None].expand(3, -1, -1) / shade0.max(), f"{a.out}/shade0.png")
        sc = scene(alb_np)
        sf = Surf(sc, dj["z"], bg_of(alb_np), dev, a)
        log(f"deshaded: shading at init {float(shade0.min()):.2f}..{float(shade0.max()):.2f}, mean {float(shade0.mean()):.2f}")
    elif a.init == "flat":
        flat = np.zeros_like(img)
        for k in range(1, int(L.max()) + 1):
            flat[L == k] = img[L == k].mean(0)
        sc = scene(flat)
        sf = Surf(sc, dj["z"], bg_of(flat), dev, a)

    layer, ch = a.target.split(":")[0], int(a.target.split(":")[1])
    net = Backbone(a.net, dev)
    net.hook(layer)
    rg = Ring(net, layer, ch, S, dev)

    def score(fr):
        state = random.getstate()
        tot = 0
        for k in range(4):
            random.seed(1000 + k)
            tot = tot + rg.act(MultiScaleCrop(224 / S, 512)(fr))
        random.setstate(state)
        return tot / 4

    held = [overhead] + [Light(math.radians(az), math.radians(50)) for az in range(0, 360, 45)]

    def evaluate(tag):
        with torch.no_grad():
            c = sf.composite()
            acts = []
            for L_ in held:
                fr = sf.shade(c, [L_])
                acts.append(float(score(fr)))
        log(f"{tag}: overhead {acts[0]:.3f}; 8 lights at 50 deg mean {np.mean(acts[1:]):.3f} [{min(acts[1:]):.3f}, {max(acts[1:]):.3f}]")
        return acts

    with torch.no_grad():
        alone = float(score(still))
    log(f"still {alone:.3f}")
    acts0 = evaluate("init")

    opt = torch.optim.Adam([
        {"params": [sf.res.spectrum], "lr": a.lr_res},
        {"params": [sf.hres], "lr": a.lr_h},
        {"params": [sf.alpha], "lr": a.lr_alpha},
    ])
    base_lr = [g["lr"] for g in opt.param_groups]
    curve = []
    if a.load:
        ck = torch.load(a.load)
        with torch.no_grad():
            sf.res.spectrum.copy_(ck["res"].to(dev))
            sf.hres.copy_(ck["hres"].to(dev))
            sf.alpha.copy_(ck["alpha"].to(dev))
        a.steps = 0
        log(f"loaded {a.load}")
    for it in range(a.steps):
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * 0.5 * (1 + math.cos(math.pi * it / a.steps))
        opt.zero_grad()
        if a.train_light == "random":
            lights = [Light.random(rng) for _ in range(a.mb)]
        else:
            az, el = map(float, a.train_light.split(","))
            lights = [Light(math.radians(az), math.radians(el))] * a.mb
        jit = torch.randn(a.mb, sf.K + 1, 2, device=dev) * a.jitter
        jit = jit.clamp(-(a.reach - 2), a.reach - 2)
        jit[:, 0] = 0
        fr = sf.render(lights, jit)
        act = rg.act(MultiScaleCrop(224 / S, 512)(fr))
        hreg = (sf.hres[1:] ** 2).mean() / 9.0
        opq = (sf.alpha[1:] ** 2).mean() / 4.0
        rim = sf.rim_loss()  # sprites() ran inside render, so this reads that pass
        pen = a.wh * hreg + a.wopq * opq + a.wrim * rim
        if a.wfid > 0:
            pen = pen + a.wfid * ((sf.render([overhead]) - still) ** 2).mean() / 0.01
        loss = (-act + act.detach().abs() * pen).sum()
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == a.steps - 1:
            curve.append((it, float(act.mean())))
            log(f"step {it + 1}/{a.steps} act {act.mean():.3f} hreg {float(hreg):.3f} opq {float(opq):.3f} rim {float(rim):.3f} "
                f"h rms {float(sf.hres[1:].pow(2).mean().sqrt()):.2f}px")

    acts = evaluate("trained")
    with torch.no_grad():
        c = sf.composite()
        tiles = [sf.shade(c, [L_])[0] for L_ in held]
        grid = torch.cat([torch.cat(tiles[r * 3:(r + 1) * 3], 2) for r in range(3)], 1)
        save(grid, f"{a.out}/lights.png")  # overhead, then az 0, 45, ... 315 at 50 deg
        save(c["alb"][0], f"{a.out}/albedo.png")
        cw = dict(c, alb=torch.ones_like(c["alb"]))
        lit, parts = sf.shade(cw, [held[3]], parts=True)
        save(lit[0], f"{a.out}/shading.png")
        hh = c["h"][0]
        save(((hh - hh.min()) / (hh.max() - hh.min()))[None].expand(3, -1, -1), f"{a.out}/height.png")
        os.makedirs(f"{a.out}/orbit", exist_ok=True)
        for i in range(48):  # light orbit, objects still: the shadows should swing round the shells
            save(sf.shade(c, [Light(2 * math.pi * i / 48, math.radians(45))])[0], f"{a.out}/orbit/f{i:04d}.png")
        os.makedirs(f"{a.out}/jitter", exist_ok=True)
        g = torch.Generator().manual_seed(7)
        for i in range(4):  # same light, every shell moved by a few px: contact shadows should follow
            jit = (torch.randn(1, sf.K + 1, 2, generator=g) * 4).clamp(-(a.reach - 2), a.reach - 2).to(dev)
            jit[:, 0] = 0
            save(sf.render([held[3]], jit)[0], f"{a.out}/jitter/f{i:04d}.png")
    if a.anim:
        # each shell drifts on its own closed path (two harmonics, random phases); the light stays put
        os.makedirs(f"{a.out}/anim", exist_ok=True)
        g = torch.Generator().manual_seed(a.anim_seed)
        cf = torch.randn(sf.K + 1, 2, 2, 2, generator=g) * torch.tensor([1.0, 0.5]).view(1, 2, 1, 1)
        cf[0] = 0
        u = torch.arange(a.anim) / a.anim
        ang = 2 * math.pi * u[:, None] * torch.tensor([1.0, 2.0])[None]  # (T, 2 harmonics)
        path = (cf[None, :, :, 0] * torch.cos(ang)[:, None, :, None] + cf[None, :, :, 1] * torch.sin(ang)[:, None, :, None]).sum(2)
        path = path * a.anim_amp / path[:, 1:].pow(2).sum(-1).mean().sqrt()
        Lt = Light(math.radians(a.anim_light), math.radians(50))
        with torch.no_grad():
            for t in range(a.anim):
                save(sf.render_ss(Lt, path[t:t + 1].to(dev).clamp(-(a.reach - 2), a.reach - 2))[0], f"{a.out}/anim/f{t:04d}.png")
        log(f"loop: {a.anim} frames, drift {a.anim_amp} px rms, supersampled")
    torch.save({"res": sf.res.spectrum.detach().cpu(), "hres": sf.hres.detach().cpu(), "alpha": sf.alpha.detach().cpu(),
                "args": vars(a)}, f"{a.out}/surf.pt")
    rep = dict(still=alone, init=acts0, trained=acts, lights=["overhead"] + [f"az{az} el50" for az in range(0, 360, 45)],
               curve=curve, args=vars(a), secs=round(time.time() - t0))
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    log(f"done: still {alone:.3f}; trained overhead {acts[0]:.3f}, lit mean {np.mean(acts[1:]):.3f}")


if __name__ == "__main__":
    main()
