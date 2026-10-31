"""Contour currents: a looping flow that runs along the level sets of the image's own coarse structure.

The velocity is u = rot grad(psi) with psi = the base image's luminance blurred at sigma px. Then
u . grad(psi) = 0: anything that is a function of psi is a steady state of the flow, so the coarse
layout (blobs, stripes, cell walls) stays where it is and keeps its shape, while finer detail slides
along it, the way grain runs along a board or water runs along a riverbed. u is divergence-free, so
nothing is compressed. The material sloshes along the contours: tau(t) = A sin(2 pi t / T), and the
frame at t is the base carried by the exact flow map Phi_{-tau(t)} (integrated with RK4 on a grid),
so the loop closes.

Drop-in for warploop2.Flow: d(t, res) returns displacements (N, 2, res, res) px with
frame_t(x) = base(x - d_t(x)).
"""

import math
import torch
import torch.nn.functional as F


def _sample(field, px, py, S):
    """Bilinear periodic sample of field (C, R, R) defined on an R grid over an S torus, at px, py (any shape, px)."""
    C, R = field.shape[0], field.shape[-1]
    gx, gy = px * (R / S), py * (R / S)
    x0, y0 = gx.floor(), gy.floor()
    fx, fy = gx - x0, gy - y0
    x0, y0 = x0.long() % R, y0.long() % R
    x1, y1 = (x0 + 1) % R, (y0 + 1) % R
    flat = field.reshape(C, R * R)
    g = lambda yy, xx: flat[:, (yy * R + xx).reshape(-1)].view(C, *px.shape)
    return (
        g(y0, x0) * (1 - fx) * (1 - fy)
        + g(y0, x1) * fx * (1 - fy)
        + g(y1, x0) * (1 - fx) * fy
        + g(y1, x1) * fx * fy
    )


class ContourFlow:
    def __init__(self, base_rgb, T, speed, sigma, dev, res=512, levels=48, phase_lag=0.0):
        """base_rgb (1, 3, S, S) in [0, 1]."""
        S = base_rgb.shape[-1]
        self.S, self.T, self.dev, self.res = S, T, dev, res
        L = (
            base_rgb[0] * torch.tensor([0.299, 0.587, 0.114], device=base_rgb.device).view(3, 1, 1)
        ).sum(0)
        # psi: luminance blurred at sigma (periodic, via FFT), on the res grid
        fy = torch.fft.fftfreq(S, device=L.device).view(-1, 1)
        fx = torch.fft.rfftfreq(S, device=L.device).view(1, -1)
        G = torch.exp(-2 * (math.pi * sigma) ** 2 * (fx * fx + fy * fy))
        Z = torch.fft.rfft2(L - L.mean()) * G
        # velocity = (dpsi/dy, -dpsi/dx), spectrally exact, then downsampled to res
        ux = torch.fft.irfft2(Z * (2j * math.pi * fy), s=(S, S))
        uy = torch.fft.irfft2(Z * (-2j * math.pi * fx), s=(S, S))
        u = torch.stack([ux, uy])[None]
        if res != S:
            u = F.interpolate(u, size=(res, res), mode="area")
        u = u[0]
        u = u / u.norm(dim=0).mean()  # mean speed 1 px per unit tau
        self.u = u.to(dev)
        self.umax = float(u.norm(dim=0).max())
        # amplitude so the mean displacement per frame is `speed`: mean |dtau/dt| = 4A/T
        self.A = speed * T / 4
        # flow maps at tau = j * A / levels, j = -levels..levels: displacement fields on the res grid
        self.levels = levels
        h = self.A / levels
        sub = max(
            1, math.ceil(h * self.umax / 0.25)
        )  # RK4 substeps so no step moves more than 0.25 px
        g = torch.arange(res, device=dev, dtype=torch.float32) * (S / res)
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        maps = {0: torch.zeros(2, res, res, device=dev)}
        for sgn in (1, -1):
            px, py = xx.clone(), yy.clone()
            for j in range(1, levels + 1):
                for _ in range(sub):
                    px, py = self._rk4(px, py, -sgn * h / sub)  # Phi_{-tau}: integrate backwards
                maps[sgn * j] = torch.stack(
                    [xx - px, yy - py]
                )  # d = x - Phi_{-tau}(x), unwrapped px
        self.maps = torch.stack(
            [maps[j] for j in range(-levels, levels + 1)]
        )  # (2L+1, 2, res, res)
        self.phase_lag = phase_lag
        d = self.d(torch.arange(T + 1, device=dev, dtype=torch.float32), res=64)
        self.speed_meas = float((d[1:] - d[:-1]).norm(dim=1).mean())
        self.max_disp = float(self.maps.norm(dim=1).max())
        self.strain = float(self._strain())
        self.drift = (0, 0)

    def _vel(self, px, py):
        v = _sample(self.u, px, py, self.S)
        return v[0], v[1]

    def _rk4(self, px, py, dt):
        k1x, k1y = self._vel(px, py)
        k2x, k2y = self._vel(px + 0.5 * dt * k1x, py + 0.5 * dt * k1y)
        k3x, k3y = self._vel(px + 0.5 * dt * k2x, py + 0.5 * dt * k2y)
        k4x, k4y = self._vel(px + dt * k3x, py + dt * k3y)
        return px + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x), py + dt / 6 * (
            k1y + 2 * k2y + 2 * k3y + k4y
        )

    def tau_index(self, t):
        """Fractional index into self.maps for times t."""
        tau = torch.sin(2 * math.pi * t.float() / self.T + self.phase_lag)  # in [-1, 1] units of A
        return (tau + 1) * self.levels

    def d(self, t, res=None):
        res = res or self.S
        q = self.tau_index(t)
        i0 = q.floor().long().clamp(0, 2 * self.levels - 1)
        w = (q - i0.float()).clamp(0, 1).view(-1, 1, 1, 1)
        m = self.maps[i0] * (1 - w) + self.maps[i0 + 1] * w  # (N, 2, R, R)
        if res != self.res:
            m = F.interpolate(m, size=(res, res), mode="bilinear", align_corners=False)
        return m

    def _strain(self):
        d = self.maps[::8]
        h = self.S / self.res
        gx = (torch.roll(d, -1, 3) - torch.roll(d, 1, 3)) / (2 * h)
        gy = (torch.roll(d, -1, 2) - torch.roll(d, 1, 2)) / (2 * h)
        return torch.cat([gx, gy], 1).norm(dim=1).amax()
