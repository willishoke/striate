"""Vector image parameterization: a stack of filled shapes with Fourier outlines.

A raster FourierImage is free to be fuzzy, and at the network's native scale it
usually is. Here the image can only be what an SVG can be: shapes painted back to
front, each a solid or linear-gradient fill inside a closed outline. Once the edge
softness anneals to zero every edge is crisp, and the result re-renders exactly at
any resolution.

Each shape is star-shaped about its center, with radius a Fourier series in angle:

    r(theta) = r0 * exp(sum_k (a_k cos k*theta + b_k sin k*theta) / k**decay)

The 1/k**decay scaling is FourierImage's frequency preconditioning moved onto the
outline: low harmonics (overall shape) move first, high ones (petals, spikes) last.

Rendering is analytic in the pixel coordinate, so the robustness transforms are
applied to the sampling grid (a random camera) rather than resampling a raster:
the network sees crisp edges at every step.
"""

import math
import random

import torch

from .param import color_correlation_normalized

# Raw parameters are optimized at unit scale with a single Adam lr; these factors
# map them to image units. Centers are in image heights (0..1 across a square).
_CENTER_SCALE = 0.1
_LOG_RADIUS_SCALE = 0.2
_HARMONIC_SCALE = 0.3


class VectorImage(torch.nn.Module):
    def __init__(
        self,
        shapes: int = 128,
        harmonics: int = 8,
        decay: float = 1.0,
        fill: str = "linear",
        learn_alpha: bool = False,
        decorrelate: bool = True,
        radius_range: tuple[float, float] = (0.03, 0.35),
        device: str = "cpu",
    ):
        super().__init__()
        if fill not in ("flat", "linear"):
            raise ValueError(f"fill must be 'flat' or 'linear', got {fill!r}")
        self.fill = fill
        self.learn_alpha = learn_alpha
        self.decorrelate = decorrelate
        k = shapes

        lo, hi = math.log(radius_range[0]), math.log(radius_range[1])
        # big shapes at the back, small in front, so nothing starts fully buried
        log_r0 = torch.empty(k).uniform_(lo, hi).sort(descending=True).values
        self.log_radius = torch.nn.Parameter(log_r0 / _LOG_RADIUS_SCALE)
        self.center = torch.nn.Parameter(torch.rand(k, 2) / _CENTER_SCALE)
        self.harm = torch.nn.Parameter(torch.randn(k, harmonics, 2) * 0.3)
        ncolors = 2 if fill == "linear" else 1
        self.color = torch.nn.Parameter(torch.randn(k, ncolors, 3) * 0.5)
        self.grad_angle = torch.nn.Parameter(torch.rand(k) * 2 * math.pi)
        self.alpha = torch.nn.Parameter(torch.full((k,), 2.0), requires_grad=learn_alpha)
        self.background = torch.nn.Parameter(torch.zeros(3))

        orders = torch.arange(1, harmonics + 1, dtype=torch.float32)
        self.register_buffer("harm_scale", _HARMONIC_SCALE / orders**decay)
        self.register_buffer("orders", orders)
        self.to(device)

    # --- geometry, in image units -------------------------------------------------

    def centers(self) -> torch.Tensor:
        return self.center * _CENTER_SCALE

    def radius(self, theta: torch.Tensor) -> torch.Tensor:
        """Outline radius of every shape at angles ``theta`` (..., K) -> (..., K)."""
        kt = theta[..., None] * self.orders  # (..., K, M)
        h = self.harm * self.harm_scale[:, None]  # (K, M, 2)
        series = (h[..., 0] * torch.cos(kt) + h[..., 1] * torch.sin(kt)).sum(-1)
        return torch.exp(self.log_radius * _LOG_RADIUS_SCALE + series)

    def fills(self) -> torch.Tensor:
        """Fill colors (K, ncolors, 3), optimized in FourierImage's decorrelated
        color space so hues drift toward natural-image statistics."""
        color = self.color
        if self.decorrelate:
            color = color @ color_correlation_normalized.to(color.device).T
        return torch.sigmoid(color)

    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha) if self.learn_alpha else torch.ones_like(self.alpha)

    # --- rendering ----------------------------------------------------------------

    def render(
        self,
        height: int,
        width: int,
        softness: float = 0.5,
        camera: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Rasterize to (1, 3, H, W) in [0, 1].

        ``softness`` is the edge width in output pixels. ``camera`` is an affine
        (A, t) applied to the sampling grid about the image center.
        """
        dev = self.center.device
        ys = (torch.arange(height, device=dev) + 0.5) / height
        xs = (torch.arange(width, device=dev) + 0.5) / height  # square pixels
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)  # (H, W, 2)
        if camera is not None:
            a, t = camera
            mid = torch.tensor([0.5 * width / height, 0.5], device=dev)
            grid = (grid - mid) @ a.T + mid + t

        if torch.is_grad_enabled():
            image = self._shade(grid, height, softness)
        else:
            # no autograd graph to keep: shade in row bands to bound memory at big sizes
            rows = max(1, 2**25 // (width * self.center.shape[0] * self.orders.numel()))
            image = torch.cat([self._shade(band, height, softness) for band in grid.split(rows)], 0)
        return image.permute(2, 0, 1)[None]

    def _shade(self, grid: torch.Tensor, height: int, softness: float) -> torch.Tensor:
        """Composite all shapes at sample points ``grid`` (..., 2) -> (..., 3)."""
        d = grid[:, :, None, :] - self.centers()  # (H, W, K, 2)
        rho = torch.sqrt((d * d).sum(-1) + 1e-12)
        theta = torch.atan2(d[..., 1], d[..., 0])
        edge = (self.radius(theta) - rho) * height / max(softness, 1e-3)
        cover = torch.sigmoid(edge) * self.opacity()  # (H, W, K)

        colors = self.fills()
        if self.fill == "linear":
            ang = self.grad_angle
            direction = torch.stack([torch.cos(ang), torch.sin(ang)], -1)  # (K, 2)
            r0 = torch.exp(self.log_radius * _LOG_RADIUS_SCALE)
            s = ((d * direction).sum(-1) / (2 * r0) + 0.5).clamp(0, 1)[..., None]
            paint = colors[:, 0] * (1 - s) + colors[:, 1] * s  # (H, W, K, 3)
        else:
            paint = colors[:, 0].expand(*grid.shape[:-1], -1, -1)

        # back-to-front "over": shape i is seen through every shape painted after it
        log_clear = torch.log1p(-cover.clamp(max=1 - 1e-6))
        after = torch.flip(torch.cumsum(torch.flip(log_clear, [-1]), -1), [-1]) - log_clear
        weight = cover * torch.exp(after)  # (H, W, K)
        image = (weight[..., None] * paint).sum(2)
        image = image + torch.exp(log_clear.sum(-1, keepdim=True)) * torch.sigmoid(self.background)
        return image

    # --- export -------------------------------------------------------------------

    @torch.no_grad()
    def to_svg(self, height: int, width: int, samples: int = 256) -> str:
        """The exact shapes as an SVG document (outlines sampled as polygons)."""
        theta = torch.linspace(0, 2 * math.pi, samples + 1, device=self.center.device)[:-1]
        radii = self.radius(theta[:, None].expand(-1, self.center.shape[0])).T  # (K, S)
        centers = self.centers()
        colors = (self.fills() * 255).round().int().tolist()
        opacity = self.opacity().tolist()
        bg = (torch.sigmoid(self.background) * 255).round().int().tolist()
        r0 = torch.exp(self.log_radius * _LOG_RADIUS_SCALE)

        def rgb(c):
            return f"rgb({c[0]},{c[1]},{c[2]})"

        h = height  # one image unit is the image height
        out = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}">'
            ),
            f'<rect width="{width}" height="{height}" fill="{rgb(bg)}"/>',
        ]
        defs = []
        for i in range(centers.shape[0]):
            cx, cy = centers[i].tolist()
            xs = cx + radii[i] * torch.cos(theta)
            ys = cy + radii[i] * torch.sin(theta)
            pts = " ".join(f"{x * h:.2f},{y * h:.2f}" for x, y in zip(xs.tolist(), ys.tolist()))
            if self.fill == "linear":
                ang = float(self.grad_angle[i])
                dx, dy = math.cos(ang) * float(r0[i]), math.sin(ang) * float(r0[i])
                defs.append(
                    f'<linearGradient id="g{i}" gradientUnits="userSpaceOnUse" '
                    f'x1="{(cx - dx) * h:.2f}" y1="{(cy - dy) * h:.2f}" '
                    f'x2="{(cx + dx) * h:.2f}" y2="{(cy + dy) * h:.2f}">'
                    f'<stop offset="0" stop-color="{rgb(colors[i][0])}"/>'
                    f'<stop offset="1" stop-color="{rgb(colors[i][1])}"/></linearGradient>'
                )
                paint = f"url(#g{i})"
            else:
                paint = rgb(colors[i][0])
            alpha = f' fill-opacity="{opacity[i]:.3f}"' if self.learn_alpha else ""
            out.append(f'<polygon points="{pts}" fill="{paint}"{alpha}/>')
        if defs:
            out.insert(1, "<defs>" + "".join(defs) + "</defs>")
        out.append("</svg>")
        return "\n".join(out)


def random_camera(
    jitter: float = 8 / 224,
    scales=(0.9, 1.1),
    degrees: float = 10.0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """A random similarity transform of the sampling grid: the vector analogue of
    TransformRobustness (jitter, scale, rotate), with no resampling blur."""
    s = random.uniform(*scales)
    ang = math.radians(random.uniform(-degrees, degrees))
    c, n = math.cos(ang) / s, math.sin(ang) / s
    a = torch.tensor([[c, -n], [n, c]], device=device)
    t = torch.tensor([random.uniform(-jitter, jitter) for _ in range(2)], device=device)
    return a, t
