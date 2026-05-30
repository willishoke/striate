"""Spectral (Fourier) image parameterization with frequency preconditioning.

This is the heart of why feature-viz images look like *anything* rather than
high-frequency adversarial noise. The optimized parameters live in the Fourier
domain and are scaled by 1/f**decay, so high frequencies are implicitly
penalized -- exactly the "decorrelated parameterization" from the Distill 2017
feature-visualization article. Color is also decorrelated through a fixed
matrix so the optimizer works in a perceptually whitened space.

Kept deliberately small and dependency-light: this module is the reference
spec we'd port to another backend (e.g. hasktorch) later.
"""

import numpy as np
import torch

# Color decorrelation matrix from the Distill / lucid feature-viz work.
_color_correlation_svd_sqrt = torch.tensor(
    [
        [0.26, 0.09, 0.02],
        [0.27, 0.00, -0.05],
        [0.27, -0.09, 0.03],
    ],
    dtype=torch.float32,
)
_max_norm = float(torch.linalg.norm(_color_correlation_svd_sqrt, dim=0).max())
color_correlation_normalized = _color_correlation_svd_sqrt / _max_norm


def _rfft2d_freqs(h: int, w: int) -> np.ndarray:
    """Radial spatial frequency for each rfft2 coefficient."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]  # rfft has (w // 2 + 1) columns
    return np.sqrt(fx * fx + fy * fy)


class FourierImage(torch.nn.Module):
    """An image parameterized by preconditioned Fourier coefficients.

    ``to_image()`` returns a valid RGB tensor in [0, 1] of shape (batch, 3, h, w).
    The optimizable parameter is ``self.spectrum``.
    """

    def __init__(
        self,
        height: int,
        width: int,
        batch: int = 1,
        sd: float = 0.01,
        decay_power: float = 1.0,
        decorrelate: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.decorrelate = decorrelate

        freqs = _rfft2d_freqs(height, width)
        scale = 1.0 / np.maximum(freqs, 1.0 / max(height, width)) ** decay_power
        # compensate for the 1/N normalization of the inverse FFT, so the
        # spatial image lands at a usable magnitude (and gradients aren't
        # crushed by the same factor). This is the key spectral scaling.
        scale = scale * np.sqrt(height * width)
        # shape (1, 1, h, w//2+1, 1): broadcasts over batch, channel, real/imag
        self.register_buffer(
            "spectrum_scale",
            torch.tensor(scale, dtype=torch.float32)[None, None, ..., None],
        )

        init = torch.randn(batch, 3, freqs.shape[0], freqs.shape[1], 2) * sd
        self.spectrum = torch.nn.Parameter(init)
        self.to(device)

    def preact(self) -> torch.Tensor:
        """The pre-sigmoid (decorrelated) image -- unbounded; the sigmoid squashes
        this into [0, 1]. Temporal smoothness penalties belong here, not on the
        squashed output: in this space there are no [0, 1] rails to clip against.
        """
        scaled = self.spectrum * self.spectrum_scale
        complex_spectrum = torch.view_as_complex(scaled.contiguous())
        image = torch.fft.irfftn(
            complex_spectrum, s=(self.height, self.width), dim=(2, 3)
        )
        image = image / 4.0  # empirical scaling so the std lands in a usable range
        if self.decorrelate:
            image = self._decorrelate_color(image)
        return image

    def to_image(self) -> torch.Tensor:
        return torch.sigmoid(self.preact())

    def _decorrelate_color(self, image: torch.Tensor) -> torch.Tensor:
        mat = color_correlation_normalized.to(image.device)
        b, c, h, w = image.shape
        flat = image.permute(0, 2, 3, 1).reshape(-1, 3)
        flat = flat @ mat.T
        return flat.reshape(b, h, w, 3).permute(0, 3, 1, 2)


def seed_spectrum_from_image(param: FourierImage, image: torch.Tensor) -> FourierImage:
    """Initialize a FourierImage's spectrum so ``to_image()`` reproduces ``image``.

    Exact inverse of the forward path (logit -> un-decorrelate -> FFT -> unscale).
    Used by progressive/octave rendering to carry a low-res result up to a higher
    resolution without restarting the optimization from noise.
    """
    eps = 1e-4
    t = image.clamp(eps, 1 - eps)
    pre = torch.log(t) - torch.log1p(-t)  # logit

    if param.decorrelate:
        mat = color_correlation_normalized.to(image.device)
        inv = torch.linalg.inv(mat)
        b, c, h, w = pre.shape
        flat = pre.permute(0, 2, 3, 1).reshape(-1, 3) @ inv.T
        pre = flat.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    spatial = pre * 4.0
    spec = torch.fft.rfftn(spatial, dim=(2, 3))
    spec_real = torch.view_as_real(spec)
    scaled = spec_real / param.spectrum_scale.to(spec_real.device)
    with torch.no_grad():
        param.spectrum.copy_(scaled)
    return param
