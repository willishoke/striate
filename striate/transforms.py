"""Transformation robustness.

The optimized image is randomly padded, jittered, scaled and rotated before
every forward pass. Forcing the activation to stay high under these transforms
is what removes adversarial high-frequency structure and yields the recognizable
feature-viz look (Distill 2017, "Feature Visualization").
"""

import random

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF


class TransformRobustness:
    def __init__(
        self,
        jitter: int = 8,
        scales=None,
        degrees=None,
        pad: int = 12,
        pad_value: float = 0.5,
    ):
        self.jitter = jitter
        self.scales = scales or [1 + (i - 5) * 0.02 for i in range(11)]
        self.degrees = degrees if degrees is not None else list(range(-10, 11))
        self.pad = pad
        self.pad_value = pad_value

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        x = F.pad(image, [self.pad] * 4, mode="constant", value=self.pad_value)
        x = self._jitter(x, self.jitter)

        scale = random.choice(self.scales)
        nh, nw = round(x.shape[2] * scale), round(x.shape[3] * scale)
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)

        if self.degrees:  # rotation is the lone grid_sample (CPU fallback on MPS); skippable
            angle = random.choice(self.degrees)
            x = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR)

        x = self._jitter(x, self.jitter // 2)
        return self._center_crop(x, h, w)

    def _jitter(self, x: torch.Tensor, d: int) -> torch.Tensor:
        if d <= 0:
            return x
        dy, dx = random.randint(-d, d), random.randint(-d, d)
        return torch.roll(x, shifts=(dy, dx), dims=(2, 3))

    def _center_crop(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        _, _, H, W = x.shape
        top, left = max((H - h) // 2, 0), max((W - w) // 2, 0)
        x = x[:, :, top : top + h, left : left + w]
        if x.shape[2] != h or x.shape[3] != w:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x
