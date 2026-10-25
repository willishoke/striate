"""The optimization loop and progressive (octave) high-res rendering."""

import torch
from tqdm import trange

from .objectives import clip_penalty
from .param import FourierImage, seed_spectrum_from_image
from .transforms import TransformRobustness


def _optimize(param, backbone, objective, transforms, steps, lr, desc, progress, clip_weight=0.0):
    optimizer = torch.optim.Adam([param.spectrum], lr=lr)
    iterator = trange(steps, desc=desc) if progress else range(steps)
    for _ in iterator:
        optimizer.zero_grad()
        image = param.to_image()
        acts = backbone(transforms(image))
        loss = objective(acts)
        if clip_weight > 0.0:
            loss = loss + clip_weight * clip_penalty(image)
        loss.backward()
        optimizer.step()
    return param.to_image().detach()


def render(
    backbone,
    objective,
    layers,
    *,
    size: int = 512,
    steps: int = 512,
    lr: float = 0.05,
    decorrelate: bool = True,
    transforms=None,
    device: str = "cpu",
    progress: bool = True,
    clip_weight: float = 0.0,
):
    """Single-resolution feature visualization -- the faithful paper reproduction."""
    backbone.clear_hooks()
    for layer in layers:
        backbone.hook(layer)
    param = FourierImage(size, size, decorrelate=decorrelate, device=device)
    transforms = transforms or TransformRobustness()
    image = _optimize(
        param, backbone, objective, transforms, steps, lr, f"{size}px", progress, clip_weight
    )
    backbone.clear_hooks()
    return image


def render_progressive(
    backbone,
    objective,
    layers,
    *,
    sizes=(256, 384, 512, 768),
    steps: int = 256,
    lr: float = 0.05,
    decorrelate: bool = True,
    device: str = "cpu",
    progress: bool = True,
):
    """Octave-style render: optimize at increasing resolutions, seeding each
    stage from the previous result. Keeps feature scale sane as you climb toward
    1080p instead of letting features shrink relative to the receptive field.

    Each entry in ``sizes`` is either an int (square) or an (h, w) tuple, so a
    16:9 schedule toward true 1920x1080 works directly.
    """
    backbone.clear_hooks()
    for layer in layers:
        backbone.hook(layer)
    transforms = TransformRobustness()

    image = None
    for size in sizes:
        h, w = (size, size) if isinstance(size, int) else size
        param = FourierImage(h, w, decorrelate=decorrelate, device=device)
        if image is not None:
            up = torch.nn.functional.interpolate(
                image, size=(h, w), mode="bicubic", align_corners=False
            ).clamp(0, 1)
            seed_spectrum_from_image(param, up)
        image = _optimize(param, backbone, objective, transforms, steps, lr, f"{w}x{h}", progress)

    backbone.clear_hooks()
    return image
