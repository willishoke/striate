"""Backbone wrapper: a modern ConvNeXt with activation hooks.

ConvNeXt is a modern CNN, so the classic conv-channel feature-viz objectives
transfer directly (unlike a ViT). Code is MIT (facebookresearch/ConvNeXt) and
BSD-3 (torchvision); the ImageNet-1k weights carry the usual unsettled
ImageNet-dataset caveat for commercial use, but feature-viz *output* is the
lowest-risk use of them. Swap to a non-ImageNet variant later if needed --
only this file changes.
"""

import torch
from torchvision import models

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_CONVNEXT = {
    "tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "small": (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1),
    "base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
}

# ConvNeXt `features` is a Sequential of stem / stage / downsample blocks.
# The four conv stages live at these indices; channel counts vary by variant.
STAGE_LAYERS = ("features.1", "features.3", "features.5", "features.7")


class Backbone(torch.nn.Module):
    def __init__(self, variant: str = "tiny", device: str = "cpu"):
        super().__init__()
        ctor, weights = _CONVNEXT[variant]
        self.model = ctor(weights=weights).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self._activations: dict[str, torch.Tensor] = {}
        self._hooks: list = []

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        mean = _IMAGENET_MEAN.to(image.device)
        std = _IMAGENET_STD.to(image.device)
        return (image - mean) / std

    def hook(self, layer_name: str) -> None:
        module = dict(self.model.named_modules())[layer_name]

        def fn(_module, _inp, out):
            self._activations[layer_name] = out

        self._hooks.append(module.register_forward_hook(fn))

    def clear_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._activations = {}

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        self._activations = {}
        self.model(self.normalize(image))
        return self._activations

    def list_layers(self) -> list[str]:
        return [name for name, _ in self.model.named_modules() if name]
