"""Backbone wrapper: a torchvision ImageNet classifier with activation hooks.

ConvNeXt (the default) is a modern CNN, so the classic conv-channel feature-viz
objectives transfer directly (unlike a ViT); GoogLeNet is the network of the
original Distill article and gives the crispest renders here. Code is MIT
(facebookresearch/ConvNeXt) and BSD-3 (torchvision); the ImageNet-1k weights
carry the usual unsettled ImageNet-dataset caveat for commercial use, but
feature-viz *output* is the lowest-risk use of them. Swap to a non-ImageNet
network later if needed -- only this file changes.
"""

import torch
from torchvision import models

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

NETS = {
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1),
    "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
    "googlenet": (models.googlenet, models.GoogLeNet_Weights.IMAGENET1K_V1),
    "inception_v3": (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1),
    "resnet50_v1": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet50_v2": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "regnet_y_1_6gf": (models.regnet_y_1_6gf, models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1),
}
# The ConvNeXt size names on their own, as the CLIs have always taken them.
NETS.update({k: NETS[f"convnext_{k}"] for k in ("tiny", "small", "base")})

# ConvNeXt `features` is a Sequential of stem / stage / downsample blocks.
# The four conv stages live at these indices; channel counts vary by variant.
STAGE_LAYERS = ("features.1", "features.3", "features.5", "features.7")


class _Done(Exception):
    """Raised from a hook once every hooked layer has fired."""


class Backbone(torch.nn.Module):
    def __init__(self, net: str = "tiny", device: str = "cpu"):
        super().__init__()
        ctor, weights = NETS[net]
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
            if len(self._activations) == len(self._hooks):
                raise _Done  # nothing past the deepest hooked layer is needed

        self._hooks.append(module.register_forward_hook(fn))

    def clear_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._activations = {}

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        self._activations = {}
        try:
            self.model(self.normalize(image))
        except _Done:
            pass
        return self._activations

    def list_layers(self) -> list[str]:
        return [name for name, _ in self.model.named_modules() if name]

    def channels(self, layer_name: str) -> int:
        """Channel count of a layer, from one forward pass at 224 px."""
        saved = self._hooks
        self._hooks = []
        self.hook(layer_name)
        try:
            return self(torch.zeros(1, 3, 224, 224, device=self.device))[layer_name].shape[1]
        finally:
            self.clear_hooks()
            self._hooks = saved
