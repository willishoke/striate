"""Feature-viz objectives. Each returns a *loss* (negated activation to maximize).

These are the three canonical objectives from the Distill article. They consume
the dict of captured activations produced by ``Backbone.forward``.
"""

import torch


class ChannelObjective:
    """Maximize the mean activation of one channel at a hooked layer.

    ``cap`` clamps each unit's activation, so once a unit fires past the cap it
    stops contributing gradient. This prevents unbounded maximization from
    driving a few activations enormous (which saturates the output to the
    sigmoid rails) and instead encourages the response to spread across the
    image -- fixing both clipping and over-sparse "blooms on black".
    """

    def __init__(self, layer: str, channel: int, cap: float | None = None):
        self.layer = layer
        self.channel = channel
        self.cap = cap

    def __call__(self, activations: dict[str, torch.Tensor]) -> torch.Tensor:
        a = activations[self.layer][:, self.channel]
        if self.cap is not None:
            a = a.clamp(max=self.cap)
        return -a.mean()


def clip_penalty(image: torch.Tensor, lo: float = 0.05, hi: float = 0.95) -> torch.Tensor:
    """Soft hinge penalty on pixels driven past the rails.

    Penalizes only pixels below ``lo`` or above ``hi`` (quadratically), leaving
    midtone contrast free. Restores highlight/shadow detail without flattening
    the image toward gray the way a global contrast penalty would.
    """
    return (torch.relu(image - hi) ** 2).mean() + (torch.relu(lo - image) ** 2).mean()


class LayerObjective:
    """DeepDream-style: maximize the mean squared activation of a whole layer."""

    def __init__(self, layer: str):
        self.layer = layer

    def __call__(self, activations: dict[str, torch.Tensor]) -> torch.Tensor:
        return -(activations[self.layer] ** 2).mean()


class NeuronObjective:
    """Maximize a single spatial unit (center by default) of one channel."""

    def __init__(self, layer: str, channel: int, x: int | None = None, y: int | None = None):
        self.layer = layer
        self.channel = channel
        self.x = x
        self.y = y

    def __call__(self, activations: dict[str, torch.Tensor]) -> torch.Tensor:
        act = activations[self.layer]
        _, _, h, w = act.shape
        x = self.x if self.x is not None else w // 2
        y = self.y if self.y is not None else h // 2
        return -act[:, self.channel, y, x].mean()
