"""Input purification: clean the image before the classifier sees it (bit-depth reduction and
an optional 3x3 blur). It is part of the model graph, so it exports to ONNX and runs on the phone.

The rounding has zero gradient. For an *adaptive* attacker we pass the gradient straight through
the rounding (BPDA, Athalye et al. 2018); without that, a gradient attack would fail only because
the attacker is handicapped, not because the defense works.
"""

from __future__ import annotations

import torch
from torch import nn


def reduce_bits(x: torch.Tensor, bits: int) -> torch.Tensor:
    levels = 2**bits - 1
    q = torch.round(x * levels) / levels
    return x + (q - x).detach()  # forward: rounded; backward: identity (BPDA)


class Purify(nn.Module):
    def __init__(self, model: nn.Module, bits: int = 4, blur: bool = False):
        super().__init__()
        self.model, self.bits, self.blur = model, bits, blur

    def forward(self, x):
        z = reduce_bits(x, self.bits)
        if self.blur:
            z = nn.functional.avg_pool2d(nn.functional.pad(z, (1, 1, 1, 1), mode="replicate"), 3, 1)
        return self.model(z)
