"""A small ResNet for 32/64 px images. Input is a float image in [0, 1], NCHW: normalisation is
inside the model so attacks and the ONNX export both work in plain pixel space."""

from __future__ import annotations

import torch
from torch import nn

MEAN = (0.5, 0.5, 0.5)
STD = (0.25, 0.25, 0.25)


class ConvBN(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, relu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        return torch.relu(x) if self.relu else x


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.a = ConvBN(cin, cout, stride)
        self.b = ConvBN(cout, cout, 1, relu=False)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Conv2d(cin, cout, 1, stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        s = x if self.skip is None else self.skip_bn(self.skip(x))
        return torch.relu(self.b(self.a(x)) + s)


class SmallResNet(nn.Module):
    """Stem + 3 stages of one residual block + linear head. width=16 gives about 0.18M params."""

    def __init__(self, num_classes: int = 10, width: int = 16):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        w = width
        self.stem = ConvBN(3, w)
        self.blocks = nn.Sequential(Block(w, w, 1), Block(w, 2 * w, 2), Block(2 * w, 4 * w, 2))
        self.fc = nn.Linear(4 * w, num_classes)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.blocks(self.stem(x))
        return self.fc(x.mean(dim=(2, 3)))


def fold_bn(model: nn.Module) -> nn.Module:
    """Return a copy in eval mode with every BatchNorm folded into the conv before it.
    This is what the exported ONNX graph looks like, and what int8 conversion quantizes."""
    import copy

    from torch.nn.utils.fusion import fuse_conv_bn_eval

    m = copy.deepcopy(model).eval()
    for mod in m.modules():
        if isinstance(mod, ConvBN):
            mod.conv = fuse_conv_bn_eval(mod.conv, mod.bn)
            mod.bn = nn.Identity()
        elif isinstance(mod, Block) and mod.skip is not None:
            mod.skip = fuse_conv_bn_eval(mod.skip, mod.skip_bn)
            mod.skip_bn = nn.Identity()
    return m


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
