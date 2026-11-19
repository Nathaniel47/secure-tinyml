"""Activation clipping: cap every ReLU output at a value chosen from clean data, so a small
input change cannot cause a huge hidden response. Post-hoc: no retraining, one pass over
calibration images. Also shrinks the range the int8 quantizer has to cover."""

from __future__ import annotations

import copy

import torch
from torch import nn

from securetinyml.models.resnet_small import Block, ConvBN


def _act_modules(model: nn.Module):
    for m in model.modules():
        if (isinstance(m, ConvBN) and m.relu) or isinstance(m, Block):
            yield m


@torch.no_grad()
def clip_activations(model: nn.Module, calib: torch.Tensor, pct: float = 99.9) -> nn.Module:
    """Return a copy whose activations are clipped at the `pct` percentile seen on `calib`."""
    m = copy.deepcopy(model).eval()
    mods = list(_act_modules(m))
    for a in mods:
        a.rec = []
    m(calib)
    for a in mods:
        v = torch.cat(a.rec)
        a.clip = float(torch.quantile(v[:: max(1, len(v) // 1_000_000)], pct / 100))
        a.rec = None
    return m


def clip_values(model: nn.Module) -> list[float | None]:
    return [a.clip for a in _act_modules(model)]
