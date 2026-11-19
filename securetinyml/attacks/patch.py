"""Adversarial patch: a visible square, optimised by gradient descent so that, pasted at a
random place, it pushes images towards a chosen target class. Images float NCHW in [0, 1]."""

from __future__ import annotations

import torch
from torch import nn


def patch_side(image_side: int, area_fraction: float) -> int:
    return max(1, round(image_side * area_fraction**0.5))


def apply_patch(x: torch.Tensor, patch: torch.Tensor, pos: list[tuple[int, int]]) -> torch.Tensor:
    """Paste `patch` (3, s, s) into each image at its (row, col). Returns a new tensor."""
    out = x.clone()
    s = patch.shape[-1]
    for k, (i, j) in enumerate(pos):
        out[k, :, i : i + s, j : j + s] = patch
    return out


def random_positions(n: int, image_side: int, s: int, g: torch.Generator):
    hi = image_side - s + 1
    ij = torch.randint(0, hi, (n, 2), generator=g)
    return [(int(a), int(b)) for a, b in ij]


def train_patch(
    model,
    x: torch.Tensor,
    target: int,
    area_fraction: float,
    steps: int = 300,
    bs: int = 128,
    lr: float = 0.1,
    seed: int = 0,
) -> torch.Tensor:
    """Gradient descent on the cross-entropy towards `target`, random position per image."""
    g = torch.Generator().manual_seed(seed)
    side = x.shape[-1]
    s = patch_side(side, area_fraction)
    param = torch.randn(3, s, s, generator=g).requires_grad_(True)  # sigmoid keeps it in [0, 1]
    opt = torch.optim.Adam([param], lr=lr)
    tgt = torch.full((bs,), target)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (bs,), generator=g)
        xb = apply_patch(x[idx], torch.sigmoid(param), random_positions(bs, side, s, g))
        loss = nn.functional.cross_entropy(model(xb), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.sigmoid(param).detach()


def patch_success_rate(predict, x, y, patch, target: int, seed: int = 123) -> float:
    """Fraction of non-target images classified as `target` with the patch at a random place.
    `predict` maps float NCHW tensors to predicted labels (tensor)."""
    keep = y != target
    xk = x[keep]
    g = torch.Generator().manual_seed(seed)
    pos = random_positions(len(xk), x.shape[-1], patch.shape[-1], g)
    return float((predict(apply_patch(xk, patch, pos)) == target).float().mean())
