"""FGSM and PGD (L-infinity), written by hand. Images are float NCHW in [0, 1].

`model` is any callable returning logits and differentiable (fp32 model or fake-quant surrogate).
"""

from __future__ import annotations

import torch
from torch import nn


def _grad_sign(model, x, y):
    x = x.clone().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x), y)
    (g,) = torch.autograd.grad(loss, x)
    return g.sign()


def fgsm(model, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """One step of size eps along the sign of the loss gradient."""
    return (x + eps * _grad_sign(model, x, y)).clamp(0, 1).detach()


def pgd(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 10,
    alpha: float | None = None,
    random_start: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Many small steps, each projected back into the eps-box around x and into [0, 1]."""
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    g = torch.Generator().manual_seed(seed)
    adv = x.clone()
    if random_start:
        adv = (adv + (torch.rand(x.shape, generator=g) * 2 - 1) * eps).clamp(0, 1)
    for _ in range(steps):
        adv = adv + alpha * _grad_sign(model, adv, y)
        adv = torch.min(torch.max(adv, x - eps), x + eps).clamp(0, 1).detach()
    return adv


def random_search(predict, x, y, eps: float, queries: int = 300, seed: int = 0):
    """Query-only L-infinity attack (a simplified Square Attack) for models with no usable
    gradient, such as the real int8 graph. `predict` maps float NCHW numpy to logits.
    Each step flips a random square of the perturbation to +-eps and keeps it if the margin
    (true logit minus best other logit) drops. x: numpy NCHW in [0, 1]; y: numpy labels."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n, c, h, w = x.shape
    delta = np.tile(rng.choice([-eps, eps], size=(n, c, 1, w)), (1, 1, h, 1)).astype(np.float32)

    def margin(adv):
        z = predict(adv)
        true = z[np.arange(n), y]
        z = z.copy()
        z[np.arange(n), y] = -np.inf
        return true - z.max(1)

    adv = np.clip(x + delta, 0, 1).astype(np.float32)
    best = margin(adv)
    for _ in range(queries):
        s = max(2, int(round(h * rng.uniform(0.1, 0.4))))
        i, j = rng.integers(0, h - s + 1), rng.integers(0, w - s + 1)
        cand = delta.copy()  # float32
        cand[:, :, i : i + s, j : j + s] = rng.choice([-eps, eps], size=(n, c, 1, 1))
        cadv = np.clip(x + cand, 0, 1).astype(np.float32)
        m = margin(cadv)
        better = m < best
        delta[better], best[better] = cand[better], m[better]
        adv = np.clip(x + delta, 0, 1)
    return adv.astype(np.float32)
