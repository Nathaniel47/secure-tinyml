"""BadNets-style backdoor: stamp a small trigger on a few training images and relabel them.

Pure numpy (uint8 NHWC images) so it also runs in CI without torch.
"""

from __future__ import annotations

import numpy as np


def add_trigger(x: np.ndarray, size: int = 4) -> np.ndarray:
    """Return a copy with a size x size black-and-white checker in the bottom-right corner."""
    out = x.copy()
    ys, xs = np.mgrid[:size, :size]
    checker = (((ys + xs) % 2) * 255).astype(np.uint8)
    out[..., -size:, -size:, :] = checker[..., None]
    return out


def poison_dataset(
    x: np.ndarray, y: np.ndarray, rate: float, target: int, seed: int, size: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poison `rate` of the whole set: pick images whose true class is not `target`, stamp the
    trigger and relabel them as `target`. Returns (x, y, poisoned_indices); inputs untouched."""
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(y != target)
    n = int(round(rate * len(y)))
    idx = np.sort(rng.choice(candidates, size=n, replace=False))
    x2, y2 = x.copy(), y.copy()
    x2[idx] = add_trigger(x[idx], size)
    y2[idx] = target
    return x2, y2, idx


def attack_success_rate(predict, x_test: np.ndarray, y_test: np.ndarray, target: int, size=4):
    """Fraction of non-target test images that the model sends to `target` once triggered.
    `predict` maps uint8 NHWC images to predicted labels."""
    keep = y_test != target
    return float((predict(add_trigger(x_test[keep], size)) == target).mean())
