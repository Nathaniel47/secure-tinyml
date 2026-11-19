"""One wrapper for all defenses: `build(model, "purify+clip", calib)` returns an fp32 model with
the defenses inside the graph, so one evaluator and one ONNX export handle every variant.
Adversarial training changes the weights, not the graph, so it is a different checkpoint."""

from __future__ import annotations

import torch
from torch import nn

NAMES = ("none", "purify", "purify_blur", "clip")


def build(model: nn.Module, defense: str, calib: torch.Tensor) -> nn.Module:
    from securetinyml.defenses.act_clip import clip_activations
    from securetinyml.defenses.purify import Purify

    out = model.eval()
    for part in [p for p in defense.split("+") if p and p != "none"]:
        if part == "clip":
            out = clip_activations(out, calib)
        elif part == "purify":
            out = Purify(out, bits=4)
        elif part == "purify_blur":
            out = Purify(out, bits=5, blur=True)
        else:
            raise ValueError(f"unknown defense {part!r}; choose from {NAMES}")
    return out.eval()
