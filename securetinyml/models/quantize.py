"""ONNX export, static int8 quantization (what ships), and a differentiable fake-quant surrogate.

The real int8 model has rounding steps with zero gradient, so a gradient attack cannot be run
through it. `FakeQuantSurrogate` copies the same rounding but passes gradients straight through
(straight-through estimator), so an attacker can differentiate it. It is a post-training
approximation of the int8 graph, not the same arithmetic: see LEARNING.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from securetinyml.models.resnet_small import fold_bn


def export_onnx(model: nn.Module, path: Path, side: int) -> None:
    model = model.eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, side, side),
        str(path),
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )


def quantize_int8(fp32_path: Path, out_path: Path, calib: np.ndarray) -> None:
    """Static post-training quantization, QDQ format, per-channel int8 weights, uint8
    activations, min/max calibration on `calib` (float NCHW in [0, 1])."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib[i : i + 1] for i in range(len(calib)))

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"x": x.astype(np.float32)}

    prepped = out_path.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(prepped))
    quantize_static(
        str(prepped),
        str(out_path),
        Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    prepped.unlink()


class _STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def fake_quant_act(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Asymmetric uint8 grid between lo and hi, rounding with straight-through gradient."""
    scale = max(hi - lo, 1e-8) / 255
    q = _STERound.apply((x.clamp(lo, hi) - lo) / scale)
    return q * scale + lo


def fake_quant_weight(w: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-channel int8."""
    flat = w.detach().abs().flatten(1).amax(1).clamp_min(1e-8)
    scale = (flat / 127).view(-1, *([1] * (w.dim() - 1)))
    return _STERound.apply(w / scale).clamp(-127, 127) * scale


class _QLayer(nn.Module):
    """Wraps a Conv2d/Linear: fake-quantizes its input (range from calibration) and weights."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.lo, self.hi = 0.0, 1.0
        self.calibrating = False
        self._seen = False

    def forward(self, x):
        if self.calibrating:
            lo, hi = x.min().item(), x.max().item()
            self.lo, self.hi = (lo, hi) if not self._seen else (min(self.lo, lo), max(self.hi, hi))
            self._seen = True
            return self.layer(x)
        x = fake_quant_act(x, min(self.lo, 0.0), max(self.hi, 0.0))
        w = fake_quant_weight(self.layer.weight)
        if isinstance(self.layer, nn.Conv2d):
            return nn.functional.conv2d(
                x, w, self.layer.bias, self.layer.stride, self.layer.padding
            )
        return nn.functional.linear(x, w, self.layer.bias)


class FakeQuantSurrogate(nn.Module):
    def __init__(self, model: nn.Module, calib: torch.Tensor):
        super().__init__()
        self.m = fold_bn(model)
        self.qlayers: list[_QLayer] = []
        self._wrap(self.m)
        for q in self.qlayers:
            q.calibrating = True
        with torch.no_grad():
            self.m(calib)
        for q in self.qlayers:
            q.calibrating = False
        for p in self.parameters():
            p.requires_grad_(False)

    def _wrap(self, mod: nn.Module) -> None:
        for name, child in mod.named_children():
            if isinstance(child, nn.Conv2d | nn.Linear):
                q = _QLayer(child)
                setattr(mod, name, q)
                self.qlayers.append(q)
            else:
                self._wrap(child)

    def forward(self, x):
        return self.m(x)
