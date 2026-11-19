import numpy as np
import pytest

torch = pytest.importorskip("torch")

from securetinyml.models.quantize import (  # noqa: E402
    FakeQuantSurrogate,
    fake_quant_act,
    fake_quant_weight,
)
from securetinyml.models.resnet_small import SmallResNet, count_params, fold_bn  # noqa: E402


def test_fake_quant_act_lands_on_256_levels_and_passes_gradient():
    x = torch.linspace(-1, 1, 1000, requires_grad=True)
    q = fake_quant_act(x, -1.0, 1.0)
    assert len(torch.unique(q.detach())) <= 256
    assert (q - x).abs().max() <= (2 / 255) / 2 + 1e-6
    q.sum().backward()
    assert torch.allclose(x.grad[1:-1], torch.ones(len(x) - 2))  # straight-through inside the range


def test_fake_quant_weight_is_int8_per_channel():
    w = torch.randn(4, 3, 3, 3)
    q = fake_quant_weight(w)
    scale = w.abs().flatten(1).amax(1) / 127
    steps = (q / scale.view(-1, 1, 1, 1)).round()
    assert torch.allclose(q, steps * scale.view(-1, 1, 1, 1), atol=1e-6)
    assert steps.abs().max() <= 127


def test_fold_bn_matches_original_in_eval_mode():
    torch.manual_seed(0)
    m = SmallResNet(10).eval()
    for mod in m.modules():  # non-trivial BN statistics
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_(0, 0.1)
            mod.running_var.uniform_(0.5, 1.5)
    x = torch.rand(4, 3, 32, 32)
    assert torch.allclose(m(x), fold_bn(m)(x), atol=1e-4)
    assert count_params(m) < 300_000


def test_surrogate_is_close_to_fp32_and_differentiable():
    torch.manual_seed(0)
    m = SmallResNet(10).eval()
    x = torch.rand(32, 3, 32, 32)
    s = FakeQuantSurrogate(m, x)
    a, b = m(x), s(x)
    assert (a.argmax(1) == b.argmax(1)).float().mean() > 0.8
    xg = x[:2].clone().requires_grad_(True)
    s(xg).sum().backward()
    assert xg.grad.abs().sum() > 0


def test_onnx_export_and_int8_agree_with_fp32(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from securetinyml.models.quantize import export_onnx, quantize_int8

    torch.manual_seed(0)
    m = SmallResNet(10).eval()
    x = torch.rand(16, 3, 32, 32)
    export_onnx(m, tmp_path / "m.onnx", 32)
    quantize_int8(tmp_path / "m.onnx", tmp_path / "m8.onnx", x.numpy())
    f32 = ort.InferenceSession(str(tmp_path / "m.onnx"))
    i8 = ort.InferenceSession(str(tmp_path / "m8.onnx"))
    ref = m(x).detach().numpy()
    assert np.allclose(f32.run(None, {"x": x.numpy()})[0], ref, atol=1e-4)
    out8 = i8.run(None, {"x": x.numpy()})[0]
    assert out8.shape == ref.shape
    assert (tmp_path / "m8.onnx").stat().st_size < (tmp_path / "m.onnx").stat().st_size
