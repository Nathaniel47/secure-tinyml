import numpy as np
import pytest

torch = pytest.importorskip("torch")

from securetinyml.defenses import build  # noqa: E402
from securetinyml.defenses.act_clip import clip_activations, clip_values  # noqa: E402
from securetinyml.defenses.adv_train import fast_adv_batch  # noqa: E402
from securetinyml.defenses.purify import reduce_bits  # noqa: E402
from securetinyml.models.resnet_small import SmallResNet  # noqa: E402


def test_reduce_bits_levels_and_straight_through_gradient():
    x = torch.rand(1000, requires_grad=True)
    q = reduce_bits(x, 4)
    assert len(torch.unique(q.detach())) <= 16
    assert (q - x).abs().max() <= 0.5 / 15 + 1e-6
    q.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_clip_limits_activations_and_leaves_original_untouched():
    torch.manual_seed(0)
    m = SmallResNet(10).eval()
    x = torch.rand(16, 3, 32, 32)
    c = clip_activations(m, x, pct=90)
    vals = clip_values(c)
    assert len(vals) == 7 and all(v is not None and v > 0 for v in vals)
    assert all(v is None for v in clip_values(m))
    assert all(a.rec is None for a in c.modules() if hasattr(a, "rec"))
    # a low percentile clips a lot and changes the output; a higher one clips less
    low, high = clip_activations(m, x, pct=10), clip_activations(m, x, pct=99.9)
    assert (low(x) - m(x)).abs().max() > (high(x) - m(x)).abs().max()
    assert all(a <= b for a, b in zip(clip_values(low), clip_values(high), strict=True))


def test_build_composes_defenses_and_rejects_unknown():
    m = SmallResNet(10).eval()
    x = torch.rand(4, 3, 32, 32)
    for name in ("none", "purify", "purify_blur", "clip", "purify+clip"):
        assert build(m, name, x)(x).shape == (4, 10)
    with pytest.raises(ValueError):
        build(m, "magic", x)


def test_fast_adv_batch_stays_in_budget():
    torch.manual_seed(0)
    m = SmallResNet(10).eval()
    x, y = torch.rand(8, 3, 32, 32), torch.randint(0, 10, (8,))
    xa = fast_adv_batch(m, x, y, 4 / 255, torch.Generator().manual_seed(0))
    assert (xa - x).abs().max() <= 4 / 255 + 1e-6 and xa.min() >= 0 and xa.max() <= 1


def test_defended_graph_exports_to_onnx_and_matches(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from securetinyml.models.quantize import export_onnx

    torch.manual_seed(0)
    x = torch.rand(8, 3, 32, 32)
    d = build(SmallResNet(10).eval(), "purify_blur+clip", x)
    export_onnx(d, tmp_path / "d.onnx", 32)
    out = ort.InferenceSession(str(tmp_path / "d.onnx")).run(None, {"x": x.numpy()})[0]
    assert np.allclose(out, d(x).detach().numpy(), atol=1e-3)
