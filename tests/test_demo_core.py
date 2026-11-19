import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
Image = pytest.importorskip("PIL.Image")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import demo_core as core  # noqa: E402


def test_preprocess_crops_square_and_scales_to_unit_range():
    img = Image.fromarray(np.full((50, 80, 3), 255, np.uint8))
    x = core.preprocess(img)
    assert x.shape == (1, 3, 32, 32) and x.dtype == np.float32
    assert np.allclose(x, 1.0)


def test_paste_patch_changes_only_its_square_and_clamps_to_the_image(tmp_path):
    Image.fromarray(np.full((10, 10, 3), 255, np.uint8)).save(tmp_path / "p.png")
    x = np.zeros((1, 3, 32, 32), np.float32)
    out = core.paste_patch(x, tmp_path / "p.png", 5, 7)
    assert out[0, :, 7:17, 5:15].min() == 1 and out.sum() == 3 * 100
    far = core.paste_patch(x, tmp_path / "p.png", 99, 99)  # clamped so it still fits
    assert far[0, :, 22:32, 22:32].min() == 1 and far.sum() == 3 * 100
    assert x.sum() == 0


def test_classify_with_shipped_models_returns_probabilities():
    if not (core.ASSETS / "fp32.onnx").exists():
        pytest.skip("demo assets not built (python app/make_assets.py)")
    s = core.Sessions()
    shown, res = core.classify(s, Image.fromarray(np.zeros((40, 40, 3), np.uint8)), None, 0, 0)
    assert shown.shape == (32, 32, 3)
    for r in res.values():
        assert abs(sum(r["probs"].values()) - 1) < 1e-6
