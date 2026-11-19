"""Demo logic with no torch and no Gradio: numpy, Pillow and ONNX Runtime only, so it runs on a
free CPU host and is testable in CI. The Gradio page in demo.py is a thin shell around this."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ASSETS = Path(__file__).parent / "assets"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SIDE = 32
MODELS = {  # label -> file in assets/
    "fp32": "fp32.onnx",
    "int8": "int8.onnx",
    "defended int8": "defended_int8.onnx",
}


def available_models() -> dict[str, Path]:
    return {k: ASSETS / v for k, v in MODELS.items() if (ASSETS / v).exists()}


def available_patches() -> dict[str, Path]:
    return {
        p.stem.removeprefix("patch_") + "% of image": p for p in sorted(ASSETS.glob("patch_*.png"))
    }


def preprocess(img: Image.Image) -> np.ndarray:
    """Centre square crop, resize to 32x32, float NCHW in [0, 1] (same as training)."""
    img = img.convert("RGB")
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((SIDE, SIDE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255


def paste_patch(x: np.ndarray, patch_png: Path, px: int, py: int) -> np.ndarray:
    """Overwrite a square of the image with the patch; (px, py) is its top-left corner."""
    patch = (
        np.asarray(Image.open(patch_png).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
    )
    side = patch.shape[-1]
    px, py = int(np.clip(px, 0, SIDE - side)), int(np.clip(py, 0, SIDE - side))
    out = x.copy()
    out[0, :, py : py + side, px : px + side] = patch
    return out


class Sessions:
    def __init__(self):
        self.s = {
            k: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            for k, p in available_models().items()
        }

    def run(self, x: np.ndarray) -> dict[str, dict]:
        out = {}
        for k, s in self.s.items():
            t0 = time.perf_counter()
            z = s.run(None, {"x": x})[0][0].astype(np.float64)
            ms = (time.perf_counter() - t0) * 1000
            p = np.exp(z - z.max())
            p /= p.sum()
            out[k] = {"probs": {c: float(v) for c, v in zip(CLASSES, p, strict=True)}, "ms": ms}
        return out


def classify(sessions: Sessions, img: Image.Image, patch: Path | None, px: int, py: int):
    """Returns the 32x32 image actually classified (as uint8 HWC) and per-model results."""
    x = preprocess(img)
    if patch is not None:
        x = paste_patch(x, patch, px, py)
    shown = (x[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return shown, sessions.run(x)
