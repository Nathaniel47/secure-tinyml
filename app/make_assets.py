"""Build the demo's assets from trained models (needs torch; run on the laptop or Colab).

    python app/make_assets.py --defended advtrain_none

Copies the fp32/int8 ONNX models and the chosen defended int8 model into app/assets/, trains
one adversarial patch per size against the fp32 model (target: airplane) and saves each as a
small PNG, and writes a few CIFAR-10 test images into app/examples/ (gitignored: CIFAR-10 is
not redistributed, see README).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from securetinyml.attacks.patch import train_patch  # noqa: E402
from securetinyml.config import PATCH_FRACTIONS  # noqa: E402
from securetinyml.data.common import load_splits  # noqa: E402
from securetinyml.data.load import get_dataset  # noqa: E402
from securetinyml.models.resnet_small import SmallResNet  # noqa: E402
from securetinyml.models.train import MODELS, to_tensor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defended", default="advtrain_none", help="stem under models/defended/")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    assets = ROOT / "app" / "assets"
    assets.mkdir(exist_ok=True)
    shutil.copy(MODELS / f"cifar10_fp32_seed{a.seed}.onnx", assets / "fp32.onnx")
    shutil.copy(MODELS / f"cifar10_fp32_seed{a.seed}_int8.onnx", assets / "int8.onnx")
    dsrc = MODELS / "defended" / f"cifar10_{a.defended}_seed{a.seed}_int8.onnx"
    if dsrc.exists():
        shutil.copy(dsrc, assets / "defended_int8.onnx")
    else:
        print("defended model not found:", dsrc)

    d = get_dataset("cifar10")
    sp = load_splits("cifar10", a.seed)
    model = SmallResNet(10)
    model.load_state_dict(torch.load(MODELS / f"cifar10_fp32_seed{a.seed}.pt"))
    model.eval()
    xtr = to_tensor(d["x_train"][sp["train"]][:5000])
    for frac in PATCH_FRACTIONS:
        patch = train_patch(model, xtr, 0, frac, steps=300, seed=a.seed)
        png = (patch.detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(png).save(assets / f"patch_{round(frac * 100)}.png")
        print("patch", frac, "saved")

    ex = ROOT / "app" / "examples"
    ex.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for i, k in enumerate(rng.choice(len(d["y_test"]), 8, replace=False)):
        name = d["classes"][d["y_test"][k]]
        Image.fromarray(d["x_test"][k]).resize((128, 128), Image.NEAREST).save(
            ex / f"{i}_{name}.png"
        )


if __name__ == "__main__":
    main()
