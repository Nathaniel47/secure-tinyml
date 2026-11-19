"""Build the files the phone page needs (all gitignored; CIFAR-10 images are not redistributed):

    python web/make_testset.py [--n 200]

Writes web/models/ (copies of the fp32 and int8 ONNX models), web/testset.bin (raw uint8 RGB,
N x 32 x 32 x 3), and web/testset.json with the labels and what ONNX Runtime on this laptop
predicts for each image, so the page can report how often the phone agrees with the laptop.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from securetinyml.data.load import get_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    models = ROOT / "web" / "models"
    models.mkdir(exist_ok=True)
    for f in ("cifar10_fp32_seed0.onnx", "cifar10_fp32_seed0_int8.onnx"):
        shutil.copy(ROOT / "models" / f, models / f)

    d = get_dataset("cifar10")
    sel = np.sort(np.random.default_rng(123).choice(len(d["y_test"]), a.n, replace=False))
    x, y = d["x_test"][sel], d["y_test"][sel]  # uint8 NHWC
    (ROOT / "web" / "testset.bin").write_bytes(x.tobytes())
    xf = (x.astype(np.float32) / 255).transpose(0, 3, 1, 2)
    meta = {"labels": y.tolist(), "indices": sel.tolist()}
    for key, f in (
        ("pred_fp32", "cifar10_fp32_seed0.onnx"),
        ("pred_int8", "cifar10_fp32_seed0_int8.onnx"),
    ):
        s = ort.InferenceSession(str(models / f), providers=["CPUExecutionProvider"])
        meta[key] = s.run(None, {"x": xf})[0].argmax(1).tolist()
    (ROOT / "web" / "testset.json").write_text(json.dumps(meta))
    acc = {k: float(np.mean(np.array(meta[k]) == y)) for k in ("pred_fp32", "pred_int8")}
    print("laptop accuracy on these images:", acc)


if __name__ == "__main__":
    main()
