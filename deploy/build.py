"""Assemble the two things that get hosted, into folders you can push or upload as they are.

    python deploy/build.py space  out/space   # Hugging Face Space (Gradio demo)
    python deploy/build.py site   out/site    # static phone page (GitHub Pages)

Nothing here publishes anything; see DEPLOY.md for the manual steps. Neither output contains
CIFAR-10/GTSRB images (licence unverified): the Space has no example images, and the static
page has no test set.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRADIO_VERSION = "6.32.0"  # the version the demo was tested with
SITE_MODELS = ("cifar10_fp32_seed0.onnx", "cifar10_fp32_seed0_int8.onnx")

SPACE_README = f"""---
title: Secure TinyML
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: {GRADIO_VERSION}
app_file: app/demo.py
pinned: false
---

# Secure TinyML demo

A small CIFAR-10 ResNet in full precision, int8, and an adversarially trained int8 version,
with pre-computed adversarial patches. Upload an image, paste a patch, and compare the three.

Research demo, not a security product. CIFAR-10 images are 32x32, so photos are cropped and
shrunk first and accuracy on ordinary photos is poor. No CIFAR-10 images are included.
Code, results and limitations: https://github.com/Nathaniel47/secure-tinyml
"""

SPACE_REQUIREMENTS = """onnxruntime>=1.18
numpy>=1.26
pillow>=10
"""  # gradio itself is installed by the Space from sdk_version


def _fresh(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def build_space(out: Path) -> None:
    out = _fresh(out)
    shutil.copytree(
        ROOT / "app",
        out / "app",
        ignore=shutil.ignore_patterns("examples", "__pycache__", "make_assets.py"),
    )
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")
    (out / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")


def build_site(out: Path) -> None:
    out = _fresh(out)
    for page in ("index.html", "demo.html"):
        shutil.copy(ROOT / "web" / page, out / page)
    (out / "models").mkdir()
    for f in SITE_MODELS:
        shutil.copy(ROOT / "models" / f, out / "models" / f)
    assets = ROOT / "app" / "assets"
    shutil.copy(assets / "defended_int8.onnx", out / "models" / "defended_int8.onnx")
    (out / "patches").mkdir()
    for f in sorted(assets.glob("patch_*.png")):
        shutil.copy(f, out / "patches" / f.name)
    (out / ".nojekyll").write_text("")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("space", "site"):
        raise SystemExit(__doc__)
    {"space": build_space, "site": build_site}[sys.argv[1]](Path(sys.argv[2]))
    print("built", sys.argv[1], "->", sys.argv[2])


if __name__ == "__main__":
    main()
