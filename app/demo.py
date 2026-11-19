"""Gradio demo: classify an image with the fp32, int8 and defended int8 models side by side,
optionally with a pre-computed adversarial patch pasted on. CPU only, from exported ONNX files.

    python app/demo.py            # then open http://127.0.0.1:7860

Assets (models, patches, examples) are built by `python app/make_assets.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import demo_core as core  # noqa: E402

sessions = core.Sessions()
patches = core.available_patches()
examples = sorted((core.ASSETS.parent / "examples").glob("*.png"))


def run(img, patch_name, px, py):
    if img is None:
        raise gr.Error("Upload an image or pick an example first.")
    patch = patches.get(patch_name) if patch_name and patch_name != "none" else None
    shown, res = core.classify(sessions, Image.fromarray(img), patch, px, py)
    upscaled = Image.fromarray(shown).resize((256, 256), Image.NEAREST)
    labels = []
    for name in core.MODELS:
        if name in res:
            top = dict(sorted(res[name]["probs"].items(), key=lambda kv: -kv[1])[:3])
            labels.append(gr.update(value=top, label=f"{name} ({res[name]['ms']:.1f} ms)"))
        else:
            labels.append(gr.update(value=None, label=f"{name} (not built)"))
    return [upscaled, *labels]


with gr.Blocks(title="Secure TinyML") as ui:
    gr.Markdown(
        "# Secure TinyML\nA small CIFAR-10 ResNet, full precision vs int8 vs a defended int8 "
        "version. Add a pre-computed adversarial patch (trained to say *airplane*) and see which "
        "models it fools. CIFAR-10 has 10 classes at 32x32 pixels, so photos are cropped and "
        "shrunk first: expect poor accuracy on anything that is not a CIFAR-style image."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Image", sources=["upload", "clipboard"])
            if examples:
                gr.Examples([str(p) for p in examples], inputs=inp, label="Examples")
            patch_name = gr.Dropdown(["none", *patches], value="none", label="Adversarial patch")
            px = gr.Slider(0, 26, value=20, step=1, label="Patch x (pixels from the left)")
            py = gr.Slider(0, 26, value=20, step=1, label="Patch y (pixels from the top)")
            go = gr.Button("Classify", variant="primary")
        with gr.Column():
            shown = gr.Image(label="What the models see (32x32)", interactive=False)
            outs = [gr.Label(num_top_classes=3, label=n) for n in core.MODELS]
    go.click(run, [inp, patch_name, px, py], [shown, *outs])

if __name__ == "__main__":
    ui.launch()
