import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deploy_build", ROOT / "deploy" / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_space_has_front_matter_demo_and_no_example_images(tmp_path):
    (ROOT / "app" / "examples").mkdir(exist_ok=True)  # gitignored folder, may not exist in CI
    (ROOT / "app" / "examples" / "x.png").write_bytes(b"x")
    try:
        build.build_space(tmp_path / "s")
    finally:
        (ROOT / "app" / "examples" / "x.png").unlink()
    s = tmp_path / "s"
    text = (s / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n") and "app_file: app/demo.py" in text
    assert (s / "app" / "demo.py").exists() and (s / "app" / "demo_core.py").exists()
    assert not (s / "app" / "examples").exists()
    assert "gradio" not in (s / "requirements.txt").read_text()  # the Space installs it itself


def test_site_has_page_and_models_but_no_test_set(tmp_path):
    build.build_site(tmp_path / "w")
    w = tmp_path / "w"
    assert (w / "index.html").exists() and (w / "demo.html").exists()
    assert (w / ".nojekyll").exists()
    assert (w / "models" / "defended_int8.onnx").exists()
    assert sorted(f.name for f in (w / "patches").glob("*.png")) == [
        "patch_10.png",
        "patch_4.png",
        "patch_7.png",
    ]
    assert not list(w.glob("testset*"))
    for f in build.SITE_MODELS:
        assert (w / "models" / f).exists()
