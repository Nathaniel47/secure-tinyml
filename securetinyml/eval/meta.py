"""Record what produced a results file: library versions, git commit and the run's settings.
One entry per results file in results/run_meta.json."""

from __future__ import annotations

import json
import platform
import subprocess
from importlib import metadata

from securetinyml.data.common import ROOT


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def write_run_meta(results_file: str, **settings) -> None:
    path = ROOT / "results" / "run_meta.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta[results_file] = {
        **settings,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "libraries": {
            p: _version(p) for p in ("torch", "numpy", "onnx", "onnxruntime", "pandas", "pillow")
        },
    }
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
