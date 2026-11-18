"""Shared helpers: hashing, resizing, seeded splits and split digests (numpy + Pillow only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
SPLITS = ROOT / "data" / "splits"
DIGESTS = ROOT / "data" / "split_digests.json"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resize_rgb(img: Image.Image, side: int) -> np.ndarray:
    """Resize to a square RGB uint8 array (side, side, 3). Aspect ratio is not kept."""
    return np.asarray(img.convert("RGB").resize((side, side), Image.BILINEAR), dtype=np.uint8)


def stratified_split(
    labels: np.ndarray, fractions: dict[str, float], seed: int, groups: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Split indices per class so each part keeps the class mix.

    If `groups` is given, whole groups (e.g. one physical traffic sign photographed 30 times)
    go to one part, so near-identical images never straddle train and validation.
    Fractions must sum to 1. Returns sorted index arrays.
    """
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    names = list(fractions)
    out: dict[str, list[int]] = {n: [] for n in names}
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if groups is None:
            units = [[i] for i in rng.permutation(idx)]
        else:
            gids = rng.permutation(np.unique(groups[idx]))
            units = [idx[groups[idx] == g].tolist() for g in gids]
        total = len(idx)
        targets = np.cumsum([fractions[n] * total for n in names])
        filled, k = 0, 0
        for u in units:
            while k < len(names) - 1 and filled >= targets[k]:
                k += 1
            out[names[k]].extend(u)
            filled += len(u)
    return {n: np.array(sorted(v), dtype=np.int64) for n, v in out.items()}


def split_digest(splits: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(splits):
        h.update(name.encode())
        h.update(np.ascontiguousarray(splits[name], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def save_splits(dataset: str, seed: int, splits: dict[str, np.ndarray]) -> str:
    """Save indices to data/splits (gitignored) and record a digest in data/split_digests.json
    (committed), so a rebuilt split can be checked without committing the indices."""
    SPLITS.mkdir(parents=True, exist_ok=True)
    np.savez(SPLITS / f"{dataset}_seed{seed}.npz", **splits)
    digest = split_digest(splits)
    book = json.loads(DIGESTS.read_text()) if DIGESTS.exists() else {}
    book[f"{dataset}_seed{seed}"] = {"digest": digest, **{k: len(v) for k, v in splits.items()}}
    DIGESTS.write_text(json.dumps(book, indent=2, sort_keys=True) + "\n")
    return digest


def load_splits(dataset: str, seed: int) -> dict[str, np.ndarray]:
    with np.load(SPLITS / f"{dataset}_seed{seed}.npz") as z:
        return {k: z[k] for k in z.files}
