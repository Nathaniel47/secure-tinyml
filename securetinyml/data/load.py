"""Load raw CIFAR-10, GTSRB and PlantVillage into small uint8 arrays cached in data/interim.

Every loader returns a dict: x_train, y_train, x_test, y_test (uint8 NHWC / int64),
`classes` (names) and, where it matters, `groups_train` (see stratified_split).
Raw archives are downloaded by hand or by `python -m securetinyml.data.fetch`.
"""

from __future__ import annotations

import csv
import io
import pickle
import re
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from securetinyml.config import DATASETS
from securetinyml.data.common import INTERIM, RAW, resize_rgb

CIFAR_CLASSES = (
    "airplane automobile bird cat deer dog frog horse ship truck".split()
)  # order of the official batches.meta


def load_cifar10(archive: Path | None = None) -> dict:
    archive = archive or RAW / "cifar10" / "cifar-10-python.tar.gz"
    xs, xt, yt = [], None, None
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            name = Path(m.name).name
            if name.startswith("data_batch") or name == "test_batch":
                d = pickle.load(tar.extractfile(m), encoding="bytes")  # noqa: S301
                x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                if name == "test_batch":
                    xt, yt = x, np.array(d[b"labels"])
                else:
                    xs.append((name, x, np.array(d[b"labels"])))
    xs.sort(key=lambda t: t[0])
    return {
        "x_train": np.concatenate([t[1] for t in xs]).astype(np.uint8),
        "y_train": np.concatenate([t[2] for t in xs]).astype(np.int64),
        "x_test": xt.astype(np.uint8),
        "y_test": yt.astype(np.int64),
        "classes": list(CIFAR_CLASSES),
    }


def _crop_resize(img: Image.Image, roi: tuple[int, int, int, int], side: int) -> np.ndarray:
    x1, y1, x2, y2 = roi
    return resize_rgb(img.crop((x1, y1, x2 + 1, y2 + 1)), side)


def load_gtsrb(raw_dir: Path | None = None, side: int | None = None) -> dict:
    """Training images come with a per-class CSV (';' separated, with a sign bounding box).
    Filenames are `<track>_<frame>.ppm`: 30 frames of one physical sign form a track,
    which becomes the group id so tracks are never split across train and validation."""
    raw_dir = raw_dir or RAW / "gtsrb"
    side = side or DATASETS["gtsrb"]
    train_zip = zipfile.ZipFile(raw_dir / "GTSRB_Final_Training_Images.zip")
    test_zip = zipfile.ZipFile(raw_dir / "GTSRB_Final_Test_Images.zip")
    gt_zip = zipfile.ZipFile(raw_dir / "GTSRB_Final_Test_GT.zip")

    x, y, g, sizes = [], [], [], []
    for name in sorted(n for n in train_zip.namelist() if re.search(r"GT-\d{5}\.csv$", n)):
        folder = name.rsplit("/", 1)[0]
        cls = int(Path(name).stem.split("-")[1])
        rows = csv.DictReader(io.StringIO(train_zip.read(name).decode()), delimiter=";")
        for r in rows:
            img = Image.open(io.BytesIO(train_zip.read(f"{folder}/{r['Filename']}")))
            roi = (int(r["Roi.X1"]), int(r["Roi.Y1"]), int(r["Roi.X2"]), int(r["Roi.Y2"]))
            x.append(_crop_resize(img, roi, side))
            y.append(cls)
            g.append(cls * 1000 + int(r["Filename"].split("_")[0]))
            sizes.append((int(r["Width"]), int(r["Height"])))

    gt_name = next(n for n in gt_zip.namelist() if n.endswith(".csv"))
    xt, yt = [], []
    for r in csv.DictReader(io.StringIO(gt_zip.read(gt_name).decode()), delimiter=";"):
        member = next(n for n in test_zip.namelist() if n.endswith("/" + r["Filename"]))
        img = Image.open(io.BytesIO(test_zip.read(member)))
        roi = (int(r["Roi.X1"]), int(r["Roi.Y1"]), int(r["Roi.X2"]), int(r["Roi.Y2"]))
        xt.append(_crop_resize(img, roi, side))
        yt.append(int(r["ClassId"]))
    return {
        "x_train": np.stack(x),
        "y_train": np.array(y, dtype=np.int64),
        "groups_train": np.array(g, dtype=np.int64),
        "x_test": np.stack(xt),
        "y_test": np.array(yt, dtype=np.int64),
        "classes": [f"class_{i:02d}" for i in range(43)],
        "orig_sizes": np.array(sizes),
    }


def load_plantvillage(
    folder: Path | None = None, n_classes: int = 10, side: int | None = None, seed: int = 0
) -> dict:
    """Read an ImageFolder-style PlantVillage copy (one sub-folder per class, e.g.
    `Tomato___Early_blight/`), keep the `n_classes` largest classes and resize.
    There is no official split, so a stratified 80/20 train/test split is made with `seed`."""
    folder = folder or RAW / "plantvillage"
    side = side or DATASETS["plantvillage"]
    exts = {".jpg", ".jpeg", ".png"}
    per_class = {
        d.name: sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
        for d in folder.iterdir()
        if d.is_dir()
    }
    keep = sorted(sorted(per_class, key=lambda k: (-len(per_class[k]), k))[:n_classes])
    rng = np.random.default_rng(seed)
    xs, ys, xt, yt = [], [], [], []
    for ci, name in enumerate(keep):
        files = per_class[name]
        order = rng.permutation(len(files))
        n_test = int(round(0.2 * len(files)))
        for rank, i in enumerate(order):
            arr = resize_rgb(Image.open(files[i]), side)
            (xt if rank < n_test else xs).append(arr)
            (yt if rank < n_test else ys).append(ci)
    return {
        "x_train": np.stack(xs),
        "y_train": np.array(ys, dtype=np.int64),
        "x_test": np.stack(xt),
        "y_test": np.array(yt, dtype=np.int64),
        "classes": keep,
    }


LOADERS = {"cifar10": load_cifar10, "gtsrb": load_gtsrb, "plantvillage": load_plantvillage}


def get_dataset(name: str, refresh: bool = False) -> dict:
    """Load from the cached data/interim/<name>.npz, building it from raw data if needed."""
    cache = INTERIM / f"{name}.npz"
    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["classes"] = list(d["classes"])
        return d
    d = LOADERS[name]()
    INTERIM.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{**d, "classes": np.array(d["classes"])})
    return d
