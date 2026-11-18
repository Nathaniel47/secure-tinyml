import io
import tarfile
import zipfile

import numpy as np
from PIL import Image

from securetinyml.data import common, load


def test_split_is_disjoint_complete_and_stratified():
    y = np.repeat(np.arange(5), 100)
    s = common.stratified_split(y, {"train": 0.9, "val": 0.1}, seed=1)
    assert len(set(s["train"]) & set(s["val"])) == 0
    assert len(s["train"]) + len(s["val"]) == len(y)
    assert np.bincount(y[s["val"]]).tolist() == [10] * 5


def test_split_is_reproducible_and_seed_dependent():
    y = np.repeat(np.arange(4), 50)
    a = common.stratified_split(y, {"train": 0.8, "val": 0.2}, seed=3)
    b = common.stratified_split(y, {"train": 0.8, "val": 0.2}, seed=3)
    c = common.stratified_split(y, {"train": 0.8, "val": 0.2}, seed=4)
    assert common.split_digest(a) == common.split_digest(b)
    assert common.split_digest(a) != common.split_digest(c)


def test_group_split_keeps_groups_together():
    y = np.repeat(np.arange(3), 60)
    g = np.repeat(np.arange(6), 30)  # 2 groups of 30 per class
    s = common.stratified_split(y, {"train": 0.5, "val": 0.5}, seed=0, groups=g)
    assert not set(g[s["train"]]) & set(g[s["val"]])


def test_fractions_must_sum_to_one():
    try:
        common.stratified_split(np.zeros(10, int), {"a": 0.5, "b": 0.4}, seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_digest_changes_if_an_index_changes():
    a = {"train": np.array([1, 2, 3]), "val": np.array([4])}
    b = {"train": np.array([1, 2, 4]), "val": np.array([3])}
    assert common.split_digest(a) != common.split_digest(b)


def test_resize_rgb_shape_and_dtype():
    img = Image.fromarray(np.zeros((50, 30, 3), np.uint8))
    out = common.resize_rgb(img, 32)
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8


def _make_cifar_tar(path):
    import pickle

    def batch(n, label):
        return pickle.dumps({b"data": np.full((n, 3072), label, np.uint8), b"labels": [label] * n})

    with tarfile.open(path, "w:gz") as t:
        for name, label in [("data_batch_1", 1), ("data_batch_2", 2), ("test_batch", 3)]:
            raw = batch(4, label)
            info = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
            info.size = len(raw)
            t.addfile(info, io.BytesIO(raw))


def test_load_cifar10_layout(tmp_path):
    p = tmp_path / "c.tar.gz"
    _make_cifar_tar(p)
    d = load.load_cifar10(p)
    assert d["x_train"].shape == (8, 32, 32, 3)
    assert d["y_train"].tolist() == [1] * 4 + [2] * 4
    assert d["y_test"].tolist() == [3] * 4
    assert len(d["classes"]) == 10


def _ppm_bytes(size=40):
    buf = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), 128, np.uint8)).save(buf, format="PPM")
    return buf.getvalue()


def test_load_gtsrb_groups_and_resize(tmp_path):
    hdr = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n"
    with zipfile.ZipFile(tmp_path / "GTSRB_Final_Training_Images.zip", "w") as z:
        for cls in (0, 1):
            folder = f"GTSRB/Final_Training/Images/{cls:05d}"
            rows = hdr
            for track in (0, 1):
                for frame in (0, 1):
                    fn = f"{track:05d}_{frame:05d}.ppm"
                    z.writestr(f"{folder}/{fn}", _ppm_bytes())
                    rows += f"{fn};40;40;5;5;34;34;{cls}\n"
            z.writestr(f"{folder}/GT-{cls:05d}.csv", rows)
    with zipfile.ZipFile(tmp_path / "GTSRB_Final_Test_Images.zip", "w") as z:
        z.writestr("GTSRB/Final_Test/Images/00000.ppm", _ppm_bytes())
    with zipfile.ZipFile(tmp_path / "GTSRB_Final_Test_GT.zip", "w") as z:
        z.writestr(
            "GT-final_test.csv",
            hdr.replace("Filename", "Filename") + "00000.ppm;40;40;5;5;34;34;1\n",
        )
    d = load.load_gtsrb(tmp_path, side=32)
    assert d["x_train"].shape == (8, 32, 32, 3)
    assert len(np.unique(d["groups_train"])) == 4  # 2 classes x 2 tracks
    assert d["y_test"].tolist() == [1]


def test_load_plantvillage_keeps_largest_classes(tmp_path):
    for name, n in [("A", 10), ("B", 6), ("C", 2)]:
        (tmp_path / name).mkdir()
        for i in range(n):
            Image.fromarray(np.full((20, 20, 3), i, np.uint8)).save(tmp_path / name / f"{i}.jpg")
    d = load.load_plantvillage(tmp_path, n_classes=2, side=16)
    assert d["classes"] == ["A", "B"]
    assert len(d["y_test"]) == 2 + 1 and len(d["y_train"]) == 8 + 5
    assert d["x_train"].shape[1:] == (16, 16, 3)
