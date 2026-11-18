"""Download CIFAR-10 and GTSRB into data/raw and verify them.

    python -m securetinyml.data.fetch

PlantVillage has no stable direct link: download it by hand (Mendeley record
https://data.mendeley.com/datasets/tywbtsjrjv/1 or a Kaggle copy), unzip so each class is a
sub-folder, and place it at data/raw/plantvillage/. See the README.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from securetinyml.data.common import RAW, md5_file, sha256_file

ERDA = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
# CIFAR-10 md5 is the value I remember from the torchvision project (the Toronto page shows
# none I could read), so a mismatch means check the file, not necessarily corruption.
# GTSRB publishes no checksum; the sha256 printed after download is recorded in the README.
FILES = {
    "cifar10/cifar-10-python.tar.gz": (
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "md5",
        "c58f30108f718f92721af3b95e74349a",
    ),
    "gtsrb/GTSRB_Final_Training_Images.zip": (
        f"{ERDA}/GTSRB_Final_Training_Images.zip",
        None,
        None,
    ),
    "gtsrb/GTSRB_Final_Test_Images.zip": (f"{ERDA}/GTSRB_Final_Test_Images.zip", None, None),
    "gtsrb/GTSRB_Final_Test_GT.zip": (f"{ERDA}/GTSRB_Final_Test_GT.zip", None, None),
}


def fetch(rel: str, url: str, algo: str | None, expected: str | None) -> Path:
    dest = RAW / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (fixed https URLs above)
    if algo == "md5" and md5_file(dest) != expected:
        raise RuntimeError(f"{rel}: md5 mismatch, delete the file and retry")
    print(f"{rel}: ok, sha256 {sha256_file(dest)[:16]}")
    return dest


if __name__ == "__main__":
    for rel, spec in FILES.items():
        fetch(rel, *spec)
