"""Build the cached arrays and the seeded train/val splits for every available dataset.

python -m securetinyml.data.prepare            # all datasets whose raw data is present
python -m securetinyml.data.prepare cifar10
"""

from __future__ import annotations

import sys

from securetinyml.config import DATASET_ORDER, SEEDS
from securetinyml.data.common import RAW, save_splits, stratified_split
from securetinyml.data.load import get_dataset

VAL_FRACTION = 0.1


def make_splits(d: dict, seed: int) -> dict:
    parts = stratified_split(
        d["y_train"],
        {"train": 1 - VAL_FRACTION, "val": VAL_FRACTION},
        seed,
        groups=d.get("groups_train"),
    )
    return {**parts}


def prepare(name: str) -> None:
    d = get_dataset(name)
    for seed in SEEDS:
        digest = save_splits(name, seed, make_splits(d, seed))
        print(f"{name} seed {seed}: digest {digest}")


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in DATASET_ORDER if (RAW / n).exists()]
    for n in names:
        prepare(n)
