from pathlib import Path

from securetinyml import config

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_order_matches_plan():
    assert config.DATASET_ORDER[0] == "cifar10"
    assert set(config.DATASET_ORDER) == set(config.DATASETS)


def test_seeds_are_three_distinct():
    assert len(set(config.SEEDS)) == 3


def test_strengths_are_sane():
    assert all(0 < f < 0.5 for f in config.PATCH_FRACTIONS)
    assert all(0 < r < 0.5 for r in config.POISON_RATES)
    assert list(config.EPS_255) == sorted(config.EPS_255)


def test_env_example_has_no_values():
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            assert line.split("=", 1)[1] == ""


def test_env_is_gitignored():
    assert ".env" in (ROOT / ".gitignore").read_text().splitlines()
