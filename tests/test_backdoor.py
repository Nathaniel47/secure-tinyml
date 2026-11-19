import numpy as np

from securetinyml.attacks.backdoor import add_trigger, attack_success_rate, poison_dataset


def _data(n=200, side=16):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (n, side, side, 3), dtype=np.uint8), rng.integers(0, 5, n)


def test_trigger_only_touches_corner():
    x, _ = _data(4)
    t = add_trigger(x, size=4)
    assert (t[:, :-4] == x[:, :-4]).all() and (t[:, :, :-4] == x[:, :, :-4]).all()
    assert not (t[:, -4:, -4:] == x[:, -4:, -4:]).all()
    assert set(np.unique(t[:, -4:, -4:])) <= {0, 255}


def test_poison_rate_target_and_inputs_untouched():
    x, y = _data()
    x0, y0 = x.copy(), y.copy()
    x2, y2, idx = poison_dataset(x, y, rate=0.05, target=2, seed=1)
    assert len(idx) == 10
    assert (y0[idx] != 2).all() and (y2[idx] == 2).all()
    assert (x == x0).all() and (y == y0).all()
    other = np.setdiff1d(np.arange(len(y)), idx)
    assert (x2[other] == x[other]).all() and (y2[other] == y[other]).all()


def test_poison_is_deterministic_per_seed():
    x, y = _data()
    a = poison_dataset(x, y, 0.1, 0, seed=3)[2]
    b = poison_dataset(x, y, 0.1, 0, seed=3)[2]
    c = poison_dataset(x, y, 0.1, 0, seed=4)[2]
    assert (a == b).all() and not (a == c).all()


def test_attack_success_rate_counts_only_non_target():
    x, y = _data(50)
    always_target = lambda imgs: np.full(len(imgs), 2)  # noqa: E731
    assert attack_success_rate(always_target, x, y, target=2) == 1.0
    never = lambda imgs: np.full(len(imgs), 1)  # noqa: E731
    assert attack_success_rate(never, x, y, target=2) == 0.0
