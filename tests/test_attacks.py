import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from securetinyml.attacks.evasion import fgsm, pgd, random_search  # noqa: E402
from securetinyml.attacks.patch import (  # noqa: E402
    apply_patch,
    patch_side,
    patch_success_rate,
    train_patch,
)


@pytest.fixture()
def toy():
    """A fixed linear 3-class model on 3x8x8 images; labels are its own clean predictions,
    so clean accuracy is 100% and any drop is caused by the attack."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(192, 3))
    x = torch.rand(64, 3, 8, 8)
    y = model(x).argmax(1)
    return model, x, y


def acc(model, x, y):
    return (model(x).argmax(1) == y).float().mean().item()


def test_fgsm_zero_eps_is_identity_and_respects_budget(toy):
    model, x, y = toy
    assert torch.equal(fgsm(model, x, y, 0.0), x)
    adv = fgsm(model, x, y, 0.1)
    assert (adv - x).abs().max() <= 0.1 + 1e-6
    assert adv.min() >= 0 and adv.max() <= 1


def test_accuracy_falls_as_eps_grows(toy):
    model, x, y = toy
    a = [acc(model, fgsm(model, x, y, e), y) for e in (0.0, 0.05, 0.3)]
    assert a[0] == 1.0 and a[0] >= a[1] >= a[2] and a[2] < 0.5


def test_pgd_within_budget_and_at_least_as_strong_as_fgsm(toy):
    model, x, y = toy
    adv = pgd(model, x, y, eps=0.1, steps=10)
    assert (adv - x).abs().max() <= 0.1 + 1e-6
    assert adv.min() >= 0 and adv.max() <= 1
    assert acc(model, adv, y) <= acc(model, fgsm(model, x, y, 0.1), y)


def test_pgd_is_deterministic_per_seed(toy):
    model, x, y = toy
    assert torch.equal(pgd(model, x, y, 0.1, seed=1), pgd(model, x, y, 0.1, seed=1))


def test_random_search_uses_queries_only_and_stays_in_budget(toy):
    model, x, y = toy
    predict = lambda a: model(torch.from_numpy(a)).detach().numpy()  # noqa: E731
    xn, yn = x.numpy(), y.numpy()
    adv = random_search(predict, xn, yn, eps=0.2, queries=60)
    assert abs(adv - xn).max() <= 0.2 + 1e-6
    before = (predict(xn).argmax(1) == yn).mean()
    after = (predict(adv).argmax(1) == yn).mean()
    assert after < before


def test_apply_patch_changes_only_the_patch_region():
    x = torch.zeros(2, 3, 8, 8)
    out = apply_patch(x, torch.ones(3, 2, 2), [(1, 1), (6, 6)])
    assert out[0, :, 1:3, 1:3].min() == 1 and out[1, :, 6:8, 6:8].min() == 1
    assert out.sum() == 2 * 3 * 4
    assert x.sum() == 0


def test_patch_side_matches_area_fraction():
    assert patch_side(32, 0.0625) == 8
    assert patch_side(32, 0.10) == 10


def test_trained_patch_raises_target_rate(toy):
    model, x, y = toy
    predict = lambda a: model(a).argmax(1)  # noqa: E731
    patch0 = torch.rand(3, 6, 6)
    patch = train_patch(model, x, target=0, area_fraction=0.5, steps=200, bs=32, seed=0)
    base = patch_success_rate(predict, x, y, patch0, target=0)
    after = patch_success_rate(predict, x, y, patch, target=0)
    assert after > base and after > 0.5
