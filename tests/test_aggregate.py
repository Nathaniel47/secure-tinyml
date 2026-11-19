import math

import pytest

pd = pytest.importorskip("pandas")

from securetinyml.eval.aggregate import mean_ci, summarize, t_crit  # noqa: E402


def test_mean_ci_matches_hand_calculation():
    m, ci, n = mean_ci([0.78, 0.80, 0.82])
    assert n == 3 and m == pytest.approx(0.80)
    # sd = 0.02, half-width = t(2) * sd / sqrt(3)
    assert ci == pytest.approx(4.303 * 0.02 / math.sqrt(3), rel=1e-3)


def test_single_seed_has_no_interval():
    m, ci, n = mean_ci([0.5])
    assert m == 0.5 and math.isnan(ci) and n == 1


def test_t_crit_falls_towards_normal():
    assert t_crit(2) > t_crit(9) > t_crit(10) > t_crit(100) == 1.96


def test_summarize_groups_over_seeds():
    df = pd.DataFrame(
        {
            "attack": ["pgd", "pgd", "pgd", "fgsm"],
            "seed": [0, 1, 2, 0],
            "value": [0.1, 0.2, 0.3, 0.9],
        }
    )
    out = summarize(df, ["attack"]).set_index("attack")
    assert out.loc["pgd", "mean"] == pytest.approx(0.2) and out.loc["pgd", "seeds"] == 3
    assert out.loc["fgsm", "seeds"] == 1 and math.isnan(out.loc["fgsm", "ci95"])
