import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from securetinyml.eval import charts  # noqa: E402


def _summary(datasets=("cifar10", "gtsrb")):
    rows = []
    for ds in datasets:
        for (weights, defense), _ in charts.CLEAN_LABELS.items():
            for kind in ("non-adaptive", "adaptive"):
                rows.append(
                    dict(
                        dataset=ds,
                        weights=weights,
                        defense=defense,
                        attack="pgd10",
                        strength=2,
                        adaptivity=kind,
                        evaluated_on="int8",
                        metric="accuracy",
                        mean=0.5,
                        ci95=0.9,
                        seeds=3,
                    )  # fmt: skip
                )
    return pd.DataFrame(rows)


def test_clean_defense_chart_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(charts, "RESULTS", tmp_path)
    charts.defense_chart_clean(_summary())
    assert (tmp_path / "chart_defenses_clean.png").stat().st_size > 0


def test_clean_defense_chart_handles_one_dataset_and_missing_defense(tmp_path, monkeypatch):
    monkeypatch.setattr(charts, "RESULTS", tmp_path)
    df = _summary(("gtsrb",))
    df = df[df.defense != "clip"]  # reindex gives NaN for the missing row; must not crash
    charts.defense_chart_clean(df)
    assert (tmp_path / "chart_defenses_clean.png").exists()
