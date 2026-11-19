"""Combine the per-seed result CSVs into mean +- 95% confidence interval tables.

    python -m securetinyml.eval.aggregate

Reads results/attacks_<dataset>_seed<k>.csv and results/defenses_<dataset>_seed<k>.csv, writes
results/summary_attacks.csv and results/summary_defenses.csv. The interval is a t-interval over
seeds (so it reflects training randomness only: 3 seeds give a wide interval, which is honest).
Rows with a single seed get no interval (NaN), not a fake one.
"""

from __future__ import annotations

import glob
import math

import pandas as pd

from securetinyml.data.common import ROOT

RESULTS = ROOT / "results"
# two-sided 95% critical values of Student's t, by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def t_crit(df: int) -> float:
    return T95.get(df, 2.228 if df <= 10 else 1.96)


def mean_ci(values) -> tuple[float, float, int]:
    v = [float(x) for x in values]
    n = len(v)
    m = sum(v) / n
    if n < 2:
        return m, math.nan, n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m, t_crit(n - 1) * sd / math.sqrt(n), n


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and 95% CI half-width of `value` over seeds, one row per key combination."""
    rows = []
    for k, g in df.groupby(keys, dropna=False, sort=True):
        m, ci, n = mean_ci(g["value"])
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,), strict=True)),
                     "mean": round(m, 4), "ci95": round(ci, 4), "seeds": n})  # fmt: skip
    return pd.DataFrame(rows)


def load(prefix: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(RESULTS / f"{prefix}_*_seed*.csv")))
    return (
        pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    )


def main() -> None:
    att = load("attacks")
    if len(att):
        keys = ["dataset", "attack", "strength", "crafted_on", "evaluated_on", "metric"]
        summarize(att, keys).to_csv(RESULTS / "summary_attacks.csv", index=False)
    dfn = load("defenses")
    if len(dfn):
        keys = ["dataset", "weights", "defense", "attack", "strength", "adaptivity",
                "evaluated_on", "metric"]  # fmt: skip
        summarize(dfn, keys).to_csv(RESULTS / "summary_defenses.csv", index=False)
    print("attacks rows:", len(att), "defense rows:", len(dfn))


if __name__ == "__main__":
    main()
