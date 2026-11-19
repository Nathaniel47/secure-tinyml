"""Charts drawn only from results/summary_*.csv (run `python -m securetinyml.eval.aggregate`
first). Error bars are 95% t-intervals over seeds; a bar with no whisker has one seed.

    python -m securetinyml.eval.charts
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from securetinyml.data.common import ROOT  # noqa: E402

RESULTS = ROOT / "results"
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#777777"  # colour-blind-safe pair plus neutral


def _err(df):
    return df["ci95"].fillna(0).to_numpy() * 100


def pgd_chart(att: pd.DataFrame) -> None:
    d = att[(att.attack == "pgd10") & (att.crafted_on == "fp32") & (att.metric == "accuracy")]
    sets = sorted(d.dataset.unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(4.5 * len(sets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], sets, strict=True):
        for ev, col, off in (("fp32", BLUE, -0.2), ("int8", ORANGE, 0.2)):
            g = d[(d.dataset == ds) & (d.evaluated_on == ev)].sort_values("strength")
            x = np.arange(len(g))
            ax.bar(x + off, g["mean"] * 100, 0.4, yerr=_err(g), color=col, label=ev, capsize=3)
        ax.set_xticks(np.arange(len(g)), [f"{int(s)}/255" for s in g.strength])
        ax.set_title(f"{ds}: accuracy under PGD-10")
        ax.set_xlabel("attack strength (eps)")
        ax.set_ylabel("accuracy (%)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "chart_pgd_fp32_vs_int8.png", dpi=150)


def defense_chart(dfn: pd.DataFrame, eps: int = 2) -> None:
    d = dfn[
        (dfn.attack == "pgd10")
        & (dfn.strength == eps)
        & (dfn.evaluated_on == "int8")
        & (dfn.metric == "accuracy")
    ].copy()
    d["name"] = d.weights + "\n" + d.defense
    sets = sorted(d.dataset.unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(6.5 * len(sets), 3.8), squeeze=False)
    for ax, ds in zip(axes[0], sets, strict=True):
        names = list(d[d.dataset == ds].name.unique())
        for kind, col, off in (("non-adaptive", GREY, -0.2), ("adaptive", ORANGE, 0.2)):
            g = d[(d.dataset == ds) & (d.adaptivity == kind)].set_index("name").reindex(names)
            ax.bar(np.arange(len(names)) + off, g["mean"] * 100, 0.4, yerr=_err(g), color=col,
                   label=kind, capsize=3)  # fmt: skip
        ax.set_xticks(np.arange(len(names)), names, fontsize=7)
        ax.set_title(f"{ds}: int8 accuracy under PGD-10, eps {eps}/255")
        ax.set_ylabel("accuracy (%)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "chart_defenses.png", dpi=150)


CLEAN_LABELS = {
    ("fp32", "none"): "No defense",
    ("fp32", "clip"): "Activation clipping",
    ("fp32", "purify"): "Purify (4-bit)",
    ("fp32", "purify_blur"): "Purify (5-bit + blur)",
    ("advtrain", "none"): "Adversarial training",
}


def defense_chart_clean(dfn: pd.DataFrame, eps: int = 2) -> None:
    """Plain-labelled version of defense_chart for the portfolio: five defenses, one row each."""
    d = dfn[
        (dfn.attack == "pgd10")
        & (dfn.strength == eps)
        & (dfn.evaluated_on == "int8")
        & (dfn.metric == "accuracy")
    ]
    sets = sorted(d.dataset.unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(5.4 * len(sets), 3.6), squeeze=False)
    labels = list(CLEAN_LABELS.values())
    y = np.arange(len(labels))[::-1]
    for ax, ds in zip(axes[0], sets, strict=True):
        for kind, col, off in (("non-adaptive", GREY, 0.2), ("adaptive", ORANGE, -0.2)):
            g = d[(d.dataset == ds) & (d.adaptivity == kind)].set_index(["weights", "defense"])
            g = g.reindex(list(CLEAN_LABELS))
            mean = g["mean"].to_numpy() * 100
            err = np.minimum(_err(g), np.minimum(mean, 100 - mean))  # keep whiskers inside 0-100
            ax.barh(y + off, mean, 0.38, xerr=err, color=col, capsize=2.5, label=kind)
            for yi, m, e in zip(y + off, mean, err, strict=True):
                ax.text(m + e + 1.5, yi, f"{m:.0f}%", va="center", fontsize=8)
        ax.set_yticks(y, labels, fontsize=9)
        ax.set_xlim(0, 108)
        ax.set_xlabel("accuracy under attack (%)")
        ax.set_title({"cifar10": "CIFAR-10", "gtsrb": "GTSRB"}.get(ds, ds), fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
    axes[0][0].legend(loc="upper right", frameon=False, fontsize=8.5)
    title = f"int8 model accuracy under PGD-10 (eps {eps}/255); whiskers show 95% interval"
    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    fig.savefig(RESULTS / "chart_defenses_clean.png", dpi=150)


def patch_backdoor_chart(att: pd.DataFrame) -> None:
    d = att[(att.metric == "target_success_rate") & (att.attack.isin(["patch", "backdoor"]))]
    d = d[(d.crafted_on.isin(["fp32", "-"]))]
    sets = sorted(d.dataset.unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(6 * len(sets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], sets, strict=True):
        g = d[d.dataset == ds].copy()
        g["name"] = g.attack + " " + g.strength.astype(str)
        names = list(g.name.unique())
        for ev, col, off in (("fp32", BLUE, -0.2), ("int8", ORANGE, 0.2)):
            h = g[g.evaluated_on == ev].set_index("name").reindex(names)
            ax.bar(np.arange(len(names)) + off, h["mean"] * 100, 0.4, yerr=_err(h), color=col,
                   label=ev, capsize=3)  # fmt: skip
        ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{ds}: targeted attack success (backdoor 0 = clean model)")
        ax.set_ylabel("success (%)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "chart_patch_backdoor.png", dpi=150)


def main() -> None:
    att, dfn = RESULTS / "summary_attacks.csv", RESULTS / "summary_defenses.csv"
    if att.exists():
        a = pd.read_csv(att)
        pgd_chart(a)
        patch_backdoor_chart(a)
    if dfn.exists():
        defense_chart(pd.read_csv(dfn))
        defense_chart_clean(pd.read_csv(dfn))
    print("charts written to", RESULTS)


if __name__ == "__main__":
    main()
