"""Cross-check the hand-written attacks against the Adversarial Robustness Toolbox (ART).

    python -m securetinyml.eval.art_check cifar10 --seed 0 --n 500

Same model, same images, same eps: ART's FGSM/PGD and patch against ours. If our code and ART
disagree a lot, one of them has a bug. Writes results/art_check_<dataset>_seed<seed>.csv.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from securetinyml.attacks.evasion import fgsm, pgd
from securetinyml.config import DATASETS
from securetinyml.data.common import ROOT
from securetinyml.data.load import get_dataset
from securetinyml.eval.meta import write_run_meta
from securetinyml.models.resnet_small import SmallResNet
from securetinyml.models.train import MODELS, to_tensor


def main() -> None:
    import art
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()

    d = get_dataset(a.dataset)
    nc, side = len(d["classes"]), DATASETS[a.dataset]
    model = SmallResNet(nc)
    model.load_state_dict(torch.load(MODELS / f"{a.dataset}_fp32_seed{a.seed}.pt"))
    model.eval()
    sel = np.sort(np.random.default_rng(a.seed).choice(len(d["y_test"]), a.n, replace=False))
    x, y = to_tensor(d["x_test"][sel]), torch.from_numpy(d["y_test"][sel])
    clf = PyTorchClassifier(
        model,
        torch.nn.CrossEntropyLoss(),
        input_shape=(3, side, side),
        nb_classes=nc,
        clip_values=(0.0, 1.0),
    )

    def acc(xx):
        with torch.no_grad():
            return (model(torch.as_tensor(xx)).argmax(1) == y).float().mean().item()

    y1h = np.eye(nc, dtype=np.float32)[y.numpy()]  # ART would otherwise attack its own predictions
    rows = []

    def add(attack, eps, impl, value):
        rows.append(
            dict(
                dataset=a.dataset,
                seed=a.seed,
                attack=attack,
                strength=eps,
                implementation=impl,
                accuracy=round(float(value), 4),
                n=len(y),
                art_version=art.__version__,
            )
        )

    add("none", 0, "ours", acc(x))
    for e in (2, 4, 8):
        eps = e / 255
        add("fgsm", e, "ours", acc(fgsm(model, x, y, eps)))
        add("fgsm", e, "art", acc(FastGradientMethod(clf, eps=eps).generate(x.numpy(), y=y1h)))
        add("pgd10", e, "ours", acc(pgd(model, x, y, eps, steps=10)))
        art_pgd = ProjectedGradientDescent(
            clf, eps=eps, eps_step=2.5 * eps / 10, max_iter=10, num_random_init=1, verbose=False
        )
        add("pgd10", e, "art", acc(art_pgd.generate(x.numpy(), y=y1h)))
        print("eps", e, "done", flush=True)

    out = ROOT / "results" / f"art_check_{a.dataset}_seed{a.seed}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_run_meta(out.name, dataset=a.dataset, seed=a.seed, n=a.n, art_version=art.__version__)
    print("wrote", out)


if __name__ == "__main__":
    main()
