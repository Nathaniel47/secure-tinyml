"""Attack grid on a trained model: fp32 vs int8, one CSV row per (attack, strength, crafted on,
evaluated on). Uses the test set, seed taken from the model file name.

    python -m securetinyml.eval.run_attacks cifar10 --seed 0 [--n 1000]

Terms: "crafted on" is the model the attacker differentiates (fp32, or the fake-quant surrogate)
or queries (int8, black-box); "evaluated on" is the model whose accuracy is measured.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import onnxruntime as ort
import torch

from securetinyml.attacks.backdoor import attack_success_rate
from securetinyml.attacks.evasion import fgsm, pgd, random_search
from securetinyml.attacks.patch import apply_patch, random_positions, train_patch
from securetinyml.config import DATASETS, EPS_255, PATCH_FRACTIONS
from securetinyml.data.common import ROOT, load_splits
from securetinyml.data.load import get_dataset
from securetinyml.eval.meta import write_run_meta
from securetinyml.models.quantize import FakeQuantSurrogate
from securetinyml.models.resnet_small import SmallResNet
from securetinyml.models.train import MODELS, to_tensor

RESULTS = ROOT / "results"


class Onnx:
    def __init__(self, path):
        self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def logits(self, x: np.ndarray, bs: int = 500) -> np.ndarray:
        return np.concatenate(
            [
                self.s.run(None, {"x": x[i : i + bs].astype(np.float32)})[0]
                for i in range(0, len(x), bs)
            ]
        )

    def pred(self, x) -> np.ndarray:
        return self.logits(np.asarray(x)).argmax(1)


def batched(fn, x, y, bs=250):
    return torch.cat([fn(x[i : i + bs], y[i : i + bs]) for i in range(0, len(x), bs)])


def load_model(dataset: str, tag: str, seed: int, nc: int):
    m = SmallResNet(nc)
    m.load_state_dict(torch.load(MODELS / f"{dataset}_{tag}_seed{seed}.pt"))
    return m.eval(), Onnx(MODELS / f"{dataset}_{tag}_seed{seed}_int8.onnx")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000, help="test images used for attacks")
    ap.add_argument(
        "--bd",
        type=float,
        nargs="*",
        default=None,
        help="backdoor poison rates (default: all found)",
    )
    ap.add_argument("--target", type=int, default=0)
    a = ap.parse_args()

    d = get_dataset(a.dataset)
    nc, side = len(d["classes"]), DATASETS[a.dataset]
    sp = load_splits(a.dataset, a.seed)
    model, int8 = load_model(a.dataset, "fp32", a.seed, nc)
    rng = np.random.default_rng(a.seed)
    sel = np.sort(rng.choice(len(d["y_test"]), a.n, replace=False))
    x, y = to_tensor(d["x_test"][sel]), torch.from_numpy(d["y_test"][sel])
    calib = to_tensor(d["x_train"][sp["train"]][:500])
    surrogate = FakeQuantSurrogate(model, calib)

    def acc32(xx):
        with torch.no_grad():
            return (model(xx).argmax(1) == y).float().mean().item()

    def acc8(xx):
        return float((int8.pred(xx.numpy()) == y.numpy()).mean())

    rows = []

    def add(attack, strength, crafted, evaluated, value, metric="accuracy"):
        rows.append(
            dict(
                dataset=a.dataset,
                seed=a.seed,
                attack=attack,
                strength=strength,
                crafted_on=crafted,
                evaluated_on=evaluated,
                metric=metric,
                value=round(float(value), 4),
                n=len(y),
            )
        )  # noqa: C408

    add("none", 0, "-", "fp32", acc32(x))
    add("none", 0, "-", "int8", acc8(x))
    with torch.no_grad():
        agree = float((model(x).argmax(1).numpy() == int8.pred(x.numpy())).mean())
    add("none", 0, "-", "fp32-vs-int8", agree, "prediction_agreement")

    for e in EPS_255:
        eps = e / 255
        for name, atk in (
            ("fgsm", lambda m, xx, yy, eps=eps: fgsm(m, xx, yy, eps)),
            ("pgd10", lambda m, xx, yy, eps=eps: pgd(m, xx, yy, eps, steps=10)),
        ):
            for crafted, src in (("fp32", model), ("surrogate", surrogate)):
                adv = batched(lambda xx, yy, src=src, atk=atk: atk(src, xx, yy), x, y)
                add(name, e, crafted, "fp32", acc32(adv))
                add(name, e, crafted, "int8", acc8(adv))
                # transfer: of the examples that fool the source model, how many fool int8?
                with torch.no_grad():
                    fooled = (src(adv).argmax(1) != y).numpy()
                fooled8 = int8.pred(adv.numpy()) != y.numpy()
                if fooled.any():
                    add(name, e, crafted, "int8", fooled8[fooled].mean(), "transfer_rate")
        n_bb = len(y)  # was 200 (and mislabelled n=1000) in the first run
        adv = random_search(lambda z: int8.logits(z), x[:n_bb].numpy(), y[:n_bb].numpy(), eps)
        add(
            "random_search300",
            e,
            "int8-blackbox",
            "int8",
            (int8.pred(adv) == y[:n_bb].numpy()).mean(),
        )
        print("eps", e, "done")

    # adversarial patch (targeted): trained against fp32 or the surrogate on training images
    xtr = to_tensor(d["x_train"][sp["train"]][:5000])
    for frac in PATCH_FRACTIONS:
        for crafted, src in (("fp32", model), ("surrogate", surrogate)):
            patch = train_patch(src, xtr, a.target, frac, steps=300, seed=a.seed)
            keep = y != a.target
            g = torch.Generator().manual_seed(123)
            pos = random_positions(int(keep.sum()), side, patch.shape[-1], g)
            xp = apply_patch(x[keep], patch, pos)
            with torch.no_grad():
                add(
                    "patch",
                    frac,
                    crafted,
                    "fp32",
                    (model(xp).argmax(1) == a.target).float().mean(),
                    "target_success_rate",
                )
            add(
                "patch",
                frac,
                crafted,
                "int8",
                (int8.pred(xp.numpy()) == a.target).mean(),
                "target_success_rate",
            )
        print("patch", frac, "done")

    # backdoor: models trained on poisoned data (see securetinyml.models.train --poison-rate)
    xt_u8, yt_u8 = d["x_test"], d["y_test"]
    rates = a.bd
    if rates is None:
        rates = [
            r
            for r in (0.01, 0.02, 0.05)
            if (MODELS / f"{a.dataset}_bd{int(r * 100)}_seed{a.seed}_int8.onnx").exists()
        ]
    if rates:  # baseline: how often the CLEAN model already sends triggered images to the target
        for nm, p in (("fp32", lambda u8: model(to_tensor(u8)).argmax(1).detach().numpy()),
                      ("int8", lambda u8: int8.pred(to_tensor(u8).numpy()))):  # fmt: skip
            add("backdoor", 0, "-", nm, attack_success_rate(p, xt_u8, yt_u8, a.target),
                "target_success_rate")  # fmt: skip
    for rate in rates:
        tag = f"bd{int(rate * 100)}"
        bm, b8 = load_model(a.dataset, tag, a.seed, nc)

        def p32(u8, bm=bm):
            with torch.no_grad():
                return bm(to_tensor(u8)).argmax(1).numpy()

        def p8(u8, b8=b8):
            return b8.pred(to_tensor(u8).numpy())

        for nm, p in (("fp32", p32), ("int8", p8)):
            add(
                "backdoor",
                rate,
                "-",
                nm,
                attack_success_rate(p, xt_u8, yt_u8, a.target),
                "target_success_rate",
            )
            add("backdoor", rate, "-", nm, (p(xt_u8) == yt_u8).mean(), "clean_accuracy")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"attacks_{a.dataset}_seed{a.seed}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_run_meta(out.name, dataset=a.dataset, seed=a.seed, n=a.n, backdoor_rates=rates)
    print("wrote", out)


if __name__ == "__main__":
    main()
