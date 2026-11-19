"""Defense grid: every defense variant against PGD, measured on the fp32 and int8 versions.

    python -m securetinyml.eval.run_defenses cifar10 --seed 0 --n 500

Attacker knowledge, written to the `adaptivity` column:
- "non-adaptive": the attack is crafted on the undefended clean model and only then fed to the
  defended model. It tests the defense against an attacker who does not know about it.
- "adaptive": the attack is crafted on the defended model itself (white-box; the purifier's
  rounding is bypassed with a straight-through gradient). A defense is only worth believing if
  it survives this one.
- "black-box": query-only random search on the defended int8 model, a check against gradient
  masking.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time

import numpy as np
import onnxruntime as ort
import torch

from securetinyml.attacks.evasion import pgd, random_search
from securetinyml.config import DATASETS
from securetinyml.data.common import ROOT, load_splits
from securetinyml.data.load import get_dataset
from securetinyml.defenses import build
from securetinyml.eval.meta import write_run_meta
from securetinyml.eval.run_attacks import Onnx, batched
from securetinyml.models.quantize import export_onnx, quantize_int8
from securetinyml.models.resnet_small import SmallResNet
from securetinyml.models.train import MODELS, to_tensor

VARIANTS = [  # (weights, defense)
    ("fp32", "none"),
    ("fp32", "purify"),
    ("fp32", "purify_blur"),
    ("fp32", "clip"),
    ("fp32", "purify+clip"),
    ("advtrain", "none"),
    ("advtrain", "clip"),
]
EPS = (2, 4)
RESULTS = ROOT / "results"


def latency_ms(sess, side, runs=100):
    x = np.random.default_rng(0).random((1, 3, side, side), dtype=np.float32)
    for _ in range(10):
        sess.run(None, {"x": x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"x": x})
        t.append((time.perf_counter() - t0) * 1000)
    return statistics.median(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()

    d = get_dataset(a.dataset)
    nc, side = len(d["classes"]), DATASETS[a.dataset]
    sp = load_splits(a.dataset, a.seed)
    rng = np.random.default_rng(a.seed)
    sel = np.sort(rng.choice(len(d["y_test"]), a.n, replace=False))
    x, y = to_tensor(d["x_test"][sel]), torch.from_numpy(d["y_test"][sel])
    calib = to_tensor(d["x_train"][sp["train"]][:500])
    tmp = MODELS / "defended"
    tmp.mkdir(exist_ok=True)

    def load(weights):
        m = SmallResNet(nc)
        m.load_state_dict(torch.load(MODELS / f"{a.dataset}_{weights}_seed{a.seed}.pt"))
        return m.eval()

    base = load("fp32")  # what a non-adaptive attacker has
    nonadaptive = {}
    for e in EPS:
        nonadaptive[e] = batched(lambda xx, yy, e=e: pgd(base, xx, yy, e / 255, steps=10), x, y)

    rows = []

    def add(weights, defense, attack, e, adaptivity, ev, value, metric="accuracy"):
        rows.append(
            dict(
                dataset=a.dataset,
                seed=a.seed,
                weights=weights,
                defense=defense,
                attack=attack,
                strength=e,
                adaptivity=adaptivity,
                evaluated_on=ev,
                metric=metric,
                value=round(float(value), 4),
                n=len(y),
            )
        )

    for weights, defense in VARIANTS:
        dm = build(load(weights), defense, calib)
        stem = f"{a.dataset}_{weights}_{defense.replace('+', '_')}_seed{a.seed}"
        export_onnx(dm, tmp / f"{stem}.onnx", side)
        quantize_int8(tmp / f"{stem}.onnx", tmp / f"{stem}_int8.onnx", calib.numpy())
        q = Onnx(tmp / f"{stem}_int8.onnx")

        def acc32(xx, dm=dm):
            with torch.no_grad():
                return (dm(xx).argmax(1) == y).float().mean().item()

        def acc8(xx, q=q):
            return float((q.pred(xx.numpy()) == y.numpy()).mean())

        add(weights, defense, "none", 0, "-", "fp32", acc32(x))
        add(weights, defense, "none", 0, "-", "int8", acc8(x))
        s32 = ort.InferenceSession(str(tmp / f"{stem}.onnx"))
        s8 = ort.InferenceSession(str(tmp / f"{stem}_int8.onnx"))
        add(
            weights, defense, "cost", 0, "-", "int8",
            (tmp / f"{stem}_int8.onnx").stat().st_size / 1024, "size_kb",
        )  # fmt: skip
        add(weights, defense, "cost", 0, "-", "fp32", latency_ms(s32, side), "latency_ms_b1")
        add(weights, defense, "cost", 0, "-", "int8", latency_ms(s8, side), "latency_ms_b1")

        for e in EPS:
            adv = nonadaptive[e]
            add(weights, defense, "pgd10", e, "non-adaptive", "fp32", acc32(adv))
            add(weights, defense, "pgd10", e, "non-adaptive", "int8", acc8(adv))
            adv = batched(lambda xx, yy, e=e, dm=dm: pgd(dm, xx, yy, e / 255, steps=10), x, y)
            add(weights, defense, "pgd10", e, "adaptive", "fp32", acc32(adv))
            add(weights, defense, "pgd10", e, "adaptive", "int8", acc8(adv))
            adv = random_search(lambda z, q=q: q.logits(z), x.numpy(), y.numpy(), e / 255)
            add(
                weights, defense, "random_search300", e, "black-box", "int8",
                (q.pred(adv) == y.numpy()).mean(),
            )  # fmt: skip
        print(weights, defense, "done", flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"defenses_{a.dataset}_seed{a.seed}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_run_meta(out.name, dataset=a.dataset, seed=a.seed, n=a.n, variants=VARIANTS, eps=EPS)
    print("wrote", out)


if __name__ == "__main__":
    main()
