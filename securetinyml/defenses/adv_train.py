"""Adversarial training, cheap version: fine-tune a clean model on one-step attacked images
("fast" adversarial training with a random start, Wong et al. 2020). A full PGD loop costs
3-8x normal training; this costs about 2x, which is what a CPU laptop can afford.

    python -m securetinyml.defenses.adv_train cifar10 --epochs 6 --eps 4 --seed 0
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from securetinyml.config import DATASETS
from securetinyml.data.common import load_splits
from securetinyml.data.load import get_dataset
from securetinyml.models.resnet_small import SmallResNet
from securetinyml.models.train import MODELS, accuracy, augment, to_tensor


def fast_adv_batch(model, x, y, eps, g):
    delta = (torch.rand(x.shape, generator=g) * 2 - 1) * eps
    xa = (x + delta).clamp(0, 1).requires_grad_(True)
    (grad,) = torch.autograd.grad(nn.functional.cross_entropy(model(xa), y), xa)
    xa = xa.detach() + 1.25 * eps * grad.sign()
    return torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1).detach()


def adv_finetune(model, x, y, epochs, eps, seed, flip=True, lr=0.05, bs=128, log=print, ckpt=None):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * ((len(x) + bs - 1) // bs)
    )
    start = 0
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        g.set_state(st["g"])
        start = st["epoch"]
        log(f"resumed after epoch {start}")
    for ep in range(start, epochs):
        model.train()
        perm = torch.randperm(len(x), generator=g)
        t0, tot = time.time(), 0.0
        for k in range(0, len(x), bs):
            idx = perm[k : k + bs]
            xb = augment(x[idx], flip, g)
            xa = fast_adv_batch(model, xb, y[idx], eps, g)
            loss = nn.functional.cross_entropy(model(xa), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        log(f"epoch {ep + 1}/{epochs} adv-loss {tot / len(x):.3f} ({time.time() - t0:.0f}s)")
        if ckpt is not None:
            state = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict())
            torch.save({**state, "g": g.get_state(), "epoch": ep + 1}, ckpt)
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--eps", type=float, default=4, help="in 1/255")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from securetinyml.models.quantize import export_onnx, quantize_int8

    d = get_dataset(a.dataset)
    sp = load_splits(a.dataset, a.seed)
    model = SmallResNet(len(d["classes"]))
    model.load_state_dict(torch.load(MODELS / f"{a.dataset}_fp32_seed{a.seed}.pt"))
    x = to_tensor(d["x_train"][sp["train"]])
    y = torch.from_numpy(d["y_train"][sp["train"]])
    model = adv_finetune(
        model,
        x,
        y,
        a.epochs,
        a.eps / 255,
        a.seed,
        flip=a.dataset == "cifar10",
        ckpt=MODELS / f"ckpt_{a.dataset}_advtrain_seed{a.seed}.pt",
    )
    xt, yt = to_tensor(d["x_test"]), torch.from_numpy(d["y_test"])
    print(f"test acc {accuracy(model, xt, yt):.4f}")
    stem = f"{a.dataset}_advtrain_seed{a.seed}"
    torch.save(model.state_dict(), MODELS / f"{stem}.pt")
    export_onnx(model, MODELS / f"{stem}.onnx", DATASETS[a.dataset])
    quantize_int8(MODELS / f"{stem}.onnx", MODELS / f"{stem}_int8.onnx", x[:500].numpy())
    print("saved", stem)


if __name__ == "__main__":
    main()
