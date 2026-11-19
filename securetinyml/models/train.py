"""Train the small ResNet on one dataset (CPU is fine for CIFAR-10 at this size).

    python -m securetinyml.models.train cifar10 --epochs 8 --seed 0 [--poison-rate 0.05]

Writes models/<dataset>_<tag>_seed<seed>.pt (gitignored), and a .onnx plus an int8 .onnx.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn

from securetinyml.attacks.backdoor import poison_dataset
from securetinyml.config import DATASETS
from securetinyml.data.common import ROOT, load_splits
from securetinyml.data.load import get_dataset
from securetinyml.models.resnet_small import SmallResNet

MODELS = ROOT / "models"


def to_tensor(x_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float().div(255)


def augment(x: torch.Tensor, flip: bool, g: torch.Generator) -> torch.Tensor:
    """Random crop (pad 4) and, if `flip`, horizontal flip. Traffic signs are not flipped."""
    n, _, h, w = x.shape
    xp = nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2,), generator=g).tolist()
    x = xp[:, :, i : i + h, j : j + w]
    if flip:
        mask = torch.rand(n, generator=g) < 0.5
        x = torch.where(mask.view(-1, 1, 1, 1), x.flip(3), x)
    return x


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, bs: int = 500) -> float:
    model.eval()
    hits = sum(
        (model(x[i : i + bs]).argmax(1) == y[i : i + bs]).sum().item() for i in range(0, len(x), bs)
    )
    return hits / len(x)


def train(
    x, y, num_classes, epochs, seed, flip=True, width=16, lr=0.1, bs=128, log=print, ckpt=None
):
    """If `ckpt` is a path, state is saved after every epoch and training resumes from it."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = SmallResNet(num_classes, width)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = epochs * ((len(x) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()
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
            loss = loss_fn(model(augment(x[idx], flip, g)), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        log(f"epoch {ep + 1}/{epochs} loss {tot / len(x):.3f} ({time.time() - t0:.0f}s)")
        if ckpt is not None:
            state = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict())
            torch.save({**state, "g": g.get_state(), "epoch": ep + 1}, ckpt)
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poison-rate", type=float, default=0.0)
    ap.add_argument("--target", type=int, default=0, help="backdoor target class")
    a = ap.parse_args()

    from securetinyml.models.quantize import export_onnx, quantize_int8

    d = get_dataset(a.dataset)
    sp = load_splits(a.dataset, a.seed)
    xtr, ytr = d["x_train"][sp["train"]], d["y_train"][sp["train"]]
    tag = "fp32"
    if a.poison_rate > 0:
        xtr, ytr, _ = poison_dataset(xtr, ytr, a.poison_rate, a.target, seed=a.seed)
        tag = f"bd{int(a.poison_rate * 100)}"
    nc = len(d["classes"])
    MODELS.mkdir(exist_ok=True)
    model = train(
        to_tensor(xtr),
        torch.from_numpy(ytr),
        nc,
        a.epochs,
        a.seed,
        flip=a.dataset == "cifar10",
        ckpt=MODELS / f"ckpt_{a.dataset}_{tag}_seed{a.seed}.pt",
    )
    xv, yv = to_tensor(d["x_train"][sp["val"]]), torch.from_numpy(d["y_train"][sp["val"]])
    xt, yt = to_tensor(d["x_test"]), torch.from_numpy(d["y_test"])
    print(f"val acc {accuracy(model, xv, yv):.4f}  test acc {accuracy(model, xt, yt):.4f}")
    MODELS.mkdir(exist_ok=True)
    stem = f"{a.dataset}_{tag}_seed{a.seed}"
    torch.save(model.state_dict(), MODELS / f"{stem}.pt")
    export_onnx(model, MODELS / f"{stem}.onnx", DATASETS[a.dataset])
    calib = to_tensor(d["x_train"][sp["train"]][:500]).numpy()  # clean images only
    quantize_int8(MODELS / f"{stem}.onnx", MODELS / f"{stem}_int8.onnx", calib)
    print("saved", stem)


if __name__ == "__main__":
    main()
