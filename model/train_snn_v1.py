"""train_snn_v1.py — Spike-time SNN baseline for SCG (256->H->K FC LIF).

Architecture (mirrors Mirsadeghi et al. 2026 SS III but in 1-D and smaller):
  Input:  (B, 256) int8 SCG window
   -> direct encoding (x_t = x for all t in [0,T))
  FC1:    256 -> H linear, no bias
   -> LIF1: v <- beta*v + I; spike if v >= theta; soft reset
  FC2:    H -> K linear, no bias  (K = n_classes = 3 by default; 5 supported)
   -> LIF2: integrate-and-fire output neurons
  Output: spike count over T timesteps -> logits

Surrogate gradient: fast-sigmoid (Zenke & Ganguli 2018), |x| denominator,
no exp() needed. Only addition + multiplication (BPTT-friendly).

Usage:
    python model/train_snn_v1.py --data data_excl100 --epochs 60 --T 32 --H 64 \
        --tag snn_v1
    python model/train_snn_v1.py --data data_excl150_5class --epochs 60 \
        --T 32 --H 64 --n-classes 5 --tag snn_5class
"""
from __future__ import annotations
import argparse
import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Surrogate-gradient spike function
# ---------------------------------------------------------------------------
class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid backward.  Slope hyperparam k=10."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = grad_output / (1.0 + 10.0 * x.abs()) ** 2
        return grad


spike_fn = FastSigmoidSpike.apply


# ---------------------------------------------------------------------------
# 1-D direct-encoded LIF SNN
# ---------------------------------------------------------------------------
class SCGSnn(nn.Module):
    """256 → H → 3 fully-connected SNN.

    Uses direct (constant) input encoding: same input current x for all T
    timesteps, so the input layer is purely analog and only the hidden /
    output layers actually spike.  Matches the simplest deployable pattern.
    """

    def __init__(self, n_in: int = 256, n_hidden: int = 64, n_classes: int = 3,
                 beta: float = 0.9, threshold: float = 1.0, T: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden, bias=False)
        self.fc2 = nn.Linear(n_hidden, n_classes, bias=False)
        self.beta = beta
        self.threshold = threshold
        self.T = T
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 256) or (B, 256) float in roughly [-1, 1]."""
        if x.dim() == 3:
            x = x.squeeze(1)
        B = x.size(0)
        device = x.device

        # Pre-compute the constant input current to both layers
        I1_const = self.fc1(x)  # (B, H)

        v1 = torch.zeros(B, self.n_hidden, device=device)
        v2 = torch.zeros(B, self.n_classes, device=device)
        spike_count = torch.zeros(B, self.n_classes, device=device)

        for _ in range(self.T):
            # Hidden LIF
            v1 = self.beta * v1 + I1_const
            s1 = spike_fn(v1 - self.threshold)
            v1 = v1 - s1 * self.threshold  # soft reset

            # Output LIF
            I2 = self.fc2(s1)
            v2 = self.beta * v2 + I2
            s2 = spike_fn(v2 - self.threshold)
            v2 = v2 - s2 * self.threshold

            spike_count = spike_count + s2

        # Use spike count as logits; softmax-cross-entropy handles the scale
        return spike_count


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NPZDataset(Dataset):
    def __init__(self, path: Path):
        d = np.load(path)
        self.X = torch.from_numpy(d["X"].astype(np.float32) / 127.0)
        self.y = torch.from_numpy(d["y"]).long()

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_balanced_sampler(y: torch.Tensor, n_classes: int = 3,
                          power: float = 0.5) -> WeightedRandomSampler:
    counts = torch.bincount(y, minlength=n_classes).float()
    counts = counts.clamp(min=1.0)  # avoid div0 if any class missing
    weights = (1.0 / counts[y]) ** power
    return WeightedRandomSampler(weights, num_samples=len(y), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int = 3
             ) -> Tuple[float, float, np.ndarray]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(1)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
        correct += (pred == y).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total, cm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data_excl100"))
    p.add_argument("--out", type=Path, default=Path("model/ckpt"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--T", type=int, default=32, help="number of SNN timesteps")
    p.add_argument("--H", type=int, default=64, help="hidden-layer neuron count")
    p.add_argument("--n-classes", type=int, default=3,
                   help="number of output classes (3 default, 5 supported)")
    p.add_argument("--beta", type=float, default=0.9, help="LIF leak")
    p.add_argument("--threshold", type=float, default=1.0, help="LIF firing threshold")
    p.add_argument("--tag", type=str, default="snn_v1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = NPZDataset(args.data / "train.npz")
    val_ds   = NPZDataset(args.data / "val.npz")
    K = args.n_classes
    print(f"train={len(train_ds)} val={len(val_ds)}  n_classes={K}")
    print(f"  train per-class={torch.bincount(train_ds.y, minlength=K).tolist()}")
    print(f"  val   per-class={torch.bincount(val_ds.y,   minlength=K).tolist()}")

    sampler = make_balanced_sampler(train_ds.y, n_classes=K)
    train_dl = DataLoader(train_ds, args.bs, sampler=sampler, drop_last=True,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds, 512, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = SCGSnn(n_in=256, n_hidden=args.H, n_classes=K,
                   beta=args.beta, threshold=args.threshold, T=args.T).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SNN 256->{args.H}->{K}  T={args.T}  beta={args.beta} theta={args.threshold}  "
          f"params={n_params}  dev={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0; n_seen = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            train_loss += loss.item() * y.numel()
            n_seen += y.numel()
        sch.step()
        val_loss, val_acc, cm = evaluate(model, val_dl, device, n_classes=K)
        per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1) * 100
        print(f"ep {ep:02d}  train_loss={train_loss/n_seen:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
              f"per_class={per_class}",
              flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state": model.state_dict(),
                        "val_acc": val_acc, "epoch": ep,
                        "arch": args.tag,
                        "n_in": 256, "H": args.H, "n_classes": K,
                        "beta": args.beta, "threshold": args.threshold,
                        "T": args.T},
                       args.out / f"best_{args.tag}.pt")
    print(f"\nbest val_acc = {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
