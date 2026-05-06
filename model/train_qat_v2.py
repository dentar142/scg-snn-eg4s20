"""train_qat_v2.py — Round 21+: enhanced training to push acc ≥ 90%.

Improvements vs v1:
  * 4x wider channels: 1 -> 16 -> 32 -> 32 -> 3 (was 1->8->16->16->3)
  * Optional kernel-7 first layer (better SCG morphology capture)
  * Gaussian noise + time-warp data augmentation
  * Class-balanced WeightedRandomSampler (fixes BG dominance)
  * Cosine LR + warm-restart
  * 60 epochs (was 30)
  * Mixup intra-class

Compatible with the v1 export pipeline (`export_weights.py`) — same
INT8 QAT semantics, just bigger architecture.

Resource budget on EG4S20:
  L0: 1x16x5    = 80 B
  L1: 16x32x5   = 2,560 B
  L2: 32x32x5   = 5,120 B
  L3: 32x3x1    = 96 B
  Total weights = ~7.9 KB (fits in 1 B9K block; 4 B9K used = 13% BRAM)
  Activations: max 16x128 = 2 KB ping-pong = 4 KB

Usage:
    python model/train_qat_v2.py --epochs 60 --bs 256 --augment
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
# FakeQuant + QConv1d (kept identical to v1 so export pipeline reuses)
# ---------------------------------------------------------------------------
class FakeQuant(nn.Module):
    def __init__(self, num_bits: int = 8, momentum: float = 0.1):
        super().__init__()
        self.num_bits = num_bits
        self.momentum = momentum
        self.register_buffer("running_absmax", torch.tensor(1.0))
        self.register_buffer("initialized", torch.tensor(False))
        self.qmax = 2 ** (num_bits - 1) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            curr = x.detach().abs().max().clamp_min(1e-8)
            if not self.initialized:
                self.running_absmax.copy_(curr)
                self.initialized.fill_(True)
            else:
                self.running_absmax.mul_(1 - self.momentum).add_(curr * self.momentum)
        scale = self.running_absmax / self.qmax
        q = torch.clamp(torch.round(x / scale), -self.qmax, self.qmax)
        return (q * scale - x).detach() + x


class QConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, p: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, padding=p, stride=stride, bias=True)
        self.bn = nn.BatchNorm1d(c_out)
        self.act_q = FakeQuant()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight
        absmax = w.detach().abs().max().clamp_min(1e-8)
        scale = absmax / 127
        wq = torch.clamp(torch.round(w / scale), -127, 127) * scale
        wq = (wq - w).detach() + w
        x = F.conv1d(x, wq, self.conv.bias, padding=self.conv.padding[0],
                     stride=self.stride)
        x = self.bn(x)
        x = F.relu(x)
        x = self.act_q(x)
        return x


# ---------------------------------------------------------------------------
# Bigger architecture: 1 -> 16 -> 32 -> 32 -> 3
# ---------------------------------------------------------------------------
class SCGNetV2(nn.Module):
    """Three downsample modes:
       * default       : 3x maxpool stride-2 between layers (matches v1..v6)
       * no_pool=True  : no downsample (matches dummy v1_nopool)
       * stride2=True  : conv stride-2 instead of maxpool (FPGA-friendly)
    """
    def __init__(self, n_classes: int = 3, k_first: int = 5,
                 channels=(16, 32, 32), no_pool: bool = False,
                 stride2: bool = False):
        super().__init__()
        c0, c1, c2 = channels
        self.no_pool = no_pool
        self.stride2 = stride2
        s = 2 if stride2 else 1
        self.in_q = FakeQuant()
        self.l0 = QConv1d(1, c0, k_first, k_first // 2, stride=s)
        self.l1 = QConv1d(c0, c1, 5, 2, stride=s)
        self.l2 = QConv1d(c1, c2, 5, 2, stride=s)
        self.l3 = QConv1d(c2, n_classes, 1, 0, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_q(x)
        if self.stride2 or self.no_pool:
            # No explicit pool — downsampling is in conv stride (or absent)
            x = self.l0(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        else:
            x = F.max_pool1d(self.l0(x), 2)
            x = F.max_pool1d(self.l1(x), 2)
            x = F.max_pool1d(self.l2(x), 2)
            x = self.l3(x)
        return x.mean(dim=2)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def augment(x: torch.Tensor) -> torch.Tensor:
    """Apply per-sample augmentations during training (fully vectorized)."""
    B, C, L = x.shape
    out = x.clone()
    # 50% Gaussian noise — vectorized
    mask = (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1)
    out = out + mask.float() * torch.randn_like(out) * 0.04   # sigma ≈ 5/127
    # Random circular shift ±10 samples (fast time-shift augment, no resampling)
    shifts = torch.randint(-10, 11, (B,), device=x.device)
    rolled = torch.zeros_like(out)
    for s_val in shifts.unique().tolist():
        idx = (shifts == s_val).nonzero(as_tuple=True)[0]
        rolled[idx] = torch.roll(out[idx], shifts=int(s_val), dims=2)
    out = rolled
    # 20% mixup intra-batch
    if random.random() < 0.2:
        perm = torch.randperm(B, device=x.device)
        lam = 0.7 + random.random() * 0.3
        out = lam * out + (1 - lam) * out[perm]
    return out.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Dataset wrapper
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
    """power=1.0 → full inverse-freq (over-samples minorities);
       power=0.5 → sqrt-inverse (mild rebalance, retains majority signal);
       power=0.0 → uniform (= no rebalance)."""
    counts = torch.bincount(y, minlength=n_classes).float()
    weights = (1.0 / counts[y]) ** power
    return WeightedRandomSampler(weights, num_samples=len(y), replacement=True)


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, np.ndarray]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    cm = np.zeros((3, 3), dtype=np.int64)
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
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("model/ckpt"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--k-first", type=int, default=5)
    p.add_argument("--channels", type=int, nargs=3, default=[16, 32, 32],
                   help="three intermediate channels (c0,c1,c2)")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--no-pool", action="store_true", help="FPGA-aligned: skip MaxPool, run all layers at len 256")
    p.add_argument("--stride2", action="store_true", help="FPGA-friendly: stride-2 conv instead of maxpool")
    p.add_argument("--tag", type=str, default="v2", help="checkpoint suffix")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = NPZDataset(args.data / "train.npz")
    val_ds   = NPZDataset(args.data / "val.npz")
    print(f"train={len(train_ds)} val={len(val_ds)}")

    sampler = make_balanced_sampler(train_ds.y)
    train_dl = DataLoader(train_ds, args.bs, sampler=sampler, drop_last=True,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds, 512, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = SCGNetV2(k_first=args.k_first, channels=tuple(args.channels),
                     no_pool=args.no_pool, stride2=args.stride2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_conv_params = sum(p.numel() for n, p in model.named_parameters() if "conv.weight" in n)
    print(f"channels=1->{args.channels}->3  no_pool={args.no_pool}  params total={n_params}  conv weights={n_conv_params}B")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0; n_seen = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            if args.augment:
                x = augment(x)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.numel()
            n_seen += y.numel()
        sch.step()
        val_loss, val_acc, cm = evaluate(model, val_dl, device)
        print(f"ep {ep:02d}  train_loss={train_loss/n_seen:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
              f"per_class={cm.diagonal()/cm.sum(axis=1)*100}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state": model.state_dict(),
                        "val_acc": val_acc, "epoch": ep,
                        "arch": args.tag, "k_first": args.k_first,
                        "channels": [1] + list(args.channels) + [3]},
                       args.out / f"best_{args.tag}.pt")
    print(f"\nbest val_acc = {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
