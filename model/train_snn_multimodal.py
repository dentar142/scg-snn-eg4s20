"""train_snn_multimodal.py — multi-modal SNN for FOSTER 5-modality corpus.

Extends SCGSnn to handle (B, 5, 256) input (5 simultaneously-recorded mechanical
/ acoustic cardiac modalities: PVDF, PZT, ACC, PCG, ERB).  ECG is NOT in input
(only used for label derivation in the dataset pipeline).

Architecture:
  Input:  (B, 5, 256) int8 → flatten to (B, 5*256=1280)
  FC1:    1280 → H linear (5x larger fan-in than single-modal)
  LIF1, FC2, LIF2: same as single-modal SCGSnn
  Output: (B, K) spike count

Usage:
    python model/train_snn_multimodal.py --data data_foster_multi \
        --epochs 60 --bs 256 --H 64 --tag snn_foster
"""
from __future__ import annotations
import argparse
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output / (1.0 + 10.0 * x.abs()) ** 2


spike_fn = FastSigmoidSpike.apply


class MultiModalSCGSnn(nn.Module):
    """(B, C, L) input -> flatten C*L -> H -> K LIF SNN."""
    def __init__(self, n_in: int = 256, n_channels: int = 5,
                 n_hidden: int = 64, n_classes: int = 3,
                 beta: float = 0.9, threshold: float = 1.0, T: int = 32):
        super().__init__()
        self.n_in = n_in
        self.n_channels = n_channels
        self.fc1 = nn.Linear(n_channels * n_in, n_hidden, bias=False)
        self.fc2 = nn.Linear(n_hidden, n_classes, bias=False)
        self.beta = beta
        self.threshold = threshold
        self.T = T
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → (B, C*L)
        if x.dim() == 3:
            B = x.size(0)
            x = x.reshape(B, -1)
        I1_const = self.fc1(x)
        device = x.device
        v1 = torch.zeros(I1_const.size(0), self.n_hidden, device=device)
        v2 = torch.zeros(I1_const.size(0), self.n_classes, device=device)
        spike_count = torch.zeros(I1_const.size(0), self.n_classes, device=device)
        for _ in range(self.T):
            v1 = self.beta * v1 + I1_const
            s1 = spike_fn(v1 - self.threshold)
            v1 = v1 - s1 * self.threshold
            I2 = self.fc2(s1)
            v2 = self.beta * v2 + I2
            s2 = spike_fn(v2 - self.threshold)
            v2 = v2 - s2 * self.threshold
            spike_count = spike_count + s2
        return spike_count


class NPZDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X may be (N, C, L) or (N, 1, L); both OK
        self.X = torch.from_numpy(X.astype(np.float32) / 127.0)
        self.y = torch.from_numpy(y).long()

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_balanced_sampler(y: torch.Tensor, n_classes: int = 3,
                          power: float = 0.5) -> WeightedRandomSampler:
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1.0)
    weights = (1.0 / counts[y]) ** power
    return WeightedRandomSampler(weights, num_samples=len(y), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int = 3):
    model.eval()
    correct = total = 0
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    loss_sum = 0.0
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
    p.add_argument("--data", type=Path, required=True,
                   help="dir with all.npz containing X (N,C,L), y, sid")
    p.add_argument("--out", type=Path, default=Path("model/ckpt"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--n-classes", type=int, default=3)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--tag", type=str, default="snn_mm")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data / "all.npz", allow_pickle=True)
    X = d["X"]; y = d["y"]; sid = d["sid"]
    print(f"loaded: X={X.shape} y={y.shape} unique_sid={len(set(sid.tolist()))}")
    print(f"  modalities={[str(s) for s in d.get('modalities', [])]}")
    print(f"  per-class={np.bincount(y, minlength=args.n_classes).tolist()}")

    # Subject-stratified train/val (random per-subject split, NOT subject-disjoint here —
    # use cross_val.py for that). For initial sanity check only.
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(X))
    n_val = int(args.val_fraction * len(X))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    train_ds = NPZDataset(X_tr, y_tr)
    val_ds = NPZDataset(X_va, y_va)
    sampler = make_balanced_sampler(train_ds.y, args.n_classes)
    train_dl = DataLoader(train_ds, args.bs, sampler=sampler, drop_last=True, num_workers=0)
    val_dl = DataLoader(val_ds, 512, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    n_channels = X.shape[1]
    model = MultiModalSCGSnn(n_in=X.shape[2], n_channels=n_channels,
                             n_hidden=args.H, n_classes=args.n_classes,
                             beta=args.beta, threshold=args.threshold, T=args.T).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MultiModal SNN ({n_channels}*{X.shape[2]}={n_channels*X.shape[2]})->{args.H}->{args.n_classes}  "
          f"params={n_params}  dev={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0; n_seen = 0
        train_correct = 0
        for x, yt in train_dl:
            x, yt = x.to(device), yt.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, yt, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            train_loss += loss.item() * yt.numel()
            n_seen += yt.numel()
            train_correct += (logits.argmax(1) == yt).sum().item()
        sch.step()
        train_acc = train_correct / max(n_seen, 1)
        val_loss, val_acc, cm = evaluate(model, val_dl, device, args.n_classes)
        print(f"ep {ep:02d} train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}% "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state": model.state_dict(),
                        "val_acc": val_acc, "epoch": ep,
                        "arch": args.tag,
                        "n_in": X.shape[2], "n_channels": n_channels,
                        "H": args.H, "n_classes": args.n_classes,
                        "beta": args.beta, "threshold": args.threshold,
                        "T": args.T},
                       args.out / f"best_{args.tag}.pt")
    print(f"\nbest val_acc = {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
