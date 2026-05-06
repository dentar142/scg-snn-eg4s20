"""
INT8 quantization-aware training for SCG-CNN on EG4S20
======================================================

Trains a tiny 1D-CNN with per-tensor symmetric INT8 quantization
of weights and activations. Designed so the resulting model can be
deployed bit-exactly in Verilog using INT8 multiply, INT32 accumulate,
and a per-layer right-shift requantization step.

Architecture (kept aggressively small for EG4S20 BRAM budget):
    Input  : (B, 1, 256) int8
    L0 Conv1d(1->8, k=5, p=2) + BN + ReLU + MaxPool2 -> (B, 8, 128)
    L1 Conv1d(8->16, k=5, p=2) + BN + ReLU + MaxPool2 -> (B, 16, 64)
    L2 Conv1d(16->16, k=5, p=2) + BN + ReLU + MaxPool2 -> (B, 16, 32)
    L3 Conv1d(16->3, k=1, p=0)                          -> (B, 3, 32)
    GAP(dim=2)                                          -> (B, 3)
    Output : 3 logits

Usage:
    python train_qat.py --data data/ --out ckpt/ --epochs 30
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Fake-quant primitive
# ---------------------------------------------------------------------------
class FakeQuant(nn.Module):
    """Per-tensor symmetric INT8 fake-quant with EMA-tracked range."""

    def __init__(self, num_bits: int = 8, momentum: float = 0.1):
        super().__init__()
        self.num_bits = num_bits
        self.momentum = momentum
        self.register_buffer("running_absmax", torch.tensor(1.0))
        self.register_buffer("initialized", torch.tensor(False))
        self.qmax = 2 ** (num_bits - 1) - 1   # 127 for INT8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            curr = x.detach().abs().max().clamp_min(1e-8)
            if not self.initialized:
                self.running_absmax.copy_(curr)
                self.initialized.fill_(True)
            else:
                self.running_absmax.mul_(1 - self.momentum).add_(curr * self.momentum)
        scale = self.running_absmax / self.qmax
        # Straight-through estimator
        q = torch.clamp(torch.round(x / scale), -self.qmax, self.qmax)
        return (q * scale - x).detach() + x


class QConv1d(nn.Module):
    """Conv1d whose weights and outputs are INT8 fake-quantized."""

    def __init__(self, c_in: int, c_out: int, k: int, p: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, padding=p, bias=True)
        self.bn = nn.BatchNorm1d(c_out)
        self.act_q = FakeQuant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights at every fwd
        w = self.conv.weight
        absmax = w.detach().abs().max().clamp_min(1e-8)
        scale = absmax / 127
        wq = torch.clamp(torch.round(w / scale), -127, 127) * scale
        wq = (wq - w).detach() + w   # STE
        x = F.conv1d(x, wq, self.conv.bias, padding=self.conv.padding[0])
        x = self.bn(x)
        x = F.relu(x)
        x = self.act_q(x)
        return x


class SCGNet(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.in_q = FakeQuant()
        self.l0 = QConv1d(1, 8, 5, 2)
        self.l1 = QConv1d(8, 16, 5, 2)
        self.l2 = QConv1d(16, 16, 5, 2)
        self.l3 = QConv1d(16, n_classes, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_q(x)
        x = F.max_pool1d(self.l0(x), 2)
        x = F.max_pool1d(self.l1(x), 2)
        x = F.max_pool1d(self.l2(x), 2)
        x = self.l3(x)
        return x.mean(dim=2)   # GAP over time


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def load_npz(path: Path) -> TensorDataset:
    d = np.load(path)
    X = torch.from_numpy(d["X"].astype(np.float32) / 127.0)   # int8 -> [-1,1]
    y = torch.from_numpy(d["y"]).long()
    return TensorDataset(X, y)


def evaluate(model: SCGNet, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("ckpt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    train_ds = load_npz(args.data / "train.npz")
    val_ds = load_npz(args.data / "val.npz")
    train_dl = DataLoader(train_ds, args.bs, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, args.bs, shuffle=False)

    device = torch.device(args.device)
    model = SCGNet().to(device)

    # Class-balanced cross-entropy
    y_all = train_ds.tensors[1].numpy()
    cls_w = torch.tensor([1.0 / max((y_all == c).sum(), 1) for c in range(3)], dtype=torch.float32)
    cls_w = (cls_w / cls_w.sum() * 3.0).to(device)
    print(f"class weights = {cls_w.tolist()}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=cls_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.numel()
        sch.step()

        val_loss, val_acc = evaluate(model, val_dl, device)
        print(f"ep {ep:02d}  train_loss={train_loss/len(train_ds):.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state": model.state_dict(),
                        "val_acc": val_acc,
                        "epoch": ep}, args.out / "best.pt")

    print(f"best val_acc = {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
