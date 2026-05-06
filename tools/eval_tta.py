"""eval_tta.py - test-time augmentation: average logits over time-shifted copies.

Often pushes accuracy 0.5-2% higher without retraining.
Usage:
    python tools/eval_tta.py --ckpt model/ckpt/best_v4.pt --shifts 0,2,-2,4,-4
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_qat_v2 import SCGNetV2  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, default=REPO / "data/val.npz")
    p.add_argument("--shifts", type=str, default="0,2,-2,4,-4,6,-6")
    p.add_argument("--bs", type=int, default=512)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch_channels = ckpt.get("channels", [1, 16, 32, 32, 3])
    inner = tuple(arch_channels[1:-1])
    model = SCGNetV2(channels=inner)
    model.load_state_dict(ckpt["state"])
    model.eval()

    val = np.load(args.data)
    X = torch.from_numpy(val["X"].astype(np.float32) / 127.0)
    y = torch.from_numpy(val["y"]).long()

    shifts = [int(s) for s in args.shifts.split(",")]
    print(f"channels={arch_channels}  shifts={shifts}  n_val={len(X)}")

    all_logits = torch.zeros(len(X), 3)
    for s in shifts:
        Xs = torch.roll(X, shifts=s, dims=2)
        for i in range(0, len(Xs), args.bs):
            all_logits[i:i + args.bs] += model(Xs[i:i + args.bs])
    all_logits /= len(shifts)
    pred = all_logits.argmax(1)
    acc = (pred == y).float().mean().item() * 100
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, pp in zip(y.numpy(), pred.numpy()):
        cm[t, pp] += 1
    print(f"TTA acc = {acc:.2f}%")
    print("CM:")
    for r in cm: print(" ", r)


if __name__ == "__main__":
    main()
