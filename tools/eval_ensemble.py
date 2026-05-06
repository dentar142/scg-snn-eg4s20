"""eval_ensemble.py — average logits across multiple checkpoints + TTA."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_qat_v2 import SCGNetV2  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--data", type=Path, default=REPO / "data/val.npz")
    p.add_argument("--shifts", type=str, default="0,2,-2,4,-4,6,-6,8,-8")
    args = p.parse_args()

    val = np.load(args.data)
    X = torch.from_numpy(val["X"].astype(np.float32) / 127.0)
    y = torch.from_numpy(val["y"]).long()

    shifts = [int(s) for s in args.shifts.split(",")]
    all_logits = torch.zeros(len(X), 3)
    n_models = 0

    for ck_path in args.ckpts:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        inner = tuple(ck["channels"][1:-1])
        m = SCGNetV2(channels=inner)
        m.load_state_dict(ck["state"]); m.eval()
        per_model = torch.zeros(len(X), 3)
        for s in shifts:
            Xs = torch.roll(X, shifts=s, dims=2)
            for i in range(0, len(Xs), 512):
                per_model[i:i+512] += m(Xs[i:i+512])
        per_model /= len(shifts)
        # Per-model accuracy
        pp = per_model.argmax(1)
        acc1 = (pp == y).float().mean().item() * 100
        print(f"  {Path(ck_path).name}: TTA acc = {acc1:.2f}%")
        all_logits += per_model
        n_models += 1

    all_logits /= n_models
    pred = all_logits.argmax(1)
    acc = (pred == y).float().mean().item() * 100
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, pp in zip(y.numpy(), pred.numpy()):
        cm[t, pp] += 1
    print(f"\nENSEMBLE ({n_models} models, {len(shifts)} shifts) acc = {acc:.2f}%")
    print("CM:")
    for r in cm: print(" ", r)
    diag = cm.diagonal() / cm.sum(axis=1) * 100
    print(f"Per-class: BG={diag[0]:.2f}% Sys={diag[1]:.2f}% Dia={diag[2]:.2f}%")


if __name__ == "__main__":
    main()
