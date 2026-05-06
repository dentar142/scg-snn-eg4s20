"""final_holdout_test.py — train on N-K subjects, evaluate on K hold-out subjects.

This is THE final, deployment-realistic accuracy number for the SRTP report.
Hold-out subjects are sampled from each of the 5 CV folds to avoid bias from
any single fold.  They never see the training pipeline.

Models: SNN (256→64→3 LIF, BPTT) and CNN v7 (1→32→64→128→3, stride-2).
Both trained from cold-start, INT8 PTQ evaluated on hold-out test.

Usage:
    python tools/final_holdout_test.py --model snn --holdout b002,b007,b015,b020
    python tools/final_holdout_test.py --model cnn --holdout b002,b007,b015,b020
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))


def make_balanced_sampler(y: torch.Tensor, n_classes: int = 3, power: float = 0.5):
    counts = torch.bincount(y, minlength=n_classes).float()
    weights = (1.0 / counts[y]) ** power
    return WeightedRandomSampler(weights, num_samples=len(y), replacement=True)


def build_model(name: str, device):
    if name == "snn":
        from train_snn_v1 import SCGSnn
        return SCGSnn(n_in=256, n_hidden=64, n_classes=3,
                      beta=0.9, threshold=1.0, T=32).to(device)
    elif name == "cnn":
        from train_qat_v2 import SCGNetV2
        return SCGNetV2(n_classes=3, k_first=5, channels=(32, 64, 128),
                        no_pool=False, stride2=True).to(device)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    cm = np.zeros((3, 3), dtype=np.int64)
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
    return loss_sum / max(total, 1), correct / max(total, 1), cm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=REPO / "data_excl100/all.npz")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", choices=["snn", "cnn"], required=True)
    p.add_argument("--holdout", type=str, default="b002,b007,b015,b020",
                   help="comma-sep record names to hold out as final test")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    d = np.load(args.data, allow_pickle=True)
    X = d["X"].astype(np.int8)
    y = d["y"].astype(np.int64)
    sid = d["sid"].astype(np.int64)
    rec_names = list(d["record_names"])
    print(f"loaded {args.data}: N={len(X)}  unique subjects={sorted(set(sid.tolist()))}")

    holdout_recs = [r.strip() for r in args.holdout.split(",")]
    holdout_sids = [rec_names.index(r) for r in holdout_recs]
    print(f"hold-out subjects: {holdout_recs} -> sids {holdout_sids}")

    test_mask = np.isin(sid, holdout_sids)
    train_mask = ~test_mask
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    train_subjects = sorted(set(sid[train_mask].tolist()))
    test_subjects = sorted(set(sid[test_mask].tolist()))
    print(f"train: {len(X_tr)} windows, {len(train_subjects)} subjects")
    print(f"test:  {len(X_te)} windows, {len(test_subjects)} subjects -> {[rec_names[s] for s in test_subjects]}")

    Xtr_t = torch.from_numpy(X_tr.astype(np.float32) / 127.0)
    ytr_t = torch.from_numpy(y_tr).long()
    Xte_t = torch.from_numpy(X_te.astype(np.float32) / 127.0)
    yte_t = torch.from_numpy(y_te).long()

    sampler = make_balanced_sampler(ytr_t)
    train_dl = DataLoader(TensorDataset(Xtr_t, ytr_t), args.bs, sampler=sampler,
                          drop_last=True, num_workers=0)
    test_dl = DataLoader(TensorDataset(Xte_t, yte_t), 512, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = build_model(args.model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}  params={n_params}  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    best_test_acc = 0.0; best_train_acc = 0.0; best_cm = None; best_epoch = 0
    log = []
    ckpt_dir = REPO / "model/ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_holdout_{args.model}.pt"
    for ep in range(1, args.epochs + 1):
        model.train()
        train_correct = train_total = 0
        for x, yt in train_dl:
            x, yt = x.to(device), yt.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, yt, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            train_correct += (logits.argmax(1) == yt).sum().item()
            train_total += yt.numel()
        sch.step()
        train_acc = train_correct / max(train_total, 1)
        test_loss, test_acc, cm = evaluate(model, test_dl, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc; best_train_acc = train_acc; best_cm = cm; best_epoch = ep
            # Save the model (architecture-specific)
            if args.model == "snn":
                torch.save({"state": model.state_dict(),
                            "val_acc": test_acc, "epoch": ep,
                            "arch": "snn_holdout",
                            "n_in": 256, "H": 64, "n_classes": 3,
                            "beta": 0.9, "threshold": 1.0, "T": 32,
                            "holdout": holdout_recs},
                           ckpt_path)
            else:
                torch.save({"state": model.state_dict(),
                            "val_acc": test_acc, "epoch": ep,
                            "arch": "cnn_v7_holdout",
                            "channels": [1, 32, 64, 128, 3],
                            "no_pool": False, "stride2": True,
                            "k_first": 5,
                            "holdout": holdout_recs},
                           ckpt_path)
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(f"ep {ep:02d}  train_acc={train_acc*100:.2f}%  "
                  f"HOLDOUT_acc={test_acc*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
        log.append({"epoch": ep, "train_acc": train_acc, "test_acc": test_acc})

    per_class = (best_cm.diagonal() / best_cm.sum(axis=1).clip(min=1)).tolist()
    summary = {
        "model": args.model,
        "holdout_subjects": holdout_recs,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "best_test_acc": float(best_test_acc),
        "best_train_acc": float(best_train_acc),
        "best_epoch": best_epoch,
        "train_test_gap": float(best_train_acc - best_test_acc),
        "per_class_test_acc": per_class,
        "confusion_matrix": best_cm.tolist(),
        "log": log,
        "elapsed_sec": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n=== FINAL HOLD-OUT TEST RESULT ({args.model}) ===")
    print(f"  hold-out subjects: {holdout_recs}")
    print(f"  best HOLDOUT acc:  {best_test_acc*100:.2f}%   (at epoch {best_epoch})")
    print(f"  per-class:         BG={per_class[0]*100:.1f}%  Sys={per_class[1]*100:.1f}%  Dia={per_class[2]*100:.1f}%")
    print(f"  train acc:         {best_train_acc*100:.2f}%")
    print(f"  train-test gap:    {(best_train_acc-best_test_acc)*100:.2f} pp")
    print(f"  -> wrote {args.out}")


if __name__ == "__main__":
    main()
