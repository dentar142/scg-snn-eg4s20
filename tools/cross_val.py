"""cross_val.py — K-fold subject-disjoint cross-validation for SCG SNN / CNN.

Reads `data_excl100/all.npz` which contains
  X    (N, 1, 256) int8
  y    (N,)        int64
  sid  (N,)        int32   — subject id (0..len(record_names)-1)
  record_names    (S,) object  — per-id record name (e.g., 'b001')

Splits the *unique* subject IDs into K folds (deterministic, sorted), so each
fold's val set has zero subject overlap with its train set.  This is the gold-
standard evaluation for physiological-signal classifiers.

Usage:
    python tools/cross_val.py --model snn --folds 5 --epochs 30 \
        --data data_excl100/all.npz --out doc/cv_snn.json
    python tools/cross_val.py --model cnn --folds 5 --epochs 30 \
        --data data_excl100/all.npz --out doc/cv_cnn.json
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import statistics
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


def make_balanced_sampler(y: torch.Tensor, n_classes: int = 3,
                          power: float = 0.5) -> WeightedRandomSampler:
    counts = torch.bincount(y, minlength=n_classes).float()
    counts = counts.clamp(min=1.0)
    weights = (1.0 / counts[y]) ** power
    return WeightedRandomSampler(weights, num_samples=len(y), replacement=True)


def build_model(name: str, device, n_classes: int = 3):
    if name == "snn":
        from train_snn_v1 import SCGSnn
        return SCGSnn(n_in=256, n_hidden=64, n_classes=n_classes,
                      beta=0.9, threshold=1.0, T=32).to(device)
    elif name == "cnn":
        from train_qat_v2 import SCGNetV2
        return SCGNetV2(n_classes=n_classes, k_first=5,
                        channels=(32, 64, 128), no_pool=False, stride2=False).to(device)
    raise ValueError(name)


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
    return loss_sum / max(total, 1), correct / max(total, 1), cm


def train_one_fold(model_name: str, X_tr, y_tr, X_va, y_va, *,
                   epochs: int, bs: int, lr: float, device, log_prefix: str,
                   n_classes: int = 3):
    device = torch.device(device)
    model = build_model(model_name, device, n_classes=n_classes)

    # Convert to float in [-1, 1]; CNN uses (B, 1, 256), SNN does too -- both OK
    Xtr_t = torch.from_numpy(X_tr.astype(np.float32) / 127.0)
    ytr_t = torch.from_numpy(y_tr).long()
    Xva_t = torch.from_numpy(X_va.astype(np.float32) / 127.0)
    yva_t = torch.from_numpy(y_va).long()

    sampler = make_balanced_sampler(ytr_t, n_classes=n_classes)
    train_dl = DataLoader(TensorDataset(Xtr_t, ytr_t), bs, sampler=sampler,
                          drop_last=True, num_workers=0)
    val_dl = DataLoader(TensorDataset(Xva_t, yva_t), 512, shuffle=False,
                        num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0; best_cm = None; best_train_acc = 0.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_correct = train_total = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.numel()
        sch.step()
        train_acc = train_correct / max(train_total, 1)
        val_loss, val_acc, cm = evaluate(model, val_dl, device, n_classes=n_classes)
        if val_acc > best_acc:
            best_acc = val_acc; best_cm = cm; best_train_acc = train_acc
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            elapsed = time.time() - t0
            print(f"  {log_prefix} ep {ep:02d}  train_acc={train_acc*100:.2f}%  "
                  f"val_acc={val_acc*100:.2f}%  ({elapsed:.0f}s)", flush=True)
    return best_acc, best_train_acc, best_cm


def per_class_f1(cm: np.ndarray) -> list:
    """Return per-class F1 from a confusion matrix (rows=truth, cols=pred)."""
    K = cm.shape[0]
    f1s = []
    for c in range(K):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        if 2 * tp + fp + fn == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return f1s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=REPO / "data_excl100/all.npz")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", choices=["snn", "cnn"], required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n-classes", type=int, default=None,
                   help="number of classes; if None, inferred from y")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    d = np.load(args.data, allow_pickle=True)
    X = d["X"].astype(np.int8)             # (N, 1, 256)
    y = d["y"].astype(np.int64)
    sid = d["sid"].astype(np.int64)
    rec_names = d["record_names"]
    K = args.n_classes if args.n_classes is not None else int(y.max()) + 1
    print(f"loaded {args.data}: N={len(X)}  K={K}  "
          f"unique subjects={sorted(set(sid.tolist()))}")
    print(f"  records: {[str(r) for r in rec_names]}")
    print(f"  per-class={np.bincount(y, minlength=K).tolist()}")

    # Subject-disjoint K-fold: assign each unique subject id to one fold
    unique_sids = sorted(set(sid.tolist()))
    n_folds = args.folds
    rng = np.random.RandomState(args.seed)
    perm = list(unique_sids)
    rng.shuffle(perm)
    fold_assign = {s: i % n_folds for i, s in enumerate(perm)}
    folds = [[s for s in unique_sids if fold_assign[s] == k] for k in range(n_folds)]
    print(f"\n{n_folds}-fold subject-disjoint split:")
    for k, fs in enumerate(folds):
        names = [str(rec_names[s]) for s in fs]
        n_win = int((np.isin(sid, fs)).sum())
        print(f"  fold {k}: subjects {names}  ({n_win} windows)")

    fold_results = []
    t_total = time.time()
    for k in range(n_folds):
        val_subjects = folds[k]
        val_mask = np.isin(sid, val_subjects)
        train_mask = ~val_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        print(f"\n=== Fold {k+1}/{n_folds} ===  train={len(X_tr)}  val={len(X_va)}  "
              f"val_subjects={[str(rec_names[s]) for s in val_subjects]}")
        best_val_acc, best_train_acc, cm = train_one_fold(
            args.model, X_tr, y_tr, X_va, y_va,
            epochs=args.epochs, bs=args.bs, lr=args.lr,
            device=args.device, log_prefix=f"[F{k+1}]", n_classes=K)
        per_class = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist()
        f1s = per_class_f1(cm)
        fold_results.append({
            "fold": k,
            "val_subjects": [str(rec_names[s]) for s in val_subjects],
            "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
            "best_val_acc": float(best_val_acc),
            "best_train_acc": float(best_train_acc),
            "train_val_gap": float(best_train_acc - best_val_acc),
            "per_class_acc": per_class,
            "per_class_f1": f1s,
            "confusion_matrix": cm.tolist(),
        })
        print(f"  [done] Fold {k+1}: best_val={best_val_acc*100:.2f}%  "
              f"train={best_train_acc*100:.2f}%  gap={(best_train_acc-best_val_acc)*100:.2f}pp")

    # Aggregate
    accs = [r["best_val_acc"] for r in fold_results]
    train_accs = [r["best_train_acc"] for r in fold_results]
    # Mean per-class F1 across folds
    mean_f1_per_class = [
        statistics.mean(r["per_class_f1"][c] for r in fold_results)
        for c in range(K)
    ]
    macro_f1_per_fold = [statistics.mean(r["per_class_f1"]) for r in fold_results]
    summary = {
        "model": args.model,
        "n_classes": K,
        "folds": args.folds,
        "epochs": args.epochs,
        "data_source": str(args.data),
        "fold_results": fold_results,
        "val_acc": {
            "mean": statistics.mean(accs),
            "std":  statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "min":  min(accs), "max": max(accs),
        },
        "train_acc": {
            "mean": statistics.mean(train_accs),
            "std":  statistics.stdev(train_accs) if len(train_accs) > 1 else 0.0,
        },
        "mean_train_val_gap": statistics.mean(
            [r["train_val_gap"] for r in fold_results]),
        "mean_per_class_f1": mean_f1_per_class,
        "macro_f1": {
            "mean": statistics.mean(macro_f1_per_fold),
            "std":  statistics.stdev(macro_f1_per_fold) if len(macro_f1_per_fold) > 1 else 0.0,
        },
        "elapsed_sec": time.time() - t_total,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n=== {args.model.upper()} {n_folds}-fold subject-disjoint CV "
          f"(K={K} classes) ===")
    print(f"  mean val_acc   = {summary['val_acc']['mean']*100:.2f} +/- "
          f"{summary['val_acc']['std']*100:.2f} %")
    print(f"  range          = [{summary['val_acc']['min']*100:.2f}, "
          f"{summary['val_acc']['max']*100:.2f}] %")
    print(f"  mean train_acc = {summary['train_acc']['mean']*100:.2f} +/- "
          f"{summary['train_acc']['std']*100:.2f} %")
    print(f"  mean gap       = {summary['mean_train_val_gap']*100:.2f} pp")
    print(f"  macro F1       = {summary['macro_f1']['mean']*100:.2f} +/- "
          f"{summary['macro_f1']['std']*100:.2f} %")
    print(f"  per-class F1   = {[f'{f*100:.1f}' for f in mean_f1_per_class]}")
    print(f"  elapsed        = {summary['elapsed_sec']/60:.1f} min")
    print(f"\n-> wrote {args.out}")


if __name__ == "__main__":
    main()
