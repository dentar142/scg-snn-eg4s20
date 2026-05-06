"""finetune_ssl.py — Fine-tune SSL-pretrained encoder for SCG 3-class classifn.

Loads `model/ckpt/ssl_<tag>.pt`, replaces projection head with a linear
classifier, and runs the SAME 5-fold subject-disjoint CV as cross_val.py
so we can directly compare against the cold-start SNN baseline (85.48 %).

Usage:
    python model/finetune_ssl.py --ssl-ckpt model/ckpt/ssl_cebs_cebs.pt \
        --data data_excl100/all.npz --out doc/cv_ssl_finetune.json \
        --folds 5 --epochs 30
"""
from __future__ import annotations
import argparse
import json
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
from pretrain_ssl import ConvEncoder  # noqa: E402


class SCGClassifier(nn.Module):
    def __init__(self, encoder: ConvEncoder, n_classes: int = 3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.feat_dim, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def make_balanced_sampler(y: torch.Tensor, n_classes: int = 3, power: float = 0.5):
    counts = torch.bincount(y, minlength=n_classes).float()
    weights = (1.0 / counts[y]) ** power
    return WeightedRandomSampler(weights, num_samples=len(y), replacement=True)


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


def train_one_fold(ssl_ckpt_path: Path, X_tr, y_tr, X_va, y_va, *,
                   epochs: int, bs: int, lr: float, freeze_epochs: int,
                   device, log_prefix: str):
    device = torch.device(device)

    # Build encoder, load SSL weights
    ssl = torch.load(ssl_ckpt_path, map_location="cpu", weights_only=False)
    enc = ConvEncoder(channels=tuple(ssl.get("channels", (32, 64, 128))))
    enc.load_state_dict(ssl["encoder_state"])
    model = SCGClassifier(enc, n_classes=3).to(device)

    Xtr_t = torch.from_numpy(X_tr.astype(np.float32) / 127.0)
    ytr_t = torch.from_numpy(y_tr).long()
    Xva_t = torch.from_numpy(X_va.astype(np.float32) / 127.0)
    yva_t = torch.from_numpy(y_va).long()

    sampler = make_balanced_sampler(ytr_t)
    train_dl = DataLoader(TensorDataset(Xtr_t, ytr_t), bs, sampler=sampler,
                          drop_last=True, num_workers=0)
    val_dl = DataLoader(TensorDataset(Xva_t, yva_t), 512, shuffle=False,
                        num_workers=0)

    # Phase 1 (freeze_epochs): only train the head
    if freeze_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0; best_cm = None; best_train_acc = 0.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == freeze_epochs + 1:
            # Phase 2: unfreeze encoder, restart optimizer with smaller LR
            for p in model.encoder.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs - freeze_epochs)

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
        val_loss, val_acc, cm = evaluate(model, val_dl, device)
        if val_acc > best_acc:
            best_acc = val_acc; best_cm = cm; best_train_acc = train_acc
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            elapsed = time.time() - t0
            phase = "frozen" if ep <= freeze_epochs else "full"
            print(f"  {log_prefix} ep {ep:02d} ({phase})  train_acc={train_acc*100:.2f}%  "
                  f"val_acc={val_acc*100:.2f}%  ({elapsed:.0f}s)", flush=True)
    return best_acc, best_train_acc, best_cm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=REPO / "data_excl100/all.npz")
    p.add_argument("--ssl-ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--freeze-epochs", type=int, default=5,
                   help="freeze encoder for the first N epochs (linear-probe warm-up)")
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
    rec_names = d["record_names"]
    print(f"loaded {args.data}: N={len(X)}  unique subjects={sorted(set(sid.tolist()))}")

    unique_sids = sorted(set(sid.tolist()))
    K = args.folds
    rng = np.random.RandomState(args.seed)
    perm = list(unique_sids); rng.shuffle(perm)
    folds = [[s for s in unique_sids if (perm.index(s) % K) == k] for k in range(K)]
    print(f"\n{K}-fold subject-disjoint split:")
    for k, fs in enumerate(folds):
        print(f"  fold {k}: subjects {[str(rec_names[s]) for s in fs]}  "
              f"({int((np.isin(sid, fs)).sum())} windows)")

    fold_results = []
    t_total = time.time()
    for k in range(K):
        val_subjects = folds[k]
        val_mask = np.isin(sid, val_subjects)
        train_mask = ~val_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        print(f"\n=== Fold {k+1}/{K} ===  train={len(X_tr)}  val={len(X_va)}  "
              f"val_subjects={[str(rec_names[s]) for s in val_subjects]}")
        best_val_acc, best_train_acc, cm = train_one_fold(
            args.ssl_ckpt, X_tr, y_tr, X_va, y_va,
            epochs=args.epochs, bs=args.bs, lr=args.lr,
            freeze_epochs=args.freeze_epochs,
            device=args.device, log_prefix=f"[F{k+1}]")
        per_class = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist()
        fold_results.append({
            "fold": k,
            "val_subjects": [str(rec_names[s]) for s in val_subjects],
            "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
            "best_val_acc": float(best_val_acc),
            "best_train_acc": float(best_train_acc),
            "train_val_gap": float(best_train_acc - best_val_acc),
            "per_class_acc": per_class,
            "confusion_matrix": cm.tolist(),
        })
        print(f"  [done] Fold {k+1}: best_val={best_val_acc*100:.2f}%  "
              f"train={best_train_acc*100:.2f}%  gap={(best_train_acc-best_val_acc)*100:.2f}pp")

    accs = [r["best_val_acc"] for r in fold_results]
    train_accs = [r["best_train_acc"] for r in fold_results]
    summary = {
        "ssl_ckpt": str(args.ssl_ckpt),
        "folds": K, "epochs": args.epochs, "freeze_epochs": args.freeze_epochs,
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
        "mean_train_val_gap": statistics.mean([r["train_val_gap"] for r in fold_results]),
        "elapsed_sec": time.time() - t_total,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n=== SSL fine-tuned {K}-fold subject-disjoint CV ===")
    print(f"  mean val_acc   = {summary['val_acc']['mean']*100:.2f} +/- "
          f"{summary['val_acc']['std']*100:.2f} %")
    print(f"  range          = [{summary['val_acc']['min']*100:.2f}, "
          f"{summary['val_acc']['max']*100:.2f}] %")
    print(f"  mean train_acc = {summary['train_acc']['mean']*100:.2f} %")
    print(f"  mean gap       = {summary['mean_train_val_gap']*100:.2f} pp")
    print(f"  elapsed        = {summary['elapsed_sec']/60:.1f} min")
    print(f"\n-> wrote {args.out}")


if __name__ == "__main__":
    main()
