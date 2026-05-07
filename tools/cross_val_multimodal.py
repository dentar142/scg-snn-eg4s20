"""cross_val_multimodal.py — K-fold subject-disjoint CV for multi-modal SNN.

Reads `*/all.npz` with X (N, C, L), y, sid, record_names. Builds a MultiModalSCGSnn
that flattens C*L into the input layer.

Usage:
    python tools/cross_val_multimodal.py --data data_foster_multi/all.npz \
        --out doc/cv_snn_foster.json --folds 5 --epochs 30
"""
from __future__ import annotations
import argparse, json, random, statistics, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_snn_multimodal import MultiModalSCGSnn  # noqa: E402


def make_balanced_sampler(y: torch.Tensor, n_classes: int = 3, power: float = 0.5):
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1.0)
    weights = (1.0 / counts[y]) ** power
    return WeightedRandomSampler(weights, num_samples=len(y), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int = 3):
    model.eval()
    correct = total = 0; cm = np.zeros((n_classes, n_classes), dtype=np.int64)
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


def train_one_fold(X_tr, y_tr, X_va, y_va, *, n_classes, n_channels, n_in,
                   epochs, bs, lr, device, log_prefix, H=64, T=32):
    device = torch.device(device)
    model = MultiModalSCGSnn(n_in=n_in, n_channels=n_channels, n_hidden=H,
                             n_classes=n_classes, beta=0.9, threshold=1.0, T=T).to(device)

    Xtr_t = torch.from_numpy(X_tr.astype(np.float32) / 127.0)
    ytr_t = torch.from_numpy(y_tr).long()
    Xva_t = torch.from_numpy(X_va.astype(np.float32) / 127.0)
    yva_t = torch.from_numpy(y_va).long()
    sampler = make_balanced_sampler(ytr_t, n_classes)
    train_dl = DataLoader(TensorDataset(Xtr_t, ytr_t), bs, sampler=sampler,
                          drop_last=True, num_workers=0)
    val_dl = DataLoader(TensorDataset(Xva_t, yva_t), 512, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0; best_cm = None; best_train_acc = 0.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tc = tt = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            tc += (logits.argmax(1) == y).sum().item(); tt += y.numel()
        sch.step()
        train_acc = tc / max(tt, 1)
        val_loss, val_acc, cm = evaluate(model, val_dl, device, n_classes)
        if val_acc > best_acc:
            best_acc = val_acc; best_cm = cm; best_train_acc = train_acc
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  {log_prefix} ep {ep:02d} train_acc={train_acc*100:.2f}% "
                  f"val_acc={val_acc*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)
    return best_acc, best_train_acc, best_cm


def per_class_f1(cm: np.ndarray) -> list:
    K = cm.shape[0]
    out = []
    for c in range(K):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        out.append(0.0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--n-classes", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    d = np.load(args.data, allow_pickle=True)
    X = d["X"].astype(np.int8); y = d["y"].astype(np.int64); sid = d["sid"].astype(np.int64)
    rec_names = list(d["record_names"])
    n_channels = X.shape[1]; n_in = X.shape[2]
    K = args.n_classes if args.n_classes is not None else int(y.max()) + 1
    print(f"loaded {args.data}: N={len(X)} shape={X.shape} K={K} channels={n_channels}")
    print(f"  unique subjects: {len(set(sid.tolist()))}")

    unique_sids = sorted(set(sid.tolist()))
    n_folds = args.folds
    rng = np.random.RandomState(args.seed)
    perm = list(unique_sids); rng.shuffle(perm)
    fold_assign = {s: i % n_folds for i, s in enumerate(perm)}
    folds = [[s for s in unique_sids if fold_assign[s] == k] for k in range(n_folds)]
    print(f"\n{n_folds}-fold subject-disjoint:")
    for k, fs in enumerate(folds):
        names = [str(rec_names[s]) for s in fs]
        n_win = int(np.isin(sid, fs).sum())
        print(f"  fold {k}: {len(fs)} subjects ({names[0]}..{names[-1]}) ({n_win} windows)")

    fold_results = []
    t_total = time.time()
    for k in range(n_folds):
        val_subjects = folds[k]
        val_mask = np.isin(sid, val_subjects)
        train_mask = ~val_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        print(f"\n=== Fold {k+1}/{n_folds} === train={len(X_tr)} val={len(X_va)}")
        best_val, best_tr, cm = train_one_fold(
            X_tr, y_tr, X_va, y_va,
            n_classes=K, n_channels=n_channels, n_in=n_in,
            epochs=args.epochs, bs=args.bs, lr=args.lr,
            device=args.device, log_prefix=f"[F{k+1}]", H=args.H, T=args.T)
        f1 = per_class_f1(cm)
        macro_f1 = sum(f1) / len(f1)
        fold_results.append({
            "fold": k, "val_subjects": [str(rec_names[s]) for s in val_subjects],
            "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
            "best_val_acc": float(best_val), "best_train_acc": float(best_tr),
            "train_val_gap": float(best_tr - best_val),
            "per_class_f1": f1, "macro_f1": macro_f1,
            "confusion_matrix": cm.tolist(),
        })
        print(f"  [done] Fold {k+1}: val={best_val*100:.2f}% train={best_tr*100:.2f}% "
              f"gap={(best_tr-best_val)*100:.2f}pp macroF1={macro_f1*100:.2f}%")

    accs = [r["best_val_acc"] for r in fold_results]
    macro = [r["macro_f1"] for r in fold_results]
    summary = {
        "data": str(args.data), "folds": n_folds, "epochs": args.epochs,
        "n_channels": int(n_channels), "n_in": int(n_in), "n_classes": int(K),
        "fold_results": fold_results,
        "val_acc": {"mean": statistics.mean(accs),
                    "std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                    "min": min(accs), "max": max(accs)},
        "macro_f1": {"mean": statistics.mean(macro),
                     "std": statistics.stdev(macro) if len(macro) > 1 else 0.0,
                     "min": min(macro), "max": max(macro)},
        "elapsed_sec": time.time() - t_total,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n=== {n_folds}-fold subject-disjoint CV (multi-modal) ===")
    print(f"  mean val_acc = {summary['val_acc']['mean']*100:.2f} +/- "
          f"{summary['val_acc']['std']*100:.2f} %")
    print(f"  mean macro-F1= {summary['macro_f1']['mean']*100:.2f} +/- "
          f"{summary['macro_f1']['std']*100:.2f} %")
    print(f"  range = [{summary['val_acc']['min']*100:.2f}, "
          f"{summary['val_acc']['max']*100:.2f}] %")
    print(f"  elapsed = {summary['elapsed_sec']/60:.1f} min")
    print(f"\n-> wrote {args.out}")


if __name__ == "__main__":
    main()
