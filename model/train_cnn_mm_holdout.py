"""train_cnn_mm_holdout.py - subject-disjoint multimodal CNN baseline.

Mirrors model/train_qat.py SCGNet but for FOSTER 5-channel input. Trains with
the same hold-out as the SNN deployed model for fair comparison. Two size
configs:
  --size small : matches SCGNet shape (~2K params, ~5x smaller than SNN)
  --size match : wider channels, params ~= SNN H=32 T=16 (~41K)

Usage:
    python model/train_cnn_mm_holdout.py --data data_foster_multi \\
        --holdout sub003 sub006 sub009 sub013 sub020 sub021 sub024 sub026 \\
        --epochs 50 --size match --tag cnn_mm_match
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        absmax = x.detach().abs().max().clamp_min(1e-8)
        scale = absmax / 127
        return torch.clamp(torch.round(x / scale), -127, 127) * scale
    @staticmethod
    def backward(ctx, g): return g


def fq(x): return FakeQuant.apply(x)


class QConv1d(nn.Module):
    def __init__(self, c_in, c_out, k, p):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, padding=p, bias=True)
        self.bn = nn.BatchNorm1d(c_out)
    def forward(self, x):
        w = self.conv.weight
        absmax = w.detach().abs().max().clamp_min(1e-8)
        scale = absmax / 127
        wq = torch.clamp(torch.round(w / scale), -127, 127) * scale
        wq = (wq - w).detach() + w
        x = F.conv1d(x, wq, self.conv.bias, padding=self.conv.padding[0])
        x = self.bn(x); x = F.relu(x); x = fq(x)
        return x


class MultiModalCNN(nn.Module):
    """5-channel SCGNet variants for FOSTER."""
    def __init__(self, n_classes: int = 3, size: str = "small"):
        super().__init__()
        if size == "small":
            chans = [5, 8, 16, 16]
        elif size == "match":
            # ~41K params target
            chans = [5, 32, 64, 64]
        else:
            raise ValueError(size)
        self.l0 = QConv1d(chans[0], chans[1], 5, 2)
        self.l1 = QConv1d(chans[1], chans[2], 5, 2)
        self.l2 = QConv1d(chans[2], chans[3], 5, 2)
        self.l3 = QConv1d(chans[3], n_classes, 1, 0)
    def forward(self, x):
        x = fq(x)
        x = F.max_pool1d(self.l0(x), 2)
        x = F.max_pool1d(self.l1(x), 2)
        x = F.max_pool1d(self.l2(x), 2)
        x = self.l3(x)
        return x.mean(dim=2)


def make_balanced_sampler(y, n_classes=3, power=0.5):
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1.0)
    weights = (1.0 / counts[y]) ** power
    return WeightedRandomSampler(weights, num_samples=len(y), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device, n_classes=3):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / max(total, 1), cm


def per_class_f1(cm):
    f1 = []
    for c in range(cm.shape[0]):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        f1.append(0.0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn))
    return f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--holdout", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("model/ckpt"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--size", choices=["small", "match"], default="match")
    p.add_argument("--tag", type=str, default="cnn_mm")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    d = np.load(args.data / "all.npz", allow_pickle=True)
    X = d["X"]; y = d["y"]; sid = d["sid"]
    rec_names = list(map(str, d["record_names"]))
    name_to_sid = {n: i for i, n in enumerate(rec_names)}
    holdout_sids = [name_to_sid[h] for h in args.holdout]
    val_mask = np.isin(sid, holdout_sids)
    train_mask = ~val_mask
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    print(f"hold-out subjects: {args.holdout}")
    print(f"  train: {len(X_tr)}  val: {len(X_va)}")

    Xtr = torch.from_numpy(X_tr.astype(np.float32) / 127.0)
    Xva = torch.from_numpy(X_va.astype(np.float32) / 127.0)
    ytr = torch.from_numpy(y_tr).long()
    yva = torch.from_numpy(y_va).long()
    sampler = make_balanced_sampler(ytr)
    train_dl = DataLoader(TensorDataset(Xtr, ytr), args.bs, sampler=sampler,
                          drop_last=True, num_workers=0)
    val_dl = DataLoader(TensorDataset(Xva, yva), 512, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = MultiModalCNN(n_classes=3, size=args.size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MultiModal CNN ({args.size})  params={n_params}  dev={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_acc = 0.0; best_cm = None; best_ep = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        n_seen = correct = 0
        for x, yb in train_dl:
            x, yb = x.to(device), yb.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            correct += (logits.argmax(1) == yb).sum().item(); n_seen += yb.numel()
        sch.step()
        train_acc = correct / max(n_seen, 1)
        val_acc, cm = evaluate(model, val_dl, device)
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(f"  ep {ep:02d} train={train_acc*100:.2f}% val={val_acc*100:.2f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc; best_cm = cm; best_ep = ep
            torch.save({"state": model.state_dict(), "val_acc": val_acc, "epoch": ep,
                        "size": args.size, "n_params": n_params,
                        "holdout_subjects": args.holdout},
                       args.out / f"best_{args.tag}.pt")

    f1 = per_class_f1(best_cm)
    manifest = {
        "tag": args.tag, "size": args.size, "n_params": n_params,
        "best_val_acc": float(best_acc), "best_epoch": best_ep,
        "macro_f1": sum(f1) / len(f1), "per_class_f1": f1,
        "confusion_matrix": best_cm.tolist(),
        "holdout_subjects": args.holdout,
        "n_train_windows": int(len(X_tr)), "n_val_windows": int(len(X_va)),
    }
    out_manifest = args.out / f"best_{args.tag}_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"\nbest val acc = {best_acc*100:.2f}%  macro-F1 = {manifest['macro_f1']*100:.2f}%")
    print(f"-> {out_manifest}")


if __name__ == "__main__":
    main()
