"""train_snn_mm_holdout.py — train multi-modal SNN with strict subject-disjoint
hold-out for FPGA deployment evaluation.

Trains MultiModalSCGSnn on FOSTER multi-modal corpus, holding out a fixed list
of subjects entirely (no leakage between train and val sets at the subject
level). Saves the best ckpt + emits a JSON manifest of train/val subject sets.

Usage (default = fold-0 of existing 5-fold CV):
    python model/train_snn_mm_holdout.py --data data_foster_multi \
        --holdout sub003 sub006 sub009 sub013 sub020 sub021 sub024 sub026 \
        --epochs 60 --H 32 --tag snn_mm_h32_holdout
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_snn_multimodal import (  # noqa: E402
    MultiModalSCGSnn, NPZDataset, make_balanced_sampler, evaluate,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("model/ckpt"))
    p.add_argument("--holdout", nargs="+", required=True,
                   help="subject record names (e.g. sub003 sub006 ...) used as val set")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--n-classes", type=int, default=3)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--tag", type=str, default="snn_mm_h32_holdout")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data / "all.npz", allow_pickle=True)
    X = d["X"]; y = d["y"]; sid = d["sid"]
    rec_names = list(map(str, d["record_names"]))
    n_channels = X.shape[1]; n_in = X.shape[2]
    print(f"loaded: X={X.shape} y={y.shape} unique_sid={len(set(sid.tolist()))}")

    name_to_sid = {n: i for i, n in enumerate(rec_names)}
    holdout_sids = []
    for h in args.holdout:
        if h not in name_to_sid:
            sys.exit(f"ERROR: holdout subject '{h}' not found in record_names")
        holdout_sids.append(name_to_sid[h])
    train_subjects = sorted(set(rec_names) - set(args.holdout))
    print(f"  hold-out (val): {args.holdout} -> sids {holdout_sids}")
    print(f"  train subjects: {len(train_subjects)} ({train_subjects[0]}..{train_subjects[-1]})")

    val_mask = np.isin(sid, holdout_sids)
    train_mask = ~val_mask
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    print(f"  train: {len(X_tr)} windows | val: {len(X_va)} windows")

    train_ds = NPZDataset(X_tr, y_tr)
    val_ds = NPZDataset(X_va, y_va)
    sampler = make_balanced_sampler(train_ds.y, args.n_classes)
    train_dl = DataLoader(train_ds, args.bs, sampler=sampler, drop_last=True, num_workers=0)
    val_dl = DataLoader(val_ds, 512, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = MultiModalSCGSnn(n_in=n_in, n_channels=n_channels,
                             n_hidden=args.H, n_classes=args.n_classes,
                             beta=args.beta, threshold=args.threshold, T=args.T).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MultiModal SNN ({n_channels}*{n_in}={n_channels*n_in})->{args.H}->{args.n_classes}  "
          f"params={n_params}  dev={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0; best_cm = None; best_ep = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        n_seen = train_correct = 0
        for x, yt in train_dl:
            x, yt = x.to(device), yt.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, yt, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            n_seen += yt.numel()
            train_correct += (logits.argmax(1) == yt).sum().item()
        sch.step()
        train_acc = train_correct / max(n_seen, 1)
        val_loss, val_acc, cm = evaluate(model, val_dl, device, args.n_classes)
        gap = train_acc - val_acc
        print(f"ep {ep:02d} train={train_acc*100:.2f}% val={val_acc*100:.2f}% "
              f"gap={gap*100:+.2f}pp ({time.time()-t0:.0f}s)", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc; best_cm = cm; best_ep = ep
            torch.save({"state": model.state_dict(),
                        "val_acc": val_acc, "epoch": ep, "arch": args.tag,
                        "n_in": n_in, "n_channels": n_channels, "H": args.H,
                        "n_classes": args.n_classes, "beta": args.beta,
                        "threshold": args.threshold, "T": args.T,
                        "holdout_subjects": args.holdout,
                        "train_subjects": train_subjects},
                       args.out / f"best_{args.tag}.pt")

    manifest = {
        "tag": args.tag, "best_val_acc": float(best_acc), "best_epoch": best_ep,
        "holdout_subjects": args.holdout, "n_holdout": len(args.holdout),
        "train_subjects": train_subjects, "n_train_subjects": len(train_subjects),
        "n_train_windows": int(len(X_tr)), "n_val_windows": int(len(X_va)),
        "confusion_matrix": best_cm.tolist() if best_cm is not None else None,
        "H": args.H, "T": args.T, "epochs": args.epochs, "seed": args.seed,
    }
    (args.out / f"best_{args.tag}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nbest val_acc = {best_acc*100:.2f}% @ ep {best_ep}")
    print(f"-> {args.out}/best_{args.tag}.pt")
    print(f"-> {args.out}/best_{args.tag}_manifest.json")


if __name__ == "__main__":
    main()
