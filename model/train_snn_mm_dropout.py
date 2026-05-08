"""train_snn_mm_dropout.py - FOSTER multi-modal SNN with modality dropout
+ phase alignment, for graceful operation under missing modalities at
deploy time (e.g. transfer to a single-modal dataset like CEBSDB).

Design:
  - Same arch as train_snn_mm_aligned.py (H=32 T=16, learnable tau,
    channel-shift augmentation)
  - During training, with prob 0.5 per batch, zero out a random subset of
    channels (1 to n-1 channels). This forces the network to produce
    sensible output from any modality subset.
  - Validation evaluates with all modalities active (sanity).
  - For deploy on CEBSDB: load this ckpt, place CEBSDB SCG into ACC slot
    (idx 2), zero other 4 -> evaluate.

Usage:
    python model/train_snn_mm_dropout.py --data data_foster_multi \\
        --holdout sub003 sub006 sub009 sub013 sub020 sub021 sub024 sub026 \\
        --epochs 60 --H 32 --T 16 --tag snn_mm_h32t16_dropout
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_snn_multimodal import MultiModalSCGSnn, make_balanced_sampler  # noqa: E402
from train_snn_mm_aligned import (  # noqa: E402
    AlignedMultiModalSCGSnn, ChannelShiftAugmentedDataset, evaluate_aligned,
)


class ModalityDropoutDataset(Dataset):
    """Wraps ChannelShiftAugmentedDataset and additionally zeros out a
    random subset of channels with probability `p_drop`. Number of
    channels zeroed is drawn uniformly in [1, n_channels-1].
    """
    def __init__(self, X, y, shift_max=15, training=True,
                 p_drop=0.5, n_channels=5):
        self.inner = ChannelShiftAugmentedDataset(X, y, shift_max, training)
        self.p_drop = p_drop
        self.n_channels = n_channels
        self.training = training

    def __len__(self): return len(self.inner)

    def __getitem__(self, i):
        x, y = self.inner[i]   # (C, L) tensor float
        if self.training and torch.rand(1).item() < self.p_drop:
            # Choose how many to drop (1..n_channels-1) and which
            n_drop = int(torch.randint(1, self.n_channels, (1,)).item())
            perm = torch.randperm(self.n_channels)[:n_drop]
            x = x.clone()
            x[perm] = 0.0
        return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--holdout", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("model/ckpt"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-tau", type=float, default=2e-2)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--n-classes", type=int, default=3)
    p.add_argument("--shift-max", type=int, default=15)
    p.add_argument("--p-drop", type=float, default=0.5)
    p.add_argument("--max-tau", type=float, default=15.0)
    p.add_argument("--tag", type=str, default="snn_mm_dropout")
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
    n_chan = int(X.shape[1])
    print(f"hold-out: {args.holdout}")
    print(f"  train {len(X_tr)} | val {len(X_va)}  shift_max={args.shift_max}  "
          f"p_drop={args.p_drop}")

    train_ds = ModalityDropoutDataset(X_tr, y_tr, args.shift_max,
                                      training=True, p_drop=args.p_drop,
                                      n_channels=n_chan)
    val_ds = ChannelShiftAugmentedDataset(X_va, y_va, 0, training=False)
    sampler = make_balanced_sampler(torch.from_numpy(y_tr).long(), args.n_classes)
    train_dl = DataLoader(train_ds, args.bs, sampler=sampler, drop_last=True, num_workers=0)
    val_dl = DataLoader(val_ds, 512, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    base = MultiModalSCGSnn(n_in=X.shape[2], n_channels=n_chan,
                            n_hidden=args.H, n_classes=args.n_classes,
                            beta=0.9, threshold=1.0, T=args.T).to(device)
    model = AlignedMultiModalSCGSnn(base, max_tau=args.max_tau, init_zero=True).to(device)

    fc_params = list(model.base.parameters())
    tau_params = [model.tau_raw]
    opt = torch.optim.AdamW([
        {"params": fc_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": tau_params, "lr": args.lr_tau, "weight_decay": 0.0},
    ])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0; best_cm = None; best_ep = 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        n_seen = correct = 0
        for x, yt in train_dl:
            x, yt = x.to(device), yt.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, yt, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(fc_params, max_norm=2.0)
            opt.step()
            n_seen += yt.numel()
            correct += (logits.argmax(1) == yt).sum().item()
        sch.step()
        train_acc = correct / max(n_seen, 1)

        model.export_int_tau()
        val_acc, cm = evaluate_aligned(model, val_dl, device)
        model.quantize_inference = False

        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            tau_disp = ", ".join(f"{t:+.1f}" for t in model.tau.tolist())
            print(f"  ep {ep:02d} train={train_acc*100:.2f}% val={val_acc*100:.2f}%  "
                  f"tau=[{tau_disp}]  ({time.time()-t0:.0f}s)", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc; best_cm = cm; best_ep = ep
            model.export_int_tau()
            torch.save({
                "state": model.base.state_dict(),
                "tau_int": model.tau_int.tolist(),
                "tau_float": model.tau.tolist(),
                "val_acc": val_acc, "epoch": ep, "arch": args.tag,
                "n_in": X.shape[2], "n_channels": n_chan, "H": args.H,
                "n_classes": args.n_classes, "beta": 0.9, "threshold": 1.0,
                "T": args.T, "shift_max_train": args.shift_max,
                "p_drop_train": args.p_drop,
                "holdout_subjects": args.holdout,
            }, args.out / f"best_{args.tag}.pt")
            model.quantize_inference = False

    model.export_int_tau()
    tau_int = model.tau_int.tolist()
    print(f"\nbest val_acc = {best_acc*100:.2f}% @ ep {best_ep}  tau_int={tau_int}")
    manifest = {
        "tag": args.tag, "best_val_acc": float(best_acc), "best_epoch": best_ep,
        "tau_int_at_best": tau_int, "tau_float": model.tau.tolist(),
        "shift_max_train": args.shift_max, "p_drop_train": args.p_drop,
        "holdout_subjects": args.holdout,
        "n_train_windows": int(len(X_tr)), "n_val_windows": int(len(X_va)),
        "confusion_matrix": best_cm.tolist() if best_cm is not None else None,
        "H": args.H, "T": args.T, "epochs": args.epochs, "seed": args.seed,
    }
    (args.out / f"best_{args.tag}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"-> {args.out}/best_{args.tag}_manifest.json")


if __name__ == "__main__":
    main()
