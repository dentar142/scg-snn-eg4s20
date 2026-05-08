"""train_snn_mm_aligned.py - subject-disjoint multi-modal SNN with phase
alignment (A: per-channel shift augmentation + B: learnable per-channel
integer time offset).

A. AUGMENTATION: during training, each batch's per-channel signal is rolled
   by an independent random integer shift in [-shift_max, +shift_max]. This
   forces fc1 to learn alignment-robust features.

B. LEARNABLE OFFSET tau_c (one per channel) implemented via differentiable
   bilinear-interpolation shift. Trained as float in [-15, +15], rounded
   to integer at deploy. On FPGA, this becomes 5 small adders modifying
   each channel-bank's read pointer (one adder/bank, 0 BRAM cost).

Usage:
    python model/train_snn_mm_aligned.py --data data_foster_multi \
        --holdout sub003 sub006 sub009 sub013 sub020 sub021 sub024 sub026 \
        --epochs 50 --H 32 --T 16 --tag snn_mm_h32t16_aligned --shift-max 15
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
from train_snn_multimodal import (  # noqa: E402
    MultiModalSCGSnn, make_balanced_sampler, evaluate as base_evaluate,
)


# ======== A. Channel-shift augmenting dataset ========
class ChannelShiftAugmentedDataset(Dataset):
    """During training, roll each channel by a random int shift in
    [-shift_max, +shift_max]. Validation/inference disable augmentation
    (set training=False)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, shift_max: int = 15,
                 training: bool = True):
        self.X = X.astype(np.int8)
        self.y = y.astype(np.int64)
        self.shift_max = shift_max
        self.training = training

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()  # (C, L)
        if self.training and self.shift_max > 0:
            for c in range(x.shape[0]):
                s = np.random.randint(-self.shift_max, self.shift_max + 1)
                if s != 0:
                    x[c] = np.roll(x[c], s)
        # int8 -> [-1, 1] float
        return torch.from_numpy(x.astype(np.float32) / 127.0), torch.tensor(int(self.y[i]))


# ======== B. Aligned wrapper around MultiModalSCGSnn ========
class AlignedMultiModalSCGSnn(nn.Module):
    """Wraps MultiModalSCGSnn with per-channel learnable time offsets
    tau_c implemented via differentiable bilinear interpolation. At
    inference, tau is rounded to integer (FPGA-friendly).
    """
    def __init__(self, base: MultiModalSCGSnn, max_tau: float = 15.0,
                 init_zero: bool = True):
        super().__init__()
        self.base = base
        self.max_tau = max_tau
        self.n_channels = base.n_channels
        # Parameter is unconstrained scalar per channel; pass through tanh
        # so effective tau in [-max_tau, +max_tau] for stable training.
        init = torch.zeros(self.n_channels) if init_zero else 0.01 * torch.randn(self.n_channels)
        self.tau_raw = nn.Parameter(init)
        # Quantize-at-inference flag
        self.quantize_inference = False
        # Cached integer tau (set when quantize_inference enabled)
        self.register_buffer("tau_int", torch.zeros(self.n_channels, dtype=torch.long))

    @property
    def tau(self):
        return self.max_tau * torch.tanh(self.tau_raw)

    def export_int_tau(self):
        with torch.no_grad():
            t = self.tau.round().long()
        self.tau_int.copy_(t)
        self.quantize_inference = True

    def shift_channels(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) -> (B, C, L) with each channel shifted by tau[c].

        Uses linear interpolation in floating-point during training; at
        inference (quantize_inference=True) uses integer roll (bit-exact
        with FPGA).
        """
        B, C, L = x.shape
        if self.quantize_inference:
            out = torch.empty_like(x)
            for c in range(C):
                t = int(self.tau_int[c].item())
                out[:, c] = torch.roll(x[:, c], shifts=t, dims=-1)
            return out
        # Differentiable shift via gather + linear interp
        tau = self.tau  # (C,)
        out = []
        idx = torch.arange(L, device=x.device, dtype=torch.float32)
        for c in range(C):
            tc = tau[c]
            src = idx - tc                      # (L,)
            src_floor = torch.floor(src).long()
            src_ceil = src_floor + 1
            frac = (src - src_floor.float()).unsqueeze(0)  # (1, L)
            src_floor_w = src_floor % L
            src_ceil_w = src_ceil % L
            xc = x[:, c]                        # (B, L)
            v0 = xc.gather(1, src_floor_w.unsqueeze(0).expand(B, L))
            v1 = xc.gather(1, src_ceil_w.unsqueeze(0).expand(B, L))
            out.append((1 - frac) * v0 + frac * v1)
        return torch.stack(out, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shift_channels(x)
        return self.base(x)


@torch.no_grad()
def evaluate_aligned(model, loader, device, n_classes=3):
    model.eval()
    correct = total = 0
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1), cm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--holdout", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("model/ckpt"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-tau", type=float, default=2e-2,
                   help="higher LR for tau since few params")
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--n-classes", type=int, default=3)
    p.add_argument("--shift-max", type=int, default=15)
    p.add_argument("--max-tau", type=float, default=15.0)
    p.add_argument("--tag", type=str, default="snn_mm_aligned")
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
    print(f"hold-out: {args.holdout}")
    print(f"  train {len(X_tr)} | val {len(X_va)}  shift_max={args.shift_max}")

    train_ds = ChannelShiftAugmentedDataset(X_tr, y_tr, args.shift_max, training=True)
    val_ds   = ChannelShiftAugmentedDataset(X_va, y_va, 0, training=False)
    sampler = make_balanced_sampler(torch.from_numpy(y_tr).long(), args.n_classes)
    train_dl = DataLoader(train_ds, args.bs, sampler=sampler, drop_last=True, num_workers=0)
    val_dl   = DataLoader(val_ds, 512, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    base = MultiModalSCGSnn(n_in=X.shape[2], n_channels=X.shape[1],
                            n_hidden=args.H, n_classes=args.n_classes,
                            beta=0.9, threshold=1.0, T=args.T).to(device)
    model = AlignedMultiModalSCGSnn(base, max_tau=args.max_tau, init_zero=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AlignedSNN H={args.H} T={args.T}  params={n_params}  "
          f"(tau adds {model.n_channels} float params)")

    # Two param groups: small lr for fc1/fc2, larger for tau
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

        # Eval with quantized integer tau (matches FPGA behavior)
        model.export_int_tau()
        val_acc, cm = evaluate_aligned(model, val_dl, device)
        # Also re-enable float tau for next training step
        model.quantize_inference = False

        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            tau_disp = ", ".join(f"{t:+.1f}" for t in model.tau.tolist())
            print(f"  ep {ep:02d} train={train_acc*100:.2f}% val={val_acc*100:.2f}%  "
                  f"tau=[{tau_disp}]  ({time.time()-t0:.0f}s)", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc; best_cm = cm; best_ep = ep
            # Save with int tau already exported
            model.export_int_tau()
            torch.save({
                "state": model.base.state_dict(),
                "tau_int": model.tau_int.tolist(),
                "tau_float": model.tau.tolist(),
                "val_acc": val_acc, "epoch": ep, "arch": args.tag,
                "n_in": X.shape[2], "n_channels": X.shape[1], "H": args.H,
                "n_classes": args.n_classes, "beta": 0.9, "threshold": 1.0,
                "T": args.T, "shift_max_train": args.shift_max,
                "holdout_subjects": args.holdout,
            }, args.out / f"best_{args.tag}.pt")
            model.quantize_inference = False

    # Final integer tau
    model.export_int_tau()
    tau_int = model.tau_int.tolist()
    print(f"\nbest val_acc = {best_acc*100:.2f}% @ ep {best_ep}")
    print(f"final tau_int = {tau_int}")

    manifest = {
        "tag": args.tag, "best_val_acc": float(best_acc), "best_epoch": best_ep,
        "tau_int_at_best": tau_int,
        "tau_float": model.tau.tolist(),
        "shift_max_train": args.shift_max,
        "holdout_subjects": args.holdout,
        "n_train_windows": int(len(X_tr)), "n_val_windows": int(len(X_va)),
        "confusion_matrix": best_cm.tolist() if best_cm is not None else None,
        "H": args.H, "T": args.T, "epochs": args.epochs, "seed": args.seed,
    }
    (args.out / f"best_{args.tag}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"-> {args.out}/best_{args.tag}_manifest.json")


if __name__ == "__main__":
    main()
