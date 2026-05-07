"""stdp_personalize.py - per-subject online calibration proof-of-concept.

Mimics what an on-chip STDP / Hebbian update would deliver: take the deployed
H=32 T=16 ckpt, freeze fc1 (the big BRAM ROM), allow only fc2 (96 INT8
weights, fits in tiny RAM) to adapt to N=100 calibration windows of each
hold-out subject. Measure acc lift on the remaining windows.

Why fc2 only:
  - fc1 is 41 KB BRAM ROM, expensive to make writable
  - fc2 is 96 INT8 weights = 12 LUT-RAM, easily writable on FPGA
  - Adapting fc2 corresponds to changing per-class spike-count thresholds
    (decision-boundary calibration), not feature extraction

The actual update rule used here is gradient-based fine-tuning of fc2 via
the same fast-sigmoid surrogate as training. STDP would substitute a local
Hebbian rule:
    Δw[c, h] = +η · pre[h] · (post[c] - target[c])
The accuracy lift trend should be similar; gradient is the upper bound.

Output: doc/stdp_personalize.json + doc/figs/stdp_per_subject.png + writeup.
"""
from __future__ import annotations
import argparse, json, sys, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_snn_multimodal import MultiModalSCGSnn  # noqa: E402


def make_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = MultiModalSCGSnn(n_in=int(ck["n_in"]), n_channels=int(ck["n_channels"]),
                         n_hidden=int(ck["H"]), n_classes=int(ck["n_classes"]),
                         beta=float(ck.get("beta", 0.9)),
                         threshold=float(ck.get("threshold", 1.0)),
                         T=int(ck["T"])).to(device)
    m.load_state_dict(ck["state"])
    return m


@torch.no_grad()
def eval_acc(m, X, y, device, batch=512):
    m.eval()
    correct = total = 0
    Xt = torch.from_numpy(X.astype(np.float32) / 127.0)
    yt = torch.from_numpy(y).long()
    for i in range(0, len(X), batch):
        x = Xt[i:i+batch].to(device); ybt = yt[i:i+batch].to(device)
        pred = m(x).argmax(1)
        correct += (pred == ybt).sum().item(); total += ybt.numel()
    return correct / max(total, 1)


def calibrate_one_subject(base_ckpt_path, X_cal, y_cal, X_test, y_test,
                          device, n_epochs=10, lr=5e-3):
    """Return baseline_acc, cal_acc on test set."""
    m = make_model(base_ckpt_path, device)
    base_acc = eval_acc(m, X_test, y_test, device)
    # Freeze fc1
    for p in m.fc1.parameters():
        p.requires_grad_(False)
    # Train only fc2
    opt = torch.optim.Adam([p for p in m.fc2.parameters()], lr=lr)
    Xc = torch.from_numpy(X_cal.astype(np.float32) / 127.0).to(device)
    yc = torch.from_numpy(y_cal).long().to(device)
    if len(np.unique(y_cal)) < 2:
        return base_acc, base_acc, "skipped: monoclass calibration set"
    m.train()
    for _ in range(n_epochs):
        # Single batch (calibration is small, 100 windows)
        logits = m(Xc)
        loss = F.cross_entropy(logits, yc, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True); loss.backward()
        opt.step()
    cal_acc = eval_acc(m, X_test, y_test, device)
    return base_acc, cal_acc, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--holdout", nargs="+", required=True)
    p.add_argument("--n-cal", type=int, default=100)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("doc/stdp_personalize.json"))
    p.add_argument("--fig-dir", type=Path, default=Path("doc/figs"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    d = np.load(args.data, allow_pickle=True)
    X = d["X"]; y = d["y"]; sid = d["sid"]
    rec_names = list(map(str, d["record_names"]))
    name_to_sid = {n: i for i, n in enumerate(rec_names)}

    rows = []
    for sub in args.holdout:
        s = name_to_sid[sub]
        m = sid == s
        if m.sum() < args.n_cal + 100:
            continue
        Xs = X[m]; ys = y[m]
        # Use first n_cal windows as calibration; rest as test (mimics 30 s of
        # on-board recording before deployment-active phase).
        # Stratify by class on calibration set: take up to n_cal/3 per class.
        per_class = args.n_cal // 3
        cal_idx = []
        for c in range(3):
            cls_idx = np.where(ys == c)[0]
            cal_idx.extend(cls_idx[:per_class].tolist())
        cal_idx = sorted(cal_idx)
        test_idx = sorted(set(range(len(Xs))) - set(cal_idx))
        X_cal, y_cal = Xs[cal_idx], ys[cal_idx]
        X_test, y_test = Xs[test_idx], ys[test_idx]
        print(f"\n--- {sub} ---  cal {len(X_cal)}  test {len(X_test)}")
        base_acc, cal_acc, status = calibrate_one_subject(
            args.ckpt, X_cal, y_cal, X_test, y_test, device,
            n_epochs=args.n_epochs, lr=args.lr)
        delta = (cal_acc - base_acc) * 100
        print(f"  base = {base_acc*100:.2f}%  after_cal = {cal_acc*100:.2f}%  "
              f"Δ = {delta:+.2f} pp  ({status})")
        rows.append({
            "subject": sub, "n_cal": len(X_cal), "n_test": len(X_test),
            "base_acc": float(base_acc), "cal_acc": float(cal_acc),
            "delta_pp": float(delta), "status": status,
            "cal_class_dist": [int((y_cal == c).sum()) for c in range(3)],
        })

    # Plot
    rows.sort(key=lambda r: r["base_acc"])
    subs = [r["subject"] for r in rows]
    base = [r["base_acc"] * 100 for r in rows]
    cal = [r["cal_acc"] * 100 for r in rows]
    x = np.arange(len(subs)); w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, base, w, color="C7", label="Pre-cal (deployed ckpt)")
    plt.bar(x + w/2, cal,  w, color="C2",
            label=f"Post-cal ({args.n_cal} windows, {args.n_epochs} epoch)")
    for i, r in enumerate(rows):
        plt.text(x[i] + w/2, cal[i] + 0.4, f"{r['delta_pp']:+.1f}",
                 ha="center", fontsize=9)
    plt.xticks(x, subs, rotation=30); plt.ylabel("Test acc (%)")
    plt.title(f"Per-subject calibration: fc2-only update, n_cal={args.n_cal}")
    plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(args.fig_dir / "stdp_per_subject.png", dpi=150)
    plt.close()

    summary = {
        "ckpt": str(args.ckpt),
        "n_cal_target": args.n_cal, "n_epochs": args.n_epochs, "lr": args.lr,
        "results": rows,
        "mean_base_acc": float(np.mean([r["base_acc"] for r in rows])),
        "mean_cal_acc": float(np.mean([r["cal_acc"] for r in rows])),
        "mean_delta_pp": float(np.mean([r["delta_pp"] for r in rows])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Summary ===")
    print(f"mean base = {summary['mean_base_acc']*100:.2f}%")
    print(f"mean cal  = {summary['mean_cal_acc']*100:.2f}%")
    print(f"mean Δ    = {summary['mean_delta_pp']:+.2f} pp")
    print(f"-> {args.out}")
    print(f"-> {args.fig_dir}/stdp_per_subject.png")


if __name__ == "__main__":
    main()
