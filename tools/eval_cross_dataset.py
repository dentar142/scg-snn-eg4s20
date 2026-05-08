"""eval_cross_dataset.py - cross-dataset transfer test FOSTER -> CEBSDB.

Evaluates a FOSTER-trained multi-modal SNN ckpt on CEBSDB val set
(single SCG channel) by mapping CEBSDB SCG into the FOSTER ACC slot
(channel idx 2) and zeroing the other 4 channels.

Compares:
  1. Aligned ckpt (no modality dropout) -> 0-shot baseline (predicted to fail)
  2. Modality-dropout ckpt -> 0-shot transfer
  3. Modality-dropout ckpt + STDP per-subject calibration

Usage:
    python tools/eval_cross_dataset.py \\
        --aligned-ckpt model/ckpt/best_snn_mm_h32t16_aligned.pt \\
        --dropout-ckpt model/ckpt/best_snn_mm_h32t16_dropout.pt \\
        --cebs-data data_excl100/val.npz
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_snn_multimodal import MultiModalSCGSnn  # noqa: E402

ACC_SLOT = 2   # FOSTER channel order: [PVDF, PZT, ACC, PCG, ERB]


def make_model(ckpt, device, baked_aligned=False):
    """Build MultiModalSCGSnn. If baked_aligned, the W1 already has tau
    permuted in. Otherwise we apply tau_int via input shift here."""
    H = int(ckpt["H"]); T = int(ckpt["T"])
    n_in = int(ckpt["n_in"]); n_chan = int(ckpt["n_channels"])
    m = MultiModalSCGSnn(n_in=n_in, n_channels=n_chan, n_hidden=H,
                         n_classes=int(ckpt["n_classes"]),
                         beta=float(ckpt.get("beta", 0.9)),
                         threshold=float(ckpt.get("threshold", 1.0)),
                         T=T).to(device)
    state = ckpt["state"]
    # Some ckpts have nested base.fc1.weight; flatten if needed
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("base.", "")] = v
    m.load_state_dict(new_state)
    return m


def cebs_to_foster_5ch(X_cebs, n_channels=5, slot=ACC_SLOT, tau_int=None):
    """X_cebs: (N, 1, 256). Return (N, 5, 256) with CEBSDB SCG in ACC slot,
    other channels zero. If tau_int is given, apply np.roll(channel, tau)."""
    N, _, L = X_cebs.shape
    out = np.zeros((N, n_channels, L), dtype=np.int8)
    sig = X_cebs[:, 0, :]
    if tau_int is not None and len(tau_int) == n_channels:
        sig = np.roll(sig, tau_int[slot], axis=-1)
    out[:, slot, :] = sig
    return out


@torch.no_grad()
def eval_acc(model, X, y, device, batch=512):
    model.eval()
    Xt = torch.from_numpy(X.astype(np.float32) / 127.0)
    yt = torch.from_numpy(y).long()
    correct = total = 0
    cm = np.zeros((3, 3), dtype=np.int64)
    for i in range(0, len(X), batch):
        x = Xt[i:i+batch].to(device); yb = yt[i:i+batch].to(device)
        pred = model(x).argmax(1)
        correct += (pred == yb).sum().item(); total += yb.numel()
        for t, p in zip(yb.cpu().numpy(), pred.cpu().numpy()):
            cm[int(t), int(p)] += 1
    return correct / max(total, 1), cm


def stdp_calibrate(model, X_cal, y_cal, device, n_epochs=10, lr=5e-3):
    for p in model.fc1.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(model.fc2.parameters()), lr=lr)
    Xc = torch.from_numpy(X_cal.astype(np.float32) / 127.0).to(device)
    yc = torch.from_numpy(y_cal).long().to(device)
    if len(np.unique(y_cal)) < 2:
        return False
    model.train()
    for _ in range(n_epochs):
        logits = model(Xc)
        loss = F.cross_entropy(logits, yc, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True); loss.backward()
        opt.step()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aligned-ckpt", type=Path, required=True)
    p.add_argument("--dropout-ckpt", type=Path, required=True)
    p.add_argument("--cebs-data", type=Path, default=Path("data_excl100/val.npz"))
    p.add_argument("--out", type=Path, default=Path("doc/cross_dataset_cebsdb.json"))
    p.add_argument("--n-cal", type=int, default=300)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    d = np.load(args.cebs_data, allow_pickle=True)
    X = d["X"]; y = d["y"]; sid = d["sid"]
    print(f"CEBSDB: X={X.shape}  y={y.shape}  unique sid={len(set(sid.tolist()))}")
    print(f"  per-class: BG={int((y==0).sum())} Sys={int((y==1).sum())} Dia={int((y==2).sum())}")

    results = {}
    for ckpt_name, ckpt_path in [("aligned_no_dropout", args.aligned_ckpt),
                                 ("dropout_aligned", args.dropout_ckpt)]:
        if not ckpt_path.exists():
            print(f"  [skip] {ckpt_name}: missing {ckpt_path}")
            continue
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        tau_int = ck.get("tau_int", None)
        print(f"\n=== {ckpt_name}  tau_int={tau_int} ===")

        # Map CEBSDB single channel to FOSTER 5-ch with ACC-slot shifted by tau
        X_5ch = cebs_to_foster_5ch(X, n_channels=int(ck["n_channels"]),
                                    tau_int=tau_int)
        # Zero-shot eval
        m_zs = make_model(ck, device)
        zs_acc, zs_cm = eval_acc(m_zs, X_5ch, y, device)
        print(f"  0-shot acc = {zs_acc*100:.2f}%")

        # STDP calibration: stratified per-subject 100 windows
        per_sub_results = []
        unique_sids = sorted(set(sid.tolist()))
        n_per_class = args.n_cal // 3
        for s in unique_sids:
            ms = sid == s
            if ms.sum() < args.n_cal + 100: continue
            Xs = X_5ch[ms]; ys = y[ms]
            cal_idx = []
            for c in range(3):
                cls_idx = np.where(ys == c)[0]
                cal_idx.extend(cls_idx[:n_per_class].tolist())
            cal_idx = sorted(set(cal_idx))
            test_idx = sorted(set(range(len(Xs))) - set(cal_idx))
            if len(cal_idx) < 30 or len(test_idx) < 100: continue
            X_cal, y_cal = Xs[cal_idx], ys[cal_idx]
            X_test, y_test = Xs[test_idx], ys[test_idx]
            # Fresh model per subject
            m = make_model(ck, device)
            base, _ = eval_acc(m, X_test, y_test, device)
            ok = stdp_calibrate(m, X_cal, y_cal, device, args.n_epochs)
            cal, _ = eval_acc(m, X_test, y_test, device)
            per_sub_results.append({
                "sid": int(s), "n_cal": len(X_cal), "n_test": len(X_test),
                "base_acc": float(base), "cal_acc": float(cal),
                "delta_pp": float((cal - base) * 100),
            })
            print(f"    sid={s:3d}  n_test={len(X_test):4d}  "
                  f"base={base*100:.2f}%  cal={cal*100:.2f}%  d={cal-base:+.4f}")
        if per_sub_results:
            mean_base = float(np.mean([r["base_acc"] for r in per_sub_results]))
            mean_cal  = float(np.mean([r["cal_acc"]  for r in per_sub_results]))
            mean_d    = float(np.mean([r["delta_pp"] for r in per_sub_results]))
            print(f"  mean per-subject base = {mean_base*100:.2f}%  cal = {mean_cal*100:.2f}%  "
                  f"d = {mean_d:+.2f} pp")
        else:
            mean_base = mean_cal = mean_d = float("nan")

        results[ckpt_name] = {
            "ckpt": str(ckpt_path),
            "tau_int": tau_int,
            "zero_shot_acc": float(zs_acc),
            "zero_shot_cm": zs_cm.tolist(),
            "per_subject_calibration": per_sub_results,
            "mean_base_acc_per_subject": mean_base,
            "mean_cal_acc_per_subject": mean_cal,
            "mean_delta_pp": mean_d,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n=== Cross-dataset summary ===")
    print(f"CEBSDB self-trained baseline: 5-fold CV 85.48 ± 2.02% (single-modal SCG, ref §6)")
    for k, v in results.items():
        print(f"\n{k}:")
        print(f"  0-shot transfer:                {v['zero_shot_acc']*100:.2f}%")
        if not np.isnan(v.get("mean_cal_acc_per_subject", float("nan"))):
            print(f"  + STDP per-subject calibration: {v['mean_cal_acc_per_subject']*100:.2f}% "
                  f"(delta {v['mean_delta_pp']:+.2f} pp)")
    print(f"\n-> wrote {args.out}")


if __name__ == "__main__":
    main()
