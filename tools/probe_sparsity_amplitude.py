"""probe_sparsity_amplitude.py — for each ckpt in a sweep, measure:
  (1) average LIF1 spike count per inference (sparsity / energy proxy)
  (2) accuracy under input amplitude perturbation (scale ∈ {0.5, 0.7, 1.0, 1.3, 1.5})

Saves JSON summary to doc/sweep_sparsity_amplitude.json.

Usage:
    python tools/probe_sparsity_amplitude.py \
        --ckpt-dir model/ckpt/sweep --data data_foster_multi/all.npz \
        --holdout sub003 sub006 sub009 sub013 sub020 sub021 sub024 sub026 \
        --out doc/sweep_sparsity_amplitude.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_snn_multimodal import MultiModalSCGSnn  # noqa: E402


@torch.no_grad()
def evaluate_with_sparsity(model, X_t, y_t, device, scale=1.0, batch=512):
    """Evaluate accuracy + measure mean LIF1 spike count per inference.

    Patches MultiModalSCGSnn.forward in-place to count L1 spikes.
    Returns (acc, mean_spikes_per_inference, total_spike_ops).
    """
    model.eval()
    H = model.n_hidden
    T = model.T
    # max possible spikes = T * H per sample (if every step fires every neuron)
    correct = total = 0
    spike_sum = 0.0
    n_samples = 0
    for i in range(0, len(X_t), batch):
        x = X_t[i:i+batch].to(device) * scale
        y = y_t[i:i+batch].to(device)
        # Recreate forward with spike counter (verbatim from MultiModalSCGSnn)
        if x.dim() == 3:
            B = x.size(0)
            xf = x.reshape(B, -1)
        else:
            xf = x
            B = xf.size(0)
        I1_const = model.fc1(xf)
        v1 = torch.zeros(B, model.n_hidden, device=device)
        v2 = torch.zeros(B, model.n_classes, device=device)
        out = torch.zeros(B, model.n_classes, device=device)
        l1_spikes = torch.zeros(B, device=device)
        threshold = model.threshold; beta = model.beta
        for _ in range(T):
            v1 = beta * v1 + I1_const
            s1 = (v1 >= threshold).float()
            l1_spikes = l1_spikes + s1.sum(dim=1)  # spikes per sample this step
            v1 = v1 - s1 * threshold
            I2 = model.fc2(s1)
            v2 = beta * v2 + I2
            s2 = (v2 >= threshold).float()
            v2 = v2 - s2 * threshold
            out = out + s2
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        spike_sum += l1_spikes.sum().item()
        n_samples += B
    return correct / max(total, 1), spike_sum / max(n_samples, 1), spike_sum, n_samples * T * H


def load_ckpt_into_model(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    H = int(ck["H"]); T = int(ck["T"])
    n_in = int(ck["n_in"]); n_chan = int(ck["n_channels"])
    model = MultiModalSCGSnn(n_in=n_in, n_channels=n_chan, n_hidden=H,
                             n_classes=int(ck["n_classes"]),
                             beta=float(ck.get("beta", 0.9)),
                             threshold=float(ck.get("threshold", 1.0)), T=T).to(device)
    model.load_state_dict(ck["state"])
    return model, H, T, n_in * n_chan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, default=Path("model/ckpt/sweep"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--holdout", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scales", nargs="+", type=float, default=[0.5, 0.7, 1.0, 1.3, 1.5])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    d = np.load(args.data, allow_pickle=True)
    X = d["X"]; y = d["y"]; sid = d["sid"]
    rec_names = list(map(str, d["record_names"]))
    name_to_sid = {n: i for i, n in enumerate(rec_names)}
    holdout_sids = [name_to_sid[h] for h in args.holdout]
    mask = np.isin(sid, holdout_sids)
    Xh, yh = X[mask], y[mask]
    print(f"hold-out subjects: {len(args.holdout)}  windows: {len(Xh)}")
    Xh_t = torch.from_numpy(Xh.astype(np.float32) / 127.0)
    yh_t = torch.from_numpy(yh).long()

    ckpts = sorted([p for p in args.ckpt_dir.glob("best_sweep_H*_T*.pt")])
    print(f"found {len(ckpts)} ckpts in {args.ckpt_dir}")

    results = []
    for ck_path in ckpts:
        tag = ck_path.stem  # best_sweep_H{H}_T{T}
        print(f"\n--- {tag} ---")
        model, H, T, n_in_total = load_ckpt_into_model(ck_path, device)
        n_params = sum(p.numel() for p in model.parameters())
        # Sparsity at scale=1
        acc, mean_spikes, total_spikes, max_spikes = evaluate_with_sparsity(
            model, Xh_t, yh_t, device, scale=1.0)
        sparsity = 1.0 - mean_spikes / (T * H)
        print(f"  acc={acc*100:.2f}%  mean_L1_spikes={mean_spikes:.1f}/inference "
              f"(max {T*H}={T}×{H})  sparsity={sparsity*100:.1f}%")
        # Amplitude robustness
        amp = []
        for s in args.scales:
            a, _, _, _ = evaluate_with_sparsity(model, Xh_t, yh_t, device, scale=s)
            amp.append({"scale": s, "acc": a})
            print(f"  scale={s:.1f}  acc={a*100:.2f}%")
        results.append({
            "tag": tag, "H": H, "T": T, "n_params": n_params,
            "n_in_total": n_in_total,
            "val_acc_unperturbed": acc,
            "mean_l1_spikes_per_inference": mean_spikes,
            "max_l1_spikes": T * H,
            "sparsity": sparsity,
            "amplitude_robustness": amp,
        })

    summary = {
        "data": str(args.data),
        "holdout_subjects": args.holdout,
        "n_holdout_windows": int(len(Xh)),
        "scales": args.scales,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n-> wrote {args.out}")


if __name__ == "__main__":
    main()
