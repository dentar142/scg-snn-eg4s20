"""calibrate_abstention.py — calibrate margin-based abstention for the
deployed H=32 T=16 multimodal SNN.

For each hold-out window, compute spike-count distribution at output, then
margin = top1 - top2. Sweep abstention threshold tau over the empirical
margin distribution; for each tau report:
    - coverage (fraction kept)
    - acc_kept (acc on the kept subset)
    - acc_rejected (acc on the abstained subset, lower bound on safety)
    - per-class kept rate (does abstention disproportionately drop one class?)

Goal: find tau that lifts kept-set acc to >= 97% with reasonable coverage
(e.g., >= 80%). This is the FPGA-cheap clinical-safety mechanism.

Usage:
    python tools/calibrate_abstention.py --ckpt model/ckpt/sweep/best_sweep_H32_T16.pt \
        --data data_foster_multi/all.npz \
        --holdout sub003 sub006 sub009 sub013 sub020 sub021 sub024 sub026 \
        --out doc/abstention_h32_t16.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_snn_multimodal import MultiModalSCGSnn  # noqa: E402

CLASS_NAMES = ["BG", "Sys", "Dia"]


@torch.no_grad()
def forward_with_spike_counts(model, X_t, device, batch=512):
    """Run forward T steps, return per-sample (B, K) integer spike counts."""
    model.eval()
    out = []
    for i in range(0, len(X_t), batch):
        x = X_t[i:i+batch].to(device)
        if x.dim() == 3:
            B = x.size(0); xf = x.reshape(B, -1)
        else:
            xf = x; B = xf.size(0)
        I1 = model.fc1(xf)
        v1 = torch.zeros(B, model.n_hidden, device=device)
        v2 = torch.zeros(B, model.n_classes, device=device)
        cnt = torch.zeros(B, model.n_classes, device=device)
        for _ in range(model.T):
            v1 = model.beta * v1 + I1
            s1 = (v1 >= model.threshold).float()
            v1 = v1 - s1 * model.threshold
            I2 = model.fc2(s1)
            v2 = model.beta * v2 + I2
            s2 = (v2 >= model.threshold).float()
            v2 = v2 - s2 * model.threshold
            cnt = cnt + s2
        out.append(cnt.cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--holdout", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fig-dir", type=Path, default=Path("doc/figs"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    d = np.load(args.data, allow_pickle=True)
    X = d["X"]; y = d["y"]; sid = d["sid"]
    rec_names = list(map(str, d["record_names"]))
    name_to_sid = {n: i for i, n in enumerate(rec_names)}
    holdout_sids = [name_to_sid[h] for h in args.holdout]
    mask = np.isin(sid, holdout_sids)
    Xh, yh, sidh = X[mask], y[mask], sid[mask]
    print(f"hold-out: {len(args.holdout)} subjects, {len(Xh)} windows")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = MultiModalSCGSnn(n_in=int(ck["n_in"]), n_channels=int(ck["n_channels"]),
                             n_hidden=int(ck["H"]), n_classes=int(ck["n_classes"]),
                             beta=float(ck.get("beta", 0.9)),
                             threshold=float(ck.get("threshold", 1.0)),
                             T=int(ck["T"])).to(device)
    model.load_state_dict(ck["state"])

    Xh_t = torch.from_numpy(Xh.astype(np.float32) / 127.0)
    spike_counts = forward_with_spike_counts(model, Xh_t, device)
    sorted_counts = np.sort(spike_counts, axis=1)[:, ::-1]   # descending
    top1 = sorted_counts[:, 0]
    top2 = sorted_counts[:, 1]
    margin = top1 - top2
    preds = spike_counts.argmax(1)
    correct = (preds == yh)
    print(f"baseline acc = {correct.mean()*100:.2f}%")
    print(f"margin distribution: min={margin.min():.0f} max={margin.max():.0f} "
          f"median={np.median(margin):.0f}")

    # Sweep tau
    taus = sorted(set(margin.tolist()))
    sweep = []
    for tau in taus:
        keep = margin >= tau
        if keep.sum() == 0: continue
        acc_keep = correct[keep].mean()
        coverage = keep.mean()
        rej = ~keep
        acc_rej = correct[rej].mean() if rej.sum() > 0 else float("nan")
        per_class_keep = []
        for c in range(3):
            cm = (yh == c) & keep
            per_class_keep.append({
                "class": c,
                "keep_n": int(cm.sum()),
                "keep_acc": float(correct[cm].mean()) if cm.sum() > 0 else float("nan"),
            })
        sweep.append({
            "tau": int(tau), "coverage": float(coverage),
            "acc_kept": float(acc_keep), "acc_rejected": float(acc_rej),
            "per_class_keep": per_class_keep,
            "n_kept": int(keep.sum()), "n_rejected": int(rej.sum()),
        })

    # Find smallest tau achieving acc_kept >= 0.97 with coverage >= 0.80
    target = next((s for s in sweep
                   if s["acc_kept"] >= 0.97 and s["coverage"] >= 0.80), None)
    if target is None:
        target = max(sweep, key=lambda s: s["acc_kept"] * s["coverage"])
        target_note = "fallback: max(acc_kept * coverage)"
    else:
        target_note = "first tau hitting acc_kept >= 97% AND coverage >= 80%"

    # ===== Plot 1: coverage vs acc_kept =====
    plt.figure(figsize=(7, 5))
    cov = [s["coverage"] * 100 for s in sweep]
    acc = [s["acc_kept"] * 100 for s in sweep]
    plt.plot(cov, acc, color="C0", linewidth=2)
    plt.scatter(target["coverage"] * 100, target["acc_kept"] * 100, s=80,
                color="red", zorder=5,
                label=f"tau={target['tau']} (kept {target['coverage']*100:.1f}%, "
                      f"acc {target['acc_kept']*100:.2f}%)")
    plt.axhline(94.43, color="gray", linestyle=":", alpha=0.6,
                label="baseline 94.43% (no abstention)")
    plt.xlabel("Coverage (% of windows kept)")
    plt.ylabel("Accuracy on kept windows (%)")
    plt.title("Margin-based abstention: coverage-accuracy trade-off "
              "(H=32 T=16, hold-out 8 subjects)")
    plt.grid(alpha=0.3); plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(args.fig_dir / "abstention_coverage_acc.png", dpi=150)
    plt.close()

    # ===== Plot 2: margin histogram split by correct/wrong =====
    plt.figure(figsize=(7, 5))
    bins = range(int(margin.min()), int(margin.max()) + 2)
    plt.hist(margin[correct], bins=bins, alpha=0.6, color="C2",
             label=f"Correct ({correct.sum()})")
    plt.hist(margin[~correct], bins=bins, alpha=0.6, color="C3",
             label=f"Wrong ({(~correct).sum()})")
    plt.axvline(target["tau"], color="black", linestyle="--",
                label=f"tau={target['tau']}")
    plt.xlabel("Margin = top1 - top2 spike count")
    plt.ylabel("Window count")
    plt.title("Margin distribution by correctness (H=32 T=16, T=16 max margin)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(args.fig_dir / "abstention_margin_hist.png", dpi=150)
    plt.close()

    # ===== Plot 3: per-subject coverage at the chosen tau =====
    plt.figure(figsize=(8, 5))
    keep_at_target = margin >= target["tau"]
    sub_data = []
    for s in holdout_sids:
        m = sidh == s
        sub_data.append({
            "name": rec_names[s],
            "n": int(m.sum()),
            "coverage": float(keep_at_target[m].mean()),
            "acc_kept": float(correct[m & keep_at_target].mean()) if (m & keep_at_target).sum() > 0 else 0.0,
            "acc_baseline": float(correct[m].mean()),
        })
    subs = [r["name"] for r in sub_data]
    cov_per = [r["coverage"] * 100 for r in sub_data]
    acc_keep_per = [r["acc_kept"] * 100 for r in sub_data]
    acc_base_per = [r["acc_baseline"] * 100 for r in sub_data]
    x = np.arange(len(subs)); w = 0.35
    plt.bar(x - w/2, acc_base_per, w, color="C7", label="Acc no-abstain")
    plt.bar(x + w/2, acc_keep_per, w, color="C0", label="Acc on kept")
    for i, c in enumerate(cov_per):
        plt.text(x[i] + w/2, acc_keep_per[i] + 0.5, f"{c:.0f}%",
                 ha="center", fontsize=8)
    plt.xticks(x, subs, rotation=30)
    plt.ylabel("Accuracy (%)")
    plt.title(f"Per-subject acc at tau={target['tau']} (numbers = coverage)")
    plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(args.fig_dir / "abstention_per_subject.png", dpi=150)
    plt.close()

    summary = {
        "ckpt": str(args.ckpt),
        "holdout_subjects": args.holdout,
        "n_windows": int(len(Xh)),
        "baseline_acc": float(correct.mean()),
        "margin_stats": {"min": int(margin.min()), "max": int(margin.max()),
                         "median": int(np.median(margin)), "mean": float(margin.mean())},
        "recommended_tau": int(target["tau"]),
        "recommendation_rationale": target_note,
        "at_recommended_tau": target,
        "per_subject_at_recommended_tau": sub_data,
        "sweep_taus": sweep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Recommendation ===")
    print(f"tau = {target['tau']}  ({target_note})")
    print(f"  coverage  = {target['coverage']*100:.2f}%  ({target['n_kept']}/{target['n_kept']+target['n_rejected']})")
    print(f"  acc_kept  = {target['acc_kept']*100:.2f}%")
    print(f"  acc_rej   = {target['acc_rejected']*100:.2f}%")
    print(f"  per-class keep: " + ", ".join(
        f"{CLASS_NAMES[r['class']]}: {r['keep_n']}/{int(np.sum(yh==r['class']))} "
        f"acc={r['keep_acc']*100:.2f}%" for r in target["per_class_keep"]))
    print(f"\n-> wrote {args.out}")
    print(f"-> figs: {args.fig_dir}/abstention_*.png")


if __name__ == "__main__":
    main()
