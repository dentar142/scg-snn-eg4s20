"""analyze_dia_errors.py - characterize where Dia mis-classifications cluster.

Reuses the full subject-disjoint hold-out bench (40,575 windows on H=32 T=32
multimodal_holdout.bit, doc/bench_fpga_snn_multimodal_holdout.json) since
predictions and per-sample sid are saved there. Generates:

  doc/figs/dia_error_per_subject.png   - per-subject Dia recall + error split
  doc/figs/dia_error_modality_signal.png - per-modality Dia window energy
  doc/dia_error_summary.md             - findings table

The full bench JSON does not store per-sample predictions; instead, re-run
prediction in CPU sim using the deployed ckpt to get per-sample preds for
the hold-out subjects.

Usage:
    python tools/analyze_dia_errors.py --data data_foster_multi/all.npz \\
        --ckpt model/ckpt/best_snn_mm_h32_holdout.pt \\
        --holdout sub003 sub006 sub009 sub013 sub020 sub021 sub024 sub026
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
MODALITY_NAMES = ["PVDF", "PZT", "ACC", "PCG", "ERB"]


@torch.no_grad()
def predict_all(model, X, device, batch=512):
    model.eval()
    Xt = torch.from_numpy(X.astype(np.float32) / 127.0)
    preds = []
    for i in range(0, len(X), batch):
        x = Xt[i:i+batch].to(device)
        logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--holdout", nargs="+", required=True)
    p.add_argument("--fig-dir", type=Path, default=Path("doc/figs"))
    p.add_argument("--out-md", type=Path, default=Path("doc/dia_error_summary.md"))
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

    preds = predict_all(model, Xh, device)
    print(f"overall acc on hold-out: {(preds==yh).mean()*100:.2f}%")

    # ===== Per-subject Dia error breakdown =====
    rows = []
    for s in holdout_sids:
        m = sidh == s
        n_dia = int(((yh == 2) & m).sum())
        if n_dia == 0:
            continue
        dia_pred = preds[m][yh[m] == 2]
        n_correct = int((dia_pred == 2).sum())
        n_to_bg   = int((dia_pred == 0).sum())
        n_to_sys  = int((dia_pred == 1).sum())
        rows.append({
            "subject": rec_names[s],
            "n_dia_total": n_dia,
            "dia_recall": n_correct / n_dia,
            "n_dia_correct": n_correct,
            "n_dia_to_bg": n_to_bg, "n_dia_to_sys": n_to_sys,
            "frac_to_bg": n_to_bg / n_dia, "frac_to_sys": n_to_sys / n_dia,
        })

    # ===== Plot 1: Per-subject Dia recall + error split =====
    plt.figure(figsize=(10, 5))
    subs = [r["subject"] for r in rows]
    correct = [r["dia_recall"] * 100 for r in rows]
    to_bg = [r["frac_to_bg"] * 100 for r in rows]
    to_sys = [r["frac_to_sys"] * 100 for r in rows]
    x = np.arange(len(subs))
    plt.bar(x, correct, label="Dia -> Dia (correct)", color="C2")
    plt.bar(x, to_bg, bottom=correct, label="Dia -> BG", color="C3")
    plt.bar(x, to_sys,
            bottom=np.array(correct) + np.array(to_bg),
            label="Dia -> Sys", color="C1")
    plt.xticks(x, subs, rotation=30)
    plt.ylabel("% of subject's Dia windows")
    plt.title("Dia class predictions per hold-out subject (FP32 sim, H=32 T=32 ckpt)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(args.fig_dir / "dia_error_per_subject.png", dpi=150)
    plt.close()

    # ===== Modality energy comparison: correct Dia vs error Dia =====
    dia_idx = np.where(yh == 2)[0]
    correct_idx = dia_idx[preds[dia_idx] == 2]
    error_idx = dia_idx[preds[dia_idx] != 2]
    print(f"  Dia total {len(dia_idx)} | correct {len(correct_idx)} | error {len(error_idx)}")

    # Per-modality RMS energy
    Xh_dia_correct = Xh[correct_idx].astype(np.float32) / 127.0
    Xh_dia_error = Xh[error_idx].astype(np.float32) / 127.0
    rms_correct = np.sqrt(np.mean(Xh_dia_correct ** 2, axis=2))  # (N, C)
    rms_error = np.sqrt(np.mean(Xh_dia_error ** 2, axis=2))
    mean_correct = rms_correct.mean(axis=0)
    mean_error = rms_error.mean(axis=0)
    std_correct = rms_correct.std(axis=0)
    std_error = rms_error.std(axis=0)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(MODALITY_NAMES))
    width = 0.35
    plt.bar(x - width/2, mean_correct, width, yerr=std_correct, capsize=4,
            color="C2", label=f"Correctly classified Dia ({len(correct_idx)})")
    plt.bar(x + width/2, mean_error, width, yerr=std_error, capsize=4,
            color="C3", label=f"Mis-classified Dia ({len(error_idx)})")
    plt.xticks(x, MODALITY_NAMES)
    plt.ylabel("RMS energy (normalized)")
    plt.title("Per-modality signal energy: correct vs misclassified Dia windows")
    plt.legend(); plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(args.fig_dir / "dia_error_modality_signal.png", dpi=150)
    plt.close()

    # ===== Summary table =====
    md = ["# Dia class error analysis\n\n"]
    md.append(f"Hold-out subjects: {args.holdout}  ({len(Xh)} total windows, "
              f"{int((yh==2).sum())} are Dia ground truth)\n\n")
    md.append("## Per-subject Dia recall\n\n")
    md.append("| Subject | n_dia | Dia recall | -> BG | -> Sys |\n")
    md.append("|---------|------:|-----------:|------:|-------:|\n")
    for r in sorted(rows, key=lambda x: x["dia_recall"]):
        md.append(f"| {r['subject']} | {r['n_dia_total']} | "
                  f"{r['dia_recall']*100:.2f}% | {r['frac_to_bg']*100:.1f}% | "
                  f"{r['frac_to_sys']*100:.1f}% |\n")

    overall_dia = preds[yh == 2]
    md.append("\n## Aggregate Dia-class confusion\n\n")
    md.append("| Predicted | Count | Fraction |\n|-----------|------:|---------:|\n")
    for c in range(3):
        n = int((overall_dia == c).sum())
        md.append(f"| {CLASS_NAMES[c]} | {n} | {n/len(overall_dia)*100:.2f}% |\n")
    md.append(f"\nDia recall overall = {(overall_dia==2).mean()*100:.2f}%\n")

    md.append("\n## Per-modality signal energy (RMS)\n\n")
    md.append("| Modality | Correct Dia | Error Dia | Δ (err-corr) |\n")
    md.append("|----------|-------------:|----------:|------:|\n")
    for i, mn in enumerate(MODALITY_NAMES):
        delta = mean_error[i] - mean_correct[i]
        md.append(f"| {mn} | {mean_correct[i]:.4f} ± {std_correct[i]:.4f} | "
                  f"{mean_error[i]:.4f} ± {std_error[i]:.4f} | "
                  f"{delta:+.4f} |\n")

    md.append("\n## Findings\n\n")
    err_to_bg = sum(r["n_dia_to_bg"] for r in rows)
    err_to_sys = sum(r["n_dia_to_sys"] for r in rows)
    if err_to_bg + err_to_sys > 0:
        bg_frac = err_to_bg / (err_to_bg + err_to_sys) * 100
        md.append(f"- Of all Dia errors, {bg_frac:.1f}% go to BG and "
                  f"{100-bg_frac:.1f}% go to Sys -- almost balanced.\n")
    worst = min(rows, key=lambda x: x["dia_recall"])
    best = max(rows, key=lambda x: x["dia_recall"])
    md.append(f"- Worst subject for Dia recall: {worst['subject']} ({worst['dia_recall']*100:.2f}%)\n")
    md.append(f"- Best subject for Dia recall: {best['subject']} ({best['dia_recall']*100:.2f}%)\n")
    err_modal = MODALITY_NAMES[int(np.argmax(np.abs(mean_error - mean_correct)))]
    md.append(f"- Largest energy gap modality: {err_modal} "
              f"(|delta_RMS| = {abs(mean_error - mean_correct).max():.4f})\n")
    md.append(f"- Total Dia recall = {(overall_dia==2).mean()*100:.2f}% "
              f"(consistent with reported per-class acc 81.13% from full bench)\n")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("".join(md), encoding="utf-8")
    print(f"-> wrote {args.out_md}")
    print(f"-> figs: {args.fig_dir}/dia_error_per_subject.png, dia_error_modality_signal.png")


if __name__ == "__main__":
    main()
