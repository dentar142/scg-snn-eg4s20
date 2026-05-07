"""analyze_subjects.py - per-subject difficulty profiling for FOSTER hold-out.

For each subject, computes from raw CSV:
  - mean heart rate (bpm) and HR std (variability)
  - R-peak detection quality (median R-peak amp / baseline std)
  - per-modality in-band SNR (5-50 Hz signal RMS / 100-450 Hz noise RMS)
  - inter-cycle alignment quality (avg cosine sim of consecutive cycles
    centered on R-peak, in PVDF channel)

Then correlates each metric with the deployed model's per-subject acc on
hold-out windows.

Output: doc/figs/subject_difficulty_*.png + doc/subject_difficulty.md
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

REPO = Path(__file__).resolve().parents[1]
TARGET_FS = 1000


def bandpass(x, fs, lo, hi, order=4):
    sos_b, sos_a = butter(order, [lo, hi], btype="band", fs=fs)
    return filtfilt(sos_b, sos_a, x)


def load_foster_csv_decimated(path):
    SOURCE_FS = 10000
    data = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float32)
    ecg = data[:, 1]; sigs = data[:, 2:7].T  # (5, T)
    factor = SOURCE_FS // TARGET_FS
    return ecg[::factor], sigs[:, ::factor]


def detect_r_peaks(ecg, fs):
    e = bandpass(ecg, fs, 5.0, 25.0)
    e2 = e**2
    height = np.percentile(e2, 99)
    peaks, _ = find_peaks(e2, distance=int(0.3 * fs), height=height)
    return peaks


def per_subject_features(csv_path, fs=TARGET_FS):
    ecg_raw, sigs_raw = load_foster_csv_decimated(csv_path)
    # ECG band-pass + R-peaks (same as production pipeline)
    ecg = bandpass(ecg_raw, fs, 5.0, 25.0)
    rpks = detect_r_peaks(ecg_raw, fs)
    if len(rpks) < 10:
        return None
    rr = np.diff(rpks) / fs * 1000   # ms
    rr_filt = rr[(rr > 300) & (rr < 1500)]   # physiological filter
    if len(rr_filt) < 5:
        return None
    hr_bpm = 60_000 / rr_filt.mean()
    hr_std = (60_000 / rr_filt).std()

    # R-peak SNR
    r_amp = ecg[rpks]
    rpk_snr = float(np.median(np.abs(r_amp))) / max(float(np.std(ecg)), 1e-6)

    # Per-modality SNR (signal 5-50 Hz / noise 100-450 Hz)
    mod_snr = []
    for c in range(5):
        sig_band = bandpass(sigs_raw[c], fs, 5.0, 50.0)
        noise_band = bandpass(sigs_raw[c], fs, 100.0, 450.0)
        sig_rms = float(np.sqrt(np.mean(sig_band**2)))
        noise_rms = float(np.sqrt(np.mean(noise_band**2)))
        mod_snr.append(sig_rms / max(noise_rms, 1e-9))

    # Inter-cycle alignment in PVDF (channel 0, primary SCG)
    pvdf = bandpass(sigs_raw[0], fs, 5.0, 50.0)
    L = 256
    cycles = []
    for r in rpks:
        if r - 50 < 0 or r - 50 + L > len(pvdf): continue
        win = pvdf[r - 50: r - 50 + L]
        cycles.append(win / (np.linalg.norm(win) + 1e-9))
    if len(cycles) >= 5:
        cycles = np.stack(cycles)
        # mean cosine sim of consecutive cycles
        cs = np.array([float(cycles[i] @ cycles[i + 1]) for i in range(len(cycles) - 1)])
        align = float(cs.mean())
    else:
        align = float("nan")

    return {
        "n_samples": int(len(ecg)),
        "n_rpeaks": int(len(rpks)),
        "hr_mean_bpm": float(hr_bpm),
        "hr_std_bpm": float(hr_std),
        "rpk_snr": rpk_snr,
        "modality_snr": {n: s for n, s in zip(["PVDF","PZT","ACC","PCG","ERB"], mod_snr)},
        "intercycle_alignment": align,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir", type=Path, default=Path("data/foster/csv"))
    p.add_argument("--bench-json", type=Path,
                   default=Path("doc/bench_fpga_snn_multimodal_holdout.json"))
    p.add_argument("--subjects", nargs="+",
                   default=["sub003","sub006","sub009","sub013","sub020","sub021","sub024","sub026"])
    p.add_argument("--fig-dir", type=Path, default=Path("doc/figs"))
    p.add_argument("--out-md", type=Path, default=Path("doc/subject_difficulty.md"))
    p.add_argument("--out-json", type=Path, default=Path("doc/subject_difficulty.json"))
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    bench = json.loads(args.bench_json.read_text())
    sub_acc = {s["subject"]: s["acc"] for s in bench["per_subject"]}

    rows = []
    for sub in args.subjects:
        path = args.csv_dir / f"{sub}.csv"
        if not path.exists():
            print(f"  [skip] missing {path}")
            continue
        print(f"  computing features for {sub}...")
        feats = per_subject_features(path)
        if feats is None:
            continue
        feats["subject"] = sub
        feats["board_acc_pct"] = sub_acc.get(sub, float("nan"))
        rows.append(feats)

    rows.sort(key=lambda r: r["board_acc_pct"])
    args.out_json.write_text(json.dumps(rows, indent=2))

    # ===== Plots =====
    accs = [r["board_acc_pct"] for r in rows]
    subs = [r["subject"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [("hr_mean_bpm", "Mean HR (bpm)"),
               ("hr_std_bpm", "HR std (bpm)"),
               ("rpk_snr", "R-peak SNR (median |amp|/std baseline)"),
               ("intercycle_alignment", "Inter-cycle cos similarity (PVDF)")]
    for ax, (key, label) in zip(axes.flat, metrics):
        vals = [r[key] for r in rows]
        ax.scatter(vals, accs, s=80, c="C0")
        for i, sub in enumerate(subs):
            ax.annotate(sub, (vals[i], accs[i]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)
        # correlation
        v = np.array(vals); a = np.array(accs)
        if np.isfinite(v).all():
            corr = np.corrcoef(v, a)[0, 1]
            ax.set_title(f"{label}  (r = {corr:+.2f})")
        else:
            ax.set_title(label)
        ax.set_ylabel("Board acc (%)"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.fig_dir / "subject_difficulty_metrics.png", dpi=150)
    plt.close()

    # Per-modality SNR plot
    plt.figure(figsize=(9, 5))
    mods = ["PVDF","PZT","ACC","PCG","ERB"]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for i, m in enumerate(mods):
        vals = [r["modality_snr"][m] for r in rows]
        plt.bar(x + i * 0.16 - 0.32, vals, 0.16, label=m)
    plt.xticks(x, subs, rotation=30)
    plt.ylabel("Modality SNR (signal_5-50Hz / noise_100-450Hz)")
    plt.title("Per-modality SNR per hold-out subject "
              "(left=hardest by board acc, right=easiest)")
    plt.legend(loc="upper right"); plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(args.fig_dir / "subject_difficulty_modality_snr.png", dpi=150)
    plt.close()

    # ===== Markdown summary =====
    md = ["# Per-subject difficulty profile (hold-out)\n\n"]
    md.append("| Subject | Board acc | HR mean | HR std | R-peak SNR | Cycle align | "
              "PVDF | PZT | ACC | PCG | ERB |\n")
    md.append("|---------|----------:|--------:|-------:|-----------:|------------:|"
              "-----:|----:|----:|----:|----:|\n")
    for r in rows:
        m = r["modality_snr"]
        md.append(f"| {r['subject']} | {r['board_acc_pct']:.2f}% | "
                  f"{r['hr_mean_bpm']:.1f} | {r['hr_std_bpm']:.1f} | "
                  f"{r['rpk_snr']:.2f} | {r['intercycle_alignment']:.3f} | "
                  f"{m['PVDF']:.2f} | {m['PZT']:.2f} | {m['ACC']:.2f} | "
                  f"{m['PCG']:.2f} | {m['ERB']:.2f} |\n")

    md.append("\n## Correlations with board accuracy\n\n")
    md.append("| Feature | Pearson r |\n|---------|----------:|\n")
    accs_arr = np.array(accs)
    for key, label in metrics + [(f"modality_snr.{m}", f"SNR ({m})") for m in mods]:
        if "modality_snr" in key:
            mod = key.split(".")[1]
            v = np.array([r["modality_snr"][mod] for r in rows])
        else:
            v = np.array([r[key] for r in rows])
        if not np.isfinite(v).all(): continue
        corr = np.corrcoef(v, accs_arr)[0, 1]
        md.append(f"| {label} | {corr:+.3f} |\n")

    md.append("\n## Findings\n\n")
    worst = rows[0]; best = rows[-1]
    md.append(f"- Hardest subject: **{worst['subject']}** ({worst['board_acc_pct']:.2f}%) "
              f"— HR {worst['hr_mean_bpm']:.0f} bpm, alignment {worst['intercycle_alignment']:.3f}\n")
    md.append(f"- Easiest subject: **{best['subject']}** ({best['board_acc_pct']:.2f}%) "
              f"— HR {best['hr_mean_bpm']:.0f} bpm, alignment {best['intercycle_alignment']:.3f}\n")
    args.out_md.write_text("".join(md), encoding="utf-8")
    print(f"\n-> wrote {args.out_md}")
    print(f"-> figs: {args.fig_dir}/subject_difficulty_*.png")


if __name__ == "__main__":
    main()
