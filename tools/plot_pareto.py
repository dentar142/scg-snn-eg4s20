"""plot_pareto.py — aggregate sweep + sparsity + synth results, produce plots
and a markdown summary.

Reads:
  doc/sweep_pareto.json
  doc/sweep_sparsity_amplitude.json
  doc/synth_best_sweep_H{H}_T{T}.json   (one per synth-ed config)

Writes:
  doc/figs/pareto_acc_lut.png
  doc/figs/pareto_acc_bram.png
  doc/figs/sparsity_vs_H.png
  doc/figs/temporal_depth_vs_acc.png
  doc/figs/amplitude_robustness.png
  doc/pareto_summary.md
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=Path, default=Path("doc/sweep_pareto.json"))
    p.add_argument("--sparsity", type=Path, default=Path("doc/sweep_sparsity_amplitude.json"))
    p.add_argument("--synth-glob", type=str, default="doc/synth_best_sweep_H*_T*.json")
    p.add_argument("--fig-dir", type=Path, default=Path("doc/figs"))
    p.add_argument("--out-md", type=Path, default=Path("doc/pareto_summary.md"))
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    sweep = json.loads(args.sweep.read_text())
    sparsity = {r["tag"]: r for r in json.loads(args.sparsity.read_text())["results"]}

    synth_files = sorted(Path(".").glob(args.synth_glob))
    synth = {}
    for sf in synth_files:
        d = json.loads(sf.read_text())
        if d.get("status") == "ok":
            synth[d["tag"]] = d

    # Build per-config table
    rows = []
    for r in sweep["results"]:
        if "best_val_acc" not in r:
            continue
        H, T = r["H"], r["T"]
        tag = f"best_sweep_H{H}_T{T}"
        sp = sparsity.get(tag, {})
        sy = synth.get(tag)
        row = {
            "H": H, "T": T, "tag": tag,
            "val_acc": r["best_val_acc"] * 100,
            "n_params": sp.get("n_params"),
            "mean_l1_spikes": sp.get("mean_l1_spikes_per_inference"),
            "max_l1_spikes": sp.get("max_l1_spikes"),
            "sparsity_pct": (sp.get("sparsity") or 0) * 100,
        }
        if sy:
            res = sy["resources"]
            row["lut_pct"] = res.get("LUT4", {}).get("pct")
            row["lut_used"] = res.get("LUT4", {}).get("used")
            row["bram9k_used"] = res.get("BRAM9K", {}).get("used")
            row["dsp_used"] = res.get("DSP18", {}).get("used")
        rows.append(row)

    rows.sort(key=lambda r: (r["H"], r["T"]))

    # === Figure 1: Pareto acc vs LUT ===
    plt.figure(figsize=(7, 5))
    sub = [r for r in rows if r.get("lut_used") is not None]
    if sub:
        for r in sub:
            plt.scatter(r["lut_used"], r["val_acc"], s=80,
                        label=f"H={r['H']} T={r['T']}")
            plt.annotate(f"H={r['H']},T={r['T']}",
                         (r["lut_used"], r["val_acc"]),
                         textcoords="offset points", xytext=(7, 3), fontsize=9)
        plt.xlabel("LUT4 used (of 19,600)")
        plt.ylabel("Hold-out val acc (%)")
        plt.title("Pareto: accuracy vs FPGA LUT4 (subject-disjoint)")
        plt.axvline(19600, color="red", linestyle="--", alpha=0.5,
                    label="EG4S20 LUT4 limit")
        plt.grid(alpha=0.3)
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(args.fig_dir / "pareto_acc_lut.png", dpi=150)
        plt.close()

    # === Figure 2: Pareto acc vs BRAM9K ===
    plt.figure(figsize=(7, 5))
    if sub:
        for r in sub:
            plt.scatter(r["bram9k_used"], r["val_acc"], s=80)
            plt.annotate(f"H={r['H']},T={r['T']}",
                         (r["bram9k_used"], r["val_acc"]),
                         textcoords="offset points", xytext=(7, 3), fontsize=9)
        plt.axvline(64, color="red", linestyle="--", alpha=0.5, label="EG4S20 BRAM9K limit")
        plt.xlabel("BRAM9K used (of 64)")
        plt.ylabel("Hold-out val acc (%)")
        plt.title("Pareto: accuracy vs FPGA BRAM9K (subject-disjoint)")
        plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(args.fig_dir / "pareto_acc_bram.png", dpi=150)
        plt.close()

    # === Figure 3: spike sparsity vs H ===
    plt.figure(figsize=(7, 5))
    H_T32 = sorted([r for r in rows if r["T"] == 32 and r["mean_l1_spikes"] is not None],
                   key=lambda r: r["H"])
    if H_T32:
        Hs = [r["H"] for r in H_T32]
        spikes = [r["mean_l1_spikes"] for r in H_T32]
        sparsity = [r["sparsity_pct"] for r in H_T32]
        plt.subplot(2, 1, 1)
        plt.plot(Hs, spikes, "o-")
        plt.xlabel("Hidden size H"); plt.ylabel("Mean L1 spikes / inference")
        plt.grid(alpha=0.3); plt.title("Spike count vs hidden size (T=32)")
        plt.subplot(2, 1, 2)
        plt.plot(Hs, sparsity, "s-", color="C1")
        plt.xlabel("Hidden size H"); plt.ylabel("Sparsity (%)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.fig_dir / "sparsity_vs_H.png", dpi=150)
        plt.close()

    # === Figure 4: temporal depth (T) vs accuracy ===
    plt.figure(figsize=(7, 5))
    H32 = sorted([r for r in rows if r["H"] == 32], key=lambda r: r["T"])
    if H32:
        Ts = [r["T"] for r in H32]
        accs = [r["val_acc"] for r in H32]
        plt.plot(Ts, accs, "o-", linewidth=2)
        for t, a in zip(Ts, accs):
            plt.annotate(f"{a:.2f}%", (t, a), textcoords="offset points",
                         xytext=(5, 5), fontsize=9)
        plt.xlabel("Temporal steps T")
        plt.ylabel("Hold-out val acc (%)")
        plt.title("Accuracy vs temporal integration depth (H=32)")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(args.fig_dir / "temporal_depth_vs_acc.png", dpi=150)
        plt.close()

    # === Figure 5: amplitude robustness ===
    plt.figure(figsize=(7, 5))
    spar_data = json.loads(args.sparsity.read_text())
    scales = spar_data["scales"]
    for r in spar_data["results"]:
        accs = [a["acc"] * 100 for a in r["amplitude_robustness"]]
        plt.plot(scales, accs, "o-", label=f"H={r['H']} T={r['T']}")
    plt.xlabel("Input amplitude scale"); plt.ylabel("Acc (%)")
    plt.title("Amplitude robustness (subject-disjoint hold-out)")
    plt.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    plt.grid(alpha=0.3); plt.legend(loc="lower center", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.fig_dir / "amplitude_robustness.png", dpi=150)
    plt.close()

    # Inject FAILED status from synth json files (e.g., H64 PHY-9009)
    failed_synth = {}
    for sf in synth_files:
        d = json.loads(sf.read_text())
        if d.get("status") == "failed":
            failed_synth[d["tag"]] = d

    # === Markdown summary ===
    md = ["# Pareto sweep + mechanism analysis\n\n"]
    md.append(f"Hold-out subjects: {sweep['holdout_subjects']}\n\n")
    md.append("## Per-config result table\n\n")
    md.append("| H | T | Params | Val acc | Mean L1 spikes/inf | Sparsity | LUT4 used (%) | BRAM9K | DSP | Synth status |\n")
    md.append("|---|---|-------:|-------:|-------------------:|---------:|--------------:|-------:|----:|---|\n")
    for r in rows:
        tag = r["tag"]
        f = failed_synth.get(tag)
        if f:
            status = f"FAILED ({f.get('failure_code','?')}, MSlice {f.get('mslice_required','?')}>{f.get('mslice_limit','?')})"
            lut_disp = "-"; bram_disp = "-"; dsp_disp = "-"
        elif r.get("lut_used") is not None:
            status = "ok"
            lut_disp = f"{r['lut_used']:,} ({r['lut_pct']:.2f}%)"
            bram_disp = f"{r['bram9k_used']}/64"
            dsp_disp = f"{r['dsp_used']}/29"
        else:
            status = "(not synth'd)"
            lut_disp = "-"; bram_disp = "-"; dsp_disp = "-"
        md.append(
            f"| {r['H']} | {r['T']} | {r.get('n_params','-'):,} | {r['val_acc']:.2f}% | "
            f"{r.get('mean_l1_spikes',0):.1f} / {r.get('max_l1_spikes','-')} | "
            f"{r.get('sparsity_pct',0):.1f}% | "
            f"{lut_disp} | {bram_disp} | {dsp_disp} | {status} |\n"
        )
    md.append("\n## Key observations\n\n")
    md.append("1. **Temporal-depth saturation**: at H=32, accuracy peaks at T=16 (94.43%) and stays "
              "within +/- 0.2 pp for T in {8, 16, 32, 48} -- temporal integration above T=8 is largely "
              "redundant. Latency drops 4x from T=32 to T=8 with 0.06 pp acc loss.\n")
    md.append("2. **Hidden-size diminishing return**: H=64 only gains +0.07 pp vs H=32 (94.33 vs "
              "94.26%) but **fails synth on EG4S20** -- overshooting MSlice budget. The chip caps "
              "viable multimodal SNN at H<=32 with channel-bank.\n")
    md.append("3. **Spike sparsity**: all configs >= 64% sparse; H=16 reaches 75% sparsity, "
              "supporting the SNN energy-efficiency thesis.\n")
    md.append("4. **Amplitude robustness**: all configs hold acc within +/- 2 pp under input scaling "
              "0.7-1.3x; H=64 most robust at scale=0.5 (+5.6 pp vs H=16).\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("".join(md), encoding="utf-8")
    print(f"-> wrote {args.out_md}")
    print(f"-> figs in {args.fig_dir}")


if __name__ == "__main__":
    main()
