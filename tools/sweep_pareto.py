"""sweep_pareto.py — train multiple (H, T) configurations of the multimodal
subject-disjoint SNN to map the energy/accuracy Pareto frontier and quantify
temporal integration contribution.

Trains each config with the SAME hold-out subjects (fold-0 of cv_snn_foster),
saves ckpts under model/ckpt/sweep/, and emits a JSON summary at
doc/sweep_pareto.json.

Configs: 2D matrix
  - H sweep at T=32: H ∈ {16, 32, 64}
  - T sweep at H=32: T ∈ {8, 16, 32, 48}
  (H=32, T=32 trained once and shared between both sweeps)

Usage:
    python tools/sweep_pareto.py --data data_foster_multi --out doc/sweep_pareto.json
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOLDOUT = ["sub003", "sub006", "sub009", "sub013", "sub020", "sub021", "sub024", "sub026"]


def run_one(data: Path, H: int, T: int, epochs: int, out_dir: Path, py: str) -> dict:
    tag = f"sweep_H{H}_T{T}"
    ckpt = out_dir / f"best_{tag}.pt"
    manifest = out_dir / f"best_{tag}_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        print(f"[skip] {tag} already trained: val_acc={m['best_val_acc']*100:.2f}%")
        return {"H": H, "T": T, "tag": tag, "best_val_acc": m["best_val_acc"],
                "best_epoch": m["best_epoch"], "skipped": True,
                "n_train_windows": m["n_train_windows"], "n_val_windows": m["n_val_windows"]}
    cmd = [py, str(REPO / "model/train_snn_mm_holdout.py"),
           "--data", str(data), "--out", str(out_dir),
           "--holdout", *HOLDOUT,
           "--epochs", str(epochs), "--bs", "256", "--lr", "2e-3",
           "--H", str(H), "--T", str(T), "--tag", tag, "--seed", "42"]
    print(f"\n=== training {tag} ===\n  $ {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    m = json.loads(manifest.read_text())
    return {"H": H, "T": T, "tag": tag, "best_val_acc": m["best_val_acc"],
            "best_epoch": m["best_epoch"], "elapsed_sec": elapsed, "skipped": False,
            "n_train_windows": m["n_train_windows"], "n_val_windows": m["n_val_windows"],
            "confusion_matrix": m["confusion_matrix"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data_foster_multi"))
    p.add_argument("--out", type=Path, default=Path("doc/sweep_pareto.json"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("model/ckpt/sweep"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--py", type=str, default="D:/anaconda3/envs/scggpu/python.exe")
    args = p.parse_args()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    random.seed(0)

    # H sweep at T=32
    h_configs = [(16, 32), (32, 32), (64, 32)]
    # T sweep at H=32
    t_configs = [(32, 8), (32, 16), (32, 48)]   # (32, 32) shared with H sweep

    # Existing H=32 T=32 ckpt: also include — we copy from holdout ckpt
    existing_manifest = REPO / "model/ckpt/best_snn_mm_h32_holdout_manifest.json"
    if existing_manifest.exists():
        target = args.ckpt_dir / "best_sweep_H32_T32_manifest.json"
        if not target.exists():
            target.write_text(existing_manifest.read_text())
        target_pt = args.ckpt_dir / "best_sweep_H32_T32.pt"
        existing_pt = REPO / "model/ckpt/best_snn_mm_h32_holdout.pt"
        if existing_pt.exists() and not target_pt.exists():
            target_pt.write_bytes(existing_pt.read_bytes())
        print(f"[reuse] H=32 T=32 ckpt linked from existing holdout deployment")

    all_configs = h_configs + t_configs
    print(f"sweep matrix: {len(all_configs)} configs")
    for H, T in all_configs:
        print(f"  H={H:3d} T={T:3d}")

    results = []
    for H, T in all_configs:
        try:
            r = run_one(args.data, H, T, args.epochs, args.ckpt_dir, args.py)
            results.append(r)
            print(f"  -> val_acc {r['best_val_acc']*100:.2f}%")
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {e}")
            results.append({"H": H, "T": T, "error": str(e)})

    summary = {
        "data": str(args.data),
        "holdout_subjects": HOLDOUT,
        "n_configs": len(all_configs),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n-> wrote {args.out}")
    print(f"\n=== Pareto sweep results ===")
    print(f"{'H':>4} {'T':>4} {'val_acc':>10} {'epoch':>6}")
    for r in sorted(results, key=lambda x: (x.get("H", 0), x.get("T", 0))):
        if "best_val_acc" in r:
            print(f"{r['H']:>4} {r['T']:>4} {r['best_val_acc']*100:>9.2f}% {r['best_epoch']:>6}")


if __name__ == "__main__":
    main()
