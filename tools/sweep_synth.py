"""sweep_synth.py — synth selected Pareto vertices.

For each (H, T) ckpt, calls synth_one_config.py and aggregates results.

Usage:
    python tools/sweep_synth.py --ckpt-dir model/ckpt/sweep
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, default=Path("model/ckpt/sweep"))
    p.add_argument("--py", type=str, default="D:/anaconda3/envs/scggpu/python.exe")
    # Configs to synth (H=32 T=32 is already covered by deployed bit, skip-able)
    p.add_argument("--configs", nargs="+", default=[
        "H16_T32", "H32_T8", "H32_T16", "H32_T48", "H64_T32",
    ])
    args = p.parse_args()

    results = []
    for cfg in args.configs:
        ckpt = args.ckpt_dir / f"best_sweep_{cfg}.pt"
        if not ckpt.exists():
            print(f"[skip] {ckpt} not found")
            continue
        bit_name = f"scg_top_snn_sweep_{cfg}"
        cmd = [args.py, str(REPO / "tools/synth_one_config.py"),
               "--ckpt", str(ckpt), "--bit-name", bit_name]
        print(f"\n========================================")
        print(f"SYNTH: {cfg}")
        print(f"========================================")
        try:
            subprocess.run(cmd, check=True)
            j = REPO / "doc" / f"synth_best_sweep_{cfg}.json"
            if j.exists():
                results.append(json.loads(j.read_text()))
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {e}")
            results.append({"tag": f"best_sweep_{cfg}", "status": "failed",
                            "error": str(e)})

    # Print summary table
    print("\n=== Synth sweep summary ===")
    print(f"{'Config':<14} {'Status':<8} {'LUT':<10} {'BRAM9K':<8} {'DSP':<5}")
    for r in results:
        if r.get("status") == "ok":
            res = r["resources"]
            lut = res.get("LUT4", {})
            bram = res.get("BRAM9K", {})
            dsp = res.get("DSP18", {})
            print(f"{r['tag'][len('best_sweep_'):]:<14} ok       "
                  f"{lut.get('used','?'):<5} ({lut.get('pct',0):.1f}%) "
                  f"{bram.get('used','?'):<8} {dsp.get('used','?'):<5}")
        else:
            print(f"{r.get('tag','?')[len('best_sweep_'):]:<14} FAILED")


if __name__ == "__main__":
    main()
