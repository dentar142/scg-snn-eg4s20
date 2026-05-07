"""synth_one_config.py — for a given (H, T) sweep ckpt, do export + channel-bank
split + Anlogic synth, then return resource numbers.

Usage:
    python tools/synth_one_config.py --ckpt model/ckpt/sweep/best_sweep_H16_T32.pt
"""
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_area_report(area_path: Path) -> dict:
    """Parse build_snn/scg_top_snn_route.area for resource numbers.

    Anlogic format groups BRAM as a parent row with "out of 64" (= BRAM9K total)
    and BRAM9K/FIFO9K as sub-rows (no "out of"). BRAM32K has its own "out of 16".
    """
    txt = area_path.read_text()
    out = {}
    for key, label in [("lut", "LUT4"), ("reg", "REG"), ("le", "LE"),
                       ("bram32k", "BRAM32K"), ("dsp", "DSP18")]:
        m = re.search(rf"#{key}\s+([\d,]+)\s+out of\s+([\d,]+)", txt)
        if m:
            used = int(m.group(1).replace(",", ""))
            total = int(m.group(2).replace(",", ""))
            out[label] = {"used": used, "total": total, "pct": 100.0 * used / total}
    # BRAM9K: parent "#bram" row (total of bram9k subtypes; fifo9k usually 0)
    m = re.search(r"#bram\s+([\d,]+)\s+out of\s+([\d,]+)", txt)
    if m:
        used = int(m.group(1).replace(",", ""))
        total = int(m.group(2).replace(",", ""))
        out["BRAM9K"] = {"used": used, "total": total, "pct": 100.0 * used / total}
    # MSlice: single line, no "out of" but printed elsewhere — extract from #le row
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--py", type=str, default="D:/anaconda3/envs/scggpu/python.exe")
    p.add_argument("--td", type=str,
                   default=r"D:\Anlogic\TD_Release_2026.1_6.2.190.657\bin\td_commands_prompt.exe")
    p.add_argument("--build-dir", type=Path, default=REPO / "build_snn")
    p.add_argument("--bit-name", type=str, default=None,
                   help="if set, copy bit to build_snn/<name>.bit after synth")
    args = p.parse_args()

    tag = args.ckpt.stem  # e.g. best_sweep_H32_T48
    print(f"==> synth for {tag}")

    # 1. export weights (writes rtl/weights_snn/W1.hex, W2.hex, meta.json + patches RTL)
    cmd = [args.py, str(REPO / "model/export_snn_weights.py"),
           "--ckpt", str(args.ckpt), "--out", str(REPO / "rtl/weights_snn"),
           "--leak-shift", "4"]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 2. read meta to get H, n_chan, win_len
    meta = json.loads((REPO / "rtl/weights_snn/meta.json").read_text())
    H = meta["H"]; n_chan = meta["n_channels"]; win_len = meta["win_len"]

    # 3. channel-bank split
    cmd = [args.py, str(REPO / "tools/split_w1_channels.py"),
           "--in", str(REPO / "rtl/weights_snn/W1.hex"),
           "--out-dir", str(REPO / "rtl/weights_snn"),
           "--n-chan", str(n_chan), "--win-len", str(win_len), "--h", str(H)]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 4. synth
    log = args.build_dir.parent / f"build_snn_{tag}.log"
    print(f"  ==> Anlogic synth (log: {log})")
    t0 = time.time()
    # TD's TCL eats backslashes; use relative path with forward slashes.
    # cwd must be repo root so 'tools/build_snn.tcl' resolves.
    tcl_arg = "tools/build_snn.tcl"
    with open(log, "w") as f:
        proc = subprocess.run([args.td, tcl_arg],
                              stdout=f, stderr=subprocess.STDOUT,
                              cwd=str(REPO))
    elapsed = time.time() - t0
    print(f"  synth elapsed = {elapsed:.0f}s, exit = {proc.returncode}")

    # Detect failure from log content (TD doesn't always non-zero exit on PHY-9009)
    log_text = log.read_text(errors="ignore") if log.exists() else ""
    if "PHY-9009" in log_text or "Build complete" not in log_text:
        # Find PHY error line for context
        err_line = next((l for l in log_text.splitlines() if "PHY-" in l and "ERROR" in l),
                        "synth failed (no Build complete marker)")
        return {"tag": tag, "status": "failed", "elapsed_sec": elapsed,
                "failure_log_line": err_line, "log": str(log)}

    area_path = args.build_dir / "scg_top_snn_route.area"
    if not area_path.exists():
        return {"tag": tag, "status": "failed", "elapsed_sec": elapsed,
                "log": str(log)}

    res = parse_area_report(area_path)
    bit_path = args.build_dir / "scg_top_snn.bit"
    saved_bit = None
    if args.bit_name and bit_path.exists():
        saved_bit = args.build_dir / f"{args.bit_name}.bit"
        shutil.copy2(bit_path, saved_bit)

    result = {"tag": tag, "status": "ok",
              "H": H, "n_chan": n_chan, "win_len": win_len,
              "elapsed_sec": elapsed, "resources": res,
              "log": str(log),
              "saved_bit": str(saved_bit) if saved_bit else None}
    print(f"  -> {json.dumps(res)}")
    out_path = args.build_dir.parent / "doc" / f"synth_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
