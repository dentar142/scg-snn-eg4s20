"""bench_fpga.py — paper-grade on-board FPGA benchmark over UART.

Strategy: weights are loaded once. Then each sample = (load 256 B window
+ run + read 1 B). A second timing pair brackets ONLY the RUN→reply round trip,
which excludes the deterministic 256-byte UART upload (~22 ms @ 115200 8N1).
The remaining ~hundreds of µs is dominated by the 1-byte UART reply (87 µs)
plus the actual on-chip inference time.

Output: doc/bench_fpga.json
"""
from __future__ import annotations
import argparse
import json
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from test_inference import load_weight_blob   # noqa: E402

CMD_RST  = b"\xA0"
CMD_LD_W = b"\xA1"
CMD_LD_X = b"\xA2"
CMD_RUN  = b"\xA3"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--weights", type=Path, default=REPO / "rtl/weights")
    p.add_argument("--data", type=Path, default=REPO / "data/val.npz")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path, default=REPO / "doc/bench_fpga.json")
    args = p.parse_args()

    blob = load_weight_blob(args.weights)
    val = np.load(args.data)
    X, y = val["X"], val["y"]
    n = min(args.n, len(X))
    print(f"weights {len(blob)} B, samples = {n}")

    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    time.sleep(0.05)

    def flood_reset():
        """Drive any in-progress S_X_DATA/S_W_DATA back to S_IDLE
        by sending enough zero bytes to overflow the largest 'need' (2008),
        then a CMD_RST.  Recovers from a single dropped UART byte."""
        ser.reset_input_buffer()
        ser.write(b"\x00" * 2200); ser.flush()
        time.sleep(0.005)
        ser.write(CMD_RST); ser.flush()
        time.sleep(0.005)
        ser.reset_input_buffer()

    def upload_weights():
        hdr = CMD_LD_W + struct.pack("<H", len(blob))
        t0 = time.perf_counter()
        ser.write(hdr + blob); ser.flush()
        return (time.perf_counter() - t0) * 1e3

    # Reset + weight load (once)
    ser.write(CMD_RST); ser.flush()
    weight_upload_ms = upload_weights()
    print(f"weight upload: {weight_upload_ms:.1f} ms")

    rt_ms = []          # full sample round-trip (load_x + run + reply)
    run_ms = []         # RUN→reply only (excludes 256-byte upload)
    upload_ms = []      # 256-byte window upload only
    preds = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    timeouts = 0

    n_recoveries = 0
    for i in range(n):
        window = X[i, 0].astype(np.int8).tobytes()

        # Flush any stale RX bytes from prior iteration
        if ser.in_waiting:
            ser.reset_input_buffer()

        # 1) upload window (CMD_LD_X + 256 B) in a single write
        t_a = time.perf_counter()
        ser.write(CMD_LD_X + window); ser.flush()
        t_b = time.perf_counter()

        # Inter-step pause: give the cable time to fully drain to the FPGA
        # before the standalone CMD_RUN goes out.  At 115200 baud, the last
        # byte of the LD_X frame is on the wire ~80 µs after flush returns;
        # 5 ms is comfortably safe.
        time.sleep(0.005)

        # 2) run + receive class byte
        ser.write(CMD_RUN); ser.flush()
        resp = ser.read(1)
        t_c = time.perf_counter()
        if not resp:
            timeouts += 1
            print(f"[{i:03d}] TIMEOUT — flooding reset (recovery #{n_recoveries + 1})",
                  flush=True)
            flood_reset()
            upload_weights()              # weights may have been corrupted
            n_recoveries += 1
            if n_recoveries > 20:
                print("too many recoveries, aborting"); break
            continue
        preds[i] = resp[0] & 0x3
        valid[i] = True

        upload_ms.append((t_b - t_a) * 1e3)
        run_ms.append((t_c - t_b) * 1e3)
        rt_ms.append((t_c - t_a) * 1e3)

        if i < 3 or i == n - 1 or i % 25 == 0:
            print(f"[{i:03d}] cls={preds[i]} (truth={int(y[i])}) "
                  f"rt={rt_ms[-1]:.2f} ms (upload={upload_ms[-1]:.2f}, "
                  f"run={run_ms[-1]:.2f})", flush=True)
    ser.close()

    n_valid = int(valid.sum())
    if n_valid == 0:
        ser.close()
        sys.exit("no valid samples — check FPGA / cabling")
    acc = float((preds[valid] == y[:n][valid]).mean() * 100)
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, pp in zip(y[:n][valid], preds[valid]):
        cm[int(t), int(pp)] += 1

    summary = {
        "n_samples_attempted": n,
        "n_samples_valid": n_valid,
        "n_timeouts": timeouts,
        "n_recoveries": n_recoveries,
        "port": args.port,
        "baud": args.baud,
        "weight_upload_ms_one_shot": weight_upload_ms,
        "round_trip_ms": {
            "mean":   statistics.mean(rt_ms),
            "median": statistics.median(rt_ms),
            "p95":    float(np.percentile(rt_ms, 95)),
            "std":    statistics.pstdev(rt_ms),
            "min":    min(rt_ms),
            "max":    max(rt_ms),
        },
        "window_upload_ms": {
            "mean":   statistics.mean(upload_ms),
            "median": statistics.median(upload_ms),
            "min":    min(upload_ms),
            "max":    max(upload_ms),
        },
        "run_ms": {  # RUN cmd byte → reply byte (~= chip compute + 1B UART tx)
            "mean":   statistics.mean(run_ms),
            "median": statistics.median(run_ms),
            "p95":    float(np.percentile(run_ms, 95)),
            "std":    statistics.pstdev(run_ms),
            "min":    min(run_ms),
            "max":    max(run_ms),
        },
        "uart_byte_cost_ms": 1000.0 * 10 / args.baud,  # 10 bits per byte (8N1)
        "accuracy_percent": acc,
        "throughput_samples_per_s_round_trip": 1000.0 / statistics.mean(rt_ms),
        "throughput_samples_per_s_run_only":   1000.0 / statistics.mean(run_ms),
        "confusion_matrix": cm.tolist(),
        "class_names": ["Background", "Systolic", "Diastolic"],
        "predictions": preds.tolist(),
        "valid_mask":  valid.tolist(),
        "truths":      y[:n].tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nFPGA on-board: acc={acc:.2f}%  "
          f"rt={summary['round_trip_ms']['mean']:.2f} ms  "
          f"run-only={summary['run_ms']['mean']:.2f} ms")
    print(f"→ wrote {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
