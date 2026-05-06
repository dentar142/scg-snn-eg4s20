"""bench_fpga_snn.py — bench the SCG-SNN engine on FPGA over UART.

W1 / W2 ROMs are baked into the bitstream, so per-sample we only need:
  CMD_LD_X (0xA2) + 256 INT8 bytes
  CMD_RUN  (0xA3)         → reply 1 byte class
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

CMD_RST  = b"\xA0"
CMD_LD_X = b"\xA2"
CMD_RUN  = b"\xA3"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--data", type=Path, default=REPO / "data_excl100/val.npz")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path, default=REPO / "doc/bench_fpga_snn.json")
    args = p.parse_args()

    val = np.load(args.data)
    X, y = val["X"], val["y"]
    n = min(args.n, len(X))
    print(f"samples = {n}  data = {args.data}")

    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(0.05)
    ser.write(CMD_RST); ser.flush()

    rt_ms = []; run_ms = []
    preds = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        window = X[i, 0].astype(np.int8).tobytes()
        if ser.in_waiting:
            ser.reset_input_buffer()
        t_a = time.perf_counter()
        ser.write(CMD_LD_X + window); ser.flush()
        t_b = time.perf_counter()
        time.sleep(0.005)
        ser.write(CMD_RUN); ser.flush()
        resp = ser.read(1)
        t_c = time.perf_counter()
        if not resp:
            print(f"[{i}] TIMEOUT")
            continue
        preds[i] = resp[0] & 0x3
        valid[i] = True
        rt_ms.append((t_c - t_a) * 1e3)
        run_ms.append((t_c - t_b) * 1e3)
        if i < 3 or i % 25 == 0 or i == n - 1:
            print(f"[{i:03d}] cls={preds[i]} (truth={int(y[i])}) "
                  f"rt={rt_ms[-1]:.2f} ms (upload={(t_b - t_a)*1e3:.2f}, "
                  f"run={run_ms[-1]:.2f})", flush=True)
    ser.close()

    n_valid = int(valid.sum())
    if n_valid == 0:
        sys.exit("No valid samples")
    acc = float((preds[valid] == y[:n][valid]).mean() * 100)
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, pp in zip(y[:n][valid], preds[valid]):
        cm[int(t), int(pp)] += 1

    summary = {
        "n_samples_attempted": n,
        "n_samples_valid": n_valid,
        "round_trip_ms": {"mean": statistics.mean(rt_ms),
                          "median": statistics.median(rt_ms),
                          "min": min(rt_ms), "max": max(rt_ms)},
        "run_ms": {"mean": statistics.mean(run_ms),
                   "median": statistics.median(run_ms)},
        "accuracy_percent": acc,
        "confusion_matrix": cm.tolist(),
        "predictions": preds.tolist(),
        "valid_mask": valid.tolist(),
        "truths": y[:n].tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nFPGA SNN: acc = {acc:.2f}%  rt={summary['round_trip_ms']['mean']:.2f} ms  "
          f"run={summary['run_ms']['mean']:.2f} ms")


if __name__ == "__main__":
    main()
