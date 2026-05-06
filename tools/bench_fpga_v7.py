"""bench_fpga_v7.py — bench v7 (1->32->64->128->3 stride-2) on FPGA over UART.

The v7 RTL has bias/M0/shift hardcoded; host only needs to upload the 52 KB
weight blob once via CMD_LD_W. Then per-sample: CMD_LD_X + 256 B + CMD_RUN.
"""
from __future__ import annotations
import argparse, json, statistics, struct, sys, time
from pathlib import Path
import numpy as np
import serial

REPO = Path(__file__).resolve().parents[1]

CMD_RST  = b"\xA0"; CMD_LD_W = b"\xA1"; CMD_LD_X = b"\xA2"; CMD_RUN = b"\xA3"


def load_v7_weight_blob(weight_dir: Path) -> bytes:
    """Concatenate L0..L3 weight hex files in the order expected by RTL."""
    blob = bytearray()
    for i in range(4):
        f = weight_dir / f"L{i}_w.hex"
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                blob.append(int(line, 16))
    return bytes(blob)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--weights", type=Path, default=REPO / "rtl/weights_v7")
    p.add_argument("--data", type=Path, default=REPO / "data/val.npz")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path, default=REPO / "doc/bench_fpga_v7.json")
    args = p.parse_args()

    blob = load_v7_weight_blob(args.weights)
    val = np.load(args.data)
    X, y = val["X"], val["y"]
    n = min(args.n, len(X))
    print(f"weights {len(blob)} B, samples = {n}")

    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(0.05)
    ser.write(CMD_RST); ser.flush()

    hdr = CMD_LD_W + struct.pack("<H", len(blob))
    t0 = time.perf_counter()
    ser.write(hdr + blob); ser.flush()
    t1 = time.perf_counter()
    print(f"weight upload: {(t1 - t0)*1000:.0f} ms ({len(blob)} bytes)")

    rt_ms = []; run_ms = []; preds = np.zeros(n, dtype=np.int64)
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
            print(f"[{i}] TIMEOUT"); continue
        preds[i] = resp[0] & 0x3
        valid[i] = True
        rt_ms.append((t_c - t_a) * 1e3)
        run_ms.append((t_c - t_b) * 1e3)
        if i < 3 or i % 25 == 0 or i == n - 1:
            print(f"[{i:03d}] cls={preds[i]} (truth={int(y[i])}) "
                  f"rt={rt_ms[-1]:.2f} ms (upload={(t_b-t_a)*1e3:.2f}, run={run_ms[-1]:.2f})", flush=True)
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
        "weight_upload_ms": (t1 - t0) * 1e3,
        "round_trip_ms": {"mean": statistics.mean(rt_ms), "median": statistics.median(rt_ms)},
        "run_ms": {"mean": statistics.mean(run_ms), "median": statistics.median(run_ms)},
        "accuracy_percent": acc,
        "confusion_matrix": cm.tolist(),
        "predictions": preds.tolist(),
        "valid_mask": valid.tolist(),
        "truths": y[:n].tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nFPGA v7: acc = {acc:.2f}%  rt={summary['round_trip_ms']['mean']:.2f} ms  run={summary['run_ms']['mean']:.2f} ms")


if __name__ == "__main__":
    main()
