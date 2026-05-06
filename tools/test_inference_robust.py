"""test_inference_robust.py — Round 15 — UART client with retry & framing recovery.

If the FPGA returns 0xFF (NAK / unknown class), the host re-uploads the
window once and retries.  Round-trip latency is reported per attempt.

Use this whenever the v0 UART (no flow control, no CRC) misframes due to
metastability that briefly leaks past the 2-FF synchronizer.
"""
from __future__ import annotations
import argparse, struct, time, sys
from pathlib import Path
import numpy as np
import serial

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from test_inference import load_weight_blob, load_window  # noqa: E402

CMD_RST  = b"\xA0"; CMD_LD_W = b"\xA1"; CMD_LD_X = b"\xA2"; CMD_RUN = b"\xA3"


def send_window_with_retry(ser, window, max_retry=3):
    for attempt in range(max_retry):
        ser.reset_input_buffer()
        ser.write(CMD_LD_X + window); ser.flush()
        time.sleep(0.005)
        ser.write(CMD_RUN); ser.flush()
        resp = ser.read(1)
        if resp and resp[0] != 0xFF:
            return resp[0] & 0x3, attempt
    return None, max_retry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--weights", type=Path, default=REPO / "rtl/weights")
    p.add_argument("--window", type=Path, required=True)
    p.add_argument("--n-runs", type=int, default=5)
    args = p.parse_args()

    blob = load_weight_blob(args.weights)
    window = load_window(args.window)
    print(f"weights {len(blob)} B, window {len(window)} B")

    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    time.sleep(0.05)
    ser.write(CMD_RST); ser.flush()
    ser.write(CMD_LD_W + struct.pack("<H", len(blob)) + blob); ser.flush()

    names = ["Background", "Systolic", "Diastolic"]
    n_retry = 0
    for i in range(args.n_runs):
        t0 = time.perf_counter()
        cls, retries = send_window_with_retry(ser, window)
        t1 = time.perf_counter()
        n_retry += retries
        if cls is None:
            print(f"[{i:03d}] FAIL (3 retries exhausted)")
        else:
            mark = "" if retries == 0 else f"  ({retries} retry)"
            print(f"[{i:03d}] cls={cls} ({names[cls]})  RTT={1000*(t1-t0):.2f} ms{mark}")
    ser.close()
    print(f"\nTotal retries: {n_retry} / {args.n_runs} runs")


if __name__ == "__main__":
    main()
