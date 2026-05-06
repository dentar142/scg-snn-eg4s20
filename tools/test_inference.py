"""
test_inference.py - host-side UART client for the SCG-CNN FPGA accelerator.

Speaks the simple binary protocol implemented in scg_top.v:
    0xA0                     : reset internal pointers
    0xA1 + N(2B,LE) + N data : load weights
    0xA2 + 256 bytes         : load one INT8 SCG window
    0xA3                     : run inference; FPGA returns 1 byte (class 0/1/2)

Usage:
    pip install pyserial numpy
    python test_inference.py --port COM5 --weights rtl/weights --window data/test_window.npy
"""
from __future__ import annotations
import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial


CMD_RST  = b"\xA0"
CMD_LD_W = b"\xA1"
CMD_LD_X = b"\xA2"
CMD_RUN  = b"\xA3"


def load_weight_blob(weight_dir: Path) -> bytes:
    """Concatenate L0..L3 weight hex files into one binary blob.

    Layout matches the address table in scg_mac_array.v:
        L0_w (40 B), L1_w (640 B), L2_w (1280 B), L3_w (48 B)
    """
    blob = bytearray()
    for i in range(4):
        f = weight_dir / f"L{i}_w.hex"
        if not f.exists():
            sys.exit(f"missing {f}")
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                blob.append(int(line, 16))
    return bytes(blob)


def load_window(path: Path) -> bytes:
    """Read a 256-sample INT8 window from .npy or .hex."""
    if path.suffix == ".npy":
        x = np.load(path).astype(np.int8).reshape(-1)
    else:
        x = np.array([int(line.strip(), 16)
                      for line in path.read_text().splitlines() if line.strip()],
                     dtype=np.uint8).astype(np.int8)
    if x.size != 256:
        sys.exit(f"expected 256 samples, got {x.size}")
    return x.tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="e.g. COM5 or /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--weights", type=Path, default=Path("rtl/weights"))
    p.add_argument("--window",  type=Path, required=True)
    p.add_argument("--n-runs", type=int, default=1)
    args = p.parse_args()

    blob   = load_weight_blob(args.weights)
    window = load_window(args.window)
    print(f"weights : {len(blob)} bytes ({len(blob)/1024:.2f} KB)")
    print(f"window  : {len(window)} bytes")

    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(0.05)

    # 1) reset
    ser.write(CMD_RST); ser.flush()

    # 2) load weights (one big chunk)
    hdr = CMD_LD_W + struct.pack("<H", len(blob))
    ser.write(hdr + blob); ser.flush()

    # 3) load window + run, repeated for benchmarking
    for i in range(args.n_runs):
        ser.write(CMD_LD_X + window); ser.flush()
        t0 = time.perf_counter()
        ser.write(CMD_RUN); ser.flush()
        resp = ser.read(1)
        t1 = time.perf_counter()
        if not resp:
            sys.exit("timeout waiting for class byte")
        cls = resp[0] & 0x3
        names = ["Background", "Systolic", "Diastolic"]
        print(f"[{i:03d}] class = {cls} ({names[cls]})  round-trip = {(t1-t0)*1000:.2f} ms")

    ser.close()


if __name__ == "__main__":
    main()
