"""sim_v7_int8.py — bit-exact CPU sim of v7 stride-2 INT8 with per-channel M0.

Mirrors what the FPGA scg_mac_array_v7.v should compute.  If THIS gives
88%, then RTL has a bug.  If this gives ~23%, then the export/quantization
math is off.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_qat_v2 import SCGNetV2  # noqa


def load_int(path: Path, width_bits: int) -> np.ndarray:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line: continue
        v = int(line, 16)
        if width_bits == 16 and v >= 0x8000: v -= 0x10000
        elif width_bits == 32 and v >= 0x80000000: v -= 0x100000000
        out.append(v)
    return np.array(out)


def conv1d_stride2_int8_per_ch(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                                M0: np.ndarray, shift: np.ndarray, *, relu: bool, stride: int) -> np.ndarray:
    """x: (Cin, L_in); w: (Cout, Cin, K); same-pad; per-channel M0/shift."""
    Cin, L_in = x.shape
    Cout, _, K = w.shape
    pad = K // 2
    L_out = (L_in + stride - 1) // stride if stride > 1 else L_in
    xp = np.pad(x.astype(np.int32), ((0, 0), (pad, pad)))
    out = np.zeros((Cout, L_out), dtype=np.int32)
    for co in range(Cout):
        acc_full = np.zeros(L_in, dtype=np.int64)
        for ci in range(Cin):
            for k in range(K):
                acc_full += xp[ci, k:k + L_in] * int(w[co, ci, k])
        acc_full += int(b[co])
        scaled_full = (acc_full * int(M0[co])) >> int(shift[co])
        # Stride: take every stride-th sample
        scaled = scaled_full[::stride]
        if relu:
            scaled = np.clip(scaled, 0, 127)
        else:
            scaled = np.clip(scaled, -128, 127)
        out[co, :len(scaled)] = scaled
    return out.astype(np.int8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=REPO / "model/ckpt/best_v7_stride2.pt")
    p.add_argument("--weights", type=Path, default=REPO / "rtl/weights_v7")
    p.add_argument("--data", type=Path, default=REPO / "data/val.npz")
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    in_q_R = float(ck["state"]["in_q.running_absmax"])
    print(f"in_q_R = {in_q_R:.5f}")

    layers = []
    layer_shapes = [(1,32,5), (32,64,5), (64,128,5), (128,3,1)]
    L_strides    = [2, 2, 2, 1]
    for li, (cin, cout, k) in enumerate(layer_shapes):
        w_bytes = bytes(int(line.strip(), 16) for line in (args.weights / f"L{li}_w.hex").read_text().splitlines() if line.strip())
        w = np.frombuffer(w_bytes, dtype=np.uint8).astype(np.int8).reshape(cout, cin, k)
        b = load_int(args.weights / f"L{li}_b.mem", 32)
        m = load_int(args.weights / f"L{li}_M0.mem", 16)
        s = load_int(args.weights / f"L{li}_shift.mem", 5)
        layers.append((w, b, m, s, L_strides[li]))

    val = np.load(args.data)
    X, y = val["X"][:args.n], val["y"][:args.n]

    correct = 0
    pred_arr = np.zeros(len(X), dtype=np.int64)
    for s in range(len(X)):
        x_raw = X[s, 0].astype(np.float32) / 127.0
        x_q = np.clip(np.round(x_raw / (in_q_R / 127)), -127, 127).astype(np.int8).reshape(1, -1)
        x = x_q
        for li, (w, b, m, sh, stride) in enumerate(layers):
            relu = (li < 3)
            x = conv1d_stride2_int8_per_ch(x, w, b, m, sh, relu=relu, stride=stride)
        gap = x.astype(np.int32).sum(axis=1)
        pred_arr[s] = int(np.argmax(gap))
    correct = int((pred_arr == y).sum())
    print(f"v7 stride-2 INT8 PTQ acc on n={len(X)}: {correct}/{len(X)} = {correct/len(X)*100:.2f}%")

    # Class distribution of predictions vs truths
    print(f"  pred dist: {np.bincount(pred_arr, minlength=3).tolist()}")
    print(f"  truth dist: {np.bincount(y, minlength=3).tolist()}")


if __name__ == "__main__":
    main()
