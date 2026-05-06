"""
golden_model.py - Bit-exact Python re-implementation of what scg_mac_array.v
will do at run-time, fed by the same INT8 weights export_weights.py drops.

Used to:
  1. Validate the export pipeline (read .hex / .mem / scales.json)
  2. Compare INT8-only inference accuracy vs the FP PyTorch model
  3. Generate expected per-layer INT8 activations for RTL verification

Math model (matches scg_mac_array.v):
    acc        = sum_{ci,k} x_int8[ci, x+k] * w_int8[co, ci, k]   (INT32)
    biased     = acc + bias_int16[co]                               (INT32)
    rescaled   = (biased * M0) >> shift                             (signed)
    activation = clamp(rescaled, 0, 127) for hidden layers (ReLU+sat)
                 clamp(rescaled, -128, 127) for L3 (logits, no ReLU)
After L3 a global average over time and an argmax give the class.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_qat import SCGNet


def read_int8_hex(path: Path) -> np.ndarray:
    """Read a 1-byte-per-line hex file into a flat int8 array."""
    bytes_ = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        v = int(line, 16) & 0xFF
        bytes_.append(v if v < 128 else v - 256)
    return np.array(bytes_, dtype=np.int8)


def read_int16_mem(path: Path) -> np.ndarray:
    vals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        v = int(line, 16) & 0xFFFF
        vals.append(v if v < 32768 else v - 65536)
    return np.array(vals, dtype=np.int16)


def conv1d_int8(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                M0: int, shift: int, *, relu: bool) -> np.ndarray:
    """1D conv with kernel_size = w.shape[2], stride 1, same-padding.

    x : (Cin, L) int8
    w : (Cout, Cin, K) int8
    b : (Cout,) int16  (already rescaled by export_weights.py)
    """
    Cin, L = x.shape
    Cout, _, K = w.shape
    pad = K // 2
    xp = np.pad(x.astype(np.int32), ((0, 0), (pad, pad)))
    out = np.zeros((Cout, L), dtype=np.int32)

    # int8 * int8 -> int32 accumulation
    for co in range(Cout):
        acc = np.zeros(L, dtype=np.int32)
        for ci in range(Cin):
            for k in range(K):
                acc += xp[ci, k:k + L] * int(w[co, ci, k])
        acc += int(b[co])
        # multiply by M0 (16-bit) then arithmetic right shift
        scaled = (acc.astype(np.int64) * M0) >> shift
        if relu:
            scaled = np.clip(scaled, 0, 127)
        else:
            scaled = np.clip(scaled, -128, 127)
        out[co] = scaled
    return out.astype(np.int8)


def maxpool1d_2(x: np.ndarray) -> np.ndarray:
    Cin, L = x.shape
    L2 = L // 2
    return np.maximum(x[:, 0:2 * L2:2], x[:, 1:2 * L2:2])


def forward_int8(x256: np.ndarray, weights_dir: Path) -> int:
    """Bit-exact INT8 forward pass. Returns predicted class (0/1/2)."""
    # Layer params (Cin, Cout, K) and per-layer M0 / shift loaded from scales.json
    info = json.loads((weights_dir / "scales.json").read_text())
    layer_shapes = [(1, 8, 5), (8, 16, 5), (16, 16, 5), (16, 3, 1)]
    weights, biases = [], []
    for i, (cin, cout, k) in enumerate(layer_shapes):
        w = read_int8_hex(weights_dir / f"L{i}_w.hex").reshape(cout, cin, k)
        b = read_int16_mem(weights_dir / f"L{i}_b.mem")
        weights.append(w); biases.append(b)

    x = x256.reshape(1, 256).astype(np.int8)        # (1, 256)
    # L0 + pool
    x = conv1d_int8(x, weights[0], biases[0],
                    info[0]["M0"], info[0]["shift"], relu=True)
    x = maxpool1d_2(x)                                # (8, 128)
    # L1 + pool
    x = conv1d_int8(x, weights[1], biases[1],
                    info[1]["M0"], info[1]["shift"], relu=True)
    x = maxpool1d_2(x)                                # (16, 64)
    # L2 + pool
    x = conv1d_int8(x, weights[2], biases[2],
                    info[2]["M0"], info[2]["shift"], relu=True)
    x = maxpool1d_2(x)                                # (16, 32)
    # L3 (1x1 conv, no ReLU)
    logits = conv1d_int8(x, weights[3], biases[3],
                         info[3]["M0"], info[3]["shift"], relu=False)
    # GAP over time -> 3 logits
    gap = logits.sum(axis=1)
    return int(np.argmax(gap))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("model/ckpt/best.pt"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--weights", type=Path, default=Path("rtl/weights"))
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()

    val = np.load(args.data / "val.npz")
    X, y = val["X"], val["y"]
    n = min(args.n, len(X))
    print(f"Evaluating bit-exact INT8 model on {n} val samples...")

    # Reference: PyTorch QAT model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = SCGNet(); model.load_state_dict(ckpt["state"]); model.eval()
    with torch.no_grad():
        Xf = torch.from_numpy(X[:n].astype(np.float32) / 127.0)
        py_pred = model(Xf).argmax(1).numpy()

    int_pred = np.zeros(n, dtype=np.int64)
    for i in range(n):
        int_pred[i] = forward_int8(X[i, 0], args.weights)

    py_acc  = (py_pred  == y[:n]).mean() * 100
    int_acc = (int_pred == y[:n]).mean() * 100
    agree   = (py_pred == int_pred).mean() * 100

    print(f"PyTorch QAT (FP)   acc = {py_acc:5.2f}%")
    print(f"INT8 golden model  acc = {int_acc:5.2f}%")
    print(f"Agreement          = {agree:5.2f}%   (FP vs INT8 same prediction)")
    print(f"Δ accuracy         = {int_acc - py_acc:+.2f}%   (INT8 vs FP)")

    # First 10 sample-by-sample dump for spot-check
    print("\nfirst 10 samples:  truth | pytorch | golden_INT8")
    for i in range(10):
        print(f"  {i:02d}: {y[i]} | {py_pred[i]} | {int_pred[i]}")


if __name__ == "__main__":
    main()
