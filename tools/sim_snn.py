"""sim_snn.py — bit-exact CPU sim of the INT8/INT16 SNN that the FPGA will run.

Quantization scheme (chosen to be FPGA-cheap):
  * Input x:        INT8 in [-127, 127]                       (already from val.npz)
  * FC1 weights:    INT8 per-tensor symmetric, scale w1_s
  * FC2 weights:    INT8 per-tensor symmetric, scale w2_s
  * Membrane v:     INT24 fixed-point (integer scale)
  * Threshold θ:    INT24, derived from training θ_fp / (in_scale * w1_s)
  * Leak β:         multiply-free approximation: v ← v - (v >> k)
                    (k chosen so that 1 - 2^-k ≈ β_train)

LIF step (no multiplications):
  I_t  = sum_j (x_j * W1[i,j])         # INT8 × INT8 → INT16, accumulate INT24
  v_t  = (v_{t-1} - (v_{t-1} >> k)) + I_t        # leak as shift-subtract
  s_t  = (v_t >= θ)                              # 1 bit
  v_t  = v_t - s_t * θ                            # soft reset

Output spike count over T steps → argmax → predicted class.

Usage:
    python tools/sim_snn.py --ckpt model/ckpt/best_snn_v1.pt \
        --data data_excl100/val.npz --n 11601 --leak-shift 4
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch


def quantize_per_tensor_int8(w: np.ndarray) -> tuple[np.ndarray, float]:
    absmax = float(np.abs(w).max())
    if absmax < 1e-12:
        return np.zeros_like(w, dtype=np.int8), 1.0
    scale = absmax / 127.0
    qw = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return qw, scale


def lif_step_int(v: np.ndarray, I: np.ndarray, theta: int, leak_shift: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Single LIF step with integer math.

    v, I, theta are int (or int arrays).  leak: v -= v >> leak_shift
    Spike when v >= theta.  Soft reset v -= s*theta.
    """
    # Arithmetic right shift on int32 = >> in numpy on signed int
    v_leaked = v - (v >> leak_shift)
    v_new = v_leaked + I
    s = (v_new >= theta).astype(np.int8)
    v_new = v_new - s.astype(v_new.dtype) * theta
    return v_new, s


def run_int_snn(X: np.ndarray, W1q: np.ndarray, W2q: np.ndarray,
                theta1: int, theta2: int, leak_shift: int, T: int) -> np.ndarray:
    """X: (N, 256) int8.  Returns (N,) predicted class via spike count argmax."""
    N, _ = X.shape
    H = W1q.shape[0]
    C = W2q.shape[0]
    preds = np.zeros(N, dtype=np.int64)

    # Pre-compute constant input current to layer 1 once per sample
    # I1[i,n] = sum_j X[n,j] * W1q[i,j]  (FC layer)
    # Use int32 to avoid overflow (256 × 127 × 127 < 2^23)
    I1 = (X.astype(np.int32) @ W1q.T.astype(np.int32))      # (N, H)

    for n in range(N):
        v1 = np.zeros(H, dtype=np.int32)
        v2 = np.zeros(C, dtype=np.int32)
        spike_count = np.zeros(C, dtype=np.int32)
        i1_const = I1[n]

        for _ in range(T):
            v1, s1 = lif_step_int(v1, i1_const, theta1, leak_shift)
            # Layer-2 input current: spike vector @ W2q
            i2 = (s1.astype(np.int32) @ W2q.T.astype(np.int32))   # (C,)
            v2, s2 = lif_step_int(v2, i2, theta2, leak_shift)
            spike_count += s2

        preds[n] = int(np.argmax(spike_count))
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data_excl100/val.npz"))
    p.add_argument("--n", type=int, default=11601)
    p.add_argument("--leak-shift", type=int, default=4,
                   help="leak as v -= v >> leak_shift; k=4 ⇒ β≈0.9375 (≈train 0.9)")
    p.add_argument("--T", type=int, default=None,
                   help="override T; defaults to ckpt's T")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["state"]
    H = ck.get("H", 64)
    T = args.T if args.T is not None else ck.get("T", 32)
    threshold_fp = ck.get("threshold", 1.0)
    print(f"loaded ckpt arch={ck.get('arch','?')} val_acc={ck.get('val_acc',0)*100:.2f}%")
    print(f"  H={H}  T={T}  threshold(fp)={threshold_fp}  leak_shift={args.leak_shift}")

    W1 = state["fc1.weight"].numpy()
    W2 = state["fc2.weight"].numpy()
    W1q, w1_s = quantize_per_tensor_int8(W1)
    W2q, w2_s = quantize_per_tensor_int8(W2)
    print(f"  W1 shape={W1.shape}  scale={w1_s:.4e}")
    print(f"  W2 shape={W2.shape}  scale={w2_s:.4e}")

    # Threshold mapping:
    #   In FP: I = x_fp @ W1.T;  spike when v_fp >= θ_fp
    #   In INT: I_q = x_int8 @ W1q.T;  v_int = ∑ I_q over T leaky steps
    #   Per-step relation:  I_q = I_fp / (in_scale * w1_s)   where in_scale = 1/127
    #   So θ_int = θ_fp / (in_scale * w1_s) = θ_fp * 127 / w1_s
    in_scale = 1.0 / 127.0
    theta1_int = int(round(threshold_fp / (in_scale * w1_s)))
    # Layer 2: input is binary spike (0/1), W2q is INT8.  v_2 += s @ W2q
    # Per-step magnitude of |I2| ≤ H * 127 ≈ 8K.  θ_2 in INT just maps θ_fp by w2_s.
    # Heuristic: θ2 ∝ θ_fp / w2_s   (no input scaling since spikes are 0/1)
    theta2_int = max(1, int(round(threshold_fp / w2_s)))
    print(f"  theta1_int = {theta1_int}   theta2_int = {theta2_int}")

    # Load data
    val = np.load(args.data)
    X = val["X"][:args.n, 0].astype(np.int8)        # (n, 256)
    y = val["y"][:args.n]
    print(f"  data: X={X.shape} y={y.shape}")

    preds = run_int_snn(X, W1q, W2q, theta1_int, theta2_int, args.leak_shift, T)
    correct = int((preds == y).sum())
    print(f"INT8 SNN sim acc on n={len(y)}: {correct}/{len(y)} = {correct/len(y)*100:.2f}%")
    print(f"  pred dist : {np.bincount(preds, minlength=3).tolist()}")
    print(f"  truth dist: {np.bincount(y, minlength=3).tolist()}")


if __name__ == "__main__":
    main()
