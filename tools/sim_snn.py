"""sim_snn.py - bit-exact CPU sim of the INT8 SNN with optional abstention.

Quantization scheme (chosen to be FPGA-cheap):
  * Input x:        INT8 in [-127, 127]                       (already from val.npz)
  * FC1 weights:    INT8 per-tensor symmetric, scale w1_s
  * FC2 weights:    INT8 per-tensor symmetric, scale w2_s
  * Membrane v:     INT24 fixed-point (integer scale)
  * Threshold theta: INT24, derived from training theta_fp / (in_scale * w1_s)
  * Leak:           multiply-free approximation: v -= v >> k

LIF step (no multiplications):
  I_t  = sum_j (x_j * W1[i,j])         # INT8 x INT8 -> INT16, accumulate INT24
  v_t  = (v_{t-1} - (v_{t-1} >> k)) + I_t        # leak as shift-subtract
  s_t  = (v_t >= theta)                          # 1 bit
  v_t  = v_t - s_t * theta                       # soft reset

Output spike count over T steps -> argmax -> predicted class.

ABSTENTION (NEW):
  After argmax, compute margin = max(spike_count) - second_max(spike_count).
  If margin < tau (--abstain-tau, default 0 = disabled), output class = 3 = UNK.
  This is the FPGA-cheap selective-prediction mechanism documented in
  doc/calibration_report.md.

Usage (no abstention, identical to old behavior):
    python tools/sim_snn.py --ckpt model/ckpt/best_snn_v1.pt \
        --data data_excl100/val.npz --n 11601 --leak-shift 4

Usage (with abstention, recommended tau=3 from calibration_report):
    python tools/sim_snn.py --ckpt model/ckpt/best_holdout_snn.pt \
        --data data_excl100/holdout.npz --n 9660 --leak-shift 4 \
        --abstain-tau 3
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch


def quantize_per_tensor_int8(w):
    absmax = float(np.abs(w).max())
    if absmax < 1e-12:
        return np.zeros_like(w, dtype=np.int8), 1.0
    scale = absmax / 127.0
    qw = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return qw, scale


def lif_step_int(v, I, theta, leak_shift):
    v_leaked = v - (v >> leak_shift)
    v_new = v_leaked + I
    s = (v_new >= theta).astype(np.int8)
    v_new = v_new - s.astype(v_new.dtype) * theta
    return v_new, s


def run_int_snn(X, W1q, W2q, theta1, theta2, leak_shift, T, abstain_tau=0):
    """X: (N, N_IN) int8 (already flattened channel-major if multi-modal).
    Returns (preds, spike_counts).

    If abstain_tau > 0, sets pred=3 (UNK) when (max_sc - second_sc) < abstain_tau.
    Otherwise pred is argmax(spike_count) ∈ {0,1,2}.
    """
    N, _ = X.shape
    H = W1q.shape[0]
    C = W2q.shape[0]
    preds = np.zeros(N, dtype=np.int64)
    spike_counts = np.zeros((N, C), dtype=np.int32)

    I1 = (X.astype(np.int32) @ W1q.T.astype(np.int32))      # (N, H)

    for n in range(N):
        v1 = np.zeros(H, dtype=np.int32)
        v2 = np.zeros(C, dtype=np.int32)
        sc = np.zeros(C, dtype=np.int32)
        i1_const = I1[n]

        for _ in range(T):
            v1, s1 = lif_step_int(v1, i1_const, theta1, leak_shift)
            i2 = (s1.astype(np.int32) @ W2q.T.astype(np.int32))
            v2, s2 = lif_step_int(v2, i2, theta2, leak_shift)
            sc += s2

        spike_counts[n] = sc
        argmax_class = int(np.argmax(sc))
        if abstain_tau > 0:
            sorted_sc = np.sort(sc)
            margin = int(sorted_sc[-1] - sorted_sc[-2])
            if margin < abstain_tau:
                preds[n] = 3   # UNK / abstention class
            else:
                preds[n] = argmax_class
        else:
            preds[n] = argmax_class

    return preds, spike_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data_excl100/val.npz"))
    p.add_argument("--n", type=int, default=11601)
    p.add_argument("--leak-shift", type=int, default=4,
                   help="leak as v -= v >> leak_shift; k=4 => beta~0.9375")
    p.add_argument("--T", type=int, default=None,
                   help="override T; defaults to ckpt's T")
    p.add_argument("--abstain-tau", type=int, default=0,
                   help="if >0, output class 3 (UNK) when (top1-top2) spike count < tau. "
                        "Recommended: 3 (see doc/calibration_report.md). "
                        "0 = disable abstention (original 3-class behavior).")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["state"]
    H = ck.get("H", 64)
    T = args.T if args.T is not None else ck.get("T", 32)
    threshold_fp = ck.get("threshold", 1.0)
    print(f"loaded ckpt arch={ck.get('arch','?')} val_acc={ck.get('val_acc',0)*100:.2f}%")
    print(f"  H={H}  T={T}  threshold(fp)={threshold_fp}  leak_shift={args.leak_shift}")
    if args.abstain_tau > 0:
        print(f"  ABSTENTION ENABLED: tau={args.abstain_tau}, output class 3=UNK")

    W1 = state["fc1.weight"].numpy()
    W2 = state["fc2.weight"].numpy()
    W1q, w1_s = quantize_per_tensor_int8(W1)
    W2q, w2_s = quantize_per_tensor_int8(W2)
    print(f"  W1 shape={W1.shape}  scale={w1_s:.4e}")
    print(f"  W2 shape={W2.shape}  scale={w2_s:.4e}")

    in_scale = 1.0 / 127.0
    theta1_int = int(round(threshold_fp / (in_scale * w1_s)))
    theta2_int = max(1, int(round(threshold_fp / w2_s)))
    print(f"  theta1_int = {theta1_int}   theta2_int = {theta2_int}")

    val = np.load(args.data, allow_pickle=True)
    X_raw = val["X"][:args.n].astype(np.int8)
    y = val["y"][:args.n]
    # X may be (N, C, L). Flatten channel-major to (N, C*L) so the simulator
    # consumes the same byte order the FPGA receives over UART.
    if X_raw.ndim == 3:
        N, C, L = X_raw.shape
        X = X_raw.reshape(N, C * L)
        print(f"  data: X={X_raw.shape} -> flattened ({C}x{L}={C*L}) -> {X.shape} y={y.shape}")
    else:
        X = X_raw
        print(f"  data: X={X.shape} y={y.shape}")

    preds, spike_counts = run_int_snn(
        X, W1q, W2q, theta1_int, theta2_int,
        args.leak_shift, T, abstain_tau=args.abstain_tau,
    )

    if args.abstain_tau > 0:
        # 4-class report: 0/1/2 = predicted classes, 3 = UNK
        n_unk = int((preds == 3).sum())
        accept = preds < 3
        n_accept = int(accept.sum())
        n_correct_on_accept = int(((preds == y) & accept).sum())
        sel_acc = n_correct_on_accept / max(n_accept, 1)
        cov = n_accept / max(len(y), 1)
        n_correct_total = int((preds == y).sum())
        nonsel_acc = n_correct_total / max(len(y), 1)
        print()
        print(f"=== With abstention (tau={args.abstain_tau}) ===")
        print(f"  Coverage:           {n_accept}/{len(y)} = {cov*100:.2f}%  ({n_unk} abstained = UNK)")
        print(f"  Selective accuracy: {n_correct_on_accept}/{n_accept} = {sel_acc*100:.2f}%  (on accepted only)")
        print(f"  Non-selective acc:  {n_correct_total}/{len(y)} = {nonsel_acc*100:.2f}%  (UNK counted as wrong)")
        # Per-subject if sid available
        if "sid" in val.files:
            sid = val["sid"][:args.n].astype(np.int64)
            print()
            print("  Per-subject breakdown:")
            for s in sorted(set(sid.tolist())):
                m = sid == s
                m_accept = m & accept
                cov_s = m_accept.sum() / max(m.sum(), 1)
                if m_accept.sum() > 0:
                    sel_s = ((preds[m_accept] == y[m_accept]).sum() / m_accept.sum())
                else:
                    sel_s = 0.0
                print(f"    sid={s}: n={m.sum()}  cov={cov_s*100:.2f}%  sel_acc={sel_s*100:.2f}%")
        print(f"  pred dist : {np.bincount(preds, minlength=4).tolist()}  (last bin = UNK)")
        print(f"  truth dist: {np.bincount(y, minlength=3).tolist()}")
    else:
        correct = int((preds == y).sum())
        print(f"INT8 SNN sim acc on n={len(y)}: {correct}/{len(y)} = {correct/len(y)*100:.2f}%")
        print(f"  pred dist : {np.bincount(preds, minlength=3).tolist()}")
        print(f"  truth dist: {np.bincount(y, minlength=3).tolist()}")


if __name__ == "__main__":
    main()
