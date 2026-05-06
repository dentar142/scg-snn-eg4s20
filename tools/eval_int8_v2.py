"""eval_int8_v2.py — fully-INT8 post-training evaluation of v2/v3/v4/v5 ckpts.

Uses the QAT-tracked FakeQuant ranges to lock in symmetric per-tensor INT8.
Folds BatchNorm into Conv weights/biases and runs a strict bit-exact INT8
forward pass (no FP32 anywhere except input scaling).

Usage: python tools/eval_int8_v2.py --ckpt model/ckpt/best_v5.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_qat_v2 import SCGNetV2  # noqa: E402


def find_m0_shift(scale_ratio: float, max_shift: int = 24) -> tuple[int, int]:
    best = (0, 0, float("inf"))
    for s in range(0, max_shift + 1):
        m = int(round(scale_ratio * (1 << s)))
        if not (0 < m < (1 << 16)):
            continue
        approx = m / (1 << s)
        err = abs(approx - scale_ratio) / max(scale_ratio, 1e-12)
        if err < best[2]:
            best = (m, s, err)
    return best[0], best[1]


@torch.no_grad()
def fold_bn_quant(model: SCGNetV2):
    """Yield per-layer (w_int8, b_int32_per_co, M0_per_co[], shift_per_co[], in_scale, out_scale, relu).

    Uses PER-CHANNEL output requantization to recover the precision lost when
    BN's per-channel scale is folded into a per-tensor-quantized weight.  This
    matches what QAT's FakeQuant + BN sequence produces during eval.
    """
    layers = [model.l0, model.l1, model.l2, model.l3]
    in_scale = float(model.in_q.running_absmax) / 127
    for li, layer in enumerate(layers):
        bn = layer.bn
        eps = bn.eps
        gamma = bn.weight; beta = bn.bias
        run_mean = bn.running_mean; run_var = bn.running_var
        # bn ratio = gamma / sqrt(var+eps); per-channel
        bn_ratio = gamma / torch.sqrt(run_var + eps)         # [Cout]
        # bn folded bias
        b_fp_per_co = (layer.conv.bias - run_mean) * bn_ratio + beta   # [Cout]

        # Quantize the ORIGINAL conv weight per-tensor (matches QAT's wq exactly)
        w = layer.conv.weight
        w_absmax = w.abs().max().clamp_min(1e-8)
        w_scale = float(w_absmax) / 127
        w_int8 = torch.clamp(torch.round(w / w_scale), -127, 127).to(torch.int8)

        out_scale = float(layer.act_q.running_absmax) / 127

        # Per-channel requantization: ratio_M[co] = (in_scale * w_scale * bn_ratio[co]) / out_scale
        # Verilog hardware: each output channel has its own M0/shift pair,
        # selected from a small ROM at runtime. Doable on FPGA.
        ratio_M_per_co = (in_scale * w_scale * bn_ratio.numpy()) / out_scale
        Cout = w_int8.shape[0]
        M0_arr = np.zeros(Cout, dtype=np.int64)
        shift_arr = np.zeros(Cout, dtype=np.int64)
        for c in range(Cout):
            M0_arr[c], shift_arr[c] = find_m0_shift(abs(float(ratio_M_per_co[c])), max_shift=28)
            if ratio_M_per_co[c] < 0:
                M0_arr[c] = -M0_arr[c]

        # Bias is added at the SAME scale as acc.  Per channel:
        #   acc_int * (in_scale * w_scale) + bias_fp_per_co   should equal
        #   the FP conv+BN output (BEFORE the per-channel bn ratio applied).
        # But since we factored bn_ratio into ratio_M, the bias must be
        # converted to the SAME pre-bn_ratio scale: bias_int = round(b_fp / (in_scale*w_scale*bn_ratio))
        bias_int32 = torch.round(
            torch.from_numpy((b_fp_per_co.numpy() / (in_scale * w_scale * bn_ratio.numpy()))).double()
        ).long()

        relu = (li < 3)
        yield w_int8, bias_int32, M0_arr, shift_arr, in_scale, out_scale, relu, w_scale
        in_scale = out_scale


def conv1d_int8(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                M0: np.ndarray, shift: np.ndarray, *, relu: bool) -> np.ndarray:
    """Per-channel requantization version. M0[co] and shift[co] per output ch."""
    Cin, L = x.shape
    Cout, _, K = w.shape
    pad = K // 2
    xp = np.pad(x.astype(np.int32), ((0, 0), (pad, pad)))
    out = np.zeros((Cout, L), dtype=np.int32)
    for co in range(Cout):
        acc = np.zeros(L, dtype=np.int32)
        for ci in range(Cin):
            for k in range(K):
                acc += xp[ci, k:k + L] * int(w[co, ci, k])
        acc += int(b[co])
        m0 = int(M0[co]); sh = int(shift[co])
        scaled = (acc.astype(np.int64) * m0) >> sh
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, default=REPO / "data/val.npz")
    p.add_argument("--n", type=int, default=2000, help="how many val samples")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt.get("channels", [1, 16, 32, 32, 3])
    inner = tuple(arch[1:-1])
    model = SCGNetV2(channels=inner)
    model.load_state_dict(ckpt["state"])
    model.eval()

    layers = list(fold_bn_quant(model))
    print(f"channels={arch}  layers={len(layers)}")
    for i, (_, _, M0_arr, shift_arr, _, _, relu, ws) in enumerate(layers):
        print(f"  L{i}: M0_range=[{int(M0_arr.min())},{int(M0_arr.max())}] "
              f"shift_range=[{int(shift_arr.min())},{int(shift_arr.max())}] "
              f"relu={relu} w_scale={ws:.5f}")

    val = np.load(args.data)
    X, y = val["X"][:args.n], val["y"][:args.n]
    in_scale = float(model.in_q.running_absmax) / 127

    # Apply in_q to all val inputs: x_q_int = round(x_int8 / R)
    in_q_R = float(model.in_q.running_absmax)
    print(f"  in_q.running_absmax = {in_q_R:.5f}  =>  rescale int8 by 1/R")

    correct = 0
    pred_arr = np.zeros(len(X), dtype=np.int64)
    for s in range(len(X)):
        x_raw = X[s, 0].astype(np.float32) / 127.0
        x_q = np.clip(np.round(x_raw / (in_q_R / 127)), -127, 127).astype(np.int8)
        x = x_q.reshape(1, -1)
        for li, (w_int8, bias_int32, M0_arr, shift_arr, ins, outs, relu, _) in enumerate(layers):
            x = conv1d_int8(x, w_int8.numpy(), bias_int32.numpy(), M0_arr, shift_arr, relu=relu)
            if li < 3:
                x = maxpool1d_2(x)
        # GAP: sum over time
        gap = x.astype(np.int32).sum(axis=1)
        pred_arr[s] = int(np.argmax(gap))
    correct = int((pred_arr == y).sum())
    print(f"INT8 PTQ acc on n={len(X)}: {correct}/{len(X)} = {correct / len(X) * 100:.2f}%")


if __name__ == "__main__":
    main()
