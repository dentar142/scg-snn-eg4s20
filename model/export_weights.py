"""
Export trained INT8 weights to Verilog $readmemh files.

For each Conv1d layer we emit:
  rtl/weights/L{idx}_w.hex   : weights in row-major (Cout, Cin, K)
  rtl/weights/L{idx}_b.mem   : INT16 biases
  rtl/weights/L{idx}_s.mem   : per-layer right-shift count (one byte)

Also writes:
  rtl/weights/scales.txt     : floating-point scales for inspection
  rtl/weights/test_window.hex: a single INT8 input window for sanity test
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_qat import SCGNet


def int8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), -127, 127).astype(np.int8)


def to_hex_byte(v: int) -> str:
    return f"{(v & 0xFF):02x}"


def write_int8(path: Path, arr: np.ndarray) -> None:
    """Row-major INT8 array -> hex file, one byte per line."""
    flat = arr.reshape(-1).astype(np.int8)
    path.write_text("\n".join(to_hex_byte(int(v)) for v in flat) + "\n")


def write_int16(path: Path, arr: np.ndarray) -> None:
    flat = arr.reshape(-1).astype(np.int16)
    lines = [f"{(int(v) & 0xFFFF):04x}" for v in flat]
    path.write_text("\n".join(lines) + "\n")


def quant_layer(layer, in_scale: float) -> dict:
    """Return INT8 weights, INT16 biases, and right-shift count for a Conv1d."""
    w = layer.conv.weight.detach().cpu().numpy()
    b = layer.conv.bias.detach().cpu().numpy()

    # Fold BN into weights/bias if the layer has been trained to convergence
    bn = layer.bn
    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.cpu().numpy()
    var = bn.running_var.cpu().numpy()
    std_inv = 1.0 / np.sqrt(var + bn.eps)
    scale = (gamma * std_inv).reshape(-1, 1, 1)
    folded_w = w * scale
    folded_b = beta + (b - mean) * gamma * std_inv

    # Per-tensor symmetric weight quantization (matches train_qat.QConv1d)
    w_absmax = float(np.abs(folded_w).max() + 1e-8)
    w_scale = w_absmax / 127.0
    w_q = int8(folded_w / w_scale)

    # Activation scale comes from the layer's FakeQuant
    out_scale = float(layer.act_q.running_absmax.item()) / 127.0

    # Bias is held in INT32 logically; we store the rescaled INT16 form
    # b_int = round(folded_b / (in_scale * w_scale))
    bias_int = np.round(folded_b / (in_scale * w_scale)).astype(np.int32)
    bias_clip = np.clip(bias_int, -32768, 32767).astype(np.int16)

    # Re-quant shift: out_int = clip( (acc + bias) * M >> shift, -127, 127 )
    # where M ≈ (in_scale * w_scale) / out_scale  in fixed-point
    M = (in_scale * w_scale) / out_scale
    # Encode M as M0 * 2^-shift with M0 in [0.5, 1.0)
    if M <= 0:
        M0, shift = 0.5, 1
    else:
        shift = max(0, int(np.ceil(-np.log2(M))))
        M0 = M * (2 ** shift)
    M0_int = int(np.clip(np.round(M0 * 256), 0, 65535))   # Q0.16 with 8-bit headroom
    return {
        "w": w_q,
        "b": bias_clip,
        "shift": shift + 8,        # extra +8 because M0_int is Q1.8
        "M0": M0_int,
        "in_scale": in_scale,
        "out_scale": out_scale,
        "w_scale": w_scale,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("rtl/weights"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = SCGNet()
    model.load_state_dict(ckpt["state"])
    model.eval()

    # Input scale comes from the training-time FakeQuant on the input
    in_scale = float(model.in_q.running_absmax.item()) / 127.0
    layers = [model.l0, model.l1, model.l2, model.l3]
    info: list[dict] = []
    cur_scale = in_scale
    for i, layer in enumerate(layers):
        d = quant_layer(layer, cur_scale)
        info.append({"idx": i,
                     "in_scale": d["in_scale"],
                     "w_scale": d["w_scale"],
                     "out_scale": d["out_scale"],
                     "shift": d["shift"],
                     "M0": d["M0"]})
        write_int8(args.out / f"L{i}_w.hex", d["w"])
        write_int16(args.out / f"L{i}_b.mem", d["b"])
        (args.out / f"L{i}_s.mem").write_text(f"{d['shift']:02x} {d['M0']:04x}\n")
        cur_scale = d["out_scale"]

    (args.out / "scales.json").write_text(json.dumps(info, indent=2))

    # Export one validation window as INT8 for sanity-check on FPGA
    val = np.load(args.data / "val.npz")
    x0 = val["X"][0, 0].astype(np.int8)   # (256,)
    write_int8(args.out / "test_window.hex", x0)
    (args.out / "test_label.txt").write_text(str(int(val["y"][0])))

    print(f"Exported {len(layers)} layers to {args.out}")
    for d in info:
        print(f"  L{d['idx']}: shift={d['shift']:>3}  M0={d['M0']:>5}  out_scale={d['out_scale']:.5f}")


if __name__ == "__main__":
    main()
