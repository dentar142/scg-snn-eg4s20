"""quant_sweep.py — bit-width and granularity sweep for SCG-CNN INT quantization.

Evaluates: per-tensor symmetric vs per-channel symmetric, INT4 / INT6 / INT8.
Output: doc/quant_sweep.json + console table.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_qat import SCGNet  # noqa: E402


def sym_quant(x: np.ndarray, n_bits: int, per_channel: bool) -> np.ndarray:
    qmax = 2 ** (n_bits - 1) - 1
    if per_channel and x.ndim >= 1:
        absmax = np.maximum(np.abs(x).max(axis=tuple(range(1, x.ndim)), keepdims=True), 1e-8)
    else:
        absmax = max(np.abs(x).max(), 1e-8)
    scale = absmax / qmax
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return (q * scale).astype(x.dtype)


@torch.no_grad()
def eval_quant(model, X, y, n_bits: int, per_channel: bool) -> float:
    """Re-quantize all conv weights of the FP32 model and run inference."""
    new_state = {}
    for name, p in model.state_dict().items():
        if name.endswith(".conv.weight"):
            w = p.numpy()
            wq = sym_quant(w, n_bits, per_channel)
            new_state[name] = torch.from_numpy(wq).type_as(p)
        else:
            new_state[name] = p
    model.load_state_dict(new_state)
    Xt = torch.from_numpy(X.astype(np.float32) / 127.0)
    pred = model(Xt).argmax(1).numpy()
    return float((pred == y).mean() * 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=REPO / "model/ckpt/best.pt")
    p.add_argument("--data", type=Path, default=REPO / "data/val.npz")
    p.add_argument("--out", type=Path, default=REPO / "doc/quant_sweep.json")
    args = p.parse_args()

    val = np.load(args.data)
    X, y = val["X"][:200], val["y"][:200]
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base_model = SCGNet()
    base_model.load_state_dict(ckpt["state"])
    base_model.eval()

    # Baseline FP32
    Xt = torch.from_numpy(X.astype(np.float32) / 127.0)
    fp_acc = float((base_model(Xt).argmax(1).numpy() == y).mean() * 100)

    results = {"fp32": fp_acc, "sweep": []}
    for n_bits in (4, 6, 8):
        for per_ch in (False, True):
            # Reload fresh weights
            m = SCGNet()
            m.load_state_dict(ckpt["state"])
            m.eval()
            acc = eval_quant(m, X, y, n_bits, per_ch)
            tag = f"INT{n_bits}-{'per_channel' if per_ch else 'per_tensor'}"
            results["sweep"].append({"tag": tag, "n_bits": n_bits, "per_channel": per_ch, "accuracy_percent": acc})
            print(f"  {tag:25s} = {acc:.2f}%")

    print(f"\nBaseline FP32: {fp_acc:.2f}%")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"-> {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
