"""export_snn_weights.py - INT8 export of FC SNN weights for the FPGA RTL.

Outputs to rtl/weights_snn/:
  W1.hex          - INT8 fc1 weights (H x 256), row-major [i, j], one byte/line
  W2.hex          - INT8 fc2 weights (K x H), row-major [c, i],  one byte/line
                    (K = n_classes, auto-detected from ckpt; 3 or 5)
  meta.json       - n_classes, H, T, leak_shift, theta1_int, theta2_int,
                    w1_scale, w2_scale

The .hex format matches export_weights_v2.py so the same Verilog $readmemh
machinery works for both engines.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("rtl/weights_snn"))
    p.add_argument("--leak-shift", type=int, default=4)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["state"]
    H = int(ck.get("H", 64))
    T = int(ck.get("T", 32))
    threshold_fp = float(ck.get("threshold", 1.0))
    args.out.mkdir(parents=True, exist_ok=True)

    W1 = state["fc1.weight"].numpy()                     # (H, 256)
    W2 = state["fc2.weight"].numpy()                     # (K, H)  K=n_classes

    def q(w):
        absmax = float(np.abs(w).max())
        scale = absmax / 127.0 if absmax > 1e-12 else 1.0
        qw = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        return qw, scale

    W1q, w1_s = q(W1)
    W2q, w2_s = q(W2)
    in_scale = 1.0 / 127.0
    theta1_int = int(round(threshold_fp / (in_scale * w1_s)))
    theta2_int = max(1, int(round(threshold_fp / w2_s)))

    # Hex dumps
    with open(args.out / "W1.hex", "w") as f:
        for v in W1q.reshape(-1):
            f.write(f"{int(v) & 0xFF:02x}\n")
    with open(args.out / "W2.hex", "w") as f:
        for v in W2q.reshape(-1):
            f.write(f"{int(v) & 0xFF:02x}\n")

    meta = {
        "n_in": int(W1q.shape[1]), "H": int(W1q.shape[0]),
        "n_classes": int(W2q.shape[0]),
        "T": T,
        "leak_shift": int(args.leak_shift),
        "threshold_fp": threshold_fp,
        "w1_scale": w1_s,
        "w2_scale": w2_s,
        "theta1_int": theta1_int,
        "theta2_int": theta2_int,
        "W1_bytes": int(W1q.size),
        "W2_bytes": int(W2q.size),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"-> wrote {args.out}/W1.hex ({W1q.size} B), W2.hex ({W2q.size} B), meta.json")
    print(f"   K={int(W2q.shape[0])} H={H} T={T} leak_shift={args.leak_shift} "
          f"theta1={theta1_int} theta2={theta2_int}")


if __name__ == "__main__":
    main()
