"""export_snn_weights.py - INT8 export of FC SNN weights for the FPGA RTL.

Outputs to rtl/weights_snn/:
  W1.hex          - INT8 fc1 weights (H x N_IN), row-major [i, j], one byte/line
                    N_IN auto-detected from W1 shape; 256 for single-modal,
                    1280 (= 5 channels x 256 samples) for FOSTER multi-modal.
  W2.hex          - INT8 fc2 weights (K x H), row-major [c, i],  one byte/line
                    (K = n_classes, auto-detected from ckpt; 3 or 5)
  meta.json       - n_in, n_classes, H, T, leak_shift, theta1_int, theta2_int,
                    w1_scale, w2_scale

When --patch-rtl is given (default ON), the THETA1/THETA2 parameter defaults
inside rtl/scg_top_snn.v are rewritten in-place so the next synthesis run
bakes the correct thresholds into the bitstream without any extra TCL plumbing.

The .hex format matches export_weights_v2.py so the same Verilog $readmemh
machinery works for both engines.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def patch_rtl_thetas(rtl_path: Path, theta1_int: int, theta2_int: int,
                     n_classes: int, n_chan: int, win_len: int,
                     H: int = None, T: int = None) -> None:
    """In-place rewrite the parameter defaults in scg_top_snn.v so the next
    synth run bakes the correct thresholds + topology.  Idempotent: re-running
    with new values just overwrites the previous numbers.
    """
    text = rtl_path.read_text(encoding="utf-8")
    out = text
    out = re.sub(r"(parameter signed \[23:0\] THETA1 = 24'sd)\d+",
                 lambda m: f"{m.group(1)}{theta1_int}", out)
    out = re.sub(r"(parameter signed \[23:0\] THETA2 = 24'sd)\d+",
                 lambda m: f"{m.group(1)}{theta2_int}", out)
    out = re.sub(r"(parameter integer N_CLASSES\s*=\s*)\d+",
                 lambda m: f"{m.group(1)}{n_classes}", out)
    out = re.sub(r"(parameter integer N_CHAN\s*=\s*)\d+",
                 lambda m: f"{m.group(1)}{n_chan}", out)
    out = re.sub(r"(parameter integer WIN_LEN\s*=\s*)\d+",
                 lambda m: f"{m.group(1)}{win_len}", out)
    if H is not None:
        out = re.sub(r"(parameter integer H\s*=\s*)\d+",
                     lambda m: f"{m.group(1)}{H}", out)
    if T is not None:
        out = re.sub(r"(parameter integer T\s*=\s*)\d+",
                     lambda m: f"{m.group(1)}{T}", out)
    if out != text:
        rtl_path.write_text(out, encoding="utf-8")
        extras = ""
        if H is not None: extras += f" H={H}"
        if T is not None: extras += f" T={T}"
        print(f"-> patched {rtl_path} with THETA1={theta1_int} THETA2={theta2_int} "
              f"N_CLASSES={n_classes} N_CHAN={n_chan} WIN_LEN={win_len}{extras}")
    else:
        print(f"   (no patch needed; rtl already at correct values)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("rtl/weights_snn"))
    p.add_argument("--leak-shift", type=int, default=4)
    p.add_argument("--rtl-top", type=Path, default=REPO / "rtl/scg_top_snn.v",
                   help="path to scg_top_snn.v to patch with thresholds + topology")
    p.add_argument("--no-patch-rtl", action="store_true",
                   help="skip rewriting THETA1/THETA2/N_CLASSES/N_CHAN/WIN_LEN in scg_top_snn.v")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["state"]
    H = int(ck.get("H", 64))
    T = int(ck.get("T", 32))
    threshold_fp = float(ck.get("threshold", 1.0))
    n_chan = int(ck.get("n_channels") or 1)        # may be None for single-modal ckpts
    win_len = int(ck.get("n_in", 256))
    args.out.mkdir(parents=True, exist_ok=True)

    W1 = state["fc1.weight"].numpy()                     # (H, n_chan*win_len)
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
        "n_channels": n_chan, "win_len": win_len,
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
          f"theta1={theta1_int} theta2={theta2_int}  n_chan={n_chan} win_len={win_len}")
    # Sanity: W1 columns must equal n_chan * win_len (else shape mismatch in RTL)
    assert int(W1q.shape[1]) == n_chan * win_len, \
        f"W1 cols {W1q.shape[1]} != n_chan {n_chan} * win_len {win_len}"

    if not args.no_patch_rtl:
        patch_rtl_thetas(args.rtl_top, theta1_int, theta2_int,
                         n_classes=int(W2q.shape[0]),
                         n_chan=n_chan, win_len=win_len,
                         H=int(W1q.shape[0]), T=T)


if __name__ == "__main__":
    main()
