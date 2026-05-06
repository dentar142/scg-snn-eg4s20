"""export_weights_v2.py — bit-exact INT8 + per-channel M0/shift export for v5 model.

Outputs to rtl/weights_v5/:
  L{0..3}_w.hex      — INT8 weights, one byte per line
  L{0..3}_b.mem      — INT32 biases (8 hex chars/line, no underscores)
  L{0..3}_M0.mem     — per-output-channel M0 (16-bit signed)
  L{0..3}_shift.mem  — per-output-channel right-shift (5-bit unsigned)
  scales.json        — debug record of scales/M0/shift
  meta.json          — architecture metadata for the RTL

The math follows golden_model.py / eval_int8_v2.py exactly:
  acc_int32 = sum_{ci, k}(x_int8 * w_int8)
  biased    = acc_int32 + b_int32[co]
  scaled    = (biased * M0[co]) >>> shift[co]
  out_int8  = sat_int8(scaled)  + ReLU on hidden layers
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
from train_qat_v2 import SCGNetV2  # noqa: E402


def find_m0_shift(scale_ratio: float, max_shift: int = 28) -> tuple[int, int]:
    if scale_ratio == 0:
        return 0, 0
    sign = -1 if scale_ratio < 0 else 1
    scale_ratio = abs(scale_ratio)
    best = (0, 0, float("inf"))
    # m must fit in signed INT16: range [-32767, 32767]; we keep m positive here
    # and apply the sign separately, so cap at 2**15 - 1 = 32767.
    for s in range(0, max_shift + 1):
        m = int(round(scale_ratio * (1 << s)))
        if not (0 < m < (1 << 15)):
            continue
        approx = m / (1 << s)
        err = abs(approx - scale_ratio) / max(scale_ratio, 1e-12)
        if err < best[2]:
            best = (m * sign, s, err)
    return best[0], best[1]


@torch.no_grad()
def export(ckpt_path: Path, out_dir: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("channels", [1, 32, 64, 128, 3])
    inner = tuple(arch[1:-1])
    no_pool = ckpt.get("no_pool", False)
    model = SCGNetV2(channels=inner, no_pool=no_pool)
    model.load_state_dict(ckpt["state"])
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [model.l0, model.l1, model.l2, model.l3]
    scales = []

    in_scale = float(model.in_q.running_absmax) / 127
    print(f"Architecture: {arch}  no_pool={no_pool}")
    print(f"in_scale = {in_scale:.6f}")

    for li, layer in enumerate(layers):
        bn = layer.bn
        eps = bn.eps
        gamma = bn.weight
        beta = bn.bias
        run_mean = bn.running_mean
        run_var = bn.running_var
        bn_ratio = (gamma / torch.sqrt(run_var + eps)).numpy()    # [Cout]
        b_fp = ((layer.conv.bias - run_mean) * (gamma / torch.sqrt(run_var + eps)) + beta).numpy()

        # Quantize the ORIGINAL conv weights per-tensor (matches QAT semantics)
        w = layer.conv.weight
        w_absmax = float(w.abs().max().clamp_min(1e-8))
        w_scale = w_absmax / 127
        w_int8 = torch.clamp(torch.round(w / w_scale), -127, 127).to(torch.int8).numpy()

        out_scale = float(layer.act_q.running_absmax) / 127
        # Per-channel M0/shift, accommodating bn_ratio per channel
        ratio_M_per_co = (in_scale * w_scale * bn_ratio) / out_scale
        Cout = w_int8.shape[0]
        M0_arr = np.zeros(Cout, dtype=np.int32)
        sh_arr = np.zeros(Cout, dtype=np.int32)
        for c in range(Cout):
            m, s = find_m0_shift(float(ratio_M_per_co[c]))
            M0_arr[c] = m
            sh_arr[c] = s
        # Bias int rep: scale = in_scale * w_scale * bn_ratio[co]
        bias_int = np.round(b_fp / (in_scale * w_scale * bn_ratio)).astype(np.int64)

        # Write weight hex (Cout * Cin * K bytes, row-major [co, ci, k])
        Cout_, Cin, K = w_int8.shape
        flat_w = w_int8.reshape(-1)
        with open(out_dir / f"L{li}_w.hex", "w") as f:
            for v in flat_w:
                f.write(f"{int(v) & 0xFF:02x}\n")
        # Biases (INT32 per channel)
        with open(out_dir / f"L{li}_b.mem", "w") as f:
            for v in bias_int:
                f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")
        # M0 (INT16 signed)
        with open(out_dir / f"L{li}_M0.mem", "w") as f:
            for v in M0_arr:
                f.write(f"{int(v) & 0xFFFF:04x}\n")
        # shift (5-bit)
        with open(out_dir / f"L{li}_shift.mem", "w") as f:
            for v in sh_arr:
                f.write(f"{int(v) & 0x1F:02x}\n")

        scales.append({
            "layer": li,
            "in_scale": in_scale, "w_scale": w_scale, "out_scale": out_scale,
            "bn_ratio_min": float(bn_ratio.min()),
            "bn_ratio_max": float(bn_ratio.max()),
            "M0_min": int(M0_arr.min()), "M0_max": int(M0_arr.max()),
            "shift_min": int(sh_arr.min()), "shift_max": int(sh_arr.max()),
            "Cout": Cout_, "Cin": Cin, "K": K,
            "bias_min": int(bias_int.min()), "bias_max": int(bias_int.max()),
        })
        print(f"L{li}: Cin={Cin} Cout={Cout_} K={K}  weights={flat_w.size}B  "
              f"M0=[{M0_arr.min()},{M0_arr.max()}] shift=[{sh_arr.min()},{sh_arr.max()}]")
        in_scale = out_scale

    (out_dir / "scales.json").write_text(json.dumps(scales, indent=2))
    meta = {
        "architecture": arch,
        "no_pool": no_pool,
        "K": [5, 5, 5, 1],
        "L_in":  [256, 128, 64, 32],   # input length per layer (after prev pool)
        "L_out": [256, 128, 64, 32],   # conv-only output length (=input length, with same-pad)
        "L_pool_out": [128, 64, 32, 32],   # AFTER 2x pool (no pool on L3)
        "in_q_running_absmax": float(model.in_q.running_absmax),
        "weight_total_bytes": int(sum(s["Cin"] * s["Cout"] * s["K"] for s in scales)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n→ wrote to {out_dir.relative_to(REPO)}/  (total {meta['weight_total_bytes']}B weights)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=REPO / "model/ckpt/best_v5.pt")
    p.add_argument("--out", type=Path, default=REPO / "rtl/weights_v5")
    args = p.parse_args()
    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
