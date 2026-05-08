"""export_aligned_weights.py - export aligned-SNN ckpt as standard W1.hex
by baking integer per-channel time offsets tau_c into W1 column permutation.

Math: aligned forward y = sum_c,k W[i,c,k] * x[c, k - tau_c]
                     = sum_c,k' W[i,c,k' + tau_c] * x[c, k']  (k' = k - tau_c)
So define W'[i,c,k] = W[i,c, (k + tau_c) mod L]; this lets a baseline-arch
RTL produce identical output without any new addressing logic.

The 5 channel-bank ROMs become identical to baseline shape (H * WIN_LEN
each); only their content differs. No RTL changes needed.

Usage:
    python tools/export_aligned_weights.py --ckpt model/ckpt/best_snn_mm_h32t16_aligned.pt
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("rtl/weights_snn"))
    p.add_argument("--leak-shift", type=int, default=4)
    p.add_argument("--py", type=str, default="D:/anaconda3/envs/scggpu/python.exe")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    H = int(ck["H"]); T = int(ck["T"])
    n_chan = int(ck["n_channels"]); win_len = int(ck["n_in"])
    tau_int = list(ck.get("tau_int", []))
    if len(tau_int) != n_chan:
        sys.exit(f"ERROR: ckpt missing tau_int (got {tau_int})")
    print(f"loaded aligned ckpt: H={H} T={T} tau_int={tau_int}")

    # Apply shift to W1 in-place
    W1 = ck["state"]["fc1.weight"].numpy()  # (H, n_chan*win_len)
    W1r = W1.reshape(H, n_chan, win_len)
    W1_aligned = np.empty_like(W1r)
    for c in range(n_chan):
        # W'[i,c,k] = W[i,c, (k + tau_c) % L]
        W1_aligned[:, c, :] = np.roll(W1r[:, c, :], -tau_int[c], axis=-1)
    W1_baked = W1_aligned.reshape(H, n_chan * win_len)
    print(f"  W1 shifted by tau per channel: tau={tau_int}")

    # Replace fc1 in ckpt and dump as a temp ckpt for the existing exporter
    ck_new = dict(ck)
    ck_new["state"] = dict(ck["state"])
    ck_new["state"]["fc1.weight"] = torch.from_numpy(W1_baked).contiguous()
    # Strip the extra aligned-only fields the baseline exporter doesn't expect
    for k in ("tau_int", "tau_float", "shift_max_train"):
        ck_new.pop(k, None)
    tmp_ckpt = args.ckpt.with_name(args.ckpt.stem + "_baked.pt")
    torch.save(ck_new, tmp_ckpt)
    print(f"  wrote shifted ckpt: {tmp_ckpt}")

    # Call existing export pipeline
    cmd = [args.py, str(REPO / "model/export_snn_weights.py"),
           "--ckpt", str(tmp_ckpt), "--out", str(args.out),
           "--leak-shift", str(args.leak_shift)]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Channel-bank split
    cmd = [args.py, str(REPO / "tools/split_w1_channels.py"),
           "--in", str(args.out / "W1.hex"),
           "--out-dir", str(args.out),
           "--n-chan", str(n_chan), "--win-len", str(win_len), "--h", str(H)]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Bake aligned info into meta
    meta_path = args.out / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["tau_int_baked"] = tau_int
    meta["aligned_ckpt"] = str(args.ckpt)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  meta updated with tau_int_baked = {tau_int}")
    print("\nDone. RTL is unchanged; channel-bank ROMs are now phase-aligned.")


if __name__ == "__main__":
    main()
