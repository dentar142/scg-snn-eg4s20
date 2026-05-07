"""split_w1_channels.py — split flat W1.hex into 5 channel-bank hex files.

Input layout (PyTorch fc1.weight shape = (H, N_IN), N_IN = N_CHAN * WIN_LEN):
  W1[i, j]   where j = c * WIN_LEN + k    (c = channel, k = sample)
  Flat W1.hex contains row-major bytes: i=0,j=0..N_IN-1, i=1,j=0..N_IN-1, ...

Output (5 channel banks of size H * WIN_LEN bytes each = 8 KB for H=32,WIN=256):
  W1_ch{c}.hex  contains for c-th channel:
    bank[i * WIN_LEN + k] = W1[i, c * WIN_LEN + k]

This forces Anlogic synth to put each 8 KB bank into BRAM9K (8 BRAM9K each, 40 total).
"""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", default="rtl/weights_snn/W1.hex")
p.add_argument("--out-dir", default="rtl/weights_snn")
p.add_argument("--n-chan", type=int, default=5)
p.add_argument("--win-len", type=int, default=256)
p.add_argument("--h", type=int, default=32)
args = p.parse_args()

H, NC, WL = args.h, args.n_chan, args.win_len
N_IN = NC * WL
expected = H * N_IN

with open(args.inp) as f:
    lines = [ln.strip() for ln in f if ln.strip()]
print(f"loaded {len(lines)} hex lines from {args.inp}")
assert len(lines) == expected, f"expected {expected} entries (H={H}*N_IN={N_IN}), got {len(lines)}"

# Build banks
banks = [[] for _ in range(NC)]
for i in range(H):
    for j in range(N_IN):
        c = j // WL
        k = j % WL
        banks[c].append(lines[i * N_IN + j])
        # banks[c][i * WL + k] now holds W1[i, j=c*WL+k]

out = Path(args.out_dir)
for c in range(NC):
    bank_path = out / f"W1_ch{c}.hex"
    with open(bank_path, "w") as f:
        f.write("\n".join(banks[c]) + "\n")
    print(f"  W1_ch{c}.hex: {len(banks[c])} entries ({len(banks[c])} B)")

print(f"\ntotal: {NC} banks * {H*WL} B each = {NC*H*WL} B (matches W1.hex {expected} B)")
