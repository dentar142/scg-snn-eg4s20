"""make_holdout_npz.py — extract hold-out subjects' windows into a separate npz.

Output: data_excl100/holdout.npz with X (N, 1, 256), y (N,), sid (N,).
Used for FPGA bench against the truly-unseen test set.
"""
import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, default=Path("data_excl100/all.npz"))
p.add_argument("--out", type=Path, default=Path("data_excl100/holdout.npz"))
p.add_argument("--records", type=str, default="b002,b007,b015,b020")
args = p.parse_args()

d = np.load(args.data, allow_pickle=True)
X = d["X"]; y = d["y"]; sid = d["sid"]; rec_names = list(d["record_names"])
holdout = [r.strip() for r in args.records.split(",")]
holdout_sids = [rec_names.index(r) for r in holdout]
mask = np.isin(sid, holdout_sids)
print(f"holdout records: {holdout}")
print(f"matched sids: {[rec_names[s] for s in sorted(set(sid[mask].tolist()))]}")
print(f"windows: {int(mask.sum())} / {len(X)}")
np.savez(args.out, X=X[mask], y=y[mask], sid=sid[mask])
print(f"wrote {args.out}")
