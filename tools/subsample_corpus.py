"""subsample_corpus.py — cap each subject's windows for balanced SSL.

The mixed corpus is dominated by MIT-BIH (2.4M of 2.78M windows from 47
subjects ≈ 51K windows/subject), while CEBS subjects only have ~10K each.
Subsample so every subject contributes equally to the SSL gradient.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-subject", type=int, default=3000,
                   help="cap windows per subject")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = np.load(args.inp, allow_pickle=True)
    X = d["X"]; sid = d["sid"]; dsid = d["dataset_id"]
    rng = np.random.RandomState(args.seed)
    print(f"loaded {len(X)} windows, {len(set(sid.tolist()))} subjects")

    keep_mask = np.zeros(len(X), dtype=bool)
    for s in sorted(set(sid.tolist())):
        idx = np.where(sid == s)[0]
        if len(idx) <= args.per_subject:
            keep_mask[idx] = True
        else:
            chosen = rng.choice(idx, args.per_subject, replace=False)
            keep_mask[chosen] = True

    X2, sid2, dsid2 = X[keep_mask], sid[keep_mask], dsid[keep_mask]
    print(f"after cap @ {args.per_subject}/subject:")
    print(f"  N = {len(X2)}")
    print(f"  per dataset: cebs={int((dsid2==0).sum())}, "
          f"mit={int((dsid2==1).sum())}, apnea={int((dsid2==2).sum())}")
    print(f"  subjects = {len(set(sid2.tolist()))}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X2, sid=sid2, dataset_id=dsid2)
    print(f"-> wrote {args.out}")


if __name__ == "__main__":
    main()
