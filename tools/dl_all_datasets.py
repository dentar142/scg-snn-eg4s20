"""dl_all_datasets.py — bulk-download every relevant public cardiac signal DB.

Sets explicit proxy env (Clash on 127.0.0.1:7890) since wfdb's session does not
always inherit shell-level HTTP_PROXY.  Sequentially fetches each DB; logs to
stderr; counts files.

Targets (in priority order):
  1. CEBSDB m + p series                 — same 20 subjects, music/post conditions
  2. PhysioNet 2016 PCG Challenge        — 3,153 recordings, 763 subjects
  3. PhysioNet 2022 CirCor DigiScope     — 5,272 recordings, 942 patients
  4. MIT-BIH Arrhythmia (ECG)            — 48 records, R-peak gold standard
  5. PTB-XL ECG                          — 21,837 records, 18,885 patients
  6. Apnea-ECG                           — 70 records, 32 subjects, sleep apnea
"""
import os
import sys
import time
import traceback
from pathlib import Path

# Ensure proxy is set (wfdb session uses requests which honors these)
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTP_PROXY",  "http://127.0.0.1:7890")
os.environ.setdefault("NO_PROXY",    "localhost,127.0.0.1")

import wfdb

JOBS = [
    # (db_name, target_dir, records_or_None, description)
    ("cebsdb",          "data/cebsdb",          [f"m{i:03d}" for i in range(1,21)] + [f"p{i:03d}" for i in range(1,21)],
        "CEBSDB m + p series (40 more recordings, same 20 subjects)"),
    ("challenge-2016",  "data/physionet2016",   None,
        "PhysioNet/CinC 2016 Heart Sound (PCG) Challenge — 3,153 PCG, 763 subjects"),
    ("circor-heart-sound","data/circor",        None,
        "PhysioNet 2022 CirCor DigiScope PCG — 5,272 recordings, 942 patients"),
    ("mitdb",           "data/mitdb",           None,
        "MIT-BIH Arrhythmia Database (ECG, 48 records, gold-standard R-peaks)"),
    ("ptb-xl",          "data/ptbxl",           None,
        "PTB-XL ECG — 21,837 records, 18,885 patients"),
    ("apnea-ecg",       "data/apnea_ecg",       None,
        "Apnea-ECG — 70 records, 32 subjects, sleep apnea"),
]


def fetch(db, tgt, recs, desc):
    out = Path(tgt)
    out.mkdir(parents=True, exist_ok=True)
    n_existing = sum(1 for _ in out.iterdir())
    msg = f"[{db}] {desc}\n   target={tgt}  existing files={n_existing}  "
    if recs is not None:
        msg += f"records to fetch={len(recs)}"
    else:
        msg += "fetching FULL database"
    print(msg, flush=True)

    t0 = time.time()
    try:
        if recs is not None:
            wfdb.dl_database(db, str(out), records=recs)
        else:
            wfdb.dl_database(db, str(out))
        n_after = sum(1 for _ in out.iterdir())
        size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024**2
        print(f"   OK  files now={n_after}  size={size_mb:.1f} MB  in {time.time()-t0:.0f}s",
              flush=True)
        return True
    except Exception as e:
        print(f"   FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


def main():
    only = sys.argv[1:]                      # optional: list of DB names to limit
    summary = []
    for db, tgt, recs, desc in JOBS:
        if only and db not in only:
            continue
        ok = fetch(db, tgt, recs, desc)
        summary.append((db, ok))
    print("\n=== Download summary ===")
    for db, ok in summary:
        print(f"  {db}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
