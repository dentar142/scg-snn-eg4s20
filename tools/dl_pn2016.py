"""dl_pn2016.py — download PhysioNet/CinC 2016 Heart Sound (PCG) Challenge.

3,153 PCG recordings from 7 different research databases (Aalborg, MIT, etc.),
763 unique subjects, 30 sec average length.  Used here for CROSS-DOMAIN
generalization testing of the SCG SNN trained on CEBSDB.
"""
import sys
from pathlib import Path
import wfdb

out = Path("data/physionet2016")
out.mkdir(parents=True, exist_ok=True)
print("downloading PhysioNet 2016 PCG Challenge full set (large, may take 10-30 min) ...", flush=True)
try:
    wfdb.dl_database("challenge-2016", str(out))
except Exception as e:
    print(f"download failed: {e}", file=sys.stderr)
    sys.exit(1)
print("done")
