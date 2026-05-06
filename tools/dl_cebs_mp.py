"""dl_cebs_mp.py — download CEBSDB m+p series (music + post-music conditions).

Each of the 20 CEBS subjects has THREE recordings: b (basal sleep), m (music
exposure), p (post-music baseline).  We already have b001..b020.  Adding m
and p triples the windows-per-subject without adding new subjects.
"""
import sys
from pathlib import Path
import wfdb

out = Path("data/cebsdb")
out.mkdir(parents=True, exist_ok=True)
recs = [f"m{i:03d}" for i in range(1, 21)] + [f"p{i:03d}" for i in range(1, 21)]
print(f"downloading {len(recs)} CEBSDB records (m + p series) ...", flush=True)
try:
    wfdb.dl_database("cebsdb", str(out), records=recs)
except Exception as e:
    print(f"download failed: {e}", file=sys.stderr)
    sys.exit(1)
print("done")
