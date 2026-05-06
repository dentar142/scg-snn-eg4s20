"""dl_curl_parallel.py — parallel curl-based dataset downloader.

The wfdb library hangs when downloading large databases through Clash proxy
(the requests session pool stalls).  curl with HTTPS_PROXY is reliable.
This script: (a) lists files in each PhysioNet DB via the index page, (b)
fetches them with curl in N concurrent workers.

Usage:
    python tools/dl_curl_parallel.py cebs_mp        # CEBSDB m + p
    python tools/dl_curl_parallel.py pn2016         # PhysioNet 2016 PCG
    python tools/dl_curl_parallel.py mitdb          # MIT-BIH Arrhythmia
    python tools/dl_curl_parallel.py apnea          # Apnea-ECG
    python tools/dl_curl_parallel.py all
"""
from __future__ import annotations
import os
import sys
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROXY = "http://127.0.0.1:7890"
ROOT = "https://physionet.org/files"


def curl_get(url: str, dest: Path, timeout: int = 120) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 0:
        return True, "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sS", "-fL", "--retry", "3", "--retry-delay", "2",
           "--max-time", str(timeout),
           "--proxy", PROXY,
           "-o", str(dest), url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, r.stderr.strip()[:200]
    return True, f"ok {dest.stat().st_size}"


def fetch_set(name: str, dest_dir: Path, urls: list[str], workers: int = 6):
    print(f"\n[{name}] {len(urls)} files -> {dest_dir}", flush=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for u in urls:
            fname = u.rsplit("/", 1)[-1]
            futs[ex.submit(curl_get, u, dest_dir / fname)] = u
        for i, fut in enumerate(as_completed(futs)):
            ok, msg = fut.result()
            url = futs[fut]
            if ok:
                if msg == "exists":
                    n_skip += 1
                else:
                    n_ok += 1
            else:
                n_fail += 1
                failures.append((url, msg))
            if (i + 1) % 50 == 0 or (i + 1) == len(urls):
                print(f"   [{i+1}/{len(urls)}]  ok={n_ok} skip={n_skip} fail={n_fail}",
                      flush=True)
    if failures:
        print(f"   first 5 failures:", flush=True)
        for u, m in failures[:5]:
            print(f"     {u}: {m}", flush=True)
    return n_ok, n_skip, n_fail


def db_cebs_mp() -> tuple[Path, list[str]]:
    base = f"{ROOT}/cebsdb/1.0.0"
    recs = [f"m{i:03d}" for i in range(1, 21)] + [f"p{i:03d}" for i in range(1, 21)]
    urls = []
    for r in recs:
        urls += [f"{base}/{r}.hea", f"{base}/{r}.dat"]
    return Path("data/cebsdb"), urls


def db_mitdb() -> tuple[Path, list[str]]:
    base = f"{ROOT}/mitdb/1.0.0"
    # MIT-BIH 48 records: 100..124, 200..234 (with gaps)
    nums = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114,
            115, 116, 117, 118, 119, 121, 122, 123, 124,
            200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217,
            219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
    urls = [f"{base}/RECORDS"]
    for n in nums:
        urls += [f"{base}/{n}.hea", f"{base}/{n}.dat", f"{base}/{n}.atr"]
    return Path("data/mitdb"), urls


def db_apnea() -> tuple[Path, list[str]]:
    base = f"{ROOT}/apnea-ecg/1.0.0"
    rec_letters = ["a", "b", "c"]
    rec_lens = {"a": 20, "b": 5, "c": 10}      # a01..a20, b01..b05, c01..c10
    urls = [f"{base}/RECORDS"]
    for L in rec_letters:
        for i in range(1, rec_lens[L] + 1):
            r = f"{L}{i:02d}"
            urls += [f"{base}/{r}.hea", f"{base}/{r}.dat"]
    # plus ER (extra records)
    return Path("data/apnea_ecg"), urls


def db_pn2016() -> tuple[Path, list[str]]:
    """PhysioNet 2016 PCG Challenge — uses zip distribution: training-a..f"""
    base = f"{ROOT}/challenge-2016/1.0.0"
    urls = [f"{base}/training.zip", f"{base}/RECORDS",
            f"{base}/training/REFERENCE.csv",
            f"{base}/validation.zip"]
    return Path("data/physionet2016"), urls


JOBS = {
    "cebs_mp":  db_cebs_mp,
    "mitdb":    db_mitdb,
    "apnea":    db_apnea,
    "pn2016":   db_pn2016,
}


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    targets = sys.argv[1:]
    if "all" in targets:
        targets = list(JOBS.keys())
    print(f"will run: {targets}")
    summary = []
    for tgt in targets:
        if tgt not in JOBS:
            print(f"unknown target: {tgt}"); continue
        dest, urls = JOBS[tgt]()
        ok, skip, fail = fetch_set(tgt, dest, urls)
        summary.append((tgt, ok, skip, fail, dest))
    print("\n=== Summary ===")
    for tgt, ok, skip, fail, dest in summary:
        print(f"  {tgt}: ok={ok}  skip={skip}  fail={fail}  -> {dest}")


if __name__ == "__main__":
    main()
