"""dl_retry_failed.py — re-fetch any missing files via the user's alt proxy.

Discovers what's missing under each dataset dir vs the expected manifest, then
fetches just those files with curl through the alternate proxy.

Usage:
    python tools/dl_retry_failed.py --proxy http://10.24.79.1:5555 [cebs_mp mitdb apnea pn2016]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = "https://physionet.org/files"


def expected_cebs_mp() -> tuple[Path, list[str]]:
    base = f"{ROOT}/cebsdb/1.0.0"
    recs = [f"m{i:03d}" for i in range(1, 21)] + [f"p{i:03d}" for i in range(1, 21)]
    urls = []
    for r in recs:
        urls += [f"{base}/{r}.hea", f"{base}/{r}.dat"]
    return Path("data/cebsdb"), urls


def expected_mitdb() -> tuple[Path, list[str]]:
    base = f"{ROOT}/mitdb/1.0.0"
    nums = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114,
            115, 116, 117, 118, 119, 121, 122, 123, 124,
            200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217,
            219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
    urls = []
    for n in nums:
        urls += [f"{base}/{n}.hea", f"{base}/{n}.dat", f"{base}/{n}.atr"]
    return Path("data/mitdb"), urls


def expected_apnea() -> tuple[Path, list[str]]:
    base = f"{ROOT}/apnea-ecg/1.0.0"
    rec_lens = {"a": 20, "b": 5, "c": 10}
    urls = []
    for L, n in rec_lens.items():
        for i in range(1, n + 1):
            r = f"{L}{i:02d}"
            urls += [f"{base}/{r}.hea", f"{base}/{r}.dat", f"{base}/{r}.apn"]
    return Path("data/apnea_ecg"), urls


def expected_pn2016() -> tuple[Path, list[str]]:
    base = f"{ROOT}/challenge-2016/1.0.0"
    urls = [f"{base}/training.zip"]
    return Path("data/physionet2016"), urls


JOBS = {
    "cebs_mp": expected_cebs_mp,
    "mitdb":   expected_mitdb,
    "apnea":   expected_apnea,
    "pn2016":  expected_pn2016,
}


def curl_get(url: str, dest: Path, proxy: str, timeout: int = 600) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 1024:
        return True, "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sS", "-fL",
           "--retry", "5", "--retry-delay", "5",
           "--max-time", str(timeout),
           "--connect-timeout", "30",
           "--proxy", proxy,
           "-o", str(dest), url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if dest.exists() and dest.stat().st_size < 1024:
            dest.unlink(missing_ok=True)
        return False, r.stderr.strip()[:200]
    return True, f"ok {dest.stat().st_size}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proxy", required=True)
    p.add_argument("--workers", type=int, default=3,
                   help="be gentle — fewer workers helps low-bandwidth proxy")
    p.add_argument("targets", nargs="*", default=list(JOBS.keys()))
    args = p.parse_args()

    print(f"proxy: {args.proxy}  workers: {args.workers}")
    summary = []
    for tgt in args.targets:
        if tgt not in JOBS:
            print(f"unknown: {tgt}"); continue
        dest, urls = JOBS[tgt]()
        # Find what's missing or zero-byte
        todo = []
        for u in urls:
            fname = u.rsplit("/", 1)[-1]
            f = dest / fname
            if (not f.exists()) or f.stat().st_size < 1024:
                todo.append(u)
        print(f"\n[{tgt}] {len(todo)}/{len(urls)} files missing -> {dest}")
        if not todo:
            summary.append((tgt, 0, 0, 0))
            continue

        ok = fail = 0
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {}
            for u in todo:
                fname = u.rsplit("/", 1)[-1]
                futs[ex.submit(curl_get, u, dest / fname, args.proxy)] = u
            for i, fut in enumerate(as_completed(futs)):
                got, msg = fut.result()
                if got: ok += 1
                else:
                    fail += 1
                    failures.append((futs[fut], msg))
                if (i + 1) % 5 == 0 or (i + 1) == len(todo):
                    print(f"   [{i+1}/{len(todo)}]  ok={ok}  fail={fail}", flush=True)
        if failures:
            print(f"   first 3 failures:")
            for u, m in failures[:3]:
                print(f"     {u}: {m[:100]}")
        summary.append((tgt, ok, fail, len(todo)))

    print("\n=== Retry summary ===")
    for tgt, ok, fail, total in summary:
        print(f"  {tgt}: ok={ok}  fail={fail}  (was missing {total})")


if __name__ == "__main__":
    main()
