"""dl_pn2016_split.py — split-fetch PhysioNet 2016 PCG via training-{a..f}/ subdirs.

The 600 MB training.zip kept failing through proxy. Each training-{letter}/
subdir has individual .wav + .hea files (~50-150 MB per letter) — small chunks,
each curl call gets stable retries + resume + timeout. If one letter fails,
the others still succeed.
"""
from __future__ import annotations
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

PROXY = "http://10.24.79.1:5555"
ROOT = "https://physionet.org/files/challenge-2016/1.0.0"


def list_dir(url: str) -> list[str]:
    """Use curl to fetch the index page; extract .wav + .hea filenames."""
    r = subprocess.run(["curl", "-sS", "-fL", "-m", "60", "--proxy", PROXY, url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return []
    files = []
    for line in r.stdout.split('\n'):
        # PhysioNet directory listing: <a href="X.wav">X.wav</a>
        for token in line.split('href="'):
            if token.startswith(('a', 'b', 'c', 'd', 'e', 'f')) and (
                    '.wav"' in token or '.hea"' in token):
                fname = token.split('"')[0]
                if fname and (fname.endswith('.wav') or fname.endswith('.hea')):
                    files.append(fname)
    return sorted(set(files))


def curl_get(url: str, dest: Path, timeout: int = 600) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 1024:
        return True, "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sS", "-fL", "-C", "-",
           "--retry", "5", "--retry-delay", "5",
           "--max-time", str(timeout), "--connect-timeout", "30",
           "--proxy", PROXY,
           "-o", str(dest), url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, r.stderr.strip()[:120]
    return True, f"ok {dest.stat().st_size}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/physionet2016"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--letters", default="abcdef")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Always try REFERENCE.csv files (small)
    for L in args.letters:
        ref_url = f"{ROOT}/training-{L}/REFERENCE.csv"
        ref_dest = args.out / f"training-{L}/REFERENCE.csv"
        ok, msg = curl_get(ref_url, ref_dest)
        print(f"  REFERENCE {L}: {ok} ({msg})", flush=True)

    print("\n=== Listing training-{letter}/ contents ===")
    all_jobs = []
    for L in args.letters:
        idx_url = f"{ROOT}/training-{L}/"
        files = list_dir(idx_url)
        print(f"  training-{L}: {len(files)} files via index parse")
        if not files:
            # Fall back: probe a01..a409 by guess
            print(f"    (index parse failed for {L}; fallback ranges)")
            for i in range(1, 410):
                files.append(f"{L}{i:04d}.wav")
                files.append(f"{L}{i:04d}.hea")
        for fname in files:
            url = f"{ROOT}/training-{L}/{fname}"
            dest = args.out / f"training-{L}/{fname}"
            all_jobs.append((url, dest))

    print(f"\n=== fetching {len(all_jobs)} files ===")
    n_ok = n_fail = n_skip = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(curl_get, u, d): (u, d) for u, d in all_jobs}
        for i, fut in enumerate(as_completed(futs)):
            ok, msg = fut.result()
            if ok and msg == "exists":
                n_skip += 1
            elif ok:
                n_ok += 1
            else:
                n_fail += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(all_jobs):
                print(f"   [{i+1}/{len(all_jobs)}]  ok={n_ok} skip={n_skip} fail={n_fail}",
                      flush=True)
    print(f"\n=== final: ok={n_ok} skip={n_skip} fail={n_fail} ===")


if __name__ == "__main__":
    main()
