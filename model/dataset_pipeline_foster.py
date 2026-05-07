"""dataset_pipeline_foster.py — multi-modal SCG corpus from FOSTER (40 subjects).

FOSTER (Foster et al., 2024, OSF:3u6yb) provides 40 subjects × ~7 min recordings
with simultaneous 5-modality cardiac signals + ECG reference, sampled at 10 kHz:
  Time, ECG, PVDF, PZT, ACC, PCG, ERB

We:
  1. resample 10 kHz -> 1 kHz (factor 10 decimation, anti-alias band-pass)
  2. derive labels from ECG R-peaks (same scheme as dataset_pipeline.py:
     R+50ms ± 30ms = Sys, R+350ms ± 30ms = Dia, BG = ≥ 100ms from any event,
     drop ambiguous)
  3. produce (N, 5, 256) windows where the 5 channels are
     [PVDF, PZT, ACC, PCG, ERB] -- ECG is NOT in the input (only used for labels)
  4. save data_foster_multi/all.npz with X, y, sid, record_names

Usage:
    python model/dataset_pipeline_foster.py --out data_foster_multi --bg-exclusion-ms 100
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

WINDOW_LEN = 256          # samples
TARGET_FS = 1000          # Hz
SOURCE_FS = 10000         # Hz (FOSTER native)
SYS_OFFSET_MS = 50
DIA_OFFSET_MS = 350
LABEL_HALF_WIN_MS = 30
BG_EXCLUSION_MS = 100
STRIDE = 32

# Five mechanical/acoustic modality columns (NOT including Time + ECG).
# ECG is used ONLY for label derivation, never as input.
SIGNAL_COLS = ['PVDF', 'PZT', 'ACC', 'PCG', 'ERB']


def bandpass(sig: np.ndarray, fs: float, low: float = 5.0, high: float = 50.0) -> np.ndarray:
    nyq = 0.5 * fs
    hi = min(high, nyq * 0.99)
    b, a = butter(4, [low / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    bp = bandpass(ecg, fs, 5.0, 25.0)
    diff = np.diff(bp, prepend=bp[0])
    sq = diff ** 2
    win = max(1, int(0.08 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    thr = 0.6 * np.std(integ)
    min_dist = int(0.4 * fs)
    peaks, _ = find_peaks(integ, height=thr, distance=min_dist)
    return peaks


def label_window(center_idx: int, r_peaks: np.ndarray, fs: float,
                 bg_exclusion_ms: float = BG_EXCLUSION_MS) -> int:
    if len(r_peaks) == 0:
        return -1
    nearest = r_peaks[np.argmin(np.abs(r_peaks - center_idx))]
    delta_ms = (center_idx - nearest) * 1000.0 / fs
    half = LABEL_HALF_WIN_MS
    if abs(delta_ms - SYS_OFFSET_MS) <= half:
        return 1
    if abs(delta_ms - DIA_OFFSET_MS) <= half:
        return 2
    d_to_sys = abs(delta_ms - SYS_OFFSET_MS)
    d_to_dia = abs(delta_ms - DIA_OFFSET_MS)
    if min(d_to_sys, d_to_dia) < bg_exclusion_ms:
        return -1
    return 0


def normalize_int8(x: np.ndarray) -> np.ndarray:
    """Per-channel z-score then INT8 quantize. Input shape (5, L) → (5, L)."""
    out = np.empty_like(x, dtype=np.int8)
    for c in range(x.shape[0]):
        ch = x[c]
        mu = ch.mean(); sd = ch.std() + 1e-6
        z = (ch - mu) / sd
        out[c] = np.clip(z * 32, -127, 127).astype(np.int8)
    return out


def load_foster_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV, return (ecg_at_target_fs, signals_at_target_fs[5, T])."""
    # NB: FOSTER CSVs are huge (~280 MB / subject). Use np.loadtxt + skiprows for speed.
    data = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float32)
    # cols: Time, ECG, PVDF, PZT, ACC, PCG, ERB
    ecg = data[:, 1]
    sigs = data[:, 2:7]   # (T, 5)
    sigs = sigs.T          # (5, T)
    # Decimate 10 kHz -> 1 kHz
    factor = SOURCE_FS // TARGET_FS
    ecg = ecg[::factor]
    sigs = sigs[:, ::factor]
    # Per-channel band-pass (ECG and signals)
    ecg = bandpass(ecg, TARGET_FS, 5.0, 25.0)
    for c in range(5):
        sigs[c] = bandpass(sigs[c], TARGET_FS, 5.0, 50.0)
    return ecg, sigs


def build_record(rec_path: Path, bg_exclusion_ms: float = BG_EXCLUSION_MS
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return X (N, 5, 256) int8, y (N,) int64."""
    print(f"  loading {rec_path.name}...", flush=True)
    t0 = time.time()
    ecg, sigs = load_foster_csv(rec_path)
    print(f"    {len(ecg)} samples @ {TARGET_FS} Hz; loaded in {time.time()-t0:.0f}s",
          flush=True)
    r_peaks = detect_r_peaks(ecg, TARGET_FS)
    print(f"    {len(r_peaks)} R-peaks detected", flush=True)

    L = sigs.shape[1]
    windows = []
    labels = []
    for start in range(0, L - WINDOW_LEN, STRIDE):
        center = start + WINDOW_LEN // 2
        lbl = label_window(center, r_peaks, TARGET_FS, bg_exclusion_ms)
        if lbl < 0:
            continue
        win = sigs[:, start:start + WINDOW_LEN]   # (5, 256) float
        windows.append(normalize_int8(win))
        labels.append(lbl)
    if not windows:
        return np.zeros((0, 5, WINDOW_LEN), dtype=np.int8), np.zeros((0,), dtype=np.int64)
    return np.stack(windows), np.asarray(labels, dtype=np.int64)


def balance(X: np.ndarray, y: np.ndarray, sid: np.ndarray, max_bg_ratio: float = 3.0):
    n_sys = int((y == 1).sum())
    n_dia = int((y == 2).sum())
    cap = int(max_bg_ratio * max(n_sys, n_dia, 1))
    bg_idx = np.where(y == 0)[0]
    if len(bg_idx) > cap:
        keep = np.random.RandomState(42).choice(bg_idx, cap, replace=False)
        keep_mask = np.zeros_like(y, dtype=bool)
        keep_mask[keep] = True
        keep_mask |= (y != 0)
        return X[keep_mask], y[keep_mask], sid[keep_mask]
    return X, y, sid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir", type=Path, default=Path("data/foster/csv"))
    p.add_argument("--out", type=Path, default=Path("data_foster_multi"))
    p.add_argument("--bg-exclusion-ms", type=float, default=BG_EXCLUSION_MS)
    p.add_argument("--max-subjects", type=int, default=40)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    csvs = sorted(args.csv_dir.glob("sub*.csv"))[:args.max_subjects]
    print(f"FOSTER multi-modal pipeline: {len(csvs)} subjects, "
          f"BG_excl={args.bg_exclusion_ms} ms")

    all_X = []; all_y = []; all_sid = []
    rec_names = []
    for sid, csv in enumerate(csvs):
        rec_names.append(csv.stem)
        try:
            X, y = build_record(csv, args.bg_exclusion_ms)
            print(f"  [{csv.stem}] {len(X)} windows | "
                  f"bg={(y==0).sum()} sys={(y==1).sum()} dia={(y==2).sum()}",
                  flush=True)
            if len(X):
                all_X.append(X)
                all_y.append(y)
                all_sid.append(np.full(len(X), sid, dtype=np.int32))
        except Exception as e:
            print(f"  [{csv.stem}] SKIP ({e})", file=sys.stderr)

    X = np.concatenate(all_X)             # (N, 5, 256)
    y = np.concatenate(all_y)
    sid = np.concatenate(all_sid)
    X, y, sid = balance(X, y, sid)
    print(f"\nAfter balance: total={len(X)} bg={(y==0).sum()} "
          f"sys={(y==1).sum()} dia={(y==2).sum()}")

    np.savez(args.out / "all.npz",
             X=X, y=y, sid=sid,
             record_names=np.array(rec_names, dtype=object),
             modalities=np.array(SIGNAL_COLS, dtype=object))
    print(f"-> wrote {args.out}/all.npz  shape={X.shape}  modalities={SIGNAL_COLS}")


if __name__ == "__main__":
    main()
