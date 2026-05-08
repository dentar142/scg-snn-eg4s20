"""build_wesad_corpus.py — extract a CEBSDB-shape (single-channel SCG) corpus
from WESAD chest ACC.

Per subject (S2-S17, 15 total, 700 Hz chest sensors):
  - Resample chest ECG + chest ACC z-axis from 700 Hz to 1 kHz
  - Band-pass 5-50 Hz on ACC (SCG band), 5-25 Hz on ECG (Pan-Tompkins)
  - Pan-Tompkins R-peak detection on ECG
  - Label windows BG / Sys / Dia: same scheme as FOSTER pipeline
      Sys = R + 50 ms +/- 30 ms
      Dia = R + 350 ms +/- 30 ms
      BG  = window center >= 100 ms from any Sys/Dia event
  - 256-sample 256-ms windows, per-window int8 z-score normalize

Output: data_wesad/all.npz with (X (N, 1, 256) int8, y (N,) int64, sid (N,) int)
matching CEBSDB val.npz schema.
"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

TARGET_FS = 1000
SOURCE_FS = 700
WIN_LEN = 256
HOP = 32
SYS_OFFSET_MS = 50
DIA_OFFSET_MS = 350
EVENT_HALF_MS = 30
BG_EXCLUSION_MS = 100


def bandpass(x, fs, lo, hi, order=4):
    sos_b, sos_a = butter(order, [lo, hi], btype="band", fs=fs)
    return filtfilt(sos_b, sos_a, x).astype(np.float32)


def detect_r_peaks(ecg, fs):
    e = bandpass(ecg, fs, 5.0, 25.0)
    e2 = e * e
    peaks, _ = find_peaks(e2, distance=int(0.3 * fs),
                          height=np.percentile(e2, 99))
    return peaks


def label_window(center, r_peaks, fs):
    sys_centers = r_peaks + int(SYS_OFFSET_MS * fs / 1000)
    dia_centers = r_peaks + int(DIA_OFFSET_MS * fs / 1000)
    half = int(EVENT_HALF_MS * fs / 1000)
    bg_excl = int(BG_EXCLUSION_MS * fs / 1000)
    if any(abs(center - sc) <= half for sc in sys_centers): return 1
    if any(abs(center - dc) <= half for dc in dia_centers): return 2
    too_close = any(abs(center - sc) < bg_excl for sc in sys_centers) or \
                any(abs(center - dc) < bg_excl for dc in dia_centers)
    return -1 if too_close else 0


def normalize_int8(win):
    mu = win.mean(); std = win.std() + 1e-9
    z = (win - mu) / std
    return np.clip(z * 32, -127, 127).astype(np.int8)


def build_record(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    chest = d["signal"]["chest"]
    ecg = chest["ECG"][:, 0].astype(np.float32)
    acc = chest["ACC"]                                     # (N, 3)
    # z-axis chest (perpendicular to body) = canonical SCG axis
    acc_z = acc[:, 2].astype(np.float32)
    # Resample 700 -> 1000 Hz
    g = 100   # gcd(700, 1000) = 100
    ecg = resample_poly(ecg, TARGET_FS // g, SOURCE_FS // g)
    acc_z = resample_poly(acc_z, TARGET_FS // g, SOURCE_FS // g)
    ecg = bandpass(ecg, TARGET_FS, 5.0, 25.0)
    acc_z = bandpass(acc_z, TARGET_FS, 5.0, 50.0)
    rpks = detect_r_peaks(ecg, TARGET_FS)
    if len(rpks) < 10:
        return None, None
    L = len(acc_z)
    windows, labels = [], []
    for c in range(WIN_LEN // 2, L - WIN_LEN // 2, HOP):
        lbl = label_window(c, rpks, TARGET_FS)
        if lbl < 0:    # exclusion zone
            continue
        win = acc_z[c - WIN_LEN // 2: c + WIN_LEN // 2]
        if len(win) != WIN_LEN: continue
        windows.append(normalize_int8(win))
        labels.append(lbl)
    if not windows:
        return None, None
    return np.stack(windows), np.asarray(labels, dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/wesad/WESAD"))
    p.add_argument("--out", type=Path, default=Path("data_wesad"))
    p.add_argument("--max-bg-ratio", type=float, default=3.0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    subjects = sorted(args.root.glob("S*"))
    all_X, all_y, all_sid, names = [], [], [], []
    sid_idx = 0
    for sub in subjects:
        pkl = sub / f"{sub.name}.pkl"
        if not pkl.exists(): continue
        print(f"  {sub.name} ...", flush=True)
        X, y = build_record(pkl)
        if X is None:
            print(f"    skip (insufficient R-peaks)")
            continue
        # Class balance (limit BG <= max_bg_ratio * Sys)
        n_sys = int((y == 1).sum())
        max_bg = int(args.max_bg_ratio * n_sys)
        bg_mask = (y == 0)
        if bg_mask.sum() > max_bg and max_bg > 0:
            bg_idx = np.where(bg_mask)[0]
            keep = np.random.RandomState(42).choice(bg_idx, max_bg, replace=False)
            other = np.where(~bg_mask)[0]
            sel = np.sort(np.concatenate([keep, other]))
            X = X[sel]; y = y[sel]
        print(f"    n_win={len(X)}  per-class={np.bincount(y, minlength=3).tolist()}",
              flush=True)
        all_X.append(X)
        all_y.append(y)
        all_sid.append(np.full(len(X), sid_idx, dtype=np.int32))
        names.append(sub.name)
        sid_idx += 1

    if not all_X:
        raise SystemExit("no data")
    X = np.concatenate(all_X)[:, np.newaxis, :]   # (N, 1, 256)
    y = np.concatenate(all_y)
    sid = np.concatenate(all_sid)
    np.savez_compressed(args.out / "all.npz",
                        X=X, y=y, sid=sid, record_names=np.array(names))
    print(f"\nfinal: X={X.shape} y={y.shape} sid range 0..{sid.max()}")
    print(f"  per-class total: BG={int((y==0).sum())} Sys={int((y==1).sum())} Dia={int((y==2).sum())}")
    print(f"  -> {args.out}/all.npz")


if __name__ == "__main__":
    main()
