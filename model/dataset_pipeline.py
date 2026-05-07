"""
CEBS dataset pipeline for SCG cardiac feature extraction
========================================================

Downloads the CEBS database from PhysioNet (ODC-BY license),
extracts SCG signal, derives Systolic / Diastolic / Background labels
from ECG R-wave references, and writes train.npz / val.npz.

Usage:
    pip install wfdb numpy scipy tqdm
    python dataset_pipeline.py --out data/                       # 3-class (default)
    python dataset_pipeline.py --out data_5cls --n-classes 5     # 5-class

Output:
    data/train.npz: X (N, 1, 256) int16, y (N,) int64
    data/val.npz:   X (N, 1, 256) int16, y (N,) int64

Labels:
    n_classes=3 (default, original):
        0 = Background, 1 = Systolic, 2 = Diastolic
    n_classes=5 (fine-grained, R-peak-only, no SCG morphology dependency):
        0 = BG (>=150 ms from any event center)
        1 = Sys-onset    (R + 15 ms +- 15 ms)  isovolumic contraction onset
        2 = Sys-peak     (R + 50 ms +- 20 ms)  ejection peak (narrower than Sys)
        3 = Dia-onset    (R + 320 ms +- 15 ms) diastolic phase ramp-up
        4 = Dia-peak     (R + 350 ms +- 20 ms) mitral valve open peak
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import wfdb
except ImportError:
    sys.exit("pip install wfdb")

from scipy.signal import butter, filtfilt, find_peaks


# CEBS records (basal sleep subset — clean signal for v0)
CEBS_RECORDS = [f"b{i:03d}" for i in range(1, 21)]  # b001..b020

WINDOW_LEN = 256          # samples
TARGET_FS = 1000          # Hz (CEBS native is 5 kHz, we down-sample)
SYS_OFFSET_MS = 50        # 50 ms after R-wave = peak of systole
DIA_OFFSET_MS = 350       # 350 ms after R-wave = peak of diastole
LABEL_HALF_WIN_MS = 30    # ±30 ms around the labeled peak counts as that class
BG_EXCLUSION_MS = 100     # BG windows must be ≥ this far from any Sys/Dia event
                          # center (else dropped as boundary-ambiguous, à la
                          # Rahman et al. DCOSS-IoT 2026 §IV-D temporal exclusion)
SCG_CHANNEL_NAMES = ("scg", "I", "II", "III")  # CEBS ECG names; SCG is in 'PCG' actually


def fetch_record(rec: str, out_dir: Path) -> Path:
    """Download a single CEBS record into out_dir/cebsdb/<rec>.* if missing."""
    target = out_dir / "cebsdb"
    target.mkdir(parents=True, exist_ok=True)
    if not (target / f"{rec}.hea").exists():
        wfdb.dl_database("cebsdb", str(target), records=[rec])
    return target / rec


def bandpass(sig: np.ndarray, fs: float, low: float = 5.0, high: float = 50.0) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins-lite: bandpass + derivative + integrate + find_peaks."""
    bp = bandpass(ecg, fs, 5.0, 25.0)
    diff = np.diff(bp, prepend=bp[0])
    sq = diff ** 2
    # Moving average integration
    win = max(1, int(0.08 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    thr = 0.6 * np.std(integ)
    min_dist = int(0.4 * fs)  # 150 BPM upper bound
    peaks, _ = find_peaks(integ, height=thr, distance=min_dist)
    return peaks


def label_window(center_idx: int, r_peaks: np.ndarray, fs: float,
                 bg_exclusion_ms: float = BG_EXCLUSION_MS) -> int:
    """Decide label of the window centered at center_idx based on nearest R-wave.

    Returns -1 for windows in the boundary-ambiguous zone (drop).
    """
    if len(r_peaks) == 0:
        return 0
    nearest = r_peaks[np.argmin(np.abs(r_peaks - center_idx))]
    delta_ms = (center_idx - nearest) * 1000.0 / fs
    half = LABEL_HALF_WIN_MS
    if abs(delta_ms - SYS_OFFSET_MS) <= half:
        return 1  # Systolic
    if abs(delta_ms - DIA_OFFSET_MS) <= half:
        return 2  # Diastolic
    # BG candidate: must be far enough from BOTH Sys and Dia event centers
    d_to_sys = abs(delta_ms - SYS_OFFSET_MS)
    d_to_dia = abs(delta_ms - DIA_OFFSET_MS)
    if min(d_to_sys, d_to_dia) < bg_exclusion_ms:
        return -1  # boundary-ambiguous: drop
    return 0       # clean Background


# 5-class label scheme (fine-grained, R-peak-only)
EVENTS_5CLS = (
    # (label, center_ms, half_ms)
    (1, 15.0,  15.0),  # Sys-onset
    (2, 50.0,  20.0),  # Sys-peak
    (3, 320.0, 15.0),  # Dia-onset
    (4, 350.0, 20.0),  # Dia-peak
)


def label_window_5cls(center_idx: int, r_peaks: np.ndarray, fs: float,
                      bg_exclusion_ms: float = 150.0) -> int:
    """5-class label using only R-peak distance (no SCG morphology).

    Returns -1 if the window is in a boundary-ambiguous zone (drop).
    """
    if len(r_peaks) == 0:
        return 0
    nearest = r_peaks[np.argmin(np.abs(r_peaks - center_idx))]
    delta_ms = (center_idx - nearest) * 1000.0 / fs

    # In-window classes: first event whose [center-half, center+half] contains delta
    for lbl, c_ms, h_ms in EVENTS_5CLS:
        if abs(delta_ms - c_ms) <= h_ms:
            return lbl

    # BG candidate: must be far from EVERY event center (>= bg_exclusion_ms).
    min_d = min(abs(delta_ms - c_ms) for _, c_ms, _ in EVENTS_5CLS)
    if min_d < bg_exclusion_ms:
        return -1
    return 0


def downsample(sig: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    factor = int(round(fs_in / fs_out))
    return sig[::factor]


def normalize_int8(x: np.ndarray) -> np.ndarray:
    """Per-window z-score, then quantize to int8 in [-127,127]."""
    mu = x.mean()
    sd = x.std() + 1e-6
    z = (x - mu) / sd
    z = np.clip(z * 32, -127, 127)  # 32 ≈ 4-sigma fits in int8
    return z.astype(np.int8)


def build_record(rec_path: Path, bg_exclusion_ms: float = BG_EXCLUSION_MS,
                 n_classes: int = 3
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (windows, labels) arrays for one CEBS record."""
    rec = wfdb.rdrecord(str(rec_path))
    fs = rec.fs
    sig_names = [n.lower() for n in rec.sig_name]

    # CEBS channels: 'I' (ECG), 'PCG' (phonocardiogram), 'RESP' (resp)
    # Some records also have 'SCG' explicitly; PCG-Z accel falls under SCG family
    ecg_idx = next((i for i, n in enumerate(sig_names) if n in ("i", "ii", "iii", "ecg")), 0)
    scg_idx = next((i for i, n in enumerate(sig_names) if "scg" in n or "pcg" in n), 1)

    ecg = rec.p_signal[:, ecg_idx]
    scg = rec.p_signal[:, scg_idx]

    # Down-sample to 1 kHz
    if fs > TARGET_FS:
        f_factor = int(round(fs / TARGET_FS))
        ecg = ecg[::f_factor]
        scg = scg[::f_factor]
        fs = TARGET_FS

    # Bandpass SCG to suppress baseline drift / HF noise
    scg = bandpass(scg, fs, 5.0, 50.0)
    r_peaks = detect_r_peaks(ecg, fs)

    # Slide a 256-sample window with 32-sample stride
    stride = 32
    windows: list[np.ndarray] = []
    labels: list[int] = []
    for start in range(0, len(scg) - WINDOW_LEN, stride):
        center = start + WINDOW_LEN // 2
        win = scg[start:start + WINDOW_LEN]
        if n_classes == 3:
            lbl = label_window(center, r_peaks, fs, bg_exclusion_ms)
        elif n_classes == 5:
            lbl = label_window_5cls(center, r_peaks, fs, bg_exclusion_ms)
        else:
            raise ValueError(f"Unsupported n_classes={n_classes} (expected 3 or 5)")
        if lbl < 0:
            continue  # boundary-ambiguous: drop
        windows.append(normalize_int8(win))
        labels.append(lbl)

    if not windows:
        return np.zeros((0, WINDOW_LEN), dtype=np.int8), np.zeros((0,), dtype=np.int64)
    return np.stack(windows), np.asarray(labels, dtype=np.int64)


def balance(X: np.ndarray, y: np.ndarray, sid: np.ndarray | None = None,
            max_bg_ratio: float = 3.0, n_classes: int = 3):
    """Cap Background (class 0) to max_bg_ratio * max(non-BG counts).

    Generalized for K classes. If `sid` (per-window subject id) is given,
    also returns the filtered sid so that subject-disjoint cross-validation
    can use the post-balance set.
    """
    non_bg_counts = [int((y == c).sum()) for c in range(1, n_classes)]
    cap = int(max_bg_ratio * max(max(non_bg_counts) if non_bg_counts else 1, 1))
    bg_mask = (y == 0)
    bg_idx = np.where(bg_mask)[0]
    if len(bg_idx) > cap:
        keep = np.random.RandomState(42).choice(bg_idx, cap, replace=False)
        keep_mask = np.zeros_like(bg_mask)
        keep_mask[keep] = True
        keep_mask |= (~bg_mask)
        if sid is None:
            return X[keep_mask], y[keep_mask]
        return X[keep_mask], y[keep_mask], sid[keep_mask]
    if sid is None:
        return X, y
    return X, y, sid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--records", nargs="*", default=CEBS_RECORDS)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--bg-exclusion-ms", type=float, default=BG_EXCLUSION_MS,
                   help="BG windows must be at least this far (ms) from any "
                        "Sys/Dia event center; 0 disables temporal exclusion.")
    p.add_argument("--cebs-dir", type=Path, default=None,
                   help="If set, read CEBS records from here instead of "
                        "downloading; defaults to <out>/cebsdb.")
    p.add_argument("--n-classes", type=int, default=3, choices=[3, 5],
                   help="Label scheme: 3 (BG/Sys/Dia, original) or "
                        "5 (BG/Sys-onset/Sys-peak/Dia-onset/Dia-peak).")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    K = args.n_classes
    print(f"BG temporal exclusion: >= {args.bg_exclusion_ms:.0f} ms")
    print(f"Label scheme: n_classes={K}")
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_sid: list[np.ndarray] = []   # subject id (int index into args.records)

    src_dir = args.cebs_dir if args.cebs_dir is not None else args.out

    for sid, rec in enumerate(args.records):
        try:
            print(f"[{rec}] processing...")
            rec_path = fetch_record(rec, src_dir)
            X, y = build_record(rec_path, args.bg_exclusion_ms, n_classes=K)
            counts = np.bincount(y, minlength=K).tolist()
            print(f"[{rec}] {len(X)} windows | per-class={counts}")
            all_X.append(X)
            all_y.append(y)
            all_sid.append(np.full(len(X), sid, dtype=np.int32))
        except Exception as e:
            print(f"[{rec}] SKIP ({e})", file=sys.stderr)
            continue

    X = np.concatenate(all_X)[:, None, :]   # (N, 1, 256)
    y = np.concatenate(all_y)
    sid = np.concatenate(all_sid)
    X, y, sid = balance(X, y, sid, n_classes=K)
    counts = np.bincount(y, minlength=K).tolist()
    print(f"After balance: total={len(X)} | per-class={counts}")
    print(f"  per-subject window count: {np.bincount(sid).tolist()}")

    # all.npz: full balanced set with subject IDs preserved (for K-fold CV)
    record_names = np.array(args.records, dtype=object)
    np.savez(args.out / "all.npz", X=X, y=y, sid=sid, record_names=record_names)
    print(f"Wrote {args.out / 'all.npz'} (N={len(X)}, with sid + record_names)")

    # Subject-wise random split would be more robust; for v0 we do plain shuffle
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    n_val = int(args.val_fraction * len(X))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    np.savez(args.out / "train.npz",
             X=X[train_idx], y=y[train_idx], sid=sid[train_idx])
    np.savez(args.out / "val.npz",
             X=X[val_idx],   y=y[val_idx],   sid=sid[val_idx])
    print(f"Wrote {args.out / 'train.npz'} ({len(train_idx)}) and val.npz ({len(val_idx)}).")


if __name__ == "__main__":
    main()
