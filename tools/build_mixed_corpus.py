"""build_mixed_corpus.py — combine CEBSDB + MIT-BIH + Apnea-ECG into one
unlabeled SSL pretraining corpus.

Pipeline (per record):
  1. wfdb.rdsamp → pick the cardiac channel (SCG/PCG for CEBS; ECG for MIT/Apnea)
  2. resample to 1000 Hz (simple ::factor decimation; CEBS 5kHz, MIT 360Hz, Apnea 100Hz)
  3. bandpass 5-50 Hz (same as dataset_pipeline.py)
  4. slide 256-sample window with stride=32
  5. per-window z-score → clip to [-127, 127] int8

Output: data_mixed/all_unlabeled.npz   X (N, 1, 256) int8, dataset_id, subject_id
        record_names: list of (dataset_id, subject_id, record_name)
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Iterable
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample_poly

WINDOW_LEN = 256
TARGET_FS = 1000
STRIDE = 32


def bandpass(sig: np.ndarray, fs: float, low: float = 5.0, high: float = 50.0) -> np.ndarray:
    nyq = 0.5 * fs
    hi = min(high, nyq * 0.99)
    b, a = butter(4, [low / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def resample_to_target(sig: np.ndarray, fs_in: float, fs_out: float = TARGET_FS) -> np.ndarray:
    if abs(fs_in - fs_out) < 1:
        return sig
    if fs_in > fs_out:
        factor = int(round(fs_in / fs_out))
        return sig[::factor]
    # upsample with polyphase
    return resample_poly(sig, int(fs_out), int(fs_in))


def normalize_int8(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sd = x.std() + 1e-6
    z = (x - mu) / sd
    z = np.clip(z * 32, -127, 127)
    return z.astype(np.int8)


def windowize(sig: np.ndarray) -> np.ndarray:
    n = len(sig)
    if n < WINDOW_LEN:
        return np.zeros((0, WINDOW_LEN), dtype=np.int8)
    starts = np.arange(0, n - WINDOW_LEN + 1, STRIDE)
    out = np.zeros((len(starts), WINDOW_LEN), dtype=np.int8)
    for i, s in enumerate(starts):
        out[i] = normalize_int8(sig[s:s + WINDOW_LEN])
    return out


# --- Dataset loaders ---------------------------------------------------------

def pick_channel(rec, prefer: tuple[str, ...]) -> int:
    sig_names = [n.lower() for n in rec.sig_name]
    for p in prefer:
        for i, n in enumerate(sig_names):
            if p.lower() in n:
                return i
    return 0


def load_cebs_records(records: Iterable[str], data_dir: Path) -> list[tuple[str, np.ndarray]]:
    out = []
    for rec_name in records:
        try:
            rec = wfdb.rdrecord(str(data_dir / rec_name))
            scg_idx = pick_channel(rec, ("scg", "pcg"))
            scg = rec.p_signal[:, scg_idx]
            scg = resample_to_target(scg, rec.fs)
            scg = bandpass(scg, TARGET_FS)
            out.append((rec_name, windowize(scg)))
        except Exception as e:
            print(f"  [cebs/{rec_name}] SKIP: {e}", file=sys.stderr)
    return out


def load_mitdb_records(records: Iterable[str], data_dir: Path) -> list[tuple[str, np.ndarray]]:
    out = []
    for rec_name in records:
        try:
            rec = wfdb.rdrecord(str(data_dir / rec_name))
            ecg_idx = pick_channel(rec, ("mlii", "ml ii", "ml-ii", "ecg", "ii", "v1", "v5"))
            sig = rec.p_signal[:, ecg_idx]
            sig = resample_to_target(sig, rec.fs)
            sig = bandpass(sig, TARGET_FS)
            out.append((rec_name, windowize(sig)))
        except Exception as e:
            print(f"  [mit/{rec_name}] SKIP: {e}", file=sys.stderr)
    return out


def load_apnea_records(records: Iterable[str], data_dir: Path) -> list[tuple[str, np.ndarray]]:
    out = []
    for rec_name in records:
        try:
            rec = wfdb.rdrecord(str(data_dir / rec_name))
            ecg_idx = 0
            sig = rec.p_signal[:, ecg_idx]
            sig = resample_to_target(sig, rec.fs)
            sig = bandpass(sig, TARGET_FS)
            out.append((rec_name, windowize(sig)))
        except Exception as e:
            print(f"  [apnea/{rec_name}] SKIP: {e}", file=sys.stderr)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data_mixed"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_X = []
    all_dsid = []                # 0=cebs, 1=mit, 2=apnea
    all_sid = []                 # global subject id (unique across datasets)
    record_meta = []             # list of (dataset_id, subject_id, record_name)

    next_sid = 0

    # ------- CEBS (60 records, 20 unique subjects) -------
    print("[cebs] processing 60 records (b/m/p × 20 subjects)...")
    cebs_recs = ([f"b{i:03d}" for i in range(1, 21)] +
                 [f"m{i:03d}" for i in range(1, 21)] +
                 [f"p{i:03d}" for i in range(1, 21)])
    t0 = time.time()
    for rec_name in cebs_recs:
        sub_id = next_sid + (int(rec_name[1:]) - 1)        # b001 → 0 + (1-1) = 0
        try:
            rec = wfdb.rdrecord(str(Path("data/cebsdb") / rec_name))
            scg_idx = pick_channel(rec, ("scg", "pcg"))
            scg = rec.p_signal[:, scg_idx]
            scg = resample_to_target(scg, rec.fs)
            scg = bandpass(scg, TARGET_FS)
            wins = windowize(scg)
            if len(wins) > 0:
                all_X.append(wins)
                all_dsid.append(np.full(len(wins), 0, dtype=np.int32))
                all_sid.append(np.full(len(wins), sub_id, dtype=np.int32))
                record_meta.append((0, sub_id, rec_name, len(wins)))
        except Exception as e:
            print(f"  [cebs/{rec_name}] SKIP: {e}", file=sys.stderr)
    next_sid += 20
    print(f"  cebs done in {time.time()-t0:.0f}s, total subjects after: {next_sid}")

    # ------- MIT-BIH (48 records, 47 unique subjects since 201/202 same) -------
    print("[mit] processing 48 records...")
    mit_recs = [str(n) for n in [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                                  111, 112, 113, 114, 115, 116, 117, 118, 119,
                                  121, 122, 123, 124, 200, 201, 202, 203, 205,
                                  207, 208, 209, 210, 212, 213, 214, 215, 217,
                                  219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]]
    t0 = time.time()
    for i, rec_name in enumerate(mit_recs):
        sub_id = next_sid + i
        try:
            rec = wfdb.rdrecord(str(Path("data/mitdb") / rec_name))
            ecg_idx = pick_channel(rec, ("mlii", "ml ii", "ii", "ecg", "v1", "v5"))
            sig = rec.p_signal[:, ecg_idx]
            sig = resample_to_target(sig, rec.fs)
            sig = bandpass(sig, TARGET_FS)
            wins = windowize(sig)
            if len(wins) > 0:
                all_X.append(wins)
                all_dsid.append(np.full(len(wins), 1, dtype=np.int32))
                all_sid.append(np.full(len(wins), sub_id, dtype=np.int32))
                record_meta.append((1, sub_id, rec_name, len(wins)))
        except Exception as e:
            print(f"  [mit/{rec_name}] SKIP: {e}", file=sys.stderr)
    next_sid += len(mit_recs)
    print(f"  mit done in {time.time()-t0:.0f}s, total subjects after: {next_sid}")

    # ------- Apnea-ECG (35 records, 32 unique) -------
    print("[apnea] processing 35 records...")
    apnea_recs = ([f"a{i:02d}" for i in range(1, 21)] +
                  [f"b{i:02d}" for i in range(1, 6)] +
                  [f"c{i:02d}" for i in range(1, 11)])
    t0 = time.time()
    for i, rec_name in enumerate(apnea_recs):
        sub_id = next_sid + i
        try:
            rec = wfdb.rdrecord(str(Path("data/apnea_ecg") / rec_name))
            sig = rec.p_signal[:, 0]
            sig = resample_to_target(sig, rec.fs)
            sig = bandpass(sig, TARGET_FS)
            wins = windowize(sig)
            # subsample heavy: apnea recordings are 8 hours, way too many windows
            if len(wins) > 5000:
                idx = np.random.RandomState(42 + i).choice(len(wins), 5000, replace=False)
                wins = wins[idx]
            if len(wins) > 0:
                all_X.append(wins)
                all_dsid.append(np.full(len(wins), 2, dtype=np.int32))
                all_sid.append(np.full(len(wins), sub_id, dtype=np.int32))
                record_meta.append((2, sub_id, rec_name, len(wins)))
        except Exception as e:
            print(f"  [apnea/{rec_name}] SKIP: {e}", file=sys.stderr)
    next_sid += len(apnea_recs)
    print(f"  apnea done in {time.time()-t0:.0f}s, total subjects after: {next_sid}")

    # Concatenate
    X = np.concatenate(all_X)[:, None, :]
    dsid = np.concatenate(all_dsid)
    sid = np.concatenate(all_sid)

    print(f"\nfinal corpus:")
    print(f"  N = {len(X)} windows")
    print(f"  unique subjects = {next_sid}")
    print(f"  per dataset: cebs={int((dsid==0).sum())}, mit={int((dsid==1).sum())}, apnea={int((dsid==2).sum())}")

    np.savez(args.out / "all_unlabeled.npz",
             X=X, dataset_id=dsid, sid=sid,
             record_meta=np.array(record_meta, dtype=object))
    print(f"-> wrote {args.out}/all_unlabeled.npz")


if __name__ == "__main__":
    main()
