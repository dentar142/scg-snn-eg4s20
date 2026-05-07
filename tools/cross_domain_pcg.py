"""cross_domain_pcg.py - SCG → PCG cross-domain transfer probe.

The deployed FOSTER-trained SNN expects (B, 5, 256) input across modalities
[PVDF, PZT, ACC, PCG, ERB]. PhysioNet 2016 has only PCG. To probe whether
FOSTER's PCG channel learned anything transferable, we:

  1. Load a stratified sample of PhysioNet 2016 records (normal/abnormal labels)
  2. Resample 2 kHz -> 1 kHz, band-pass 5-50 Hz, normalize per-window to int8
  3. Cut each record into 256-ms windows
  4. Build (N, 5, 256) tensors with PCG signal on channel 3 (FOSTER PCG slot)
     and zeros (or noise) on other 4 channels
  5. Run SNN forward, collect output spike-count distribution per record
  6. Test: does spike-count entropy differ between normal vs abnormal records?

Hypothesis: if FOSTER's PCG channel learned generalizable cardiac-rhythm
features, abnormal recordings (with murmurs, irregular cycles) should produce
HIGHER prediction entropy than normal recordings. If H_abnormal > H_normal
significantly, cross-domain transfer is plausible.

Output: doc/cross_domain_pcg.json + doc/figs/cross_domain_pcg_entropy.png
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample_poly
from scipy.io import wavfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
from train_snn_multimodal import MultiModalSCGSnn  # noqa: E402

TARGET_FS = 1000
WIN_LEN = 256
PCG_CHANNEL_IDX = 3   # FOSTER ordering: [PVDF, PZT, ACC, PCG, ERB]


def bandpass(x, fs, lo, hi, order=4):
    sos_b, sos_a = butter(order, [lo, hi], btype="band", fs=fs)
    return filtfilt(sos_b, sos_a, x).astype(np.float32)


def load_record(wav_path):
    fs, data = wavfile.read(wav_path)
    data = data.astype(np.float32)
    if data.ndim > 1: data = data[:, 0]
    if fs != TARGET_FS:
        # resample to 1 kHz
        from math import gcd
        g = gcd(int(fs), TARGET_FS)
        data = resample_poly(data, TARGET_FS // g, fs // g)
    data = bandpass(data, TARGET_FS, 5.0, 50.0)
    return data


def windowize(sig, win_len=WIN_LEN, hop=WIN_LEN):
    """Cut into non-overlapping 256-sample windows; INT8-normalize per window."""
    n = len(sig); n_win = n // hop
    if n_win == 0: return None
    sig = sig[: n_win * hop].reshape(n_win, hop)
    # Per-window z-score then quantize to INT8 [-127, 127]
    mu = sig.mean(axis=1, keepdims=True)
    std = sig.std(axis=1, keepdims=True) + 1e-9
    z = (sig - mu) / std
    q = np.clip(z * 32, -127, 127).astype(np.int8)
    return q   # (n_win, 256)


def expand_to_5channel(pcg_windows):
    """Build (N, 5, 256) with PCG on channel 3 and zeros elsewhere."""
    N = len(pcg_windows)
    out = np.zeros((N, 5, WIN_LEN), dtype=np.int8)
    out[:, PCG_CHANNEL_IDX, :] = pcg_windows
    return out


@torch.no_grad()
def get_spike_counts(model, X, device, batch=512):
    model.eval()
    Xt = torch.from_numpy(X.astype(np.float32) / 127.0)
    counts = []
    for i in range(0, len(X), batch):
        x = Xt[i:i+batch].to(device)
        if x.dim() == 3:
            B = x.size(0); xf = x.reshape(B, -1)
        else:
            xf = x; B = xf.size(0)
        I1 = model.fc1(xf)
        v1 = torch.zeros(B, model.n_hidden, device=device)
        v2 = torch.zeros(B, model.n_classes, device=device)
        cnt = torch.zeros(B, model.n_classes, device=device)
        for _ in range(model.T):
            v1 = model.beta * v1 + I1
            s1 = (v1 >= model.threshold).float()
            v1 = v1 - s1 * model.threshold
            I2 = model.fc2(s1)
            v2 = model.beta * v2 + I2
            s2 = (v2 >= model.threshold).float()
            v2 = v2 - s2 * model.threshold
            cnt = cnt + s2
        counts.append(cnt.cpu().numpy())
    return np.concatenate(counts, axis=0)


def class_entropy(counts):
    """Per-window prediction entropy from spike counts (softmax-like)."""
    p = counts / (counts.sum(axis=1, keepdims=True) + 1e-9)
    p = np.clip(p, 1e-9, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])  # in [0,1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("model/ckpt/sweep/best_sweep_H32_T16.pt"))
    p.add_argument("--pn-root", type=Path, default=Path("data/physionet2016"))
    p.add_argument("--max-per-letter", type=int, default=20,
                   help="cap per training letter (a..f) for speed")
    p.add_argument("--out", type=Path, default=Path("doc/cross_domain_pcg.json"))
    p.add_argument("--fig-dir", type=Path, default=Path("doc/figs"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = MultiModalSCGSnn(n_in=int(ck["n_in"]), n_channels=int(ck["n_channels"]),
                             n_hidden=int(ck["H"]), n_classes=int(ck["n_classes"]),
                             beta=float(ck.get("beta", 0.9)),
                             threshold=float(ck.get("threshold", 1.0)),
                             T=int(ck["T"])).to(device)
    model.load_state_dict(ck["state"])

    records = []   # (rec_id, label, n_windows, ent_mean, pred_dist)
    for L in "abcdef":
        ref_path = args.pn_root / f"training-{L}" / "REFERENCE.csv"
        if not ref_path.exists(): continue
        df = [l.strip().split(",") for l in ref_path.read_text().splitlines() if l.strip()]
        # Stratify: take up to max_per_letter normal + max_per_letter abnormal
        by_label = {-1: [], 1: []}
        for rec_id, lab in df:
            by_label.setdefault(int(lab), []).append(rec_id)
        for lab in [-1, 1]:
            picked = by_label[lab][:args.max_per_letter]
            for rec_id in picked:
                wav_path = args.pn_root / f"training-{L}" / f"{rec_id}.wav"
                if not wav_path.exists(): continue
                try:
                    sig = load_record(wav_path)
                    win = windowize(sig)
                    if win is None or len(win) < 4: continue
                    X5 = expand_to_5channel(win)
                    counts = get_spike_counts(model, X5, device)
                    ent = class_entropy(counts).mean()
                    pred = counts.argmax(axis=1)
                    pred_dist = [int((pred == c).sum()) / len(pred) for c in range(3)]
                    records.append({
                        "rec_id": rec_id, "letter": L, "label": int(lab),
                        "n_windows": int(len(win)),
                        "entropy_mean": float(ent),
                        "pred_dist": pred_dist,
                    })
                except Exception as e:
                    print(f"  [skip] {rec_id}: {e}")
        print(f"  letter {L}: collected {len([r for r in records if r['letter']==L])} records")

    # Statistical test
    e_norm = np.array([r["entropy_mean"] for r in records if r["label"] == -1])
    e_abnm = np.array([r["entropy_mean"] for r in records if r["label"] == 1])
    print(f"\nNormal records:    n={len(e_norm)}  ent={e_norm.mean():.4f} ± {e_norm.std():.4f}")
    print(f"Abnormal records:  n={len(e_abnm)}  ent={e_abnm.mean():.4f} ± {e_abnm.std():.4f}")
    delta = float(e_abnm.mean() - e_norm.mean())
    # Welch's t-test (no scipy.stats import needed; manual)
    se = np.sqrt(e_norm.var(ddof=1) / len(e_norm) + e_abnm.var(ddof=1) / len(e_abnm))
    t = delta / max(se, 1e-9)
    print(f"  delta = {delta:+.4f},  t-statistic = {t:+.2f}")

    # Plot histogram
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    plt.hist(e_norm, bins=bins, alpha=0.55, color="C2",
             label=f"Normal (n={len(e_norm)}, mean {e_norm.mean():.3f})")
    plt.hist(e_abnm, bins=bins, alpha=0.55, color="C3",
             label=f"Abnormal (n={len(e_abnm)}, mean {e_abnm.mean():.3f})")
    plt.axvline(e_norm.mean(), color="C2", linestyle="--", alpha=0.7)
    plt.axvline(e_abnm.mean(), color="C3", linestyle="--", alpha=0.7)
    plt.xlabel("Per-record mean prediction entropy (normalized to [0,1])")
    plt.ylabel("Record count")
    plt.title(f"Cross-domain PCG: spike-output entropy by label (delta {delta:+.4f}, t {t:+.2f})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(args.fig_dir / "cross_domain_pcg_entropy.png", dpi=150)
    plt.close()

    summary = {
        "ckpt": str(args.ckpt), "n_records": len(records),
        "n_normal": int(len(e_norm)), "n_abnormal": int(len(e_abnm)),
        "entropy_normal": {"mean": float(e_norm.mean()), "std": float(e_norm.std())},
        "entropy_abnormal": {"mean": float(e_abnm.mean()), "std": float(e_abnm.std())},
        "delta": delta, "welch_t": float(t),
        "interpretation": (
            "If t > 2 with abnormal entropy higher, FOSTER PCG channel transferred."
            if abs(t) > 2 else
            "Effect not significant (|t| < 2); FOSTER PCG features did not transfer."
        ),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n-> wrote {args.out}")
    print(f"-> {args.fig_dir}/cross_domain_pcg_entropy.png")


if __name__ == "__main__":
    main()
