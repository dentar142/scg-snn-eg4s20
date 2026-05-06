"""bench_cpu.py — paper-grade CPU baseline benchmark.

Reports for both PyTorch FP32 and bit-exact INT8 golden model:
  * accuracy on val.npz
  * latency (mean / median / p95 / std) per sample
  * throughput
  * CPU info (model, cores, frequency)

Output: doc/bench_cpu.json (machine-readable for the report stage).
"""
from __future__ import annotations
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO / "tools"))

from train_qat import SCGNet                    # noqa: E402
from golden_model import forward_int8           # noqa: E402


def bench_pytorch(model: SCGNet, X: np.ndarray, y: np.ndarray, *, threads: int) -> dict:
    torch.set_num_threads(threads)
    Xt = torch.from_numpy(X.astype(np.float32) / 127.0)
    n = len(X)

    # Warm-up (5 samples)
    with torch.no_grad():
        for i in range(min(5, n)):
            model(Xt[i:i + 1])

    # Single-sample latency
    lat = []
    preds = np.zeros(n, dtype=np.int64)
    with torch.no_grad():
        for i in range(n):
            t0 = time.perf_counter()
            logit = model(Xt[i:i + 1])
            t1 = time.perf_counter()
            preds[i] = int(logit.argmax(1).item())
            lat.append((t1 - t0) * 1e3)   # ms
    acc = float((preds == y).mean() * 100)

    # Throughput (batch=128 if data large enough)
    bs = min(128, n)
    Xb = Xt[:bs]
    with torch.no_grad():
        for _ in range(3):
            model(Xb)
        t0 = time.perf_counter()
        n_iter = 50
        for _ in range(n_iter):
            model(Xb)
        t1 = time.perf_counter()
    thr = (bs * n_iter) / (t1 - t0)
    return {
        "accuracy_percent": acc,
        "latency_ms_mean": statistics.mean(lat),
        "latency_ms_median": statistics.median(lat),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_std": statistics.pstdev(lat),
        "latency_ms_min": min(lat),
        "latency_ms_max": max(lat),
        "throughput_samples_per_s_batch128": thr,
        "n_samples": n,
        "threads": threads,
    }


def bench_int8_golden(weights: Path, X: np.ndarray, y: np.ndarray) -> dict:
    n = len(X)
    # Warm-up
    for i in range(min(3, n)):
        forward_int8(X[i, 0], weights)

    lat = []
    preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t0 = time.perf_counter()
        preds[i] = forward_int8(X[i, 0], weights)
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1e3)
    acc = float((preds == y).mean() * 100)
    return {
        "accuracy_percent": acc,
        "latency_ms_mean": statistics.mean(lat),
        "latency_ms_median": statistics.median(lat),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_std": statistics.pstdev(lat),
        "latency_ms_min": min(lat),
        "latency_ms_max": max(lat),
        "throughput_samples_per_s": 1000.0 / statistics.mean(lat),
        "n_samples": n,
        "implementation": "pure-python int8 (numpy)",
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int = 3) -> list:
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=REPO / "model/ckpt/best.pt")
    p.add_argument("--data", type=Path, default=REPO / "data/val.npz")
    p.add_argument("--weights", type=Path, default=REPO / "rtl/weights")
    p.add_argument("--n", type=int, default=200, help="how many val samples")
    p.add_argument("--out", type=Path, default=REPO / "doc/bench_cpu.json")
    args = p.parse_args()

    val = np.load(args.data)
    X, y = val["X"], val["y"]
    n = min(args.n, len(X))
    X, y = X[:n], y[:n]

    # System info
    try:
        cpu_freq = psutil.cpu_freq()
        freq_mhz = float(cpu_freq.max) if cpu_freq and cpu_freq.max else None
    except Exception:
        freq_mhz = None
    sys_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "max_freq_mhz": freq_mhz,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }

    # Load model once
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = SCGNet()
    model.load_state_dict(ckpt["state"])
    model.eval()

    print(f"[1/3] PyTorch FP32 single-thread on {n} samples...")
    fp32_st = bench_pytorch(model, X, y, threads=1)
    print(f"      acc={fp32_st['accuracy_percent']:.2f}% "
          f"lat={fp32_st['latency_ms_mean']:.3f} ms "
          f"thr={fp32_st['throughput_samples_per_s_batch128']:.0f} samp/s")

    nthr = psutil.cpu_count(logical=True) or 8
    print(f"[2/3] PyTorch FP32 {nthr}-thread on {n} samples...")
    fp32_mt = bench_pytorch(model, X, y, threads=nthr)
    print(f"      acc={fp32_mt['accuracy_percent']:.2f}% "
          f"lat={fp32_mt['latency_ms_mean']:.3f} ms "
          f"thr={fp32_mt['throughput_samples_per_s_batch128']:.0f} samp/s")

    print(f"[3/3] INT8 golden model (Python/NumPy) on {n} samples...")
    int8 = bench_int8_golden(args.weights, X, y)
    print(f"      acc={int8['accuracy_percent']:.2f}% "
          f"lat={int8['latency_ms_mean']:.2f} ms")

    # Predictions for confusion matrix (rerun fast, no timing)
    Xt = torch.from_numpy(X.astype(np.float32) / 127.0)
    with torch.no_grad():
        fp32_pred = model(Xt).argmax(1).numpy()
    int8_pred = np.array([forward_int8(X[i, 0], args.weights) for i in range(n)])

    cm_fp = confusion_matrix(y, fp32_pred)
    cm_int = confusion_matrix(y, int8_pred)
    agreement = float((fp32_pred == int8_pred).mean() * 100)

    out = {
        "system": sys_info,
        "fp32_single_thread": fp32_st,
        "fp32_multi_thread": fp32_mt,
        "int8_golden_python": int8,
        "confusion_matrix_fp32": cm_fp,
        "confusion_matrix_int8": cm_int,
        "fp32_vs_int8_agreement_percent": agreement,
        "n_samples": n,
        "class_names": ["Background", "Systolic", "Diastolic"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n→ wrote {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
