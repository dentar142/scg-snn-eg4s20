"""bench_fpga_cross_dataset.py - on-board FPGA bench of CEBSDB single-SCG
through a 5-channel multimodal SNN bit (FOSTER-trained).

Maps CEBSDB SCG -> FOSTER ACC slot (channel idx 2), zeros other 4 channels,
sends 5*256 = 1280 bytes per window via UART. Reports overall + per-class
+ per-subject accuracy.

Usage:
    python tools/bench_fpga_cross_dataset.py --port COM28 \\
        --cebs-data data_excl100/val.npz --n 5000 \\
        --out doc/bench_fpga_dropout_aligned_cebsdb.json
"""
from __future__ import annotations
import argparse, json, statistics, sys, time
from pathlib import Path
import numpy as np
import serial

CMD_RST  = b"\xA0"
CMD_LD_X = b"\xA2"
CMD_RUN  = b"\xA3"
ACC_SLOT = 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--cebs-data", type=Path, required=True)
    p.add_argument("--n", type=int, default=5000,
                   help="cap on samples; 0 = all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-channels", type=int, default=5)
    p.add_argument("--slot", type=int, default=ACC_SLOT)
    args = p.parse_args()

    d = np.load(args.cebs_data, allow_pickle=True)
    X1 = d["X"]   # (N, 1, 256)
    y  = d["y"]
    sid = d["sid"]
    print(f"CEBSDB: X={X1.shape}  unique sid={len(set(sid.tolist()))}")
    print(f"  per-class: BG={int((y==0).sum())} Sys={int((y==1).sum())} Dia={int((y==2).sum())}")

    if X1.ndim != 3 or X1.shape[1] != 1:
        sys.exit(f"expected (N, 1, L), got {X1.shape}")
    N, _, L = X1.shape

    # Stratified subsampling
    if args.n > 0 and args.n < N:
        rng = np.random.RandomState(args.seed)
        per_class = max(1, args.n // 3)
        idx = []
        for c in range(3):
            cls_idx = np.where(y == c)[0]
            if len(cls_idx) <= per_class:
                idx.append(cls_idx)
            else:
                idx.append(rng.choice(cls_idx, per_class, replace=False))
        idx = np.concatenate(idx)
        X1 = X1[idx]; y = y[idx]; sid = sid[idx]
        print(f"  stratified subsample: {len(X1)} ({per_class}/class)")

    # Expand to 5-channel: SCG in slot, zeros elsewhere
    n = len(X1); C = args.n_channels
    X5 = np.zeros((n, C, L), dtype=np.int8)
    X5[:, args.slot, :] = X1[:, 0, :]
    print(f"  X5 = {X5.shape}, SCG -> slot {args.slot}, others zero")

    pred_mask = 0x03   # K=3 -> low 2 bits
    n_in = C * L

    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(0.05)
    ser.write(CMD_RST); ser.flush()

    rt_ms, run_ms = [], []
    preds = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    t_overall = time.time()
    for i in range(n):
        window = X5[i].reshape(-1).astype(np.int8).tobytes()
        if ser.in_waiting:
            ser.reset_input_buffer()
        t_a = time.perf_counter()
        ser.write(CMD_LD_X + window); ser.flush()
        t_b = time.perf_counter()
        time.sleep(0.005)
        ser.write(CMD_RUN); ser.flush()
        resp = ser.read(1)
        t_c = time.perf_counter()
        if not resp:
            continue
        preds[i] = resp[0] & pred_mask
        valid[i] = True
        rt_ms.append((t_c - t_a) * 1e3)
        run_ms.append((t_c - t_b) * 1e3)
        if i < 3 or i % 200 == 0 or i == n - 1:
            elapsed = time.time() - t_overall
            eta = elapsed / max(i + 1, 1) * (n - i - 1)
            print(f"[{i:05d}/{n}] cls={preds[i]} truth={int(y[i])} "
                  f"sid={int(sid[i])} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                  flush=True)
    ser.close()

    n_valid = int(valid.sum())
    if n_valid == 0:
        sys.exit("No valid samples")

    acc = float((preds[valid] == y[valid]).mean() * 100)
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, pp in zip(y[valid], preds[valid]):
        cm[int(t), int(pp)] += 1
    per_class_acc = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist()
    per_class_f1 = []
    for c in range(3):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        per_class_f1.append(0.0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn))

    per_subject = []
    for s in sorted(set(sid.tolist())):
        m = (sid == s) & valid
        if not m.any(): continue
        per_subject.append({
            "sid": int(s), "n": int(m.sum()),
            "acc": float((preds[m] == y[m]).mean() * 100),
        })

    summary = {
        "n_classes": 3, "n_samples_valid": n_valid,
        "round_trip_ms": {"mean": statistics.mean(rt_ms),
                          "median": statistics.median(rt_ms)},
        "run_ms": {"mean": statistics.mean(run_ms),
                   "median": statistics.median(run_ms)},
        "accuracy_percent": acc,
        "per_class_acc": per_class_acc,
        "per_class_f1": per_class_f1,
        "macro_f1": sum(per_class_f1) / 3,
        "confusion_matrix": cm.tolist(),
        "per_subject": per_subject,
        "n_channels_sent": C, "slot": args.slot,
        "data": str(args.cebs_data),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nFPGA cross-dataset acc = {acc:.2f}%   macro-F1 = {summary['macro_f1']*100:.2f}%   "
          f"run = {summary['run_ms']['mean']:.2f} ms")
    print("per-subject:", "  ".join(f"sid{r['sid']}={r['acc']:.1f}%" for r in per_subject[:6]),
          "...")


if __name__ == "__main__":
    main()
