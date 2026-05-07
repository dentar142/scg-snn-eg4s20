"""bench_fpga_snn_holdout.py — strict subject-disjoint FPGA bench.

Loads data_foster_multi/all.npz, filters to hold-out subject windows only,
benchmarks FPGA via UART, emits per-subject + overall results.

Usage:
    python tools/bench_fpga_snn_holdout.py --port COM27 \
        --data data_foster_multi/all.npz \
        --holdout sub003 sub006 sub009 sub013 sub020 sub021 sub024 sub026 \
        --out doc/bench_fpga_snn_multimodal_holdout.json
"""
from __future__ import annotations
import argparse, json, statistics, sys, time
from pathlib import Path
import numpy as np
import serial

CMD_RST  = b"\xA0"
CMD_LD_X = b"\xA2"
CMD_RUN  = b"\xA3"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--holdout", nargs="+", required=True)
    p.add_argument("--n", type=int, default=0,
                   help="cap on samples (0 = all hold-out windows)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for stratified subsampling when --n cap < total")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-classes", type=int, default=3)
    args = p.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X = d["X"]; y = d["y"]; sid = d["sid"]
    rec_names = list(map(str, d["record_names"]))

    name_to_sid = {n: i for i, n in enumerate(rec_names)}
    holdout_sids = []
    for h in args.holdout:
        if h not in name_to_sid:
            sys.exit(f"ERROR: holdout subject '{h}' not found")
        holdout_sids.append(name_to_sid[h])
    mask = np.isin(sid, holdout_sids)
    Xh, yh, sidh = X[mask], y[mask], sid[mask]
    print(f"hold-out subjects: {args.holdout} -> {len(args.holdout)} subjects, "
          f"{len(Xh)} windows total")

    if args.n > 0 and args.n < len(Xh):
        rng = np.random.RandomState(args.seed)
        # per-subject stratified sample
        keep_idx = []
        per_sub = max(1, args.n // len(args.holdout))
        for s in holdout_sids:
            idx_s = np.where(sidh == s)[0]
            if len(idx_s) <= per_sub:
                keep_idx.append(idx_s)
            else:
                keep_idx.append(rng.choice(idx_s, per_sub, replace=False))
        keep_idx = np.concatenate(keep_idx)
        Xh, yh, sidh = Xh[keep_idx], yh[keep_idx], sidh[keep_idx]
        print(f"  subsampled to {len(Xh)} windows ({per_sub}/subject)")

    K = args.n_classes
    pred_mask = (1 << max(2, (K - 1).bit_length())) - 1
    C, L = int(Xh.shape[1]), int(Xh.shape[2])
    n_in = C * L
    print(f"sending {n_in} bytes/sample (C={C}, L={L})  K={K}")

    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(0.05)
    ser.write(CMD_RST); ser.flush()

    n = len(Xh)
    rt_ms, run_ms = [], []
    preds = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    t_overall = time.time()
    for i in range(n):
        window = Xh[i].reshape(-1).astype(np.int8).tobytes()
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
            print(f"[{i}] TIMEOUT")
            continue
        preds[i] = resp[0] & pred_mask
        valid[i] = True
        rt_ms.append((t_c - t_a) * 1e3)
        run_ms.append((t_c - t_b) * 1e3)
        if i < 3 or i % 200 == 0 or i == n - 1:
            elapsed = time.time() - t_overall
            eta = elapsed / max(i + 1, 1) * (n - i - 1)
            print(f"[{i:05d}/{n}] cls={preds[i]} truth={int(yh[i])} "
                  f"sid={int(sidh[i])} rt={rt_ms[-1]:.1f}ms "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    ser.close()

    n_valid = int(valid.sum())
    if n_valid == 0:
        sys.exit("No valid samples")

    acc = float((preds[valid] == yh[valid]).mean() * 100)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, pp in zip(yh[valid], preds[valid]):
        cm[int(t), int(pp)] += 1
    per_class_acc = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist()
    per_class_f1 = []
    for c in range(K):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        per_class_f1.append(0.0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn))

    per_subject = []
    for s in holdout_sids:
        m = (sidh == s) & valid
        if not m.any():
            continue
        sub_acc = float((preds[m] == yh[m]).mean() * 100)
        sub_cm = np.zeros((K, K), dtype=np.int64)
        for t, pp in zip(yh[m], preds[m]):
            sub_cm[int(t), int(pp)] += 1
        per_subject.append({
            "subject": rec_names[s],
            "n": int(m.sum()),
            "acc": sub_acc,
            "confusion_matrix": sub_cm.tolist(),
        })

    summary = {
        "n_classes": K,
        "holdout_subjects": args.holdout,
        "n_holdout_subjects": len(args.holdout),
        "n_samples_attempted": n,
        "n_samples_valid": n_valid,
        "round_trip_ms": {"mean": statistics.mean(rt_ms),
                          "median": statistics.median(rt_ms),
                          "min": min(rt_ms), "max": max(rt_ms)},
        "run_ms": {"mean": statistics.mean(run_ms),
                   "median": statistics.median(run_ms)},
        "accuracy_percent": acc,
        "per_class_acc": per_class_acc,
        "per_class_f1": per_class_f1,
        "macro_f1": sum(per_class_f1) / len(per_class_f1),
        "confusion_matrix": cm.tolist(),
        "per_subject": per_subject,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nFPGA SNN (subject-disjoint hold-out): acc = {acc:.2f}%  "
          f"macro-F1 = {summary['macro_f1']*100:.2f}%  "
          f"rt = {summary['round_trip_ms']['mean']:.1f} ms  "
          f"run = {summary['run_ms']['mean']:.2f} ms")
    print("per-subject:")
    for r in per_subject:
        print(f"  {r['subject']}: n={r['n']}  acc={r['acc']:.2f}%")


if __name__ == "__main__":
    main()
