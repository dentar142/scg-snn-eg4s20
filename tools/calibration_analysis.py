"""calibration_analysis.py - Per-subject OOD detection + abstention for SCG SNN.

Loads a checkpoint, runs INT8 SNN inference on val + holdout while logging
per-sample confidence telemetry. For each candidate confidence signal,
computes histograms, ROC, and coverage-vs-selective-accuracy curves. Picks
a defensible threshold tau on the HW-cheap "margin" signal (top1 - top2
spike count) and reports per-subject coverage and selective accuracy.

Usage:
    D:/anaconda3/envs/scggpu/python.exe tools/calibration_analysis.py \
        --ckpt model/ckpt/best_snn_v1.pt \
        --id-data data_excl100/val.npz \
        --ood-data data_excl100/holdout.npz \
        --out-dir doc \
        --leak-shift 4
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def quantize_per_tensor_int8(w):
    absmax = float(np.abs(w).max())
    if absmax < 1e-12:
        return np.zeros_like(w, dtype=np.int8), 1.0
    scale = absmax / 127.0
    qw = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return qw, scale


def lif_step_int(v, I, theta, leak_shift):
    v_leaked = v - (v >> leak_shift)
    v_new = v_leaked + I
    s = (v_new >= theta).astype(np.int8)
    v_new = v_new - s.astype(v_new.dtype) * theta
    return v_new, s


def run_int_snn_instrumented(X, W1q, W2q, theta1, theta2, leak_shift, T):
    N, _ = X.shape
    H = W1q.shape[0]
    C = W2q.shape[0]

    preds = np.zeros(N, dtype=np.int64)
    spike_counts = np.zeros((N, C), dtype=np.int32)
    hidden_fr = np.zeros(N, dtype=np.int32)
    hidden_active = np.zeros(N, dtype=np.int32)
    first_out_t = np.full(N, T, dtype=np.int32)
    v2_final_max = np.zeros(N, dtype=np.int32)

    I1 = (X.astype(np.int32) @ W1q.T.astype(np.int32))

    for n in range(N):
        v1 = np.zeros(H, dtype=np.int32)
        v2 = np.zeros(C, dtype=np.int32)
        sc = np.zeros(C, dtype=np.int32)
        any_hidden_ever = np.zeros(H, dtype=np.bool_)
        i1_const = I1[n]
        any_out_fired = False

        for t in range(T):
            v1, s1 = lif_step_int(v1, i1_const, theta1, leak_shift)
            any_hidden_ever |= s1.astype(np.bool_)
            hidden_fr[n] += int(s1.sum())
            i2 = (s1.astype(np.int32) @ W2q.T.astype(np.int32))
            v2, s2 = lif_step_int(v2, i2, theta2, leak_shift)
            sc += s2
            if (not any_out_fired) and s2.sum() > 0:
                first_out_t[n] = t
                any_out_fired = True

        spike_counts[n] = sc
        hidden_active[n] = int(any_hidden_ever.sum())
        v2_final_max[n] = int(v2.max())
        preds[n] = int(np.argmax(sc))

    return {
        "preds": preds,
        "spike_counts": spike_counts,
        "hidden_fr": hidden_fr,
        "hidden_active": hidden_active,
        "first_out_t": first_out_t,
        "v2_final_max": v2_final_max,
    }


def confidence_signals(tel):
    sc = tel["spike_counts"].astype(np.float64)
    sorted_sc = np.sort(sc, axis=1)
    top1 = sorted_sc[:, -1]
    top2 = sorted_sc[:, -2]

    margin = (top1 - top2)
    sc_max = top1
    sc_sum = sc.sum(axis=1)
    eps = 1e-9
    p = (sc + eps) / (sc.sum(axis=1, keepdims=True) + 3 * eps)
    H = -(p * np.log(p)).sum(axis=1)
    entropy_neg = -H
    log3 = np.log(3.0)
    entropy_norm = 1.0 - H / log3

    hidden_fr = tel["hidden_fr"].astype(np.float64)
    hidden_active = tel["hidden_active"].astype(np.float64)
    first_out_neg = -tel["first_out_t"].astype(np.float64)

    return {
        "margin": margin,
        "sc_max": sc_max,
        "sc_sum": sc_sum,
        "entropy_neg": entropy_neg,
        "entropy_norm": entropy_norm,
        "hidden_fr": hidden_fr,
        "hidden_active": hidden_active,
        "first_out_neg": first_out_neg,
    }


def roc_auc(scores, labels_pos):
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels_pos[order].astype(np.int64)
    P = int(y.sum())
    Nn = int(len(y) - P)
    if P == 0 or Nn == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([s.max(), s.min()]), 0.5
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / P
    fpr = fps / Nn
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    thr = np.concatenate(([s[0] + 1.0], s))
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, thr, auc


def coverage_accuracy_curve(scores, correct):
    order = np.argsort(-scores, kind="mergesort")
    correct_sorted = correct[order].astype(np.int64)
    N = len(scores)
    cum_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, N + 1) / N
    sel_acc = cum_correct / np.arange(1, N + 1)
    thr = scores[order]
    return coverage, sel_acc, thr


def macro_f1(y_true, y_pred, n_classes=3):
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(f1s)), f1s


def build_int8(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["state"]
    H = ck.get("H", 64)
    T = ck.get("T", 32)
    threshold_fp = ck.get("threshold", 1.0)
    W1 = state["fc1.weight"].numpy()
    W2 = state["fc2.weight"].numpy()
    W1q, w1_s = quantize_per_tensor_int8(W1)
    W2q, w2_s = quantize_per_tensor_int8(W2)
    in_scale = 1.0 / 127.0
    theta1_int = int(round(threshold_fp / (in_scale * w1_s)))
    theta2_int = max(1, int(round(threshold_fp / w2_s)))
    return {
        "ckpt": ck, "H": H, "T": T,
        "W1q": W1q, "W2q": W2q,
        "theta1_int": theta1_int, "theta2_int": theta2_int,
        "w1_s": w1_s, "w2_s": w2_s,
    }


def evaluate_dataset(net, X_int8, y, leak_shift, label):
    print(f"  [{label}] running INT8 SNN on N={len(X_int8)} ...", flush=True)
    t0 = time.time()
    tel = run_int_snn_instrumented(
        X_int8, net["W1q"], net["W2q"],
        net["theta1_int"], net["theta2_int"],
        leak_shift, net["T"],
    )
    dt = time.time() - t0
    sigs = confidence_signals(tel)
    preds = tel["preds"]
    correct = (preds == y).astype(np.int64)
    acc = float(correct.mean())
    print(f"    [{label}] done in {dt:.1f}s, accuracy={acc*100:.2f}%", flush=True)
    return tel, sigs, preds, correct, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("model/ckpt/best_snn_v1.pt"))
    p.add_argument("--id-data", type=Path, default=Path("data_excl100/val.npz"))
    p.add_argument("--ood-data", type=Path, default=Path("data_excl100/holdout.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("doc"))
    p.add_argument("--leak-shift", type=int, default=4)
    p.add_argument("--n-id", type=int, default=11601)
    p.add_argument("--n-ood", type=int, default=9660)
    p.add_argument("--target-coverage-min", type=float, default=0.5)
    args = p.parse_args()

    REPO = Path(__file__).resolve().parents[1]
    ckpt_path = args.ckpt if args.ckpt.is_absolute() else REPO / args.ckpt
    id_path = args.id_data if args.id_data.is_absolute() else REPO / args.id_data
    ood_path = args.ood_data if args.ood_data.is_absolute() else REPO / args.ood_data
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    print("[OBJECTIVE] Per-subject OOD detection + abstention for SCG SNN")
    print(f"  ckpt:  {ckpt_path}")
    print(f"  ID:    {id_path}")
    print(f"  OOD:   {ood_path}")
    print(f"  out:   {out_dir}")

    net = build_int8(ckpt_path)
    print(f"\n[DATA] arch={net['ckpt'].get('arch')}  H={net['H']}  T={net['T']}  "
          f"theta1_int={net['theta1_int']}  theta2_int={net['theta2_int']}")

    val = np.load(id_path, allow_pickle=True)
    X_id = val["X"][:args.n_id, 0].astype(np.int8)
    y_id = val["y"][:args.n_id].astype(np.int64)
    print(f"\n[DATA] ID ({id_path.name}): N={len(X_id)}  y dist={np.bincount(y_id, minlength=3).tolist()}")

    print("\n[ANALYZE] running INT8 SNN on ID set ...")
    tel_id, sigs_id, preds_id, correct_id, acc_id = evaluate_dataset(
        net, X_id, y_id, args.leak_shift, "ID")

    ho = np.load(ood_path, allow_pickle=True)
    X_ood = ho["X"][:args.n_ood, 0].astype(np.int8)
    y_ood = ho["y"][:args.n_ood].astype(np.int64)
    sid_ood = ho["sid"][:args.n_ood].astype(np.int64)
    print(f"\n[DATA] OOD ({ood_path.name}): N={len(X_ood)}  y dist={np.bincount(y_ood, minlength=3).tolist()}  sids={sorted(set(sid_ood.tolist()))}")

    a = np.load(REPO / "data_excl100/all.npz", allow_pickle=True)
    rec_names = list(a["record_names"])
    sid_to_name = {int(s): rec_names[s] for s in set(sid_ood.tolist())}
    print(f"  sid -> name: {sid_to_name}")

    print("\n[ANALYZE] running INT8 SNN on OOD set ...")
    tel_ood, sigs_ood, preds_ood, correct_ood, acc_ood = evaluate_dataset(
        net, X_ood, y_ood, args.leak_shift, "OOD")

    print("\n[FINDING] Per-subject baseline accuracy (no abstention)")
    per_subj = {}
    for s in sorted(set(sid_ood.tolist())):
        m = sid_ood == s
        a_s = float(correct_ood[m].mean())
        f1_s, _ = macro_f1(y_ood[m], preds_ood[m])
        per_subj[sid_to_name[int(s)]] = {"acc": a_s, "macro_f1": f1_s, "n": int(m.sum())}
        print(f"  {sid_to_name[int(s)]}: n={m.sum()} acc={a_s*100:.2f}% macro_f1={f1_s*100:.2f}%")

    print("\n[ANALYZE] confidence signal AUROC (b015 = trusted, b002+b007 = OOD-like)")
    name_to_sid = {v: k for k, v in sid_to_name.items()}
    is_b015 = (sid_ood == name_to_sid["b015"]).astype(np.int64)
    label_pos_oodtask = is_b015

    auroc_table = {}
    for sig_name, sig_arr in sigs_ood.items():
        _, _, _, auc = roc_auc(sig_arr, label_pos_oodtask)
        auroc_table[sig_name] = auc
        print(f"  AUROC[{sig_name:>14}] = {auc:.4f}")

    print("\n[ANALYZE] AUROC for correct-vs-incorrect")
    auroc_correct = {}
    for sig_name, sig_arr in sigs_ood.items():
        _, _, _, auc_ci = roc_auc(sig_arr, correct_ood)
        auroc_correct[sig_name] = auc_ci
        print(f"  AUROC_correct[{sig_name:>14}] = {auc_ci:.4f}")

    print("\n[PLOT] per-subject confidence histograms ...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    sig_order = ["margin", "sc_max", "sc_sum", "entropy_neg", "entropy_norm",
                 "hidden_fr", "hidden_active", "first_out_neg"]
    colors = {"b015": "#2ca02c", "b007": "#d62728", "b002": "#1f77b4"}
    for ax, sig_name in zip(axes.flat, sig_order):
        s = sigs_ood[sig_name]
        for subj_name in ["b015", "b007", "b002"]:
            m = sid_ood == name_to_sid[subj_name]
            ax.hist(s[m], bins=40, alpha=0.45,
                    label=f"{subj_name} (n={int(m.sum())})",
                    color=colors[subj_name], density=True)
        ax.set_title(f"{sig_name}  AUROC={auroc_table[sig_name]:.3f}")
        ax.set_xlabel(sig_name); ax.set_ylabel("density")
        ax.legend(fontsize=7)
    axes.flat[-1].clear()
    names = list(auroc_table.keys())
    vals = [auroc_table[n] for n in names]
    axes.flat[-1].barh(names, vals, color="#444")
    axes.flat[-1].axvline(0.5, color="red", lw=0.6, ls="--")
    axes.flat[-1].set_xlim(0, 1)
    axes.flat[-1].set_xlabel("AUROC (b015 vs b002+b007)")
    axes.flat[-1].set_title("Signal ranking")
    fig.suptitle(f"Confidence signals - hold-out 9660 - ckpt={ckpt_path.name} acc={acc_ood*100:.2f}%", fontsize=12)
    fig.tight_layout()
    fig.savefig(figs_dir / "calib_hist_per_subject.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {figs_dir/'calib_hist_per_subject.png'}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, sig_name in zip(axes.flat, sig_order):
        s = sigs_ood[sig_name]
        ax.hist(s[correct_ood == 1], bins=40, alpha=0.55,
                label=f"correct (n={int((correct_ood==1).sum())})",
                color="#2ca02c", density=True)
        ax.hist(s[correct_ood == 0], bins=40, alpha=0.55,
                label=f"wrong (n={int((correct_ood==0).sum())})",
                color="#d62728", density=True)
        ax.set_title(f"{sig_name}  AUROC_correct={auroc_correct[sig_name]:.3f}")
        ax.set_xlabel(sig_name); ax.set_ylabel("density")
        ax.legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle("Confidence signals - correct vs incorrect on hold-out", fontsize=12)
    fig.tight_layout()
    fig.savefig(figs_dir / "calib_hist_correct_vs_wrong.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {figs_dir/'calib_hist_correct_vs_wrong.png'}")

    print("\n[PLOT] coverage-accuracy curves on hold-out ...")
    fig, ax = plt.subplots(figsize=(8, 6))
    for sig_name in ["margin", "sc_max", "sc_sum", "entropy_neg",
                     "hidden_active", "first_out_neg"]:
        cov, sel, thr = coverage_accuracy_curve(sigs_ood[sig_name], correct_ood)
        ax.plot(cov, sel, label=sig_name, alpha=0.85)
    ax.axhline(acc_ood, color="black", lw=0.8, ls=":", label=f"baseline {acc_ood*100:.1f}%")
    ax.axvline(args.target_coverage_min, color="grey", lw=0.6, ls="--",
               label=f"target cov >= {args.target_coverage_min:.2f}")
    ax.set_xlabel("Coverage (fraction accepted)")
    ax.set_ylabel("Selective accuracy on accepted samples")
    ax.set_title("Coverage-accuracy trade-off - hold-out 9660 samples")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs_dir / "calib_coverage_accuracy.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {figs_dir/'calib_coverage_accuracy.png'}")

    print("[PLOT] ROC for OOD detection ...")
    fig, ax = plt.subplots(figsize=(8, 6))
    for sig_name in ["margin", "sc_max", "sc_sum", "entropy_neg",
                     "hidden_active", "first_out_neg"]:
        fpr, tpr, _, auc = roc_auc(sigs_ood[sig_name], label_pos_oodtask)
        ax.plot(fpr, tpr, label=f"{sig_name} (AUC={auc:.3f})", alpha=0.85)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("FPR (b002/b007 mistakenly trusted)")
    ax.set_ylabel("TPR (b015 correctly trusted)")
    ax.set_title("OOD detection ROC - b015 (ID) vs b002+b007 (OOD)")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs_dir / "calib_roc_ood.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {figs_dir/'calib_roc_ood.png'}")

    print("\n[ANALYZE] threshold sweep on margin (HW-cheap recommended)")
    margin_id = sigs_id["margin"]
    margin_ood = sigs_ood["margin"]
    print(f"  margin range on ID:  [{int(margin_id.min())}, {int(margin_id.max())}]")
    print(f"  margin range on OOD: [{int(margin_ood.min())}, {int(margin_ood.max())}]")

    candidates = np.arange(0, int(max(margin_id.max(), margin_ood.max())) + 2)
    rows = []
    for tau in candidates:
        keep_id = margin_id >= tau
        keep_ood = margin_ood >= tau
        cov_id = float(keep_id.mean())
        cov_ood = float(keep_ood.mean())
        sel_acc_id = float((preds_id[keep_id] == y_id[keep_id]).mean()) if keep_id.sum() else 0.0
        sel_acc_ood = float((preds_ood[keep_ood] == y_ood[keep_ood]).mean()) if keep_ood.sum() else 0.0
        per_subj_cov = {}
        per_subj_acc = {}
        for s in sorted(set(sid_ood.tolist())):
            m = sid_ood == s
            keep_s = keep_ood & m
            per_subj_cov[sid_to_name[int(s)]] = float(keep_s.sum() / max(m.sum(), 1))
            per_subj_acc[sid_to_name[int(s)]] = (
                float((preds_ood[keep_s] == y_ood[keep_s]).mean()) if keep_s.sum() else 0.0
            )
        rows.append({
            "tau": int(tau),
            "cov_id": cov_id, "sel_acc_id": sel_acc_id,
            "cov_ood": cov_ood, "sel_acc_ood": sel_acc_ood,
            "per_subj_cov": per_subj_cov, "per_subj_acc": per_subj_acc,
        })

    print()
    print(f"  {'tau':>3} | {'covID':>6} {'selID':>6} | {'covOOD':>6} {'selOOD':>6} | "
          f"{'b015c':>5}/{'b015a':>5} | {'b007c':>5}/{'b007a':>5} | {'b002c':>5}/{'b002a':>5}")
    for r in rows[:30]:
        psc = r["per_subj_cov"]; psa = r["per_subj_acc"]
        print(f"  {r['tau']:>3} | {r['cov_id']:>6.3f} {r['sel_acc_id']:>6.3f} | "
              f"{r['cov_ood']:>6.3f} {r['sel_acc_ood']:>6.3f} | "
              f"{psc.get('b015',0):>5.2f}/{psa.get('b015',0):>5.2f} | "
              f"{psc.get('b007',0):>5.2f}/{psa.get('b007',0):>5.2f} | "
              f"{psc.get('b002',0):>5.2f}/{psa.get('b002',0):>5.2f}")

    feasible = [r for r in rows if r["sel_acc_ood"] >= 0.90 and r["cov_ood"] >= 0.50]
    if feasible:
        chosen = max(feasible, key=lambda r: r["cov_ood"])
        rule = "sel_acc_ood>=0.90 AND cov_ood>=0.50, max coverage"
    else:
        relax = [r for r in rows if r["cov_ood"] >= 0.30]
        if relax:
            chosen = max(relax, key=lambda r: r["sel_acc_ood"])
            rule = "RELAXED: cov_ood>=0.30, max sel_acc_ood"
        else:
            chosen = max(rows, key=lambda r: r["sel_acc_ood"])
            rule = "FALLBACK: max sel_acc_ood"

    print(f"\n[FINDING] chosen tau* = {chosen['tau']}")
    print(f"  rule: {rule}")
    print(f"  cov_ID  = {chosen['cov_id']*100:.2f}%   sel_acc_ID  = {chosen['sel_acc_id']*100:.2f}%")
    print(f"  cov_OOD = {chosen['cov_ood']*100:.2f}%   sel_acc_OOD = {chosen['sel_acc_ood']*100:.2f}%")
    for k, v in chosen["per_subj_cov"].items():
        print(f"  {k}: cov={v*100:.2f}%  sel_acc={chosen['per_subj_acc'][k]*100:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    subjs = list(chosen["per_subj_cov"].keys())
    cov_vals = [chosen["per_subj_cov"][s] * 100 for s in subjs]
    acc_vals = [chosen["per_subj_acc"][s] * 100 for s in subjs]
    base_acc = [per_subj[s]["acc"] * 100 for s in subjs]
    x = np.arange(len(subjs))
    w = 0.27
    ax.bar(x - w, base_acc, w, label="baseline acc (no abstention)", color="#aaaaaa")
    ax.bar(x,     cov_vals, w, label=f"coverage @ tau={chosen['tau']}", color="#1f77b4")
    ax.bar(x + w, acc_vals, w, label=f"selective acc @ tau={chosen['tau']}", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(subjs)
    ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.set_title(f"Per-subject calibration - signal=margin, tau*={chosen['tau']}")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figs_dir / "calib_per_subject_at_tau.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {figs_dir/'calib_per_subject_at_tau.png'}")

    summary = {
        "ckpt": str(ckpt_path),
        "id_data": str(id_path), "ood_data": str(ood_path),
        "leak_shift": args.leak_shift,
        "T": net["T"], "H": net["H"],
        "theta1_int": net["theta1_int"], "theta2_int": net["theta2_int"],
        "n_id": int(len(X_id)), "n_ood": int(len(X_ood)),
        "acc_id": acc_id, "acc_ood_baseline": acc_ood,
        "per_subject_baseline": per_subj,
        "auroc_ood_detection": auroc_table,
        "auroc_correct_vs_wrong": auroc_correct,
        "tau_recommended": chosen["tau"],
        "tau_rule": rule,
        "tau_metrics": {
            "cov_id": chosen["cov_id"], "sel_acc_id": chosen["sel_acc_id"],
            "cov_ood": chosen["cov_ood"], "sel_acc_ood": chosen["sel_acc_ood"],
            "per_subj_cov": chosen["per_subj_cov"],
            "per_subj_acc": chosen["per_subj_acc"],
        },
        "sweep": rows,
    }
    json_path = out_dir / "calibration_results.json"
    json_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[OUTPUT] wrote {json_path}")

    npz_path = out_dir / "calibration_telemetry.npz"
    np.savez(
        npz_path,
        id_preds=preds_id, id_y=y_id, id_correct=correct_id,
        ood_preds=preds_ood, ood_y=y_ood, ood_sid=sid_ood, ood_correct=correct_ood,
        **{f"id_sig_{k}": v for k, v in sigs_id.items()},
        **{f"ood_sig_{k}": v for k, v in sigs_ood.items()},
        ood_spike_counts=tel_ood["spike_counts"],
        id_spike_counts=tel_id["spike_counts"],
    )
    print(f"[OUTPUT] wrote {npz_path}")

    print("\n[LIMITATION]")
    print("  * Only 3 hold-out subjects (b002/b007/b015) - small sample for OOD claim.")
    print("  * Recommended tau is fit on hold-out; strict protocol re-calibrates on CV fold.")
    print("  * Selective-accuracy spec achievable largely because b015 is near-perfect.")


if __name__ == "__main__":
    main()

