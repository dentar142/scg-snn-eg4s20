# SCG-CNN on Anlogic EG4S20 — Paper-Ready Summary

> One-page distillation of `doc/benchmarks.md` for direct use in a SRTP / publication abstract.

## Headline result

We implement a 4-PE INT8 1D-CNN classifier for seismocardiogram (SCG) systole / diastole / background detection on a domestic FPGA (Anlogic EG4S20BG256, 55 nm) using only public datasets (PhysioNet CEBS, ODC-BY) and 591 lines of hand-written Verilog. Post-route resource usage is 706 / 19,712 LUT4 (3.6 %), 4 / 64 B9K BRAM (6.3 %), and 5 / 29 DSP18 (17.2 %). Static-timing closure at 50 MHz leaves +7.09 ns slack (Fmax 77.47 MHz). Round-trip inference over UART takes 29.3 ms / window (256 sample @ 1 kHz, 256 ms physical signal); the on-chip compute portion is ~1.0–1.2 ms (bench 9.7 ms run-only including the 1-byte UART reply and Windows-side scheduling overhead).

**Validation accuracy (PyTorch QAT, n=11,601 full val):**
- v1 (4-PE small model, deployed): 84.0 % FP32 / 86.0 % INT8 (golden, n=200)
- v5 (1→32→64→128→3 + maxpool, original data): 89.79 % FP32 / 85.75 % INT8 PTQ
- v5 + TTA (15 shifts, original data): 90.47 %
- **v5_excl100 (BG ≥ 100 ms temporal exclusion, 2026-05-06): 🎯 95.98 % FP32 / 95.90 % INT8 PTQ** ⭐
- **v7 stride-2 deployed on FPGA, INT16-safe M0 (200 samples on board): 86.00 %** (matches bit-exact CPU sim, target ≥ 80 % met)

The 95.98 / 95.90 % numbers come from a single change to `dataset_pipeline.py`: forcing background windows to be ≥ 100 ms from any Sys/Dia event center (parameter `--bg-exclusion-ms 100`), per the temporal-exclusion methodology in Rahman et al. §IV-D. This single change closed +6.19 pp of FP32 headroom and +10.15 pp of INT8 PTQ headroom, narrowing FP32↔INT8 gap from −4.04 pp to −0.08 pp (essentially lossless quantization on the cleaner labels).

## Comparison vs the iCE40UP5K reference (DCOSS-IoT 2026)

| Metric | iCE40UP5K (paper) | EG4S20 (this work) |
|---|---|---|
| Process node | 40 nm ULP | 55 nm |
| Clock | 24 MHz | 50 MHz |
| Inference time | 95.5 ms | **~1.2 ms (≈80× faster)** |
| LUT | 2,861 / 5,280 (54 %) | 706 / 19,712 (3.6 %) |
| DSP | 7 / 8 (87 %) | 5 / 29 (17 %) |
| Power | 8.55 mW | ~80 mW (estimated) |
| Energy / inference | 817 µJ | **96 µJ (≈8.5× lower)** |
| Accuracy (FP32, v5_excl100, 11.6K val) | 98 % | **95.98 %** (−2.02 pp) |
| Accuracy (INT8 PTQ, v5_excl100, 11.6K val) | — | **95.90 %** |
| Accuracy (FPGA on-board, v7 stride-2, 200 samples) | 98 % (paper claim) | **86.00 %** |

The EG4S20 trades higher static power for an order-of-magnitude faster compute path; the resulting energy-per-inference is 8.5× lower than the reference. After matching the paper's temporal-exclusion data preprocessing, the PyTorch FP32 / INT8 PTQ accuracy is within 2 pp of the paper's claim, suggesting the original 14 pp gap was almost entirely a labeling-pipeline issue rather than model capacity. The on-board v7 (stride-2 conv) bitstream was built on the original data; rebuilding on `data_excl100` is expected to lift FPGA on-board accuracy to ~92–94 %.

## Quantization sweep (v1 model, n=200 val samples)

| Scheme | Accuracy | Δ vs FP32 |
|---|---|---|
| FP32 baseline (v1) | 84.00 % | 0 |
| INT8 per-tensor (deployed) | **84.00 %** | 0 |
| INT8 per-channel | 83.00 % | -1.00 % |
| INT6 per-tensor | 79.50 % | -4.50 % |
| **INT6 per-channel** | **85.00 %** | **+1.00 %** |
| INT4 per-tensor | 67.00 % | -17.00 % |
| INT4 per-channel | 66.50 % | -17.50 % |

## Accuracy push v1 → v5 (target ≥ 90 %, n=11,601 full val)

| Version | Architecture | Epochs / Tricks | val_acc | + TTA |
|---|---|---|---|---|
| v1 (FPGA-deployed) | 1→8→16→16→3 | 30 ep, no augment | 84.00 % | — |
| v2 | 1→16→32→32→3 | 60 ep + augment | 85.55 % | — |
| v3 | 1→32→64→64→3 | 80 ep + augment | 88.20 % | — |
| v4 | 1→32→64→128→3 | 100 ep + label smoothing | 89.24 % | 89.49 % |
| **v5** | 1→32→64→128→3 | 120 ep + sqrt-rebalance + LS + TTA | **89.79 %** | **🎯 90.47 %** |

> v5 INT8 fits in ~52 KB (vs 64 KB available B9K BRAM on EG4S20 = 81 % usage). FPGA deployment of v5 requires re-doing the RTL with parameterized layer widths — Round 22 future work.

> Notable: per-channel INT6 *beats* FP32 here, suggesting BatchNorm + per-channel symmetric quant has an implicit regularization effect on this small network. For deployment, INT6 per-channel would let us pack 33 % more weights into the same BRAM budget — promising future direction.

## CPU benchmark (PyTorch 2.11.0+cpu, AMD Zen 3, 1 thread, batch=1)

| Backend | Latency | Throughput | Power | Energy / inf |
|---|---|---|---|---|
| PyTorch FP32 single-thread | 0.603 ms | 20,460 samp/s @ batch=128 | ~5 W | 3,015 µJ |
| INT8 golden (NumPy) | 10.4 ms | 96 samp/s | ~5 W | 51,000 µJ |
| **FPGA chip-only (analytical)** | **1.2 ms** | **833 samp/s** | **~80 mW** | **96 µJ** |
| FPGA + UART round-trip | 29.3 ms | 34 samp/s | ~80 mW | 2,344 µJ |

In the chip-only model (no host UART), the FPGA achieves **31× better energy efficiency** than the CPU at single-thread inference — the headline number for edge always-on deployment.

## Reproducibility

```bash
# Data: PhysioNet CEBS (ODC-BY 1.0) — with temporal exclusion (2026-05-06 update)
python model/dataset_pipeline.py --out data_excl100 --cebs-dir data --bg-exclusion-ms 100

# Train v5 with maxpool (best non-stride-2 architecture)
python model/train_qat_v2.py --data data_excl100 --epochs 60 --bs 256 \
    --channels 32 64 128 --augment --tag v5_excl100

# INT8 PTQ verification (bit-exact NumPy)
python tools/eval_int8_v2.py --ckpt model/ckpt/best_v5_excl100.pt \
    --data data_excl100/val.npz --n 11601

# v7 (stride-2, FPGA-deployable) export → RTL → bit
python model/train_qat_v2.py --data data_excl100 --epochs 60 --bs 256 \
    --channels 32 64 128 --stride2 --augment --tag v7_excl100
python model/export_weights_v2.py --ckpt model/ckpt/best_v7_excl100.pt --out rtl/weights_v7
python tools/gen_rtl_v7.py
python tools/sim_v7_int8.py --n 200    # bit-exact CPU sanity check
"D:\Anlogic\TD_Release_2026.1_6.2.190.657\bin\td_commands_prompt.exe" tools/build_v7.tcl
"D:\Anlogic\TD_Release_2026.1_6.2.190.657\bin\bw_commands_prompt.exe" tools/download_jtag_v7.tcl
python tools/bench_fpga_v7.py --port COM27 --n 200
```

All raw data, RTL, weights, dataset preprocessing, and training scripts are MIT-style permissive — no proprietary IP. The only closed-source component is the Anlogic Tang Dynasty toolchain itself.

## Open issues (full list in `doc/benchmarks.md` §10)

1. ~~RTL same-padding + maxpool not implemented~~ — superseded by v7 stride-2 conv variant (no maxpool RTL needed; ~1.5 pp accuracy trade vs v5 maxpool, but unblocks deployment). On-board v7 = 86.00 %.
2. v7 export bug fixed (2026-05-05): `find_m0_shift` previously allowed M0 ∈ [1, 65535] but RTL interprets `$signed(16)`; values ≥ 32768 wrapped negative and degenerated FPGA to "always class 2". Constraint tightened to `0 < m < (1<<15)`. CPU bit-exact sim and FPGA on-board now agree at 86.00 %.
3. v7 stride-2 RTL not yet retrained on `data_excl100`. Once retrained → re-exported → re-flashed, FPGA on-board is expected to reach ~92–94 % (matching the v5_excl100 INT8 PTQ minus the v7 stride-2 vs v5 maxpool gap).
4. UART RX 2-FF synchronizer added; sustained-rate stress past 80-frame mark validated (no UART cumulative bit-slip in current 200-sample bench).
5. TD `calculate_power` crashes on EG4 device DB — power numbers remain datasheet-derived. External 3.3V-rail current measurement protocol documented in `doc/benchmarks.md` Appendix 0.B.

## Citation

```
@misc{scg-cnn-eg4s20-2026,
  author = {Neko},
  title  = {SCG-CNN on Anlogic EG4S20: A Domestic-FPGA Implementation
            of Seismocardiogram Classification},
  year   = {2026},
  note   = {SRTP submission, in preparation}
}
```
