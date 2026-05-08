# SCG-SNN-on-Anlogic-EG4S20

**多模态心震图 LIF 脉冲神经网络硬件加速器，国产 FPGA EG4S20 上的 zero-leakage gold-standard 部署 + 跨数据集泛化**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data: ODC-BY](https://img.shields.io/badge/Data-FOSTER%20OSF%2FCEBSDB%20ODC--BY-orange)](https://osf.io/3u6yb/)
[![Hardware: EG4S20](https://img.shields.io/badge/Hardware-Anlogic%20EG4S20BG256-green)](https://www.anlogic.com/)
[![Verilog: 100%25 hand-written](https://img.shields.io/badge/Verilog-100%25%20hand--written-purple)](rtl/)
[![Board acc: 95.02%25](https://img.shields.io/badge/board%20acc-95.02%25-brightgreen)](doc/SRTP_FINAL_REPORT.md)
[![Cross-dataset: 87.70%25](https://img.shields.io/badge/cross--dataset-87.70%25-blue)](doc/SRTP_FINAL_REPORT.md)

<p align="center">
  <img src="./assets/toilet.svg" alt="Toilet Badge for scg-snn-eg4s20" width="256" />
  <br>
  <sub>程序化 3D 马桶徽章，颜色由 commit SHA 决定 · 仓库更新后在
  <a href="../../actions/workflows/refresh-toilet-badge.yml">Actions</a> 手动刷新 ·
  生成器：<a href="https://github.com/dentar142/StoneBadge">dentar142/StoneBadge</a>
  （fork from <a href="https://github.com/professor-lee/StoneBadge">professor-lee/StoneBadge</a>）</sub>
</p>

---

## TL;DR

- **多模态 5-channel SCG** (PVDF / PZT / ACC / PCG / ERB) → BG / Sys / Dia 三分类
- 在 **Anlogic EG4S20 国产 FPGA**（19,600 LUT4 / 64 BRAM9K / 29 DSP18）上**手写 Verilog** 部署
- **zero-leakage subject-disjoint 板上 acc = 95.02 %**（40,575 hold-out windows，8 受试者，aligned + phase-shift bake）
- **跨数据集 0-shot acc = 78.07 %**（FOSTER → CEBSDB），**+30 s STDP 校准 → 87.70 %（反超 CEBSDB 5-fold 自训）**
- **资源**：LUT 10.76 % / BRAM9K 39 / DSP **1**，**推理 8.65 ms / 窗**
- 全套 ckpt（5.3 MB FP32 + 量化 hex）+ bit + bench JSON + 17 张图 + 1100 行报告 在仓库

完整技术细节：[**doc/SRTP_FINAL_REPORT.md**](doc/SRTP_FINAL_REPORT.md)

---

## 全部 commit 链（reverse chronological）

| Hash | 内容 |
|---|---|
| `7b29ef6` | un-ignore + push 全部 5.3 MB FP32 ckpt |
| `6228796` | **modality dropout 实现 FOSTER → CEBSDB 跨数据集 87.70 %** ⭐ |
| `fe79395` | phase-aligned SNN 闭合 CNN 反超的一半 |
| `710d0b4` | cross-domain SCG → PCG 单通道转移失败（被 6228796 修复）|
| `14bbbd2` | 5 项机制深析 + CNN baseline + STDP + ADC RTL |
| `b0072eb` | 🚽 toilet badge auto-commit |
| `d1153bd` | H=32 T=16 部署 + badge workflow |
| `379a384` | Pareto 扫描 + sparsity + amplitude |
| `684b862` | **zero-leakage gold-standard 多模态部署** |
| `ad8e5c5` | report §9 placeholder |
| `1852a4e` | report §9 honest negative result |
| `a176cf3` | revert W1 single-array, H=32 multimodal |
| `eb760c1` | RTL split W1 → 3 BRAM32K banks (failed) |
| `4d06dd3` | FOSTER 5-channel + CV + calibration |
| `5e3a937` | 5-class SNN 实验 |
| `0449d45` | per-subject OOD abstention |
| `49d81a5` | comprehensive technical README |
| `08fff9b` | initial commit (single-modal SNN baseline) |

---

## 核心结果总表

### A. 板上部署 SNN bit 全谱

| Bitstream | 训练 | 评估 | **板上 acc** | LUT | BRAM9K | DSP | Latency | 部署状态 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `scg_top_snn_singlemodal_backup.bit` | CEBSDB 16 sub single-modal | 9,660 hold-out | 77.72 % | 15.9 % | 18 | 1 | 7.88 ms | fallback |
| `scg_top_snn_multimodal.bit` (leaky) | FOSTER random win | 200 win (含训练人) | 98.00 %（无效）| 10.97 % | 39 | 1 | 8.66 ms | fallback |
| `scg_top_snn_multimodal_holdout.bit` | FOSTER 32 sub | 8 sub × 40,575 win | **94.14 %** | 10.70 % | 39 | 1 | 8.65 ms | fallback |
| `scg_top_snn_sweep_H32_T16.bit` | 同上, T=16 | 8 sub × 5,000 strat | 94.54 % | 10.70 % | 38 | 1 | 9.12 ms | fallback |
| **`scg_top_snn_aligned_h32t16.bit`** ⭐ | + Aligned (A+B) | 8 sub × 5,000 strat | **95.02 %** | 10.76 % | 39 | 1 | 9.12 ms | **当前烧录** |

### B. 跨数据集 (FOSTER 训练 → CEBSDB 11,601 win 评估)

| 模型 | 0-shot CEBSDB | + STDP per-subject | vs CEBSDB 自训 baseline (85.48 %) |
|---|---:|---:|---:|
| Random | 33.33 % | — | -52.15 |
| Aligned (无 dropout) | 43.19 % | 83.81 % | -1.67 |
| **Dropout-aligned** | **78.07 %** | **87.70 %** ⭐ | **+2.22** ⭐ |

→ **训练大库 + 部署小校准** 范式可行：CEBSDB 自训费力费数据，FOSTER 预训练 + 30 s STDP 反而更好。

### C. SNN vs CNN 公平比较 (FOSTER 同 hold-out)

| | SNN H=32 T=16 (Aligned) | CNN match (32K) | Δ |
|---|---:|---:|---:|
| Params | 41,061 | 32,201 | CNN -22 % |
| Sim val acc | 94.81 % | **96.04 %** | CNN +1.23 |
| Macro-F1 | 92.58 % | 94.86 % | CNN +2.28 |
| Dia F1 | 85.83 % | 89.80 % | CNN +3.97 |
| FPGA 部署 | ✅ 1 DSP, 9.12 ms | ❌ DSP-bound, 估 20+ ms | SNN 2-5× |
| 跨数据集 0-shot | **78.07 %** (Dropout 版) | 未测 | SNN 已证 |

诚实结论：**CEBSDB 单模态低 SNR**——SNN 赢；**FOSTER 多模态丰富信号**——CNN raw 精度赢；**FPGA 维度（延迟、DSP、稀疏、可校准）**——SNN 强。

### D. Pareto 扫描（§9.6）

| H | T | Sim acc | LUT | BRAM9K | Sparsity | 综合 |
|---|---:|---:|---:|---:|---:|---|
| 16 | 32 | 93.68 % | 7.0 % (1,377) | 21 | 75.0 % | ✅ 最小 |
| 32 | 8 | 94.20 % | 10.7 % | 39 | 70.1 % | ✅ 最快 |
| **32 | 16** | **94.43 %** | 10.7 % | 38 | 70.5 % | ✅ **deployed** |
| 32 | 32 | 94.26 % | 10.7 % | 39 | 68.4 % | ✅ |
| 32 | 48 | 94.38 % | 10.7 % | 39 | 69.3 % | ✅ |
| 64 | 32 | 94.33 % | — | — | 64.6 % | ❌ **PHY-9009 (3.3× MSlice)** |

→ EG4S20 容量边界 **= H ≤ 32**；T 整合 > 16 饱和。

### E. 机制分析合集

| 章 | 主题 | 核心数字 |
|---|---|---|
| §11.1 | Dia 错分溯源 | sub021 仅 62.6 % Dia recall（+Sys 26 %）|
| §11.2 | Margin-based abstention | τ=2 → kept 89.25 %, acc 97.78 % (+3.35 pp lift) |
| §11.3 | 难受试者特征 | ACC/ERB SNR -0.50 反相关（运动 artifact）|
| §11.4 | CNN vs SNN | regime-conditional, 见上 |
| §11.5 | STDP per-subject | sub021 +3.95 pp，最难受试者最受益 |
| §11.6 | ADC-direct RTL 骨架 | 1 kHz/通道, ~150 µs ADC + 1.8 ms LIF, ~50 % 余量 |
| §11.7 | Phase alignment (A+B) | τ=[4,5,5,6,13]，板上 +0.48 pp |
| §11.8 | **Modality Dropout 跨数据集** | **0-shot 78 % / + STDP 87.70 % > CEBSDB 自训 85.48 %** ⭐ |
| §11.9 | Cross-domain SCG → PCG | 单通道转移失败（被 §11.8 修复）|

---

## 系统架构

```
┌──────────────┐  UART/USB    ┌─────────────────────────────────────┐  UART  ┌────────┐
│  Host PC     │ ──1280 B───→ │  EG4S20 FPGA  (50 MHz crystal)      │ ──1B─→ │  Host  │
│  (Python)    │              │                                     │        │        │
│              │              │  scg_top_snn  (top.v)               │        │        │
│  bench /     │              │   ├─ uart_rx → input buffer         │        │        │
│  inference   │              │   │   (X_BUF: 5 BRAM9K, 256 B/ch)   │        │        │
│              │              │   ├─ scg_snn_engine  (engine.v)     │        │        │
│              │              │   │   ├─ FC1: 1280→32 LIF           │        │        │
│              │              │   │   │   W1 = 5 channel-banks      │        │        │
│              │              │   │   │   (5 × 8 KB BRAM9K, 39 tot) │        │        │
│              │              │   │   ├─ T=16 spike accumulation    │        │        │
│              │              │   │   ├─ FC2: 32→3                  │        │        │
│              │              │   │   │   W2 = 96 INT8 in 1 BRAM9K  │        │        │
│              │              │   │   └─ argmax → 1 byte class      │        │        │
│              │              │   └─ uart_tx                        │        │        │
│              │              └─────────────────────────────────────┘        │        │
└──────────────┘                                                              └────────┘
```

**关键 RTL 创新（§9）**：

1. **Channel-bank ROM 重组** —— 突破 Anlogic TD 对 ≥ 40 KB 单一 ROM 数组的 BRAM 推断阈值。把 W1 (40 KB) 切成 5 个 8 KB 子 ROM（per-modality），每个独立触发 BRAM9K 推断，资源占用从 41,214 LUT (210 % 超限) → **2,098 LUT (10.7 %)**。
2. **Aligned τ 烘进 W1** —— 学到的 per-channel 时间偏移 τ_int = [4, 5, 5, 6, 13] 直接作为 W1 列置换 baked into hex，**RTL 完全不变**。
3. **Soft reset LIF** —— `v <- v - s * theta` 替代 hard reset，避免负无穷漂移。

---

## 数据集

| 数据集 | 受试者 | 模态 | 用途 |
|---|---|---|---|
| **FOSTER** ([OSF: 3u6yb](https://osf.io/3u6yb/)) | 40 | PVDF / PZT / ACC / PCG / ERB（同步 ECG 标签）| **当前部署训练** |
| **CEBSDB** ([PhysioNet ODC-BY](https://physionet.org/content/cebsdb/1.0.0/)) | 19 | 单 SCG (~ACC) + ECG | 跨数据集评估目标 |
| **PhysioNet 2016 PCG** ([PhysioNet](https://physionet.org/content/challenge-2016/1.0.0/)) | 3,239 records | PCG only | 跨域负面结果（§11.9）|

预处理（`model/dataset_pipeline_foster.py`）：10 kHz → 1 kHz 抗混叠下采，5–50 Hz 带通滤波，ECG R-peak 检测 (Pan-Tompkins)，Sys = R+50 ms ±30，Dia = R+350 ms ±30，BG_EXCLUSION=100 ms，per-window int8 z-score 归一化。

---

## 复现命令

### 仅 sim（用仓库已有 ckpt + npz，无需重训）

```bash
git clone https://github.com/dentar142/scg-snn-eg4s20.git
cd scg-snn-eg4s20

# 准备 FOSTER + CEBSDB（训练时用过的 npz 不在仓库，需重生成）
python tools/dl_curl_parallel.py cebs_mp                  # CEBSDB
python tools/dl_foster_osf.py                              # FOSTER (~11 GB)
python model/dataset_pipeline.py --out data_excl100 --bg-exclusion-ms 100
python model/dataset_pipeline_foster.py --out data_foster_multi --bg-exclusion-ms 100

# 任选已 train 好的 ckpt 跑分析
python tools/calibrate_abstention.py --ckpt model/ckpt/sweep/best_sweep_H32_T16.pt \
    --data data_foster_multi/all.npz \
    --holdout sub003 sub006 sub009 sub013 sub020 sub021 sub024 sub026
python tools/eval_cross_dataset.py \
    --aligned-ckpt model/ckpt/best_snn_mm_h32t16_aligned.pt \
    --dropout-ckpt model/ckpt/best_snn_mm_h32t16_dropout.pt \
    --cebs-data data_excl100/val.npz
```

### 重训（GPU，~17 min/config）

```bash
python model/train_snn_mm_aligned.py --data data_foster_multi \
    --holdout sub003 sub006 sub009 sub013 sub020 sub021 sub024 sub026 \
    --epochs 50 --H 32 --T 16 --tag snn_mm_h32t16_aligned --shift-max 15
```

### FPGA 部署（Anlogic TD 6.2.190657 + JTAG）

```bash
# 1. 量化 + 烘 τ + 拆通道 bank（替换 W1.hex）
python tools/export_aligned_weights.py --ckpt model/ckpt/best_snn_mm_h32t16_aligned.pt

# 2. 综合
python tools/synth_one_config.py --ckpt model/ckpt/best_snn_mm_h32t16_aligned.pt \
    --bit-name scg_top_snn_aligned_h32t16

# 3. 烧录 (HX4S20C 板)
"D:\Anlogic\TD_Release_2026.1_6.2.190.657\bin\bw_commands_prompt.exe" tools/download_jtag_snn.tcl

# 4. 板上 bench
python tools/bench_fpga_snn_holdout.py --port COM28 --data data_foster_multi/all.npz \
    --holdout sub003 sub006 sub009 sub013 sub020 sub021 sub024 sub026 --n 5000 \
    --out doc/bench_aligned.json
```

---

## 文件清单

```
scg-snn-eg4s20/
├── rtl/                                # 100% 手写 Verilog
│   ├── scg_top_snn.v                   # 顶层 (UART + 5-bank ROM + 引擎)
│   ├── scg_snn_engine.v                # LIF 引擎 (FSM + soft reset)
│   ├── scg_adc_stream.v                # 实时 ADC 流式骨架 (§11.6)
│   └── weights_snn/                    # INT8 量化权重 (烘 τ aligned 版本)
│       ├── W1.hex (40 KB, H=32×1280)
│       ├── W1_ch{0..4}.hex (5 × 8 KB channel bank)
│       ├── W2.hex (96 B)
│       └── meta.json (theta1/2, scale, tau_int_baked)
│
├── model/                              # 训练 + 量化导出
│   ├── train_snn_multimodal.py         # 基础多模态 SNN
│   ├── train_snn_mm_holdout.py         # subject-disjoint 训练
│   ├── train_snn_mm_aligned.py         # + 通道 shift augment + 学 τ
│   ├── train_snn_mm_dropout.py         # + modality dropout (跨数据集鲁棒)
│   ├── train_cnn_mm_holdout.py         # CNN baseline
│   ├── export_snn_weights.py           # FP32 → INT8 + RTL 参数 patch
│   ├── dataset_pipeline_foster.py      # 5-channel R-peak 标签生成
│   └── ckpt/                           # 33 个 .pt FP32 ckpt + 11 manifest
│
├── tools/                              # 分析、bench、综合脚本
│   ├── bench_fpga_snn.py / _holdout.py # UART bench
│   ├── synth_one_config.py             # 单 config 自动综合
│   ├── sweep_pareto.py + sweep_synth.py # Pareto 矩阵
│   ├── analyze_dia_errors.py           # §11.1
│   ├── calibrate_abstention.py         # §11.2
│   ├── analyze_subjects.py             # §11.3
│   ├── stdp_personalize.py             # §11.5
│   ├── export_aligned_weights.py       # §11.7 τ-bake
│   ├── eval_cross_dataset.py           # §11.8 真跨数据集
│   ├── cross_domain_pcg.py             # §11.9 单通道跨域（负面）
│   ├── plot_pareto.py                  # 5 张 Pareto 图
│   ├── split_w1_channels.py            # W1 → 5 banks
│   ├── dl_*.py                         # 数据下载 (curl/osfclient/proxy)
│   └── *.tcl                           # JTAG 下载、综合、功耗探针
│
├── constraints/scg_top.adc             # HX4S20C 板引脚约束
│
├── doc/
│   ├── SRTP_FINAL_REPORT.md            # 1100 行完整报告（§0–§14, 附录 A-C）
│   ├── bench_*.json                    # 全部 bench 结果（含每窗预测）
│   ├── synth_*.json                    # 各 config 综合资源
│   ├── sweep_pareto.json               # 6-config 扫描
│   ├── cross_dataset_cebsdb.json       # FOSTER → CEBSDB
│   ├── stdp_personalize*.json          # STDP per-subject
│   ├── abstention_h32_t16.json         # margin abstention
│   ├── subject_difficulty.{md,json}    # 难受试者特征
│   ├── dia_error_summary.md            # Dia 类错分
│   └── figs/*.png                      # 17 张分析图
│
├── build_snn/                          # 5 个部署 bit + 综合报告
│   ├── scg_top_snn_aligned_h32t16.bit  # 当前烧录 (95.02 % 板上)
│   ├── scg_top_snn_multimodal_holdout.bit (94.14 % 全测)
│   ├── scg_top_snn_sweep_H{16,32}_T*.bit (Pareto 角点)
│   └── scg_top_snn_singlemodal_backup.bit (CEBSDB 单模态 fallback)
│
├── assets/toilet.svg                   # 🚽 自动生成徽章
└── .github/workflows/refresh-toilet-badge.yml
```

---

## 已知限制

1. **静态功耗劣势**：EG4S20 55 nm vs 论文 iCE40UP5K 40 nm ULP（80 mW vs 8.55 mW），**物理硬约束**
2. **功耗未实测**：Anlogic TD `calculate_power` 在 EG4 上 coredump，且 power library 未随 TD 6.2.190657 发行；datasheet 估算仅
3. **CNN raw 精度反超**：FOSTER 多模态 CNN +1.6 pp（§11.4）；SNN 真正优势在 FPGA 维度（延迟、DSP、稀疏、可校准）
4. **跨域 PCG 单通道转移失败**：§11.9 已证 FOSTER PCG 通道与其他 4 个机械模态深度耦合；Modality Dropout（§11.8）是修复方案
5. **dropout-aligned 未实际烧板**：sim 78 % 0-shot CEBSDB 准确率未在 FPGA 上 bit-exact 验证（待外接 ADC 或额外 bench）

---

## 引用

```bibtex
@misc{scg_snn_eg4s20_2026,
  title  = {Multi-Modal SCG SNN on Anlogic EG4S20: Phase Alignment, Pareto
            Frontier, and Cross-Dataset Generalization via Modality Dropout},
  author = {Neko},
  year   = 2026,
  url    = {https://github.com/dentar142/scg-snn-eg4s20},
  note   = {SRTP final report; board acc 95.02% on FOSTER subject-disjoint
            hold-out; 0-shot 78.07% on CEBSDB cross-dataset}
}
```

对标论文（不同范式 + 不同 FPGA）：

```bibtex
@inproceedings{rahman2026scgcnn,
  title  = {At the Edge of the Heart: Real-time SCG Classification on iCE40UP5K},
  author = {Rahman, et al.},
  year   = 2026,
  booktitle = {DCOSS-IoT}
}
```

数据集致谢：FOSTER (OSF:3u6yb)；CEBSDB (PhysioNet, ODC-BY 1.0)；PhysioNet/CinC Challenge 2016。

---

**License**：MIT（代码）/ ODC-BY 1.0（CEBSDB 衍生数据）/ FOSTER 原 license（FOSTER 衍生数据）
