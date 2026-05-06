# SCG-SNN-on-Anlogic-EG4S20

**心震图（SCG）三分类 SNN 加速器在国产 FPGA 上的实现**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data: ODC-BY](https://img.shields.io/badge/Data-PhysioNet%20CEBSDB%20ODC--BY-orange)](https://physionet.org/content/cebsdb/1.0.0/)
[![Hardware: EG4S20](https://img.shields.io/badge/Hardware-Anlogic%20EG4S20BG256-green)](https://www.anlogic.com/)

---

## 1. 项目一览

在国产 Anlogic EG4S20BG256 FPGA 上用手写 Verilog 实现了 256→64→3 LIF 脉冲神经网络（SNN），完成心震图（SCG）信号的 BG/Sys/Dia 三分类。**与 SOTA CNN 论文（Rahman et al. DCOSS-IoT 2026, iCE40UP5K）对标**：在严格 subject-disjoint 评估下，SNN 实现 12× 推理加速、7× DSP 资源节省、subject-disjoint hold-out 精度 77.72 %。

| 关键指标 | 本工程 SNN | 对比 |
|---|---|---|
| 推理时延 | **7.88 ms** | vs CNN 论文 95.5 ms (12× 更快) |
| DSP 占用 | **1 / 29 (3.5 %)** | vs CNN 论文 7/8 (87 %, 7× 更少) |
| LUT 占用 | 3,121 / 19,600 (15.9 %) | vs CNN 12,387 (47.3 pp 更少) |
| 模型大小 | **16.6 KB** INT8 | vs CNN 论文 ~28 KB (省 41 %) |
| 5-fold CV 精度 | **85.48 ± 2.02 %** | 论文未做严格评估 |
| Hold-out (3 subj) | **77.72 %** | 论文未报 |
| 单次推理能耗 | ~630 µJ | vs 论文 817 µJ (-23 %) |

详细技术报告见 [`doc/SRTP_FINAL_REPORT.md`](doc/SRTP_FINAL_REPORT.md)（636 行）。

## 2. 硬件平台

| 项 | 规格 |
|---|---|
| 开发板 | HX4S20C 比赛板（康芯科技）|
| FPGA | **Anlogic EG4S20BG256**（55 nm SRAM-based）|
| 资源总量 | 19,600 LUT4 / 64 BRAM9K / 16 BRAM32K / 29 DSP18 |
| 时钟 | 50 MHz on-board crystal |
| 接口 | UART (115200 8N1, COM27) + JTAG |
| 工具链 | Anlogic Tang Dynasty (TD) v6.2.x |

## 3. 项目目录

```
.
├── README.md                              ← 本文件
├── doc/                                   ← 完整技术文档
│   ├── SRTP_FINAL_REPORT.md               ← 详细 SRTP 终结报告 ⭐
│   ├── benchmarks.md                      ← benchmark 全集
│   ├── paper_summary.md                   ← 论文级摘要
│   └── *.json                             ← 原始 benchmark 数据
├── model/                                 ← PyTorch 训练
│   ├── dataset_pipeline.py                ← 数据预处理 + temporal exclusion
│   ├── train_snn_v1.py                    ← SNN BPTT 训练 ⭐
│   ├── train_qat_v2.py                    ← CNN QAT baseline
│   ├── pretrain_ssl.py                    ← SimCLR SSL pretraining (探索性)
│   ├── finetune_ssl.py                    ← SSL fine-tune CV
│   ├── export_snn_weights.py              ← SNN INT8 权重导出
│   └── export_weights_v2.py               ← CNN INT8 权重导出
├── tools/                                 ← Python 工具链
│   ├── sim_snn.py                         ← SNN 比特级 CPU 模拟器 ⭐
│   ├── sim_v7_int8.py                     ← CNN 比特级 CPU 模拟器
│   ├── cross_val.py                       ← K-fold subject-disjoint CV
│   ├── final_holdout_test.py              ← Hold-out test runner
│   ├── make_holdout_npz.py                ← Hold-out subset 生成
│   ├── bench_fpga_snn.py                  ← FPGA UART benchmark (SNN)
│   ├── bench_fpga_v7.py                   ← FPGA UART benchmark (CNN)
│   ├── gen_rtl_v7.py                      ← CNN RTL 自动生成
│   ├── build_snn.tcl                      ← TD synthesis (SNN)
│   ├── build_v7.tcl                       ← TD synthesis (CNN)
│   ├── download_jtag_*.tcl                ← JTAG flash scripts
│   ├── dl_curl_parallel.py                ← PhysioNet 并行下载
│   └── ...                                ← 数据集相关工具
├── rtl/                                   ← FPGA RTL（手写 Verilog）
│   ├── scg_top_snn.v                      ← SNN 顶层 ⭐
│   ├── scg_snn_engine.v                   ← SNN 推理引擎（12-state FSM）⭐
│   ├── scg_top_v7.v                       ← CNN 顶层
│   ├── scg_mac_array_v7.v                 ← CNN 引擎（gen_rtl 生成）
│   ├── weights_snn/                       ← SNN INT8 权重 (W1.hex, W2.hex, meta.json)
│   └── weights_v7/                        ← CNN INT8 权重
├── constraints/                           ← TD .adc 引脚约束
│   └── scg_top.adc
├── build_v7/scg_top_v7.bit                ← 预编译 CNN 比特流（629 KB）
└── build_snn/scg_top_snn.bit              ← 预编译 SNN 比特流（649 KB）
```

> **数据集** `data/`、`data_excl100/`、`data_mixed/` 已在 `.gitignore` 排除——通过 `tools/dl_curl_parallel.py` 重新下载即可（见 §4）。

## 4. 完整复现 pipeline

### 环境
```
Python 3.11 + PyTorch 2.5+ (CUDA 12.4 推荐) + numpy + scipy + wfdb + pyserial
Anlogic Tang Dynasty 6.2.x（厂商提供）
50 MHz 板载晶振 + UART (115200 8N1)
```

### 步骤 0：克隆与基础环境
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
conda create -n scggpu python=3.11 -y
conda activate scggpu
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
pip install numpy scipy wfdb pyserial
```

### 步骤 1：下载 PhysioNet CEBSDB
```bash
python tools/dl_curl_parallel.py cebs_mp     # 60 records, ~360 MB
# 如果代理超时，可切换备用：
# python tools/dl_retry_failed.py --proxy http://your-proxy:port cebs_mp
```

### 步骤 2：数据预处理（temporal exclusion）
```bash
python model/dataset_pipeline.py --out data_excl100 --cebs-dir data --bg-exclusion-ms 100
# 输出: data_excl100/{train,val,all,holdout}.npz
```

### 步骤 3：训练 SNN（GPU 推荐，~5 min）
```bash
python model/train_snn_v1.py --data data_excl100 --epochs 60 --bs 256 --T 32 --H 64 --tag snn_v1
# 输出: model/ckpt/best_snn_v1.pt
```

### 步骤 4：5-fold subject-disjoint CV（GPU ~7 min）
```bash
python tools/cross_val.py --data data_excl100/all.npz --out doc/cv_snn.json --model snn --folds 5 --epochs 30
# 预期: mean = 85.48 ± 2.02 %
```

### 步骤 5：Hold-out 终测（最严格部署精度）
```bash
python tools/make_holdout_npz.py     # 生成 data_excl100/holdout.npz
python tools/final_holdout_test.py --model snn --epochs 60 --out doc/final_holdout_snn.json
# 预期: best HOLDOUT acc = 78.10 %
```

### 步骤 6：导出 INT8 权重 + 比特级 CPU sim 验证
```bash
python model/export_snn_weights.py --ckpt model/ckpt/best_snn_v1.pt --out rtl/weights_snn
python tools/sim_snn.py --ckpt model/ckpt/best_snn_v1.pt --data data_excl100/val.npz --n 11601 --leak-shift 4
# 预期 INT8 sim acc = 97.76 %
```

### 步骤 7：FPGA 综合（TD ~5 min）
```bash
"D:\Anlogic\TD_Release_2026.1_6.2.190.657\bin\td_commands_prompt.exe" tools/build_snn.tcl
# 输出: build_snn/scg_top_snn.bit (~649 KB)
```

### 步骤 8：JTAG 烧录
```bash
"D:\Anlogic\TD_Release_2026.1_6.2.190.657\bin\bw_commands_prompt.exe" tools/download_jtag_snn.tcl
```

### 步骤 9：板上 bench
```bash
# 工程一致性测试（200 random val sample）
python tools/bench_fpga_snn.py --port COM27 --n 200 --data data_excl100/val.npz
# 预期: acc = 96.50 %, run-only = 7.88 ms

# 论文级 hold-out 测试（9660 unseen sample, ~5 min）
python tools/bench_fpga_snn.py --port COM27 --n 9660 --data data_excl100/holdout.npz \
    --out doc/bench_fpga_snn_holdout_full.json
# 预期: acc = 77.72 %, FPGA = sim 比特级一致
```

## 5. 主要实验结果

### 5.1 三方 5-fold subject-disjoint CV
| 模型 | mean ± std | range | gap |
|---|---|---|---|
| **SNN cold-start** ⭐ | **85.48 ± 2.02 %** | 82.60–87.48 | 10.90 pp |
| CNN cold-start | 79.68 ± 3.46 % | 74.27–82.72 | 16.32 pp |
| CNN + SSL (19 sub CEBS) | 79.88 ± 4.91 % | 72.76–85.74 | 14.77 pp |
| CNN + SSL (93 sub mixed) | 78.24 ± 7.30 % | 65.63–83.55 | 15.24 pp |

### 5.2 Hold-out（3 unseen subjects, 9660 samples）
| 类 | precision | recall | F1 |
|---|---|---|---|
| BG | 78.6 % | 97.6 % | 87.1 % |
| Sys | 92.3 % | 40.6 % | 56.4 % |
| Dia | 62.9 % | 44.2 % | 51.9 % |
| **macro** | 77.9 % | 60.8 % | **65.1 %** |
| **overall acc** | — | — | **77.72 %** |

### 5.3 Per-subject 双峰分布（重要发现）
| Subject | overall acc | macro-F1 |
|---|---|---|
| **b015** | **98.77 %** ⭐ | **98.24 %** （超过原论文 97.70 %）|
| b007 | 68.63 % | 36.69 % |
| b002 | 65.45 % | 38.94 % |

→ SNN 对部分新病人（b015）几乎完美泛化，对另一些（b002/b007）几乎不工作。**临床部署需要"病人适配性筛查"**。

### 5.4 FPGA on-board verification
- 9660 / 9660 samples valid，**FPGA = CPU sim 完全一致**（误差 0）
- run-only **7.88 ms / sample**（127 inferences/sec）
- LUT 15.9 %, BRAM 28 %, **DSP 仅 3.5 %**

## 6. 与原论文对比

| 维度 | 本工程 (SNN, EG4S20) | Rahman et al. (CNN, iCE40UP5K) | 胜负 |
|---|---|---|---|
| 工艺节点 | 55 nm | 40 nm ULP | 论文 |
| 时钟 | 50 MHz | 24 MHz | +2× 本工程 |
| **DSP 占用** | **3.5 %** | **87 %** | **本工程 7×** |
| **推理时延** | **7.88 ms** | **95.5 ms** | **本工程 12×** |
| 模型大小 | 16.6 KB | ~28 KB | 本工程 -41 % |
| 静态功耗 | ~80 mW (估) | 8.55 mW | 论文（工艺差距）|
| **每次推理能耗** | **~630 µJ** | 817 µJ | **本工程 -23 %** |
| 评估严格度 | **CV + hold-out** | random-shuffle val/test | **本工程严格** |

## 7. 引用

```bibtex
@misc{scg-snn-eg4s20-2026,
  author = {Neko},
  title  = {SNN-Based SCG Classification on Domestic FPGA EG4S20:
            A Subject-Disjoint Evaluation Study},
  year   = {2026},
  howpublished = {\url{https://github.com/<your-github>/<repo-name>}}
}
```

## 8. 致谢与许可

- **数据**：[PhysioNet CEBSDB](https://physionet.org/content/cebsdb/1.0.0/)（García-González et al.，ODC-BY 1.0）
- **对标论文**：Rahman K M A, Albrecht U-V, Kulau U, et al., *At the Edge of the Heart: ULP FPGA-Based CNN for On-Device Cardiac Feature Extraction*, **DCOSS-IoT 2026**, [arXiv:2604.25799](https://arxiv.org/abs/2604.25799)
- **工具链**：Anlogic Tang Dynasty 6.2.x；PyTorch；NumPy；SciPy；wfdb

**本工程所有 RTL / Python / TCL 代码均为原创**，无任何专有 IP 复用。
代码使用 [MIT 许可](LICENSE)。数据集使用须遵守 PhysioNet CEBSDB 的 ODC-BY 协议（必须署名）。
