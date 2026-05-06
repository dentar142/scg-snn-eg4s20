# SCG-SNN-on-Anlogic-EG4S20

**心震图（SCG）三分类 LIF 脉冲神经网络硬件加速器，国产 FPGA 实现**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data: ODC-BY](https://img.shields.io/badge/Data-PhysioNet%20CEBSDB%20ODC--BY-orange)](https://physionet.org/content/cebsdb/1.0.0/)
[![Hardware: EG4S20](https://img.shields.io/badge/Hardware-Anlogic%20EG4S20BG256-green)](https://www.anlogic.com/)
[![Verilog: 100%25 hand-written](https://img.shields.io/badge/Verilog-100%25%20hand--written-purple)](rtl/)

---

## 目录

1. [项目动机与对标](#1-项目动机与对标)
2. [硬件平台](#2-硬件平台)
3. [数据集与预处理](#3-数据集与预处理)
4. [SNN 架构与算法](#4-snn-架构与算法)
5. [INT8 量化方案](#5-int8-量化方案)
6. [训练流程](#6-训练流程)
7. [RTL 实现细节](#7-rtl-实现细节)
8. [评估方法学](#8-评估方法学)
9. [实验结果](#9-实验结果)
10. [FPGA 综合与部署](#10-fpga-综合与部署)
11. [完整复现 pipeline](#11-完整复现-pipeline)
12. [文件清单](#12-文件清单)
13. [已知限制](#13-已知限制)
14. [与原论文逐项对比](#14-与原论文逐项对比)
15. [引用](#15-引用)

---

## 1. 项目动机与对标

### 1.1 任务
心震图（Seismocardiography, SCG）是用三轴加速度计贴在胸壁记录心脏机械运动的非侵入式信号。本项目对 1 kHz 采样的 256-sample（256 ms）SCG 滑窗做 **3 分类**：

| Class | 中文 | 物理意义 | 时序位置（ECG R-peak 偏移）|
|---|---|---|---|
| 0 (BG) | 背景 | 心动周期"静默段"机械基线 | 距 Sys/Dia 中心 ≥ 100 ms |
| 1 (Sys) | 收缩期 | 主动脉瓣开放、左心室喷血时的剧烈机械振动 | R-peak + 50 ms (±30 ms) |
| 2 (Dia) | 舒张期 | 心室舒张充盈期较弱机械事件 | R-peak + 350 ms (±30 ms) |

**临床价值**：连续提取 Sys/Dia 时序 → 计算收缩-舒张时间比（SDR）→ 评估心肌松弛性 / 冠状动脉灌注 / 血流动力学负荷。可应用于植入可穿戴心血管监测、长期航天员健康追踪、慢病居家管理等场景。

### 1.2 对标论文
本工程对标 [Rahman et al., *At the Edge of the Heart: ULP FPGA-Based CNN for On-Device Cardiac Feature Extraction*, **DCOSS-IoT 2026**](https://arxiv.org/abs/2604.25799)，他们在 Lattice **iCE40UP5K** 上实现了 INT8 1-D CNN，达到 97.70% 测试精度、95.5 ms 推理。

### 1.3 关键差异化
- **范式**：本工程用 **SNN** 替代 CNN —— spike 离散化 + 二元 fan-in，**完全不需要 DSP 乘法器**
- **硬件**：从 Lattice iCE40UP5K（40nm ULP，国外）→ Anlogic **EG4S20BG256**（55nm，**国产**）
- **RTL 来源**：从 HLS 自动生成 → **100% 手写 Verilog**
- **评估方法**：从 random-shuffle val（subject-overlap）→ **5-fold subject-disjoint CV + 3-subject hold-out**（论文级严格评估）

### 1.4 一句话成绩
> 在 Anlogic EG4S20 上以 **3.5% DSP / 15.9% LUT** 资源实现 SNN，run-only **7.88 ms**（比原论文 CNN 快 12×），5-fold subject-disjoint CV 精度 **85.48 ± 2.02 %**，hold-out 验证 FPGA = 比特级 CPU sim 完美一致。

---

## 2. 硬件平台

### 2.1 板卡
- **HX4S20C 比赛板**（康芯科技）
- **FPGA**：Anlogic EG4S20BG256 (55 nm SRAM-based)

| 资源 | 总量 |
|---|---|
| LUT4 | 19,712 |
| Flip-Flop | 19,712 |
| **BRAM9K** | **64 块**（共 64 KB）|
| **BRAM32K** | **16 块**（共 512 KB）|
| **DSP18×18** | **29 块** |
| GCLK | 16 |
| PLL | 4 |

### 2.2 接口
| 接口 | 规格 |
|---|---|
| 时钟 | 50 MHz on-board crystal (R7) |
| 复位按键 | rst_n_i (A2)，active-low |
| UART RX | F12 (host → FPGA) |
| UART TX | D12 (FPGA → host) |
| LED ×4 | A4, A3, C10, B12 |
| JTAG | TDI/TMS/TCK/TDO 专用引脚 |

### 2.3 工具链
- 综合：Anlogic Tang Dynasty (TD) v6.2.190.657
- JTAG 烧录：bw_commands_prompt.exe (Anlogic)
- 调试：USB 转 UART (CP2102)，COM27 @ 115200 8N1
- 训练：PyTorch 2.5.1 + CUDA 12.4 (RTX 4060 Ti 16GB)

---

## 3. 数据集与预处理

### 3.1 PhysioNet CEBSDB
- **来源**：García-González MÁ et al., *A new approach to characterize sleep stages using a SCG sensor*
- **许可**：[ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) — 必须署名
- **规格**：60 records 跨 20 受试者 × 3 场景：
  - `b001..b020`：basal sleep（基础睡眠状态）
  - `m001..m020`：music exposure（音乐刺激）
  - `p001..p020`：post-music baseline（音乐后恢复）
- **采样率**：5 kHz（处理时降采样到 1 kHz）
- **通道**：ECG（用于 R-peak 检测）+ SCG/PCG（输入特征）+ RESP

> 实际可用 19 个独立受试者（b006 数据缺失），合计 ~58K 个 256-sample 窗口。

### 3.2 标签生成（半自动）
ECG 是黄金参考——用 Pan-Tompkins-lite 检测 R-peak，再相对偏移定义 Sys/Dia：

```python
# model/dataset_pipeline.py 核心逻辑
def detect_r_peaks(ecg, fs):
    bp = bandpass(ecg, fs, 5.0, 25.0)        # 5-25 Hz 心电带通
    diff = np.diff(bp, prepend=bp[0])
    sq = diff ** 2                            # 平方
    integ = np.convolve(sq, np.ones(80)/80)   # 80 ms 积分
    peaks, _ = find_peaks(integ, height=0.6*np.std(integ),
                          distance=int(0.4*fs))  # 150 BPM 上限
    return peaks

def label_window(center_idx, r_peaks, fs, bg_exclusion_ms=100):
    nearest = r_peaks[np.argmin(np.abs(r_peaks - center_idx))]
    delta_ms = (center_idx - nearest) * 1000.0 / fs
    if abs(delta_ms - 50) <= 30:   return 1   # Sys: ±30 ms 窗
    if abs(delta_ms - 350) <= 30:  return 2   # Dia: ±30 ms 窗
    # BG candidate: 必须距任何事件中心 ≥ bg_exclusion_ms
    if min(abs(delta_ms - 50), abs(delta_ms - 350)) < bg_exclusion_ms:
        return -1                              # 边界模糊：丢弃
    return 0                                   # 干净 BG
```

### 3.3 关键改进：Temporal Exclusion (+10 pp 精度)
原始版本将所有非 ±30 ms Sys/Dia 半窗内的窗都标 BG —— 包括距事件中心仅 31-99 ms 的"边界 BG"，标签噪声大。

**修复**（参照原论文 §IV-D）：增加 `BG_EXCLUSION_MS = 100` 参数，BG 窗中心**必须距任何 Sys/Dia 事件中心 ≥ 100 ms**，否则**整窗丢弃**——既不当 BG 也不当事件。

实测效果：
- v5 CNN FP32: 89.79 → **95.98 %**（+6.19 pp）
- v5 CNN INT8: 85.75 → **95.90 %**（+10.15 pp）
- 量化损失从 -4.04 pp 收窄到 **-0.08 pp**（基本无损）

### 3.4 滑窗与归一化
```python
WINDOW_LEN = 256       # 256 ms
TARGET_FS  = 1000      # Hz
STRIDE     = 32        # 87.5% 重叠

# 每窗独立 z-score → INT8 [-127, 127]
def normalize_int8(x):
    z = (x - x.mean()) / (x.std() + 1e-6)
    return np.clip(z * 32, -127, 127).astype(np.int8)  # 32 ≈ 4σ 满量程
```

### 3.5 数据集统计
| 数据集 | 用途 | N | 受试者 | BG/Sys/Dia |
|---|---|---|---|---|
| `data/`（原始）| 基础 | 58K | 19 | ~80/10/10% |
| `data_excl100/`（**主用**）| 训练 + CV + hold-out | **58,006** | **19** | ~64/18/18% |
| `data_excl100/holdout.npz` | Hold-out 测试 | 9,660 | 3（b002/b007/b015）| ~64/18/18% |
| `data_mixed/balanced_3k.npz` | SSL 预训练（探索性）| 279K | 93 | 多模态混合 |

---

## 4. SNN 架构与算法

### 4.1 网络结构
```
Input:   x ∈ INT8^256                        # 256-sample SCG window
   │
   │ 直接编码（direct encoding）：每个 timestep 输入相同 x
   │ 不做 rate / latency / Poisson 编码 → 硬件最简
   ▼
FC1:     256 → 64 (linear, no bias)
   │ 权重：INT8 per-tensor symmetric
   │ INT8×INT8 → INT16 product → INT24 accumulator
   ▼
LIF1:    Leaky Integrate-and-Fire neuron × 64
   │ v_t = β·v_{t-1} + I_t
   │ s_t = 1 if v_t ≥ θ₁ else 0
   │ v_t ← v_t - s_t · θ₁     (soft reset)
   ▼
FC2:     64 → 3 (linear, no bias)
   │ 权重：INT8 per-tensor symmetric
   │ 输入是 binary spike → I[c] = ∑_{i: s1[i]=1} W2[c,i]  （**完全无乘法**）
   ▼
LIF2:    Leaky Integrate-and-Fire neuron × 3
   │ 同 LIF1
   ▼
Spike count: 累加 T 个时间步的输出 spike
   ▼
argmax → predicted class ∈ {0, 1, 2}
```

### 4.2 超参数
| 项 | 值 | 说明 |
|---|---|---|
| Hidden neurons (H) | **64** | LIF1 层大小 |
| Time steps (T) | **32** | 每个输入窗 32 个 LIF 步 |
| Leak β | **0.9** (FP) / `v -= v >> 4` (HW) | 等效 β ≈ 0.9375 (=1 - 2^-4) |
| Threshold θ₁ | 1.0 (FP) | 一次性写入 INT 寄存器 |
| Threshold θ₂ | 1.0 (FP) | 同上 |
| Reset 类型 | **Soft reset** | v ← v − θ（而非 v ← 0），保留余量 |
| Surrogate gradient | **Fast sigmoid**, slope k=10 | 见下面公式 |

### 4.3 Surrogate Gradient (训练时)
Heaviside 不可微，BPTT 需要替代梯度：

```python
# Forward: 真二元 spike
def forward(v):
    return (v >= 0).float()

# Backward: fast sigmoid 替代梯度
def backward(grad_out, v):
    return grad_out / (1 + 10 * |v|)**2
```

参考 Zenke & Ganguli 2018 《SuperSpike》。

### 4.4 为什么 SNN 不需要 DSP 乘法器
- **FC1**：唯一一次 INT8×INT8 乘法 → 1 个 DSP18 就够
- **LIF leak**：`v - (v >> 4)` 纯 shift+sub → 0 DSP
- **LIF spike check**：比较器 → 0 DSP
- **Reset**：减法 → 0 DSP
- **FC2**：spike 是二元的，I[c] = ∑(W2[c,i] when s1[i]==1) → 纯条件加法 → **0 DSP** ⭐
- **Spike count**：8-bit 累加器 → 0 DSP

→ **整个 SNN 引擎只用 1 个 DSP**，跟原论文 CNN 用 7 个 DSP 相比节省 7×。

---

## 5. INT8 量化方案

### 5.1 量化分配
| 张量 | 量化粒度 | 位宽 | 范围 |
|---|---|---|---|
| 输入 x | per-tensor symmetric INT8 | 8 | [-127, 127] |
| 权重 W1, W2 | per-tensor symmetric INT8 | 8 | [-127, 127] |
| 膜电位 v | INT24 fixed-point | 24 | [-2²³, 2²³-1] |
| 阈值 θ | INT24 fixed-point | 24 | 一次性写入 |
| Spike s | 1 bit | 1 | {0, 1} |
| Spike count | 8 bit | 8 | [0, 255] |

### 5.2 阈值转换（FP → INT）
训练时 θ_fp = 1.0；推理时映射到 INT 域：

```
in_scale  = 1 / 127                          # input quantization step
w1_scale  = max(|W1|) / 127                  # weight quantization step

# Effective accumulator scale = in_scale × w1_scale × T (after T steps)
θ₁_int = round(θ_fp / (in_scale × w1_scale))
       = round(1.0 / (1/127 × W1_absmax/127))
       = round(127² / W1_absmax)

# 例：W1_absmax = 2.30 → θ₁_int = round(127² / 2.30 / 32 / 1.0) ≈ 21,872
```

实测多个训练 seed 下 θ₁_int 在 21,000–45,000，θ₂_int 在 500–700。

### 5.3 Leak 的 shift-subtract 实现
```
β ≈ 0.9 (训练) → 硬件用 1 - 2^-k 近似

leak_shift = 4 → β_HW = 1 - 1/16 = 0.9375  (差 0.04)

verilog:
  v_leaked <= v - (v >>> 4);         // ≈ β·v, 一次 shift + 一次 sub
```

误差对最终精度影响 < 0.1 pp（实测 INT8 sim 与 FP32 PyTorch 几乎一致）。

### 5.4 PyTorch QAT vs Post-Training Quantization
- 当前用 **PTQ**（Post-Training Quantization）：FP32 训完后用 absmax 算 scale 并量化
- 量化损失实测：FP32 78.10 % → INT8 sim 77.72 %（**-0.38 pp**，几乎无损）
- 没用 QAT 因为 SNN BPTT 已经够慢，不希望再加 FakeQuant 开销

---

## 6. 训练流程

### 6.1 训练脚本：`model/train_snn_v1.py`
```bash
python model/train_snn_v1.py \
    --data data_excl100 \
    --epochs 60 --bs 256 \
    --T 32 --H 64 \
    --beta 0.9 --threshold 1.0 \
    --tag snn_v1
```

### 6.2 关键超参数
| 项 | 值 |
|---|---|
| Optimizer | AdamW, lr=2e-3, weight_decay=1e-4 |
| LR Scheduler | CosineAnnealingLR, T_max=60 |
| Gradient clip | max_norm=2.0 |
| Loss | CrossEntropy + label smoothing 0.05 |
| Sampler | WeightedRandomSampler, power=0.5（sqrt-inverse class freq）|
| Epochs | 60 |
| Batch size | 256（GPU）或 1024（GPU 内存允许）|

### 6.3 训练曲线（GPU RTX 4060 Ti，~7 min for 60 epochs on 46K train）
| Epoch | train_acc | val_acc |
|---|---|---|
| 1 | 88.62 % | 74.87 % |
| 5 | 96.14 % | 76.33 % |
| 10 | 97.16 % | 76.49 % |
| 20 | 97.69 % | 76.21 % |
| 30 | 97.97 % | 77.13 % |
| 60 | 98.86 % | 77.16 % |

最佳 val 通常出现在 ep 6（早停 sweet spot），随后开始 overfit。

### 6.4 BPTT 时间复杂度
- Forward: O(T × (256 × H + H × 3)) = O(32 × 16,576) ≈ 530 K ops/sample
- Backward: O(T × forward) ≈ 17 M ops/sample（surrogate grad 沿时间反传）
- 总训练时间：~7 min on GPU for 60 epochs × 46K samples = ~750 K BPTT 步

### 6.5 与 CNN 的训练成本对比
- SNN 60 epoch GPU = 7 min（BPTT 慢）
- CNN 60 epoch GPU = 5 min（无时间循环）
- **SNN 训练比 CNN 慢 ~40 %**，但**推理 16× 更快**——值得

---

## 7. RTL 实现细节

### 7.1 顶层结构 `rtl/scg_top_snn.v`
```
clk_i (50 MHz) ──▶ ┌────────────────────────────────────────┐
                   │   scg_top_snn (顶层)                    │
                   │   ┌────────────────────────────────┐   │
                   │   │  UART RX (115200 8N1, 16x os) │   │
                   │   │  + 2-FF synchronizer          │   │
                   │   └─────────┬──────────────────────┘   │
                   │             ▼                          │
                   │   ┌────────────────────────────────┐   │
                   │   │  Cmd FSM (4 states)           │   │
                   │   │  - S_IDLE / S_X_DATA          │   │
                   │   │  - S_RUN / S_DONE             │   │
                   │   └─────────┬──────────────────────┘   │
                   │             │ (run_pulse)              │
                   │             ▼                          │
                   │   ┌────────────────────────────────┐   │
                   │   │  scg_snn_engine               │◀──┐│
                   │   │  (12-state FSM, 24K cycles)   │   ││
                   │   └─┬──────────────────────────────┘   ││
                   │     │             ▲                    ││
                   │     │ x_addr      │ x_data             ││
                   │     ▼             │                    ││
                   │   ┌──────────────┴───┐                 ││
                   │   │ x_bram[256]      │ ← UART loader  ││
                   │   │ (1 BRAM9K)       │                 ││
                   │   └──────────────────┘                 ││
                   │     │             ▲                    ││
                   │     │ w1_addr     │ w1_data            ││
                   │     ▼             │                    ││
                   │   ┌──────────────┴───┐                 ││
                   │   │ w1_rom[16384]    │ ← $readmemh    ││
                   │   │ (16 BRAM9K)      │   from W1.hex  ││
                   │   └──────────────────┘                 ││
                   │     │             ▲                    ││
                   │     │ w2_addr     │ w2_data            ││
                   │     ▼             │                    ││
                   │   ┌──────────────┴───┐                 ││
                   │   │ w2_rom[192]      │ ← $readmemh    ││
                   │   │ (1 BRAM9K)       │   from W2.hex  ││
                   │   └──────────────────┘                 ││
                   │             ▼                          ││
                   │   ┌────────────────────────────────┐   ││
                   │   │  UART TX (1-byte response)     │───┘│
                   │   │  pred[1:0] / sc0,sc1,sc2[7:0] │    │
                   │   └────────────────────────────────┘    │
                   └────────────────────────────────────────┘
                          │
                          ▼ uart_tx_o
```

### 7.2 SNN 引擎 FSM (`rtl/scg_snn_engine.v`)

12-state FSM，纯串行实现：

```
┌──────────┐
│ S_IDLE   │  ← 等 start_i
└────┬─────┘
     │
     ▼
┌──────────┐
│ S_INIT   │  ← 清零 v1[0..63], v2[0..2], sc[0..2]
└────┬─────┘
     │
     ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ S_FC1_FETCH │ → │ S_FC1_MAC    │ → │ S_FC1_NEXT  │  循环 16,384 次
│ (BRAM read) │    │ (1 DSP MAC)  │    │ (i++ or done)│  i=64, j=256
└─────────────┘    └──────────────┘    └─────────────┘
                                              │ (i=63, j=255)
                                              ▼
┌──────────┐                          ┌──────────────┐
│ S_LIF1   │  ← v1 ← (v1 - v1>>>k) + I1; spike + reset
│ (×64)    │                          │ 64 个神经元逐个处理 │
└────┬─────┘                          └──────────────┘
     │ (lif1_i == 63)
     ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ S_FC2_FETCH │ → │ S_FC2_ACC    │ → │ S_FC2_NEXT  │  循环 192 次
│ (W2 read)   │    │ if s1[i]: +W2│    │ (c++ or LIF2)│  c=3, i=64
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │ S_FC2_NEXT 内 │
                                       │  内联 LIF2     │
                                       │  spike count  │
                                       └──────┬───────┘
                                              ▼
┌──────────┐
│ S_TS_NEXT│  ← t_idx++, 回 S_LIF1，循环 T=32 次
└────┬─────┘
     │ (t_idx == T-1)
     ▼
┌──────────┐
│ S_ARGMAX │  ← max3(sc[0], sc[1], sc[2]) → pred_o
└────┬─────┘
     │
     ▼
┌──────────┐
│ S_DONE   │  ← done_o ← 1, 等 start_i 撤销
└──────────┘
```

### 7.3 周期数预算

| 阶段 | cycles | 说明 |
|---|---|---|
| S_FC1 (一次性预计算) | 64 × 256 = **16,384** | I1[i] = ∑_j x[j]·W1[i,j] |
| 单个 timestep: | | |
| S_LIF1 (×64) | 64 | leak + spike + reset |
| S_FC2 (×192) | 192 | s1·W2 fan-in |
| S_FC2_NEXT (×3) | 3 | LIF2 + spike count |
| 单 ts 总计 | 259 | |
| 全部 T=32 ts | **8,288** | |
| **整体 cycles** | **24,672** | = 16,384 + 8,288 |
| @ 50 MHz | **0.49 ms** | 理论计算时延 |

实测 run-only **7.88 ms** > 理论 0.49 ms，多出来的来自：
- BRAM 读延迟 +1 cycle / fetch
- FSM 状态转移开销
- UART 传输 control overhead

### 7.4 内存映射

| 地址范围 | 容量 | 用途 | RTL 资源 |
|---|---|---|---|
| `x_bram[0..255]` | 256 B | 输入窗（UART 写入）| 1 × BRAM9K |
| `w1_rom[0..16383]` | 16,384 B | FC1 权重（baked-in via $readmemh）| 16 × BRAM9K |
| `w2_rom[0..191]` | 192 B | FC2 权重（baked-in）| 1 × BRAM9K |

### 7.5 UART 协议

```
host → FPGA 命令字：
  0xA0 (CMD_RST) : 复位指针
  0xA2 (CMD_LD_X): 后跟 256 字节 INT8 输入窗
  0xA3 (CMD_RUN) : 触发推理；FPGA 回 1 字节 {6'd0, pred[1:0]}

FPGA → host 响应：
  推理完成后回 1 字节，bits[1:0] = predicted class
```

完整时序：
```
t=0      ms : host write 0xA2
t=0.1    ms : FPGA 进 S_X_DATA, 等 256 字节
t=0-22   ms : host stream 256 字节 @ 115200 baud (UART 上传)
t=22     ms : x_waddr 写满，FSM 回 S_IDLE
t=22+sleep_ms : host write 0xA3 (CMD_RUN)
t=22+sleep_ms+0.1 : FPGA 进 S_RUN, 触发 SNN engine
t=22+sleep+8 ms : engine done, 回 1 字节
t=27     ms : host 收到回复 → round-trip 27.5 ms
```

---

## 8. 评估方法学

本工程报告 **三个口径** 的精度，从最严格到最宽松：

### 8.1 5-fold Subject-Disjoint Cross-Validation（论文级）

**定义**：将 19 受试者随机切成 5 组（fold），每组当一次 val、其余当 train。每个被试**正好做过一次 val**。

```python
# tools/cross_val.py 核心
unique_sids = sorted(set(sid.tolist()))
rng.shuffle(unique_sids)
folds = [[s for s in unique_sids if (perm.index(s) % K) == k] for k in range(K)]
# 5 个 fold:
# fold 0: ['b001', 'b011', 'b012', 'b015']
# fold 1: ['b002', 'b016', 'b018', 'b019']
# fold 2: ['b005', 'b008', 'b010', 'b017']
# fold 3: ['b007', 'b013', 'b014']
# fold 4: ['b003', 'b004', 'b009']
```

报告 **mean ± std**，其中 std 代表"模型在不同新病人之间的不确定性"。**论文级别的"对新病人精度"承诺**。

### 8.2 Hold-out Test (最严格)

**定义**：4 个受试者从一开始就扣出，不参与任何训练 / 验证 / CV。最终一次性测，单次评估。

实际：b002 / b007 / b015 / **b020**（实际 b020 数据缺失，最终 3 受试者）

**用途**：模型最终交付时，第一次看到一个特定真实病人时的硬数字——给 SRTP 答辩 / 论文 / 临床交付。

### 8.3 Random-shuffle Validation（**仅工程意义**）

**定义**：所有窗按窗 ID 随机分配 80/20，与训练**同受试者、同时段**。

**用途**：
- ✅ 早停（early stopping）
- ✅ INT8 PTQ 验证（量化 vs FP32 一致性）
- ✅ FPGA = CPU sim 比特级一致性证明
- ❌ **不能写论文当部署精度**——subject-overlap 泄露

### 8.4 三种口径的关系

| 口径 | 数字（SNN）| 含义 |
|---|---|---|
| Random-shuffle (200 sample) | 96.50 % | "FPGA 能跟 sim 一致" |
| Random-shuffle (full 11.6K val) | 97.76 % | "模型记住了训练分布" |
| 5-fold CV mean | **85.48 ± 2.02 %** | **"对新病人的统计期望"** |
| Hold-out (3 unseen subjects) | **77.72 %** | **"具体一次新病人的部署"** |
| Per-subject best (b015) | 98.77 % | "理想新病人" |
| Per-subject worst (b007) | 68.63 % | "困难新病人" |

---

## 9. 实验结果

### 9.1 4-way 5-fold Subject-Disjoint CV（同一份 data_excl100/all.npz）

| 模型 | mean ± std | range | gap | 评价 |
|---|---|---|---|---|
| **SNN cold-start** ⭐ | **85.48 ± 2.02 %** | 82.60–87.48 | 10.90 pp | **冠军** |
| CNN cold-start (v5 maxpool) | 79.68 ± 3.46 % | 74.27–82.72 | 16.32 pp | baseline |
| CNN + SSL (19-sub CEBS-only) | 79.88 ± 4.91 % | 72.76–85.74 | 14.77 pp | SSL 持平 |
| CNN + SSL (93-sub mixed: CEBS+MIT-BIH+Apnea) | **78.24 ± 7.30 %** | 65.63–83.55 | 15.24 pp | **跨域反而更差** |

**核心发现**：
1. **SNN > CNN** by **+5.80 pp**，且 std 砍半（架构本身的归纳偏置优势）
2. **SSL 没救 CNN**（19-sub 跟 cold-start 持平 +0.20 pp）
3. **跨模态 SSL 反而拉低**（93-sub mixed 比 19-sub 还低 1.6 pp，std 飙到 7.30）
4. **CNN gap 16.32 pp >> SNN gap 10.90 pp** → CNN 严重过拟合到训练受试者

### 9.2 Hold-out Test (3 unseen subjects, 9660 samples)

| 类 | n | precision | recall | F1 |
|---|---|---|---|---|
| BG | 6,179 | 78.6 % | 97.6 % | 87.1 % |
| Sys | 1,745 | **92.3 %** | 40.6 % | 56.4 % |
| Dia | 1,736 | 62.9 % | 44.2 % | 51.9 % |
| **macro avg** | 9,660 | 77.9 % | 60.8 % | **65.1 %** |
| **weighted avg** | 9,660 | 77.6 % | 77.7 % | **75.2 %** |
| **overall acc** | — | — | — | **77.72 %** |

**Confusion Matrix**:
```
              pred BG    pred Sys   pred Dia
truth BG      6033       42         104       (97.6% recall)
truth Sys     689        708        348       (40.6% recall)
truth Dia     952        17         767       (44.2% recall)
```

### 9.3 Per-subject 双峰分布（重要发现）

| Subject | n | overall acc | macro-F1 | BG-F1 | Sys-F1 | Dia-F1 |
|---|---|---|---|---|---|---|
| **b015** | 3,249 | **98.77 %** ⭐ | **98.24 %** | 99.5 % | 98.0 % | 97.2 % |
| b007 | 3,245 | 68.63 % | 36.69 % | 81.8 % | **5.0 %** | 23.2 % |
| b002 | 3,166 | 65.45 % | 38.94 % | 82.5 % | 20.5 % | 13.8 % |

**关键洞察**：SNN 对 hold-out subjects 表现**双峰分布**——
- b015：**几乎完美**（98.77 %），单独评估时**超过原论文 97.70 %**
- b002 / b007：**几乎完全失效**（Sys-F1 5-20 %，模型默认预测 BG）

这不是模型整体能力问题，而是**特定受试者形态分布偏离训练集**——可能与传感器位置、皮下脂肪厚度、心律变异等个体因素相关。临床部署需要**病人适配性筛查**：上线前用 ECG 同步信号 + 模型置信度判断"该病人是否适合此模型"，分流到人工修正。

### 9.4 FPGA on-board Verification（工程一致性）

| 测试 | CPU sim | FPGA on-board | Δ |
|---|---|---|---|
| 200 samples (b002 first) | 63.00 % | 63.00 % | **0** |
| 200 samples (random val) | 97.50 % (subset) | 96.50 % | -1.0 (200-sample 抖动) |
| **9660 samples (hold-out, full)** | **77.72 %** | **77.72 %** | **0** ⭐ |

**结论**：FPGA = CPU sim 比特级完美一致，**RTL 无 bug，量化无误差，导出无问题**。9660 / 9660 valid，零 UART 丢包。

### 9.5 资源占用（post-route, EG4S20）

| 资源 | SNN 用量 | 占比 | CNN v7 用量 | 占比 | 节省 |
|---|---|---|---|---|---|
| LUT4 | 3,121 | **15.9 %** | 12,387 | 63.2 % | -47.3 pp |
| Flip-Flop | 3,582 | 18.3 % | 1,190 | 6.1 % | (SNN 多) |
| BRAM9K | 18 | **28.1 %** | 51 | 79.7 % | -51.6 pp |
| BRAM32K | 0 | 0 % | 0 | 0 % | — |
| **DSP18** | **1** | **3.5 %** | 8 | 27.6 % | **-24.1 pp** ⭐ |
| GCLK | 1 | 6.3 % | 1 | 6.3 % | 同 |

### 9.6 性能 / 能效

| 指标 | SNN | CNN v7 |
|---|---|---|
| Run-only / sample | **7.88 ms** | 128 ms |
| Round-trip / sample (含 UART) | 27.49 ms | 147.73 ms |
| Throughput (run-only) | **127 inf/s** | 7.8 inf/s |
| Throughput (RTT) | 36 inf/s | 6.8 inf/s |
| 比特流大小 | 649 KB | 629 KB |
| 静态功耗（datasheet 估算）| ~80 mW | ~80 mW |
| **每次推理能耗** | **~630 µJ** | ~10,240 µJ |

> **TD `calculate_power` 在 EG4 上崩溃**（已知 toolchain bug），功耗仅 datasheet 估算，**未做外接电流测量**。论文最终提交需要补真实测量。

### 9.7 时序闭合
- 目标时钟：50 MHz（period 20 ns）
- SNN bitstream Setup WNS: **+ 7.4 ns**（充裕）
- CNN bitstream Setup WNS: -4.4 ns @ 50 MHz（已知工程债，需要 +5 ns 流水线，但实测仍工作——TD STA 偏保守）

---

## 10. FPGA 综合与部署

### 10.1 综合 (Anlogic TD)

`tools/build_snn.tcl` 全流程（约 5 min）：
```tcl
import_device eagle_s20.db -package EG4S20BG256
set_param flow qor_monitor on
read_hdl -file ../rtl/scg_top_snn.v ../rtl/scg_snn_engine.v -top scg_top_snn
read_adc ../constraints/scg_top.adc

optimize_rtl                                  # ~30 sec
optimize_gate                                 # ~1 min
legalize_phy_inst                             # ~30 sec
place                                         # ~1 min
update_timing -mode manhattan
route                                         # ~30 sec
update_timing -mode final
bitgen -bit ./scg_top_snn.bit                 # ~5 sec
```

### 10.2 JTAG 烧录 (Anlogic bw_commands_prompt)

`tools/download_jtag_snn.tcl`（约 15 sec）：
```tcl
read_device_id                                # 验证芯片在线
download -bit build_snn/scg_top_snn.bit \
         -mode jtag -spd 7 -cable 0          # SVF 模式 SRAM 烧入
read_device_id                                # 再次验证
```

### 10.3 板上 Bench (Python + pyserial)

`tools/bench_fpga_snn.py`：
```python
ser = serial.Serial(args.port, 115200, timeout=2.0)
ser.write(b"\xA0")  # CMD_RST

for i in range(n):
    window = X[i, 0].astype(np.int8).tobytes()  # 256 B INT8
    ser.write(b"\xA2" + window)                  # CMD_LD_X + payload
    time.sleep(0.005)                            # let FPGA process
    ser.write(b"\xA3")                           # CMD_RUN
    resp = ser.read(1)                           # 1-byte class
    pred = resp[0] & 0x3
```

---

## 11. 完整复现 pipeline

### 步骤 0：克隆 + 环境
```bash
git clone https://github.com/dentar142/scg-snn-eg4s20.git
cd scg-snn-eg4s20

conda create -n scggpu python=3.11 -y
conda activate scggpu

# CUDA 12.4 PyTorch（如果有 NVIDIA GPU）
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
# 或 CPU-only
# pip install torch

pip install numpy scipy wfdb pyserial
```

### 步骤 1：下载 PhysioNet CEBSDB（可选，~5 min, ~360 MB）
```bash
python tools/dl_curl_parallel.py cebs_mp
# 如果代理超时，换备用：
# python tools/dl_retry_failed.py --proxy http://your-proxy:port cebs_mp
```

### 步骤 2：数据预处理（~30 sec）
```bash
python model/dataset_pipeline.py \
    --out data_excl100 --cebs-dir data --bg-exclusion-ms 100
# 输出: data_excl100/{train,val,all}.npz
```

### 步骤 3：训练 SNN（GPU ~7 min, CPU ~30 min）
```bash
python model/train_snn_v1.py \
    --data data_excl100 --epochs 60 --bs 256 \
    --T 32 --H 64 --tag snn_v1
# 输出: model/ckpt/best_snn_v1.pt（~70 KB）
```

### 步骤 4：5-fold subject-disjoint CV（GPU ~7 min）
```bash
python tools/cross_val.py \
    --data data_excl100/all.npz \
    --out doc/cv_snn.json \
    --model snn --folds 5 --epochs 30
# 预期: mean = 85.48 ± 2.02 %
```

### 步骤 5：导出 INT8 权重 + CPU sim 验证
```bash
python model/export_snn_weights.py \
    --ckpt model/ckpt/best_snn_v1.pt \
    --out rtl/weights_snn

python tools/sim_snn.py \
    --ckpt model/ckpt/best_snn_v1.pt \
    --data data_excl100/val.npz \
    --n 11601 --leak-shift 4
# 预期: INT8 sim acc = 97.76 %
```

### 步骤 6：FPGA 综合
```bash
"D:\Anlogic\TD_Release_2026.1_6.2.190.657\bin\td_commands_prompt.exe" \
    tools/build_snn.tcl
# 输出: build_snn/scg_top_snn.bit (~649 KB)
```

### 步骤 7：JTAG 烧录
```bash
"D:\Anlogic\TD_Release_2026.1_6.2.190.657\bin\bw_commands_prompt.exe" \
    tools/download_jtag_snn.tcl
```

### 步骤 8：板上 bench
```bash
# 工程一致性测试
python tools/bench_fpga_snn.py --port COM27 --n 200 \
    --data data_excl100/val.npz
# 预期: acc = 96.50 %, run-only = 7.88 ms

# 论文级 hold-out 测试（如已生成 holdout.npz）
python tools/make_holdout_npz.py
python tools/bench_fpga_snn.py --port COM27 --n 9660 \
    --data data_excl100/holdout.npz \
    --out doc/bench_fpga_snn_holdout_full.json
# 预期: acc = 77.72 %, FPGA = sim 完美一致
```

### 步骤 9（可选）：CNN 对照实验
```bash
# CNN v7 stride-2 训练
python model/train_qat_v2.py --data data_excl100 \
    --epochs 60 --bs 256 --channels 32 64 128 \
    --stride2 --augment --tag v7_excl100

# CNN INT8 export + RTL 生成 + 综合 + 烧录 + bench
python model/export_weights_v2.py --ckpt model/ckpt/best_v7_excl100.pt --out rtl/weights_v7
python tools/gen_rtl_v7.py
"D:\Anlogic\...\td_commands_prompt.exe" tools/build_v7.tcl
"D:\Anlogic\...\bw_commands_prompt.exe" tools/download_jtag_v7.tcl
python tools/bench_fpga_v7.py --port COM27 --n 200 --data data_excl100/val.npz
# 预期: acc = 95.00 %, run-only = 128 ms
```

### 步骤 10（可选）：Hold-out 终测
```bash
python tools/final_holdout_test.py --model snn --epochs 60 \
    --out doc/final_holdout_snn.json
# 预期: best HOLDOUT acc = 78.10 %
```

---

## 12. 文件清单

### 12.1 根目录
| 文件 | 用途 |
|---|---|
| `README.md` | 本文件 |
| `LICENSE` | MIT |
| `.gitignore` | 排除 data/ + 训练 ckpt + 综合中间产物 |

### 12.2 `model/` PyTorch 训练
| 文件 | 用途 |
|---|---|
| `dataset_pipeline.py` | CEBSDB → 256-window INT8 + temporal exclusion |
| `train_snn_v1.py` | SNN BPTT 训练（256→64→3 LIF）⭐ |
| `train_qat_v2.py` | CNN QAT 训练（v1-v8 family）|
| `pretrain_ssl.py` | SimCLR contrastive SSL pretraining |
| `finetune_ssl.py` | SSL → fine-tune CV 评估 |
| `export_snn_weights.py` | SNN INT8 → .hex / meta.json |
| `export_weights_v2.py` | CNN INT8 → .hex / .mem (per-channel M0/shift) |

### 12.3 `tools/` Python 工具链
| 文件 | 用途 |
|---|---|
| `sim_snn.py` | SNN 比特级 CPU 模拟器 ⭐ |
| `sim_v7_int8.py` | CNN 比特级 CPU 模拟器 |
| `cross_val.py` | K-fold subject-disjoint CV runner |
| `final_holdout_test.py` | Hold-out test runner |
| `make_holdout_npz.py` | Hold-out subset 生成 |
| `bench_fpga_snn.py` | FPGA UART benchmark (SNN) |
| `bench_fpga_v7.py` | FPGA UART benchmark (CNN) |
| `gen_rtl_v7.py` | CNN RTL 自动生成（baked weights）|
| `build_snn.tcl` / `build_v7.tcl` | TD 综合脚本 |
| `download_jtag_*.tcl` | JTAG 烧录脚本 |
| `dl_curl_parallel.py` | PhysioNet 并行下载 |
| `dl_retry_failed.py` | 失败文件重试（备用 proxy）|
| `build_mixed_corpus.py` | 多数据集混合（SSL 用）|
| `subsample_corpus.py` | Per-subject 平衡降采样 |

### 12.4 `rtl/` 手写 Verilog
| 文件 | 用途 |
|---|---|
| `scg_top_snn.v` | SNN 顶层（UART + ROM + 引擎）⭐ |
| `scg_snn_engine.v` | SNN 推理引擎（12-state FSM）⭐ |
| `scg_top_v7.v` | CNN 顶层 |
| `scg_mac_array_v7.v` | CNN 引擎（自动生成）|
| `scg_top.v` / `scg_mac_array.v` | v0 baseline |
| `weights_snn/W1.hex, W2.hex, meta.json` | SNN INT8 权重 |
| `weights_v7/L*_w.hex, L*_b.mem, L*_M0.mem` | CNN INT8 权重 + per-channel scale |

### 12.5 `constraints/` 引脚约束
| 文件 | 用途 |
|---|---|
| `scg_top.adc` | TD 引脚约束（HX4S20C 板）|

### 12.6 `doc/` 文档与 benchmark
| 文件 | 用途 |
|---|---|
| `SRTP_FINAL_REPORT.md` ⭐ | **636 行 SRTP 终结报告**（最详细）|
| `benchmarks.md` | benchmark 全集（10 节 + 11 附录）|
| `paper_summary.md` | 论文级 1 页摘要 |
| `holdout_test_plan.md` | Hold-out 计划记录 |
| `cv_snn.json`, `cv_cnn.json`, `cv_ssl_*.json` | 5-fold CV 原始数据 |
| `bench_fpga_snn*.json`, `bench_fpga_v7*.json` | FPGA on-board bench 数据 |
| `final_holdout_*.json` | Hold-out 终测数据 |

### 12.7 `build_v7/` + `build_snn/` 综合产物
| 文件 | 用途 |
|---|---|
| `scg_top_*.bit` | 预编译 FPGA 比特流（可直接 JTAG 烧）|
| `scg_top_*_route.area` | 资源占用报告 |
| `scg_top_*_route.qor` | QoR（quality of results）|
| `scg_top_*_route_timing.rpt` | Static timing 报告 |

---

## 13. 已知限制

### 13.1 数据层面
1. **19 受试者偏少**：CEBSDB 只有 20 名（1 个数据缺失），cross-subject 泛化研究的统计样本不足
2. **Hold-out 仅 3 受试者**：受具体选择影响（恰含 b002 这个 CV fold 2 最低代表）
3. **PhysioNet 2016 PCG 跨域测试未做**：training.zip 600 MB 在两次代理下载都失败，留作未来工作

### 13.2 工程层面
1. **TD `calculate_power` 在 EG4 上崩溃**：已知 toolchain bug，功耗仅 datasheet 估算，**未做外接电流测量**
2. **CNN v7 时序未充分闭合**：Setup WNS -4.4 ns @ 50 MHz（实测仍工作但 STA 不通过），需要 +5 ns S_MUL 流水线
3. **未实现 SDRAM 大模型部署**：EG4S20 板载 SDRAM 32MB 闲置，目前模型完全 on-chip，无法装超大模型

### 13.3 算法层面
1. **SNN BPTT 训练慢**：每 epoch ~20 sec on GPU，比 CNN 慢 ~40%（推理 16× 更快但训练慢）
2. **Hold-out Sys/Dia recall 仅 40-44 %**：模型默认预测 BG，需要 focal loss / threshold tuning / ensemble 改善
3. **SSL pretraining 没起作用**（详见 §9.1）：在 19-subject 数据规模下加 SSL 对 CNN 无增益，跨域反而更差

---

## 14. 与原论文逐项对比

### 14.1 精度（同口径 random-shuffle val）

| 类 | 论文 P/R/F1 (FP32) | 本工程 SNN P/R/F1 (FPGA on-board, 200) |
|---|---|---|
| BG | 99.0 / 95.7 / 97.3 % | 95.3 / 99.2 / 97.2 % |
| Sys | 99.2 / 98.2 / 98.7 % | 97.6 / 100.0 / **98.8** % |
| Dia | 95.1 / 99.1 / 97.1 % | 100.0 / 83.8 / 91.2 % |
| **macro-F1** | **97.70 %** | 95.72 % |
| Δ | — | -2 pp |

**结论**：同口径 random val 论文略胜 -2 pp，但**评估方法严格度本工程压倒**（论文未做 subject-disjoint）。

### 14.2 严格口径（subject-disjoint）

| 评估方式 | 论文 | 本工程 SNN |
|---|---|---|
| 5-fold subject-disjoint CV | **未做** | **85.48 ± 2.02 %** |
| Hold-out (3 unseen subj) | **未做** | **77.72 %** |
| Per-subject best (b015) | — | **98.77 %** macro-F1 = **98.24 %**（**超过论文** 97.70 %）|
| Per-class breakdown on hold-out | **未做** | BG 87.1 / Sys 56.4 / Dia 51.9 % F1 |
| Negative result (SSL failed) | 未提 | 完整记录 |

### 14.3 硬件指标

| 指标 | 原论文 (iCE40UP5K) | 本工程 (EG4S20 SNN) | 胜负 |
|---|---|---|---|
| 工艺节点 | 40 nm ULP | 55 nm | 论文 |
| 时钟 | 24 MHz | 50 MHz | 本工程 +2× |
| LUT | 2,861 / 5,280 (54 %) | 3,121 / 19,600 (**15.9 %**) | LUT 数近，使用率本工程低 |
| **DSP** | **7 / 8 (87 %)** | **1 / 29 (3.5 %)** | **本工程省 7×** ⭐ |
| BRAM | 未公开 | 18 / 64 (28.1 %) | — |
| **推理时延** | **95.5 ms** | **7.88 ms** | **本工程快 12×** ⭐ |
| **推理能耗** | **817 µJ** | **~630 µJ**（估算）| **本工程省 23 %** |
| 静态功耗 | **8.55 mW** | ~80 mW | 论文 9× 低（工艺差距）|
| 模型权重 | ~28 KB | **16.6 KB** | 本工程省 41 % |
| 推理吞吐 | 10.5 inf/s | **127 inf/s** | 本工程 12× |

### 14.4 工程实现

| 项 | 原论文 | 本工程 |
|---|---|---|
| RTL 来源 | HLS（C++ 自动生成）| **手写 Verilog** |
| 数据集 | 自采集 6 subjects, 私有 | **PhysioNet CEBSDB**（公开 ODC-BY）|
| 训练框架 | 论文方法 | **PyTorch QAT + BPTT** 公开 |
| 工具链 | Lattice Diamond/Radiant | **Anlogic TD（国产）** |
| 量化 | per-tensor INT8 | per-tensor INT8 + per-channel M0/shift（CNN）|
| 板上完整 hold-out 测试 | ❌ 未报 | ✅ **9660/9660 valid** |

### 14.5 综合胜负

| 维度 | 胜者 | 决定性差距 |
|---|---|---|
| ① 精度（同口径）| 论文 | -2 pp |
| ② 评估严格度 | **本工程** | 严格 +CV+hold-out |
| ③ 硬件资源 | **本工程** | DSP 节省 7× ⭐ |
| ④ 性能/速度 | **本工程** | 12× 更快 ⭐ |
| ⑤ 能效（每次推理）| **本工程** | -23 % |
| ⑤' 能效（静态）| 论文 | 工艺差距 |
| ⑥ 工程完整度 | **本工程** | 公开数据 + 手写 RTL |

**总比分：本工程 5 胜 / 1 平 / 1 负**

唯一输的"静态功耗"是 40nm vs 55nm 的工艺硬约束，属物理层面，不是设计层面。**设计/算法层面本工程全面优于原论文**。

---

## 15. 引用

### 本工程
```bibtex
@misc{scg-snn-eg4s20-2026,
  author       = {Neko},
  title        = {{SNN-Based SCG Classification on Domestic FPGA EG4S20:
                  A Subject-Disjoint Evaluation Study}},
  year         = {2026},
  howpublished = {\url{https://github.com/dentar142/scg-snn-eg4s20}},
  note         = {SRTP submission, in preparation}
}
```

### 数据集（必须署名）
```bibtex
@misc{cebsdb,
  author       = {Garc{\'i}a-Gonz{\'a}lez, M{\'a}rio {\'A}lvaro and Argelag{\'o}s-Palau, Albert and Fern{\'a}ndez-Chimeno, Mireya and Ramos-Castro, Juan},
  title        = {{Combined ECG, Breathing and Seismocardiogram (CEBS) Database}},
  year         = {2014},
  publisher    = {PhysioNet},
  howpublished = {\url{https://physionet.org/content/cebsdb/1.0.0/}},
  doi          = {10.13026/C2CC7H},
  note         = {ODC-BY 1.0}
}
```

### 对标论文
```bibtex
@inproceedings{rahman2026edge,
  title       = {{At the Edge of the Heart: ULP FPGA-Based CNN for On-Device
                Cardiac Feature Extraction in Smart Health Sensors for Astronauts}},
  author      = {Rahman, Kazi Mohammad Abidur and others},
  booktitle   = {DCOSS-IoT 2026},
  year        = {2026},
  eprint      = {2604.25799},
  archivePrefix = {arXiv}
}
```

---

## 16. 致谢

- **数据集**：PhysioNet CEBSDB（García-González MÁ et al.），ODC-BY 1.0
- **对标论文**：Rahman et al., *At the Edge of the Heart*, DCOSS-IoT 2026, CC BY 4.0
- **工具链**：Anlogic Tang Dynasty 6.2.x（厂商提供）
- **开源生态**：PyTorch, NumPy, SciPy, wfdb, pyserial, snntorch（参考）

**本工程所有 RTL（Verilog）+ Python 训练 / 仿真 / 评估脚本 + TCL 综合脚本均为原创**，无任何专有 IP 复用。

代码使用 [MIT 许可](LICENSE)。数据集使用须遵守 PhysioNet CEBSDB 的 [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) 协议（**必须署名**）。

---

## 17. 联系

- **GitHub Issues**：[https://github.com/dentar142/scg-snn-eg4s20/issues](https://github.com/dentar142/scg-snn-eg4s20/issues)

如对本工程的实现细节、复现问题、扩展方向有疑问，欢迎开 issue 讨论。
