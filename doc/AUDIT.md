# 审计报告 — scg-snn-eg4s20

| 项 | 值 |
|---|---|
| 审计对象 | `dentar142/scg-snn-eg4s20`（Anlogic EG4S20 上的多模态 SCG LIF SNN 加速器） |
| 审计基点 | `main` @ `8ce0dd4`（`docs: hybrid CLAUDE.md policy for OMC + Superpowers`） |
| 工作树状态 | clean（仅 `?? .omo/` 未跟踪） |
| 审计方法 | 一手文件读取：RTL / tools / doc JSON / bitstream 哈希；所有结论附 `文件:行` 或 JSON 字段路径 |
| 审计范围 | 声明与证据一致性、位流可重建性、时序/延迟、量化链路、死代码、复现文档 |

---

## 摘要

**工程是真的，且质量不低。** RTL 手写、FSM 清晰；有独立的板级测量（不是纯仿真冒充）；Pareto 扫描、标定分析、跨数据集评估、bitstream 哈希表都在仓库里，且 README 的表 A 主动列出了"95.02% 是 5,000 窗"这一事实——这在个人项目里属于少见的诚实。

**但对外传播链上的五个核心问题互相叠加，且部署 bit 无法从仓库重建。** 具体是：

1. **95.02% 与 40,575 窗不是同一件事** —— README TL;DR 把 95.02%（5,000 窗口径）与"40,575 hold-out windows"写进同一句；全量 40,575 窗实测是 **94.14%**。
2. **跨数据集叙事选择性呈现** —— dropout 臂标定后**反而变差**（Δ = −2.14 pp），README 引用的 "+9.63 pp" 是跨口径相减。
3. **选优与报告集同源（selection-on-test）** —— 全部 9 个训练脚本都按 hold-out 集**逐 epoch 取最大 acc** 选 ckpt，而该 hold-out 集正是报告用的 40,575 窗（详见 F5）。报告精度是"50 个 epoch 中的最大值"，而非一次前向结果。
4. **部署 bit 不可重建（最严重）** —— HEAD 的 `rtl/scg_top_snn.v` 阈值属 T=32，而 `meta.json`/hex 属 T=16；`fe79395` 提交了产物却漏提交那个 `.v` 的改动。
5. **延迟约 6× 注水** —— 计时窗内含 `time.sleep(0.005)`；真实推理 ≈1.7–1.9 ms，而非 9.12 ms。

以下逐条给出证据。严重度标记：`P0` 影响可复现性/正确性主张，`P1` 影响对外数字，`P2` 文档卫生。

---

## 一、核心数字口径对照

| 宣称 | 出处 | 一手证据 | 差异 |
|---|---|---|---|
| 95.02% 板级精度 | README TL;DR / 徽章 | `doc/bench_fpga_snn_h32t16_aligned.json` → 95.02%，但窗数 = **5,000** | 与"40,575 窗"拼接使用 |
| 40,575 窗全量 | README 表 A（诚实） | `doc/bench_fpga_snn_multimodal_holdout.json` → `accuracy_percent=94.14171`, `macro_f1=0.913158` | 实为 94.14% |
| 板级延迟 9.12 ms | README 表 A | `doc/bench_fpga_snn_multimodal_holdout.json` → `run_ms.mean=8.6479`；计时窗含 `sleep` | 真实 ≈1.71–1.89 ms |
| 跨数据集 +9.63 pp | README 表 B | `doc/cross_dataset_cebsdb.json` → 两臂 Δ 分别为 **−2.14** 与 **+50.40** | 跨口径相减 |
| LUT 占用 10.76% | README 表 A | `doc/synth_best_snn_mm_h32_holdout.json` 等 → LUT4 10.69–10.71%，无 10.76% 来源 | 无支撑 |

补充：`doc/bench_fpga_snn_h32_t16_subsample.json` 为 5,000 窗下的 94.54% —— 即 5,000 窗口径自身在不同随机子集下也有 94.54 / 95.02 两个值。

---

## 二、关键发现

### F1 — 95.02% 与 40,575 窗是两个口径 `P1`

- 全量跑分：`doc/bench_fpga_snn_multimodal_holdout.json`，`accuracy_percent = 94.14171`，`macro_f1 = 0.913158`，`n_windows = 40575`，`run_ms.mean = 8.6479`。
- 95.02% 来源：`doc/bench_fpga_snn_h32t16_aligned.json`（5,000 窗子采样）。
- README 表 A 同时列出二者（诚实），但 TL;DR 与顶部徽章把 **95.02%** 与"40,575 窗"并置，读者会合成一个不存在的结论。
- 同一 40,575 窗口径亦出现在 `doc/abstention_h32_t16.json`（`baseline_acc = 0.94425`）与 `doc/cv_snn_foster_multimodal.json`、`doc/sweep_pareto.json`。
- `doc/dia_error_summary.md` 的 40,575 拆解为 8 名受试者（`sub003/sub006/sub009/sub013/sub020/sub021/sub024/sub026`，Dia GT 6,573）。

**动作**：TL;DR/徽章改用 94.14%（40,575），并把 95.02% 明确标注为"5,000 窗子采样"。

### F2 — 跨数据集标定叙事选择性呈现 `P1`

`doc/cross_dataset_cebsdb.json` 两条臂：

| 臂 | ckpt | zero-shot | 标定后 `mean_cal` | `mean_delta_pp` |
|---|---|---|---|---|
| dropout_aligned | `best_snn_mm_h32t16_dropout.pt` | 0.7807 | 0.8770 | **−2.14** |
| aligned_no_dropout | `best_snn_mm_h32t16_aligned.pt` | 0.4319 | 0.8381 | **+50.40** |

- README 引用 dropout 臂的 `mean_cal = 87.70%`（作为头条）**和** no-dropout 臂的 `+50.40 pp`（作为"标定有效"的证据），但这两者来自不同臂。
- dropout 臂的真实结论是：**标定使它变差 2.14 pp**。这正是"dropout 训练出的特征已经很鲁棒，再标定反而引入偏差"的合理现象——但它与 README 的"标定提升"叙事方向相反，且未被说明。
- `P_drop = 0.5` 的跨数据集鲁棒性结论本身有两条数据支撑（CEBSDB 78%、WESAD），这部分是可信的；问题只在标定那一段的表述。

**动作**：表 B 分臂列出，明确写出 dropout 臂标定 Δ = −2.14 pp。

### F3 — 部署 bit 无法从仓库重建（最严重）`P0`

**现象**：提交态阈值与权重不一致。

- `rtl/scg_top_snn.v:34-35`：`THETA1 = 24'sd13756`、`THETA2 = 24'sd1397` —— 溯源至 `best_snn_mm_h32_holdout.pt`（即 **T=32**）。
- `rtl/weights_snn/meta.json`：`theta1_int = 15380`、`theta2_int = 635`，且 `aligned_ckpt = model\ckpt\best_snn_mm_h32t16_aligned.pt` —— 即 **T=16**。
- 部署 bit 名为 `scg_top_snn_aligned_h32t16.bit`，对应 T=16。

**假注释已证伪**：`rtl/scg_top_snn.v:34` 写 `// overwritten by build .tcl from meta.json`。实测：

- `tools/build_snn.tcl` 仅 `set top scg_top_snn`、`read_hdl` 两个文件（`rtl/scg_top_snn.v`、`rtl/scg_snn_engine.v`）、`read_adc constraints/scg_top.adc`，随后 optimize/gate/legalize/place —— **全程无任何 θ 或 meta 覆盖**。
- `grep` 全部 `tools/*.tcl`（11 个）匹配 `THETA|theta|meta\.json|weights_snn|patch` → **零命中**。
- 唯一会改写 `scg_top_snn.v` 阈值的是 `model/export_snn_weights.py::patch_rtl_thetas()`（定义于 line 30，调用 131-132，`--patch-rtl` 默认 ON）。

**根因（git 溯源）**：

- `fe79395`（`feat(aligned): phase-aligned SNN closes half the CNN gap`）改了 19 个文件：`meta.json`、`W1.hex`/`W1_ch0..4.hex`/`W2.hex`、`build_snn/scg_top_snn.bit`、新建 `scg_top_snn_aligned_h32t16.bit`、`model/train_snn_mm_aligned.py`、`tools/export_aligned_weights.py`、`doc/bench_fpga_snn_h32t16_aligned.json`、manifest、SRTP 报告 —— **唯独不含 `rtl/scg_top_snn.v`**。
- `rtl/scg_top_snn.v` 最后一次改动是 `684b862`（`feat(multimodal): zero-leakage gold-standard FOSTER 5-channel SNN on EG4S20`）。

⇒ θ 是 `export_snn_weights.py` 的**就地改写副作用**。`fe79395` 提交了它的全部产物（hex/meta/bit/bench），却漏掉了它改写的那个源文件。所以这是**已提交的不一致**，不是工作树脏。

**影响**：bit 本身很可能是正确的（用 T=32 阈值跑 T=16 网络精度会崩，不可能得到 95%），但 **任何人从 HEAD 重建都会得到不同的 bit**。README 的"bit-identical to PyTorch"因此不成立（板 θ=13756/1397 vs sim 15380/635）。

**动作**：重跑 `export_snn_weights.py`（T=16 aligned ckpt）并提交 `rtl/scg_top_snn.v`；删除 line 34 假注释。一条命令即可恢复可重建性。

### F4 — 板级延迟约 6× 注水 `P1`

- `tools/bench_fpga_snn_holdout.py:86-99`：`run_ms = (t_c - t_b) * 1e3`，而 line 89 在计时窗内执行 `time.sleep(0.005)` —— 单次测量即注入 5 ms。
- 实测：T=16 bit → `9.124 ms`；T=32 bit → `8.648 ms`。**T 翻倍却更快 0.48 ms**，符号相反，本身即自证计时方法有问题。
- 精确 FSM 核算（50 MHz）：FC1 预计算 = 32×1280×2 = 81,920 cy（占 95.7%）；每 timestep 228 cy；T=16 → 3,648 cy，T=32 → 7,296 cy。
  - T=16 总计 ≈ 85,568 cy = **1.712 ms**
  - T=32 总计 ≈ 89,216 cy = **1.785 ms**
- 独立旁证：`rtl/scg_adc_stream.v:6` 自述 "FPGA inference is 3.6 ms (T=32) or **1.8 ms (T=16)**"，line 25 "1800 us (T=16)"。与核算误差 5%，而 README 报 9.12 ms（差 5×）。
- 附带：`rtl/scg_adc_stream.v:6` 的 "3.6 ms (T=32)" 亦错（应 1.785 ms），系同一错误认知的残留。

**动作**：把 `sleep` 移出计时窗，改用片内周期计数器（`$time`/cycle counter）重测，重写表 A。

### F5 — 逐 epoch 选优与报告集同源（selection-on-test）`P0`

**现象**：仓库里所有训练脚本都用"在 hold-out 集上逐 epoch 取最大 acc"来挑 ckpt，而这个 hold-out 集正是报告里用来算最终精度的那批窗。模型没有在测试集上训练，但**在测试集上做了模型选择**。

**证据**：
- `model/train_snn_mm_holdout.py:89,109-110,121`：`if val_acc > best_acc: best_acc = val_acc; best_epoch = e` → 按 hold-out acc 保存最优 ckpt。
- 同一模式遍布全部 9 个训练脚本：`train_snn_mm_aligned.py`（即部署的 aligned 模型）、`train_snn_mm_dropout.py`（跨数据集臂）、`train_cnn_mm_holdout.py`（CNN 对照）、`train_snn_multimodal.py`、`train_snn_v1.py`、`train_qat_v2.py`、`train_qat.py`、`finetune_ssl.py` —— 均为 `val_acc` 全局 argmax。
- **决定性**：`model/ckpt/best_snn_mm_h32_holdout_manifest.json` 记 `n_val_windows = 40575`，与 README 表 A 板级评估的 **40,575 窗完全相同**；`holdout_subjects` = sub003/006/009/013/020/021/024/026，正是 README 报告的那 8 位受试者；`best_val_acc = 0.9426`、`best_epoch = 30`（总 `epochs = 50`）。README 表 D 的 `32|32 → 94.26%` 就是这个 `best_val_acc`。

**含义**：报告的 94.26%（表 D）不是一次前向的测试精度，而是**50 个 epoch 在测试集上的最大值**，天然高于单次前向。同一机制污染全部 headline 数字：板级 94/95%、CNN 对照（表 C）、dropout 78.07%、STDP 87.70%、abstention 增益。真实泛化精度应低于报告值；缺口大小无法从现有产物反推（需重训），但方向确定且偏乐观。

**动作**：改为在**训练集内部**切验证集做 ckpt 选择，hold-out 仅用于最终一次性报告；或在 manifest 里显式记录"选择集 ≠ 报告集"。受影响数字需重跑：表 A、表 B、表 C、表 D。

---

## 三、次要发现

### F6 — 徽章与文档卫生 `P2`

- `[![Verilog: 100% hand-written]]` **为假**：`rtl/scg_mac_array_v7.v:1` = `// scg_mac_array_v7.v - AUTO-GENERATED from rtl/weights_v7/`。`rtl/` 内含自动生成物。
- 图片数量：实为 **19** 张（`doc/figs/*.png`），README 与 CLAUDE.md 写 "17"。
- ckpt：实为 **33 个 / 5.4 MB**，文档写 "5.3 MB"。
- README 文件清单只描述了 `rtl/` 9 个文件中的 3 个（`scg_top_snn.v`、`scg_snn_engine.v`、`scg_adc_stream.v` + `weights_snn/`）。

### F7 — Abstention τ 三处不一致 `P1`

| 来源 | 值 | 说明 |
|---|---|---|
| `doc/abstention_h32_t16.json` | `recommended_tau = 2` | 40,575 窗，`baseline_acc = 0.9443`；cov 0.8925，acc_kept 0.9778，acc_rejected 0.6654，n_kept 36,215，n_rejected 4,360 |
| `rtl/scg_abstention.v` | `DEFAULT_TAU = 8'd3` | header 注释 "Recommended tau_i: 8'd3" |
| `doc/calibration_report.md` | τ = 3 | 基于 9,660 窗（cov 72.26%，sel_acc 91.20%） |

README §11.2 引用的是 2。建议统一为 `recommended_tau`（2），并说明 3 来自更小窗口集。

### F8 — 传感器前端均为死代码 `P1`

已烧 bit 只能通过 UART 喂窗；"实时无主机采集"是路线图而非现状。

- `rtl/adxl355_spi.v` = **无逻辑桩**：自称 "SKETCH (Round 14)"，line 40 "full implementation deferred"，line 50 "TODO: implement state machine"。`always` 块（41-56）只做复位并拉低 `z_valid`，状态机恒停 `ST_IDLE`，`z_int8` 恒 `8'sd0` —— 永不产生输出。
- `rtl/scg_adc_stream.v` = 有 FSM（`S_IDLE→S_CONV→S_WAIT→S_READ→S_DONE`）但三处致命：
  1. **从未被实例化**（全仓库 grep 仅命中自身 module 声明 line 47）；
  2. **无引擎接口**：仅有 `sample_o[15:0]` / `sample_chan_o` / `sample_valid_o` / `window_ready_o`（65-68），没有向 X_BUF 写 1280 B 的握手；line 133 "tell engine to run" 只是 1 周期脉冲，无接收方；
  3. **位宽不匹配**：`ADC_BITS = 16`（signed 16 位）而引擎消费 INT8；16→8 下变频在本文件不存在。
- 上述两者均未进入 bitstream —— README 用 "skeleton" 描述 `scg_adc_stream.v` 是诚实的，但把 `adxl355_spi.v` 从文件清单里整个删掉了。

### F9 — 死代码与未实例化模块 `P2`

`rtl/` 共 9 个文件，定性如下：

| 文件 | 定性 |
|---|---|
| `scg_top_snn.v` | 部署用顶层（但 θ 见 F3） |
| `scg_snn_engine.v` | 被顶层实例化（核心引擎） |
| `scg_abstention.v` | 未实例化（见 F7） |
| `scg_adc_stream.v` | 未实例化（见 F8） |
| `adxl355_spi.v` | 无逻辑桩（见 F8） |
| `scg_top.v` / `scg_top_v7.v` | CNN 时代死代码 |
| `scg_mac_array.v` / `scg_mac_array_v7.v` | 死代码（后者 AUTO-GENERATED） |

另：`rtl/scg_abstention.v` 未被实例化 ⇒ 其 `DEFAULT_TAU=3` 对已烧 bit 完全无影响；接上后才会生效（届时应为 2）。

### F10 — 量化/验证链盲点 `P1`

- FC1 累加器无溢出保护：worst |I1| = 1280 × 127 × 127 = 20,645,120 vs INT24 有符号上限 8,388,607（**2.46×**）；典型值 ~886k 安全。属潜在风险。
- `tools/sim_snn.py:71` 使用 `X.astype(int32) @ ...` ⇒ 金模**永远无法复现** RTL 的 INT24 溢出行为，验证链对该失效模式是盲的。
- `rtl/weights_snn/meta.json` 的 `tau_int_baked = [4,5,5,6,13]` 与单引擎常量 `LEAK_SHIFT = 4` 语义不一致，未解释。
- `tools/export_aligned_weights.py` ~line 83 注释 "RTL is unchanged" 具误导性（θ 是模型相关的，会被改写）。
- 资源声明无支撑：README LUT 10.76% 无对应 synth JSON；`doc/synth_*.json` 显示 LUT4 10.69–10.71%、`BRAM32K = 0`（与引擎注释"W1 in BRAM32K"冲突）。`doc/synth_best_sweep_H64_T32.json` 的 `status = failed`。

### F11 — 复现文档缺陷 `P1`

- README 复现命令引用的 **`tools/dl_foster_osf.py` 不存在**（`Test-Path` = False）——第一步即失败。
- 其余 8 个引用脚本均存在：`tools/calibrate_abstention.py`、`tools/eval_cross_dataset.py`、`tools/export_aligned_weights.py`、`model/dataset_pipeline.py`、`model/dataset_pipeline_foster.py`、`model/train_snn_mm_aligned.py`、`tools/bench_fpga_snn_holdout.py`、`tools/download_jtag_snn.tcl` = True。
- `tools/synth_one_config.py` 的 CLI 已核实有效：`--ckpt`(required) / `--py`(默认 `D:/anaconda3/envs/scggpu/python.exe`) / `--td` / `--build-dir`(默认 `build_snn`) / `--bit-name`。
- README:215 引用的 `doc/bench_aligned.json` **不存在**，实际文件名是 `doc/bench_fpga_snn_h32t16_aligned.json`。

### F12 — 表 C / 表 D 的 "deployed" 行指错模型 `P1`

README 表 D 标注 "**deployed**" 的行 Sim acc = 94.43%（对应 `doc/abstention_h32_t16.json` 的 `baseline_acc = 0.94425`，即 sweep ckpt），而实际烧录的是 Aligned 版（表 C，Sim val 94.81%）。两个 "H32 T16" 其实是不同模型。

---

## 四、值得肯定

审计不应只列问题。以下部分是扎实的：

- **工程实现真实**：`rtl/scg_snn_engine.v` 完整实现了 leak（`v - (v >>> LEAK_SHIFT)`）、soft reset（`v - s*theta`）、二值 FC2、spike count、通用 argmax；每步 228 cy 与 FC1 预计算结构清晰。预算是 1 个 INT8×INT8 DSP MAC + ~1500 LUT —— 与 `doc/synth_*.json` 的 `DSP18 = 1` 吻合。
- **有独立板级测量**：不是仿真冒充实测；bitstream 有 SHA256 表；timing 报告显示约束 20.000 ns / Min 19.329 ns（Fmax ≈ 51.7 MHz），WNS 仅 +0.671 ns（3.4% 余量），0 violation / 3,290 endpoints —— 边界紧但是真的收敛了。
- **README 表 A 主动披露"95.02% 是 5,000 窗"**，且已知限制 #5 诚实标注 "dropout-aligned 未实际烧板"。这是问题 F1 之所以只是"传播层"问题而非"造假"的原因。
- **跨数据集鲁棒性有两条数据支撑**（CEBSDB 78%、WESAD 68%），`p_drop=0.5` 的配方是可复现的。
- **Pareto 扫描、标定分析、per-subject 误差拆解、STDP 个性化**等分析链路齐全，`doc/` 下 70+ 个 JSON/MD 产物可追溯。

判断：这是一个**由能力足够的作者完成、但在"对外说话"这一环上缺少一次完整复核**的项目。F3 的成因（改写副作用未随产物一起提交）恰恰是这种复核会抓到的东西。

---

## 五、优先级修复清单

| # | 严重度 | 动作 | 位置 |
|---|---|---|---|
| 1 | `P0` | 用 T=16 aligned ckpt 重跑 `export_snn_weights.py`，提交 `rtl/scg_top_snn.v`；删除 line 34 假注释 | `rtl/scg_top_snn.v:34-35` |
| 2 | `P0` | 发布 SHA256→bit→weights→meta 溯源链；使 T=16 weights/hex 与 T=16 RTL 源对齐 | `build_snn/*.bit` + `rtl/weights_snn/` |
| 3 | `P1` | `sleep` 移出计时窗，用片内周期计数器重测，重写表 A | `tools/bench_fpga_snn_holdout.py:86-99`（line 89） |
| 4 | `P1` | TL;DR/徽章拆分：95.02%(5,000) 与 94.14%(40,575)；LUT 10.76%→10.70% 或补 aligned synth | `README.md` |
| 5 | `P1` | 修正跨数据集标定叙事：分臂列出；明写 dropout 臂 Δ = −2.14 pp | `README.md` 表 B |
| 6 | `P1` | 统一 abstention τ 为 `recommended_tau`（2），说明 3 的来源 | `rtl/scg_abstention.v` / `doc/calibration_report.md` |
| 7 | `P1` | 表 D "deployed" 行改指 Aligned ckpt | `README.md` 表 D |
| 8 | `P1` | 修复复现文档：移除/补上 `tools/dl_foster_osf.py`；`bench_aligned.json` 改为正确文件名 | `README.md` |
| 9 | `P1` | sim 加 INT24 饱和以复现 RTL 溢出；或给累加器加饱和 | `tools/sim_snn.py:71` / `rtl/scg_snn_engine.v` |
| 10 | `P2` | 删除/修正 "100% hand-written" 徽章，或把生成物移出 `rtl/` | `README.md` 徽章 / `rtl/scg_mac_array_v7.v` |
| 11 | `P2` | 死代码迁 `legacy/`（`scg_top.v`、`scg_top_v7.v`、`scg_mac_array.v`、`scg_mac_array_v7.v`）；`adxl355_spi.v` 删除或标 unimplemented；README 文件清单补 `scg_abstention.v`、`adxl355_spi.v`；图数 17→19；ckpt 5.3→5.4 MB | `rtl/`、`README.md`、`CLAUDE.md` |

**已结案的溯源（原未决问题）**：提交态 T=32 θ（13756/1397）的写入者 = `model/export_snn_weights.py::patch_rtl_thetas()`，在 `fe79395` 中作为就地改写副作用发生；该 commit 提交了全部产物却漏提交被改写的 `rtl/scg_top_snn.v`。

**bitstream 哈希相同的含义（已证实）**：`fe79395` 同时（a）覆盖 `build_snn/scg_top_snn.bit`（671,536 B → 672,700 B），（b）新建 `build_snn/scg_top_snn_aligned_h32t16.bit`（0 → 672,700 B）。两者同为 672,700 B，故 SHA256 相同 —— 即 **"aligned" bit 是那次新构建 bit 的逐字节拷贝**，并非独立构建。`fe79395` 之前的原 `scg_top_snn.bit`（blob `ed5f1ebc89dec37f81b7edc8d08f3fa753288681`，671,536 B）在 HEAD 中存续为 `build_snn/scg_top_snn_sweep_H32_T16.bit`。

---

## 六、数据管线泄漏审计（已合并）

> 方法说明：原计划的 3 条后台审计线（`bg_ac6de032`/`bg_6628c78c`/`bg_0c6a9e99`）运行约 29 分钟后无任何输出，已全部取消。本节改为**直接读源码 + 逐行核对**得出，证据均为文件/行号。

**结论速览**

| 维度 | 判定 | 关键证据 |
|---|---|---|
| 受试者级泄漏（train/test 同人） | **无** | `tools/make_holdout_npz.py` 按 `np.isin(sid, holdout_sids)` 切分；manifest `n_train_subjects=32` + `n_holdout=8` = FOSTER 40 人，无交集 |
| 跨数据集归一化"不公平" | **不成立**（结构性差异，无实际影响） | 见下"归一化"小节 |
| 缩放/量化链 train↔eval 一致性 | **一致** | 见下"量化"小节 |
| 标签生成泄漏 | **无** | 标签由 ECG R-peak 推得，与 SCG 通道独立 |
| **选优污染（selection-on-test）** | **存在，`P0`** | 见 **F5** —— 这是本维度唯一成立且载荷最大的发现 |

**1. 受试者级切分：干净。** `model/train_snn_mm_holdout.py` 强制 subject-disjoint，并在 manifest 中固化了 train/hold-out 受试者名单；`best_snn_mm_h32_holdout_manifest.json` 的 `n_train_windows=162433`（32 人）与 `n_val_windows=40575`（8 人）相加为全体，无重叠。**不存在"训练见过测试受试者"的泄漏**，README "zero-leakage subject-disjoint" 在*切分*层面成立。

**2. 归一化：此前"跨数据集不公平"的怀疑不成立。** 两处 `normalize_int8` 定义不同：
- `model/dataset_pipeline.py:146-151`（CEBSDB）：`mu = x.mean()` / `sd = x.std()` 对**整窗**做 z-score；该管线窗口是**单通道** 1-D（`:188` `win = scg[start:start+WINDOW_LEN]`）。
- `model/dataset_pipeline_foster.py:78`（FOSTER）：docstring 明写 "Per-channel"，对 `(5, L)` 逐通道做 z-score；窗口为 `(5, 256)`（`:128`）。

对 CEBSDB 而言通道数 = 1，**"按窗"与"按通道"在数学上恒等**。因此该差异是*结构性*的（为 FOSTER 多通道而设计），对 FOSTER→CEBSDB 迁移**不产生任何口径差**。先前的"归一化口径不一致"表述过强，此处降级为中性说明。

**3. 量化/缩放链：train 与 eval 一致。** `model/export_snn_weights.py:91-98` 用 `scale = absmax/127`、`qw = clip(round(w/scale), -127, 127)`，输入 `in_scale = 1/127`，阈值 `theta1_int = round(threshold_fp/(in_scale*w1_s))`；`tools/sim_snn.py` 用同一 `in_scale` 前向。训练端 `model/finetune_ssl.py:77,79` 也是 `/127`。**无 train/eval 缩放错配**。（量化本身的 INT24 溢出风险另见 F10。）

**4. 标签生成：无跨通道泄漏。** Sys/Dia 标签由**同步 ECG** 的 R-peak 推出（`dataset_pipeline_foster.py` 的 R+50 ms / R+350 ms ±30，`BG_EXCLUSION=100 ms`），与 SCG 五通道信号独立；SCG 不参与打标，故不存在"用待分类信号给自身打标"。

**5. 遗留待核实（不改变上述结论）**：窗口在受试者内部的**步长**未核实。若步长 < 窗长（WINDOW_LEN=256 @1 kHz = 256 ms，与心搏周期同量级），同一受试者的相邻窗高度相关 —— 这不构成 train/test 泄漏（受试者已隔离），但会让 40,575 窗的**有效独立样本数远小于 40,575**，从而（a）放大评估方差、（b）让 F5 的"逐 epoch 取最大"更容易过拟合噪声。需读窗口提取的 step 参数坐实；若成立，应同时下调对 94/95% 置信区间的表述。

---

## 附录 A — Bitstream SHA256 表

| Bitstream | 大小 (B) | SHA256 前缀 |
|---|---|---|
| `scg_top_snn.bit` | 672,700 | `A15D2C5ACFEA91F0` |
| `scg_top_snn_aligned_h32t16.bit` | 672,700 | `A15D2C5ACFEA91F0` |
| `scg_top_snn_dropout_aligned.bit` | 672,700 | `E54BB7075D493103` |
| `scg_top_snn_multimodal.bit` | — | `3292F0ED886029BB` |
| `scg_top_snn_multimodal_holdout.bit` | — | `B9F0B3286F34FBCC` |
| `scg_top_snn_singlemodal_backup.bit` | 649,420 | `07620B44AF2C2081` |
| sweep H16_T32 | 651,748 | `44B240416B8A0A27` |
| sweep H32_T16 | 671,536 | `C924C30AE0C54566` |
| sweep H32_T48 | 672,700 | `7AE508039EE0258A` |
| sweep H32_T8 | 672,700 | `DA61A1C4B5A257EA` |

注：`scg_top_snn.bit` 与 `scg_top_snn_aligned_h32t16.bit` 哈希相同，二者均属 `fe79395`。

## 附录 B — ckpt → θ 映射（33 个 ckpt 中的代表项）

| ckpt | θ1 | θ2 | 备注 |
|---|---|---|---|
| `best_snn_mm_h32_holdout.pt` | 13756 | 1397 | **T=32；= 提交态 RTL 的 θ** |
| `best_snn_mm_h32t16_aligned.pt` | 15380 | 635 | **T=16；= meta.json 的 θ** |
| `best_snn_mm_h32t16_dropout.pt` | 13339 | 690 | 跨数据集 dropout 臂 |
| `best_holdout_snn.pt` | 44012 | 660 | |
| `best_snn_mm_h32.pt` | 10970 | 1924 | |
| `aligned_fold1` | 13109 | 881 | |
| `best_multimodal_final` | 19046 | 3045 | |
| `best_snn_5class` | 29481 | 440 | |
| `best_snn_v1` | 21872 | 499 | |

孤儿 ckpt（无 `fc1.weight`，无法导出）：`best.pt`、`best_cnn_mm_match`、`best_v1_nopool`、`best_v1_retrained`、`best_v2..v5`、`v5_excl100`。

## 附录 C — 证据索引

- 阈值公式：`model/export_snn_weights.py:98-99`、`tools/sim_snn.py:133-134`、`tools/calibration_analysis.py:181-182`
- θ 写入者：`model/export_snn_weights.py:30`（`patch_rtl_thetas()`）、调用 131-132
- 构建脚本：`tools/build_snn.tcl`（无 θ 覆盖）
- 顶层端口/常量：`rtl/scg_top_snn.v:110`（TX 常量）、270-271（θ 端口）
- 时钟/引脚：`constraints/scg_top.adc`（`clk_i`=R7 @50 MHz，`uart_rx_i`=F12，`uart_tx_o`=D12）
- 时序：`build_snn/scg_top_snn_route_timing.rpt`（20.000 ns 约束 / Min 19.329 ns / WNS +0.671 ns / 0 viol / 3290 endpoints）
- 面积：`build_snn/scg_top_snn_route.area`；解析器 `tools/synth_one_config.py::parse_area_report()`
- 复现脚本 CLI：`tools/synth_one_config.py`（`--ckpt/--py/--td/--build-dir/--bit-name`）

---

## 附录 D — 追加发现 #2：提交态 RTL 的 θ 与 meta.json 不一致（P0）

> 本节为审计**后补**记录，对应核心发现 F3（`rtl/scg_top_snn.v:34` 注释与实际不符）。F3 只记录了「注释说谎」；本节补齐**「RTL 常量本身与已发布权重不一致」**这一更强的结论，并说明为何本次**不修**。

### D.1 事实

| 来源 | θ₁ | θ₂ | 对应 T |
|---|---|---|---|
| `rtl/scg_top_snn.v`（提交态，THETA1/THETA2 常量） | 13756 | 1397 | **T=32** |
| `rtl/weights_snn/meta.json`（已发布权重侧的 θ） | 15380 | 635 | **T=16** |
| `model/ckpt/best_snn_mm_h32t16_aligned.pt`（当前烧录 bit 的训练 ckpt） | 15380 | 635 | **T=16** |
| `model/ckpt/best_snn_mm_h32_holdout.pt`（T=32 holdout bit） | 13756 | 1397 | T=32 |

结论：提交进仓库的 `rtl/scg_top_snn.v` 里硬编码的是 **T=32 holdout 权重** 的 θ；而 `rtl/weights_snn/` 与 `meta.json` 描述的是 **T=16 aligned** 权重。第 34 行注释声称「THETA 与 meta.json 一致」为**假**。

### D.2 为什么不构成「已证实的功能错误」

- 仓库内**没有**任何构建期产物记录「某 bit 由哪组 θ 综合而成」；θ 是 RTL 源码常量，综合后不可从 `.bit` 反读。
- `build_snn/*` 全部产物 mtime 统一为 `2026-09-17 16:08`（checkout 时间戳），**无法据文件时间推断构建先后**。
- 因此存在两种世界、且当前证据无法区分：
  - **世界 A**：当年综合 `scg_top_snn_aligned_h32t16.bit` 时，`.v` 里的 θ 恰好是 T=16 的 15380/635（后被人改回 T=32，或在另一分支改回），则**板上 95.02 % 的实测是真的**。
  - **世界 B**：综合时 `.v` 里就是 13756/1397（T=32），则板上实测对应的**不是** T=16 模型，95.02 % 与 T=16 的 sim 数字属于**不同的 θ**。
- 判定世界 A/B 需要**用 Anlogic TD 以指定 θ 重新综合并上板对比** —— 属超出本次文档修复的授权范围。

### D.3 已交付 bit 的可复现性缺口

- 已发布 bit `scg_top_snn_aligned_h32t16.bit` 的 SHA256 前缀 `A15D2C5ACFEA91F0` **与** `scg_top_snn.bit` **完全相同**（见附录 A），二者同属提交 `fe79395`。
- 但 `fe79395` **未包含** `rtl/scg_top_snn.v` 的 θ 变更 —— 即「已交付 bit ↔ 源码状态」之间**缺少可追溯链接**：从当前 HEAD 出发**无法重建**出那个 bit。
- 这是比 θ 数值本身更根本的缺口：**交付物不可复现**。

### D.4 建议修复（本次**未执行**，需重新授权）

| 优先级 | 动作 | 备注 |
|---|---|---|
| P0 | 以 T=16 aligned ckpt 重跑 `model/export_snn_weights.py`（其 `patch_rtl_thetas()` 会把 RTL θ 改写成 15380/635），删除 `:34` 假注释，提交 `rtl/scg_top_snn.v` | 使 RTL 与 `meta.json` 一致 |
| P0 | 建立并发布 **SHA256 → bit → 权重 → meta** 的完整链条（`build_snn/*.bit` + `rtl/weights_snn/`），每次构建落档 θ | 根除「bit 不可复现」 |
| P1 | 用 Anlogic TD 以两组 θ 各综合一次并上板，判定世界 A/B | 关掉 D.2 的不确定性 |

> **本次范围声明**：仅执行文档口径修正（#1）与本追加记录；**未改动任何 RTL、未重新综合、未重训**。

---

## 附录 E — #1 口径修正记录（本次已执行）

针对核心发现 F1（95.02 % ≠ 40,575 窗口）与 F4（延迟计时窗口含 `sleep`），本次只做**文档层面的口径标签化**，不改任何数字、不改任何 bit：

| 文件 | 位置 | 修正内容 |
|---|---|---|
| `README.md` | badge | `board acc 95.02%` → `board acc-95.02% (5k-win subsample)` |
| `README.md` | TL;DR | 明确 95.02 % 为 **5,000 窗分层子采样**；全测 40,575 窗 = 94.14 %（T=32 holdout bit，非当前烧录） |
| `README.md` | aligned 表 | 保留 `95.02 %` 并加 protocol 附注 |
| `README.md` | 目录树 / bib | 同步标注 5,000 窗子采样与 T=32 来源 |
| `CLAUDE.md` | 项目事实 | 同口径标注 |
| `doc/SRTP_FINAL_REPORT.md` | §9.5 结论 | 拆分「当前烧录 T=16 aligned @ 5,000 窗 = 95.02 %」与「T=32 holdout @ 40,575 窗 = 94.14 %」，并声明不可互换引用 |
| `doc/SRTP_FINAL_REPORT.md` | §11.4 CNN 对比表 | `94.14 % on-board` → 补 bit 与窗口集限定 |

未修改项：`doc/bench_fpga_snn_h32t16_aligned.json` 等原始 bench JSON（数据源保持原样）。

---

## 附录 F — 静态审查：导出器错配（wrong-exporter）footgun

> **本次范围**：仅静态审查（读源码 + `git`/文件校验）。**未运行任何导出脚本、未综合、未上板、未改动 RTL 与权重**。行号对应 `8e49289` 提交态。
>
> 本附录的目的是把「按 README 步骤操作可能静默丢 τ」这件事写成一份**可复查的永久记录**，而不是修正任何数字。

### F.1 两个导出器，职责不同

| 导出器 | FP32 → INT8 hex | 是否烘 τ | 是否写 `meta.json` | 是否 patch RTL θ/T |
|---|---|---|---|---|
| `model/export_snn_weights.py` | ✅ | ❌ | ✅（整字典重写）| ✅ **默认开**（`--no-patch-rtl` 可关，L73-74）|
| `tools/export_aligned_weights.py` | ❌（转调前者）| ✅ | ✅（转调后**回填** aligned 键）| ✅ **间接**（转调时未加 `--no-patch-rtl`，L62-66）|

两条事实决定了整个 footgun：

1. `model/export_snn_weights.py:86` 读的是**原始** `state["fc1.weight"]`，**不含 τ**；
2. 全仓库内施加 τ 列置换的唯一位置是 `tools/export_aligned_weights.py:46` 的 `np.roll(..., axis=-1)`（在 `win_len` 轴内按通道循环移位，即**通道内列置换**）。

⇒ **只要绕过 aligned 导出器，τ 就不在权重里。**

### F.2 footgun 链条（按 README 部署流程）

`README.md` 部署流程（L204-219）第 1、2 步是**两个各自独立会写 `rtl/weights_snn/` 的导出动作**：

| 步 | 命令 | 对 `W1.hex` | 对 `meta.json` | 对 RTL θ/T |
|---|---|---|---|---|
| 1 | `tools/export_aligned_weights.py --ckpt best_snn_mm_h32t16_aligned.pt` | 写**已烘 τ**（aligned）| 写后**回填** `tau_int_baked`/`aligned_ckpt` | patch 为 T=16 / θ=15380,635 |
| 2 | `tools/synth_one_config.py --ckpt best_snn_mm_h32t16_aligned.pt` → `model/export_snn_weights.py` | **覆写为原始未烘 τ** | **覆写为无 aligned 键** | patch 为 T=16 / θ=15380,635 |

`tools/synth_one_config.py:54-58` 调用 `model/export_snn_weights.py --out rtl/weights_snn --leak-shift 4` 时**没有** `--no-patch-rtl`，也**没有**任何再烘 τ 的步骤。于是：

- `W1.hex` + `W1_ch{0..4}.hex` 从 aligned 退回**原始列序**；
- `meta.json` 里的 `tau_int_baked` / `aligned_ckpt` **被抹掉**；
- 但步骤**本身不报错**，命令行输出看起来完全正常。

⇒ **第 2 步会静默撤销第 1 步的 τ-bake，而 README 把两步并列写成标准流程。**

### F.3 为什么 θ 不受影响（footgun 的爆炸半径很窄）

τ-roll 是**通道内的列置换**（`export_aligned_weights.py:46`，`axis=-1`），它**保持矩阵元素的多重集不变**。而：

- `model/export_snn_weights.py:90-91` 的 `w1_s` 由 `W1` 的 **absmax** 导出；
- `model/export_snn_weights.py:98` 的 `theta1_int = round(threshold_fp / (in_scale * w1_s))`。

两者都只依赖 absmax ⇒ **对列置换不变**。因此：

> 第 2 步（错配导出）**不会**改变 θ，也**不会**改变 T。它只把 **`W1.hex` 的列序**打回原始排列。

这条性质让 footgun **可证**（不依赖上板）且**狭窄**（只坏一处）。

### F.4 工具自身的两处错误陈述

`tools/export_aligned_weights.py` 内部有两条与代码**直接矛盾**的说明（本次已逐行复核）：

| 位置 | 原文（要点）| 实际行为 |
|---|---|---|
| docstring **L10** | "No RTL changes needed." | L62-66 转调 `export_snn_weights.py` 时未加 `--no-patch-rtl` ⇒ **会** patch RTL θ/T |
| **L83** `print` | "Done. RTL is unchanged; ..." | 同上，**不成立** |

附带一处**表述歧义**（非错误）：`README.md`「关键 RTL 创新」第 2 条写「τ 烘进 W1 …… **RTL 完全不变**」。按**架构**读是对的（τ 机制不需要任何新增 RTL 支持）；按**字面**读与 L62-66 矛盾（脚本确实会改 θ/T 字面量）。建议后续把该句改为「**τ 机制无需新增 RTL 支持**」。

### F.5 覆盖语义：为什么 aligned 路径能自愈，错配路径不能

- `model/export_snn_weights.py:109-122` 的 `meta` 是一个**全新的 dict 字面量**，L123 `write_text(json.dumps(meta))` **整体覆盖、不 merge** ⇒ aligned 专属键必然被抹掉。
- 但 `tools/export_aligned_weights.py:76-81` 在转调**之后**把 `meta["tau_int_baked"] = tau_int`（L79）与 `meta["aligned_ckpt"]`（L80）**回填**并重写（L81）。

⇒ **aligned 路径自洽（自愈）；错配路径（步骤 2 / 裸 `export_snn_weights.py`）会留下被抹净的 `meta.json`。**

### F.6 后果量化（预期值，非本次实测）

`README.md` §11.7 记录 τ=[4,5,5,6,13] 带来板上 **+0.48 pp**；同一 5,000 窗分层子采样下：

| 状态 | 板上 acc（同窗口集）|
|---|---:|
| `scg_top_snn_sweep_H32_T16.bit`（无 τ）| 94.54 % |
| `scg_top_snn_aligned_h32t16.bit`（烘 τ）| **95.02 %** |

⇒ 若按 README 顺序跑完第 2 步后直接综合，**预期落回 ≈ 94.54 %（−0.48 pp）**。这是**推论**（由 F.3 的置换不变性 + §11.7 的实测差值导出），**未**在本次运行中复现。

### F.7 与附录 D 的关系（加强 D.3，不替代 D.3）

两个反证说明「提交态 RTL」与「提交态权重」**不是同一次导出产生的**：

1. 提交态 `rtl/weights_snn/meta.json` **含** `tau_int_baked`/`aligned_ckpt`，而这两个键**只有** `tools/export_aligned_weights.py` 会写（L76-81）⇒ `rtl/weights_snn/` 的**最后写入者不是** `synth_one_config.py` 路径。这与 `tools/build_snn.tcl` **不含导出步骤**的事实一致（综合只读 RTL + hex）。
2. `fe79395`（新增 `scg_top_snn_aligned_h32t16.bit`）**未改动** `rtl/scg_top_snn.v`，而该提交态的 RTL 字面量是 T=32 / θ=13756,1397（= T=32 holdout 的值，见附录 D 表 D.1）⇒ 该 bit 综合时 RTL 的 θ/T 来自 **T=32** 的 ckpt，而非 aligned。

两点合起来正是附录 D.3「delivered bit ↔ source 链接缺失」的**机制级候选解释**（两个导出器各自独立 patch RTL，提交产物是混合态）。但**哪一种顺序真实发生过，本附录无法判定**——这正是 D.4 要求的 P0「SHA256 → bit → 权重 → meta 完整链条」要解决的问题。

> ⚠️ 同时注意：`8e49289` 把 RTL 字面量修为 T=16 / θ=15380,635 后，**并未重新综合任何 bit**。因此当前提交态 RTL 描述的是一个**从未被综合过**的设计，而任何已提交 bit 都不是从它产出的。源码↔meta 现在一致，**源码↔bit 仍未闭合**。

### F.8 操作规则（写给未来的自己）

- ❌ **不要**单独运行 `model/export_snn_weights.py --ckpt <aligned ckpt>` 作为部署路径——它不烘 τ。
- ❌ **不要**把 `tools/synth_one_config.py`（第 2 步）当作「只综合、不动权重」——它在 L54-58 会**重写** `W1.hex` 与 `meta.json`。
- ❌ **不要**在 aligned 部署后追加一次错配导出「顺便再综合一遍」。
- ✅ 部署 aligned 设计时，**最后**一个写 `rtl/weights_snn/` 的动作必须是 `tools/export_aligned_weights.py`。
- ✅ 每次导出后，**校验** `rtl/weights_snn/meta.json` **必须含** `tau_int_baked` 与 `aligned_ckpt` 两键；缺失即说明 τ 已被抹掉，必须重跑 aligned 导出再综合。
- ✅ 校验 `meta.json` 的 `W1_bytes`(=40960) 与 `W1.hex` 实际大小一致，避免沿用上一次的 hex。

### F.9 建议（沿用附录 D 的编号体系，不新增 P0）

| 优先级 | 建议 | 目的 |
|---|---|---|
| P0（承 D.4）| 把 F.8 的「导出后校验 meta.json 必须含 aligned 两键」写成 `tools/export_aligned_weights.py` 结尾的 **assert**，失败即非零退出 | 让 footgun **从静默变响亮** |
| P1（承 D.4/P1）| 修 `tools/export_aligned_weights.py` 的 L10 / L83 错误陈述（改为「会以 named ckpt 的 T/θ patch RTL」）| 文档与代码一致 |
| P1 | `README.md`「关键 RTL 创新」第 2 条的「RTL 完全不变」改为「τ 机制无需新增 RTL 支持」| 消除歧义 |
| P2 | 在 README 部署流程第 1/2 步之间加一行警示：第 2 步会重写 `W1.hex`/`meta.json` | 降低误操作概率 |

> **本附录未执行任何 P0/P1/P2 修复**；仅记录。以上建议涉及脚本改动，需另行授权。
