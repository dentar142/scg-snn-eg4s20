现状汇总
# Hold-out Test Set 计划（用户要求）

为避免 CV 之外的"答辩用最终精度数据"也被沾染，按下面规则保留独立测试集：

## 当前数据划分
- 19 个 CEBSDB 受试者（b001..b020 缺 b006）
- 5-fold subject-disjoint CV 用了**全部 19 个**——SNN mean = 85.48 ± 2.02 %
- 这是研究/调参用的；下次重训前必须切出 hold-out 集

## 拟定 hold-out 方案
- **Test A: 4 个 CEBSDB 受试者** 整体扣出（永不进训练 / 永不进 CV）
  - 选 b002, b007, b015, b020（覆盖原 5 fold 的不同 fold）
  - 剩 15 受试者继续做 CV 训练 + 调参
- **Test B（cross-domain）：PhysioNet 2016 PCG**
  - 完全异构数据集，永不进训练
  - 仅在最终交付前评估一次

## 报告纪律
- CV mean ± std → 用于模型/超参选择
- Test A acc → 论文/答辩中"最终精度"那一行的数字（**只在最后一次跑**）
- Test B acc → "跨域泛化"那一节的数字

## 实现 TODO
- 修改 `dataset_pipeline.py` 加 `--holdout-records b002,b007,b015,b020`
- 拆出 `data_excl100/holdout.npz` 永远不进 train/val
- 重训之前先确认 `holdout.npz` 存在且未被读过
