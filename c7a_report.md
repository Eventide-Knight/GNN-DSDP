# C7a 报告：3,000 个 train 正式训练 + 968 个 val 评估

时间：2026-11-18 20:02（Asia/Shanghai）
任务书：§3「模型」「评估与训练口径」④、§7 C7a 行
**结论：PASS —— val ROC-AUC = 0.8943（> 0.8 门槛），val PR-AUC = 0.3978，best epoch = 86。**

## 0. 数据补齐（本轮前置）

| split | 复合物 | 残基 | 正样本 | pos% | 0 skipped | id 集合核对 |
|---|---|---|---|---|---|---|
| train（图） | 3000 | 1,350,188 | 58,463 | 4.33% | ✅ | — |
| **val** | **968** | **474,342** | **19,020** | **4.01%** | ✅ | `.pt` == `.npz` == split 文件（968/968/968）|
| **test** | **363** | **130,222** | **7,290** | **5.60%** | ✅ | `.pt` == `.npz` == split 文件（363/363/363）|

- `build_dataset.py --split val`：968 个，0 skipped（1.5 min）
- `graph.py --batch --split val --workers 16`：968 个，0 skipped（87 s）；test 363 个（23 s）
- test 图**只建不用**，留给 C7b 评估；本轮训练/选模型全程只碰 train + val。

## 1. 训练配置（gnn_site/train_full.py）

| 项 | 值 |
|---|---|
| 设备 | **NVIDIA GeForce RTX 4090**（slurm 作业 159725，cn1，`-p gpu1 --gres=gpu:1 --cpus-per-task=8 -t 4:00:00`）|
| **排队时长** | **0 分钟**（提交即 RUNNING；提交时全集群 24 pending / 39 running，但恰好有空卡）|
| 训练/验证 | 3000 / 968 个复合物 |
| 模型 | GCN 4 层 46→38→28→18→8 + 输出 8→2 + log_softmax，**3,570 参数**，dropout 0 |
| 损失 | NLLLoss，`weight=[1.0, **22.09**]`（按训练集实际 neg/pos 自动算：58,463 正 / 1,350,188 总）|
| 优化器 | Adam lr=0.01（无 weight decay），梯度裁剪 10 |
| batch / epoch | 32 / **100**（patience=15，按 val PR-AUC 早停，未触发——第 86 epoch 后到 100 一直在 1 个点内浮动）|
| 随机种子 | 0 |
| 总耗时 | **186.8 s**（≈1.8 s/epoch，第一个 epoch 含数据加载 7.9 s）|

## 2. 训练曲线（节选；完整 100 行在 `reports/c7a_train.json`）

| epoch | train_loss | val_loss | val PR-AUC | val ROC-AUC | 用时 |
|---|---|---|---|---|---|
| 1 | 0.6060 | 0.5530 | 0.1838 | 0.7904 | 7.9 s |
| 2 | 0.5533 | 0.5369 | 0.2267 | **0.8040** | 1.8 s |
| 5 | 0.4949 | 0.4875 | 0.2713 | 0.8464 | 1.8 s |
| 10 | 0.4637 | 0.4858 | 0.2833 | 0.8527 | 1.7 s |
| 20 | 0.4391 | 0.4485 | 0.3341 | 0.8710 | 2.0 s |
| 50 | 0.3993 | 0.4468 | 0.3786 | 0.8857 | 1.8 s |
| 70 | 0.3888 | 0.4273 | 0.3942 | 0.8914 | 1.9 s |
| **86（best）** | **0.3791** | **0.4128** | **0.3978** | **0.8943** | 1.8 s |
| 100 | 0.3776 | 0.4188 | 0.3823 | 0.8913 | 1.7 s |

- ROC-AUC 在第 2 个 epoch 就过 0.80，之后缓慢爬到 0.894 平台（见 `reports/c7a_curve.png` 右图）。
- train_loss 单调下降到 0.378，val_loss 到 0.41 后走平 → **轻微过拟合但不严重**；模型只有 3,570 参数，主要瓶颈是容量/特征，不是过拟合。

## 3. 最终指标（best epoch 86，独立复算）

| 指标 | 数值 |
|---|---|
| **val PR-AUC** | **0.3978**（随机基线 = 正样本比例 ≈ 0.040，≈ **10× random**）|
| **val ROC-AUC** | **0.8943** |
| val loss | 0.4128 |
| 评估规模 | 968 个复合物 / 474,342 残基 / 19,020 正样本 |
| 参考 | NodeCoder 原论文 ROC-AUC ≈ 0.88，本模型 0.894 同量级 |

用独立进程重新加载 `c7a_best.pt` 在 968 个 val 图上复算：`PR-AUC=0.3978 ROC-AUC=0.8943`，与训练日志**完全一致**。

## 4. 判定与对照实验

- **PASS**：val ROC-AUC 0.8943 > 0.8 门槛。
- 任务书允许"若 ROC-AUC < 0.75 才做 1~2 组对照"——本模型 0.894，**未触发**，因此没有做 lr/隐藏维/dropout 的额外对照（也符合"不要超参搜索马拉松"）。
- test 363 **全程未参与训练和选模型**，只在第 0 步建好了图，留给 C7b。

## 5. 产物

| 文件 | 说明 |
|---|---|
| `gnn_site/c7a_best.pt` | 55 KB，含 `model_state` / `optimizer_state` / `epoch=86` / `val_pr_auc=0.3978` / `val_roc_auc=0.8943` / `args` |
| `gnn_site/reports/c7a_train.json` | 完整 100 epoch 日志 + 配置 + 类别权重 + 最终指标 |
| `gnn_site/reports/c7a_curve.png` | 左：train/val loss；右：PR-AUC / ROC-AUC 随 epoch |
| `gnn_site/reports/c7a_gpu_slurm.out` | 作业原始日志（含设备、每 epoch 行）|
| `gnn_site/train_full.py` | 正式训练脚本（不动 C6 的 train.py）|
| `gnn_site/c7a_gpu.sbatch` | slurm 提交脚本 |

## 6. 说明 / 坑

1. **GPU 排队 0 分钟**，所以第 2.5 步（可选 C2 基线）**按约定没有做**（该步只在排队时执行）。C2 仍需在 C7b 前补上，作为对比表分母。
2. 提交时 `sinfo` 显示三个 GPU 分区全 Allocated（3/0/0/3），但实际有空卡释放，作业秒进——说明 `sinfo` 的 Allocated 计数有滞后，以后照样先提交再说。
3. val 的 `p_pos_mean = 0.256`，明显高于真实 4.01%，是 `pos_weight=22` 的已知副作用；**PR-AUC / ROC-AUC 不受影响**，但将来把概率转成盒子阈值（C8）时要重新校准或直接扫阈值。
4. 100 epoch 只训练 ~3 分钟，C7b 全量 16,379 预计 ≈16,379/3,000 × 187 s ≈ **17 分钟**（同配置，不含数据加载），可行。

## 7. 下一步

**C7b**：用全量 16,379 个 train 重训，产出最终 `best_model.pt` + `predict.py` + 363 test 上的 AUC（test 图已就绪）。
