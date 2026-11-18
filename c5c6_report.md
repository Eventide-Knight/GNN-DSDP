# C5 / C6 报告：model.py（4 层 GCN）+ train.py（极小数据冒烟）

时间：2026-11-18 19:45（Asia/Shanghai）
环境：`$PY`（torch 2.8.0+cu128 / PyG 2.6.1 / sklearn 1.4.0）
依据：任务书 §3「模型」、§4 Step 4、§7 C5/C6 行；对照 `$NODECODER_DIR/src/NodeCoder/gcn/{NodeCoder.py,NodeCoder_train.py}`

## C5：模型结构

`gnn_site/model.py` → `GCN_SiteModel`，照抄 NodeCoder 的 StackedGCN 思路（GCNConv 堆叠 + ELU + log_softmax 头）：

| # | 层 | 输出形状（13gs: 418 节点） | 参数量 |
|---|---|---|---|
| — | 输入 x | [418, 46] | — |
| 1 | GCNConv(46→38) + ELU | [418, 38] | 1,786 |
| 2 | GCNConv(38→28) + ELU | [418, 28] | 1,092 |
| 3 | GCNConv(28→18) + ELU | [418, 18] | 522 |
| 4 | GCNConv(18→8) + ELU | [418, 8] | 152 |
| 5 | GCNConv(8→2) | [418, 2] | 18 |
| — | log_softmax(dim=1) | [418, 2] | — |
| | **合计** | | **3,570** |

边权重用的是 `graph.py` 的 `edge_weight = 1/d²`（GCNConv 原生支持）。

### 为什么选 8 → 2 + log_softmax（而不是 8 → 1 + sigmoid）

1. NodeCoder 原版就是 `log_softmax` 输出 + NLLLoss，任务书要求"先照抄，跑通再说"。
2. NLLLoss 支持 `weight=[w_neg, w_pos]`，是处理 ~4% 正样本不平衡最直接的方式（NodeCoder 的 weighted_loss 就是这套）。
3. 两列 logits 保留显式的背景类，`exp(out)` 直接得到"残基×类别"概率表，PR-AUC 取第 1 列即可。
4. 仍保留 `--head sigmoid`（8→1 + sigmoid），实测同一套特征/图也能跑通：

```
python model.py --head sigmoid --self-check
  output sigmoid  [411, 1]  finite=True   p(pos) min/max/mean = 0.0056 / 0.4466 / 0.0764
  RESULT: PASS
```

### 自检结果（未训练、未存 checkpoint）

| 图 | 节点 | 边 | 各层形状 | 输出 finite | 概率行和 | 结论 |
|---|---|---|---|---|---|---|
| `data/graph/train/13gs.pt`（train） | 418 | 7096 | 38/28/18/8/2 | ✅ | [1.000000, 1.000000] | PASS |
| `reports/graph_6qqw.pt`（test） | 411 | 7306 | 38/28/18/8/2 | ✅ | [1.000000, 1.000000] | PASS |

- `RESULT: PASS (params=3570, head=logsoftmax, graphs=2)`
- 无 NaN / inf（logits、每层隐藏层、输出逐项检查）
- 未训练、未落任何 checkpoint 文件

## C6：训练冒烟

`gnn_site/train.py`，命令：

```bash
python train.py --train-limit 30 --val-limit 10 --epochs 2 --batch-size 8 \
                --lr 0.01 --weight-mode linear --device cpu
```

配置：Adam(lr=0.01)、NLLLoss(weight=[1.0, **19.64**])（train 30 图：602 正 / 12424 总，neg/pos=19.64）、按 val PR-AUC 早停（patience=5，本次未触发）。

### 1. mini-batch 与 batch 向量（正确性断言）

```
[batch] graphs=8 nodes=3271 edges=61154 | batch vector (3271,) max+1=8 ok=True
```
`batch.batch.max()+1 == num_graphs == 8`，`batch.batch.numel() == x.size(0) == 3271`，`y.numel()==3271` —— 多图打包/`batch` 向量用法已验证。

### 2. loss 曲线与收敛证据

每个 mini-batch 的 loss（epoch 1 / epoch 2 各 4 步）：

| epoch | step1 | step2 | step3 | step4 | epoch 均值 |
|---|---|---|---|---|---|
| 1 | 0.9262 | 1.2006 | 0.7407 | 1.0530 | **0.9641** |
| 2 | 0.7208 | 0.6853 | 0.7102 | 0.6357 | **0.6843** |

- 8 步 loss **全部有限**（`loss_all_finite: true`）
- epoch 均值 0.964 → 0.684（**-29%**）
- 控制性过拟合（8 个图、60 步）作为"确实在学"的硬证据：**0.6861 → 0.2896（-57.8%）**

### 3. backward + step

```
[train] weight delta after training: max|dw|=7.33e-02
```
`weights_changed: true`，梯度范数每步 1.1~13.1，无 NaN 梯度、无异常中断。

### 4. 验证集（10 个 val 图，4952 残基 / 235 正样本）指标

| epoch | PR-AUC | ROC-AUC | P@0.5 | R@0.5 |
|---|---|---|---|---|
| 1 | 0.0538 | 0.5782 | 0.054 | 0.613 |
| 2 | **0.0750** | **0.6820** | 0.079 | 0.247 |

- 随机基线 PR-AUC ≈ 正样本比例 ≈ 0.047；2 个 epoch / 30 个复合物下 PR-AUC 0.075、ROC-AUC 0.682，**高于随机但远未收敛**——这是冒烟该有的量级，真实训练在 C7a。
- 指标计算走 `sklearn.metrics.average_precision_score / roc_auc_score`，作用在 `exp(log_softmax)[:,1]` 上。

### 5. 资源情况

- **全程 CPU**，无 GPU 排队问题（env_check 第 6 节提到 GPU 分区常被占满）。
- **无 OOM**：模型 3,570 参数、单 batch 8 图 / 3,271 节点 / 61,154 边，峰值显存/内存无压力。
- 整个 C6 冒烟 wall time **11.6 s**。

### 6. 判定

```
RESULT: PASS
loss_all_finite=true, epoch_loss_decreased=true, overfit_loss_decreased=true,
weights_changed=true, batch_vector_ok=true, metrics_ok=true, no_oom=true
```

## 交付文件

| 文件 | 说明 |
|---|---|
| `gnn_site/model.py` | 4 层 GCN + 输出头，自检 CLI |
| `gnn_site/train.py` | 训练循环（NLLLoss/Adam/早停/PR-AUC），冒烟 CLI |
| `gnn_site/reports/train_smoke.json` | 完整配置 + 每步 loss + 每 epoch 指标 + 断言结果 |
| `gnn_site/reports/train_smoke_curve.png` | 左：每步 train loss；右：60 步过拟合曲线 |
| `gnn_site/data/graph/val/*.pt` | 10 个 val 图（本次为跑 val 指标新建） |

## 下一步

**C7a**：用 3,000 个 train 复合物训练，验证集 **ROC-AUC > 0.8** 才放行 C7b（全量 16,379）。
C6 的 30 图 / 2 epoch 只证明管线通，不构成任何效果结论。
