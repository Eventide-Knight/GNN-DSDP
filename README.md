# GNN-DSDP：残基级 GNN 位点预测 + DSDP 对接（组内 demo）

> 一句话：把 DSDP 第一段"位点预测"从 3D-CNN 体素（PUResNet）换成**残基级 GCN**，
> 输出每个残基的配体结合概率，再把高概率残基变成对接盒子，接回 DSDP 原有 CUDA 对接引擎，
> 在 PDBBind time split 上和原版 DSDP 基线做同一口径对比。

代码与数据全部在集群：**`/share/home/huangyy/DSDP/gnn_site/`**（本文所有路径均为集群绝对路径）。

---

## 1. 结果速览（数字与 `results/table_all.json` 对齐）

### 表 1：残基级位点预测（test 363，5 Å 标签，130,222 残基 / 5.60% 正样本）

| 方法 | 口径 | ROC-AUC | PR-AUC |
|---|---|---|---|
| DSDP 3D-CNN | Cα 所在体素（任务书口径） | 0.8636 | 0.3204 |
| DSDP 3D-CNN | 3×3×3 邻域最大 | **0.9075** | **0.4925** |
| **GNN（全量 16,379 训练）** | 原始概率 | 0.8853 | 0.4049 |
| GNN | 1-hop 图邻域最大 | 0.8464 | 0.2487 |

CSV：`/share/home/huangyy/DSDP/gnn_site/results/table_residue_test363.csv`
图：`/share/home/huangyy/DSDP/gnn_site/results/fig_table1_auc.png`

### 表 2：端到端对接成功率（test 363 分层抽 40 个，同一 DSDP 引擎，exhaustiveness 384 / search_depth 40 / top_n 1）

| 管线 | n | RMSD<2 Å 成功率 | RMSD 中位数 | 盒子平均覆盖率 | 失败分类（盒子没罩住 / 罩住但没采样到） |
|---|---|---|---|---|---|
| GNN 盒子（阈值 0.5 / 36 Å） | 40 | **13/40 = 32.5%** | 5.72 Å | 0.855 | 6 / 21 |
| DSDP 原版 blind docking（`--site_npy`） | 40 | **16/40 = 40.0%** | 4.58 Å | 0.900 | 4 / 20 |

CSV：`/share/home/huangyy/DSDP/gnn_site/results/table_e2e.csv`
图：`/share/home/huangyy/DSDP/gnn_site/results/fig_e2e_rmsd.png`

### 其它

- **val 968**（选模型用）：GNN PR-AUC **0.4452** / ROC-AUC **0.9147**（best epoch 92）。
- **test 363**：PR-AUC **0.4049** / ROC-AUC **0.8853**。
- **DUD-E 102（补充考场）**：GNN PR-AUC **0.5326** / ROC-AUC **0.9141**（102 靶点、29,524 残基、
  2,092 正样本 7.1%）。`test_dataset/DUD_E_dataset` 只有 pdbqt，蛋白是本地清洗成 PDB 后跑的，
  与 test 363 不同源，**不进主表**，只作补充列：`/share/home/huangyy/DSDP/gnn_site/results/table_dude102.csv`。
- 盒子中心误差（test 363）：**中位 10.5 Å / 平均 11.5 Å**，配体覆盖 ≥90% 的复合物占 **82.1%**。
  图：`/share/home/huangyy/DSDP/gnn_site/results/fig_center_error_hist.png`
  明细：`/share/home/huangyy/DSDP/gnn_site/results/center_error_test363.csv`

---

## 2. 环境

| 项 | 值 |
|---|---|
| 解释器 | `/share/home/huangyy/miniconda3/envs/DSDP/bin/python`（3.9.25） |
| 关键包 | torch 2.8.0+cu128、PyG 2.6.1、biopython 1.83、freesasa 2.2.1、mkdssp 4.5.3、loguru、scikit-learn 1.4.0、matplotlib 3.8.2 |
| GPU | 登录节点**无 GPU**；训练/对接必须 `sbatch` 到 `gpu1/gpu2/gpu3`（RTX 4090 等） |
| 环境自检 | `/share/home/huangyy/DSDP/gnn_site/env_check.md`（含一遍可复现的检查清单） |

```bash
source /share/home/huangyy/miniconda3/etc/profile.d/conda.sh
conda activate DSDP
```

---

## 3. 数据来源与切分

- 结构：**PDBBind v2020 全量包**（已在本机，勿重新下载）
  `/share/home/huangyy/DataInEquibind/pdbbind_v2020/` → 解包到
  `/share/home/huangyy/DSDP/gnn_site/pdbbind_v2020/`（refined-set 5,316 + v2020-other-PL 14,127 = **19,443**）。
- 编号：EquiBind time split，`/share/home/huangyy/DataInEquibind/`
  - train `timesplit_no_lig_overlap_train` **16,379**
  - val `timesplit_no_lig_overlap_val` **968**
  - test `timesplit_test` **363**（= `test_dataset/PDBBind_time_split_dataset/`，与 EquiBind `timesplit_test` 完全一致）
- **覆盖率实测 100%**（16379/968/363 全命中，`gnn_site/skipped.txt` 为空）；三集合互斥。
- 蛋白一律用 PDBBind `<id>_protein.pdb`；标签配体 train/val 用 `<id>_ligand.mol2`，test 用仓库 `<id>_ligand.pdbqt`。
- 标签定义：残基重原子与配体重原子最小距离 **≤5 Å** 为正样本（train 4.30%、val 4.01%、test 5.60%）。

---

## 4. 一键复现（按顺序）

```bash
cd /share/home/huangyy/DSDP/gnn_site
PY=/share/home/huangyy/miniconda3/envs/DSDP/bin/python

# (0) 环境与数据盘点（只读检查，产出 env_check.md）
$PY -c "import torch, torch_geometric, freesasa; print(torch.__version__, torch_geometric.__version__)"

# (1) 结构索引（19,443 ID，split 覆盖率核对）
$PY collect_structures.py

# (2) 残基级 5 Å 标签（train 全量约 21 min；val/test 各 1~2 min）
$PY build_dataset.py --split train
$PY build_dataset.py --split val
$PY build_dataset.py --split test

# (3) 特征 + 图（16 进程；train 约 19 min）
$PY graph.py --batch --split train --workers 16
$PY graph.py --batch --split val   --workers 16
$PY graph.py --batch --split test  --workers 16

# (4) 训练（GPU；100 epoch 约 12 min）
sbatch /share/home/huangyy/DSDP/gnn_site/c7b_train.sbatch
#   等价命令：python train_full.py --device cuda --epochs 100 --batch-size 32 \
#              --patience 15 --seed 0 --tag c7b --ckpt-name best_model.pt

# (5) 评估（val + test + 阈值表）
$PY c7b_eval.py --model best_model.pt --device cpu

# (6) 单蛋白预测
$PY predict.py --protein 6qqw --out /tmp/predict_6qqw.csv

# (7) 盒子（val 定的规则：阈值 0.5 / 36 Å / 概率加权质心 / 单链接 8 Å）
$PY residue_to_box.py --protein 6qqw --ligand \
    /share/home/huangyy/DSDP/test_dataset/PDBBind_time_split_dataset/6qqw/6qqw_ligand.pdbqt

# (8) 端到端对接（必须 GPU 节点）
sbatch /share/home/huangyy/DSDP/gnn_site/c9_gpu.sbatch   # 1a2b / 6qqw / 6qxa
sbatch /share/home/huangyy/DSDP/gnn_site/c10_full.sbatch # 40 个蛋白两条管线

# (9) 表与图
$PY c10_figures.py
$PY c10_viz.py --ids 6g2o,6jao,6qqw,6kqi,5zk7
```

---

## 5. 关键结论与坑

- **GNN 赢「Cα 单点体素」口径，输「3×3×3 邻域」口径**（PR 0.4049 vs 0.4925），端到端也略输
  （32.5% vs 40.0%）。原因是 DSDP 的体素概率极稀疏（均值 0.0004），邻域最大值相当于"补容差"；
  GNN 概率本身就密（均值 0.23），再做最大平滑只会扩散假阳性（平滑后 PR 掉到 0.2487）。
- 端到端失败的两种模式很清楚：**GNN 40 个里有 6 个盒子没罩住**、21 个罩住了但采样没找到；
  DSDP 是 4 / 20。说明瓶颈主要不在"盒子大小"，而在"口袋选对没有"。
- 盒子参数**只在 val 968 上选**（阈值 0.5 / 36 Å / 加权质心 / 8 Å 单链接，val 覆盖≥90% 占 79.0%），
  test 只做一次性报告，未回流调参。
- 训练/对接的硬约束：登录节点无 GPU（CUDA 二进制直接 segfault），DSDP 的
  `--dataset_path` **不能**把 `site/out/log` 放进去（会被当成复合物），这两点在
  `env_check.md` 第 6 节有记录。

---

## 6. 诚实的局限（demo 定位，不是论文）

1. **没有超过 DSDP 基线**：残基级 3×3×3 口径 DSDP 更强（PR 0.4925 vs 0.4049），端到端也更强
   （40.0% vs 32.5%，n=40，样本小、未做显著性检验）。本次只是把管线换成 GNN 并跑通，
   不宣称效果更好。
2. **盒子精度有限**：中心到真实配体质心中位 **10.5 Å**、平均 11.5 Å；36 Å 盒子只保证
   **82.1%** 的复合物覆盖 ≥90% 配体。残基级粒度天然比 2 Å 体素粗，这是任务书预期内的 trade-off。
3. **概率未校准**：val 上平均预测概率约 0.23，而真实正样本只有 4.01%（`pos_weight≈22` 的副作用）。
   PR/ROC 不受影响，但"概率 0.5"不等于"50% 把握"；C8 的阈值 0.5 是**排序阈值**，不是校准概率。
4. **失败模式**：GNN 在 40 个样本里 6 个盒子完全没罩住配体（top-20 命中 0 的例子如 6kqi、5zk7、
   6qxa），DSDP 在同样蛋白上有的能救回来（6kqi：GNN 32.9 Å vs DSDP 0.8 Å）——说明
   "选错口袋"比"盒子不够大"更致命。
5. **模型很小**（3,570 参数，46 维手工特征），没有 PLM 嵌入/EGNN；train loss 到 0.36 仍在降，
   主要瓶颈是特征表达而不是过拟合。
6. **DUD-E 102 已补做**（PR 0.5326 / ROC 0.9141），但它是另一套考场：只有 pdbqt、蛋白是本地清洗成
   PDB 后特征化的，与 test 363 不同源、正样本比例也更高（7.1%），**不能混进主表比较**，只作补充列。
7. **RMSD 口径**：重原子、原子顺序一一对应、**不做叠合**（同一坐标系直接比），
   <2 Å 记成功；未做对称原子校正，长链配体可能因此偏严。

---

## 7. 文件清单（`/share/home/huangyy/DSDP/gnn_site/`）

| 文件 | 作用 |
|---|---|
| `env_check.md` | C1 环境/数据盘点 + 一键检查命令 |
| `collect_structures.py` | 扫 PDBBind v2020 建 `structure_index.json`，核对 split 覆盖率 |
| `build_dataset.py` | 5 Å 残基标签 → `data/label/<split>/*.npz` + `label_stat*.txt` |
| `features.py` | 46 维残基特征（DSSP 4 列位自写解析 + freesasa） |
| `graph.py` | Cα≤10 Å 建边、权重 1/d²、PyG `Data`；批量建图 |
| `model.py` | 4 层 GCN 46→38→28→18→8→2（3,570 参数）+ 维度自检 |
| `train.py` | C6 冒烟训练脚本（30 图 2 epoch） |
| `train_full.py` | 正式训练（NLLLoss/Adam/早停/存 best） |
| `predict.py` | 每残基概率 CSV |
| `residue_to_box.py` | C8：概率 → 聚类 → 盒子（阈值 0.5 / 36 Å） |
| `run_gnn_dsdp.py` | C9：GNN→盒子→DSDP→RMSD 一条龙 |
| `box_param_scan.py` | 盒子参数在 val 上的扫描 |
| `c2_baseline.py` | DSDP 3D-CNN 推理 + 体素→残基投影（基线） |
| `c7b_eval.py` / `c7b_boxcheck.py` | 最终评估 / 盒子覆盖分析 |
| `c10_e2e.py` | 40 蛋白两条管线端到端对比 |
| `c10_dsdp_coverage.py` | 按 DSDP 源码复算其盒子覆盖率 |
| `c10_figures.py` / `c10_viz.py` | 表/图与 3D 可视化 |
| `best_model.pt` | **最终模型**（epoch 92） |
| `data/` | 标签 npz、图 pt（train 16,379 约 4.1 GB） |
| `results/` | 表 CSV、图 PNG、可视化 |
| `reports/` | 各阶段 json/日志/曲线 |

**实验报告**：`c4_report.md`（特征/图）、`c5c6_report.md`（模型冒烟）、`c7a_report.md`（3000 版）、
`c7b_report.md`（全量 + 阈值）、`c2_report.md`（基线）、`c8c9_report.md`（盒子 + 端到端）。

---

## 8. 图（需在 VS Code Remote 里打开）

| 图 | 路径 | 内容 |
|---|---|---|
| 表 1 主图 | `/share/home/huangyy/DSDP/gnn_site/results/fig_table1_auc.png` | 4 种口径 ROC/PR 柱状图 |
| 端到端 | `/share/home/huangyy/DSDP/gnn_site/results/fig_e2e_rmsd.png` | RMSD 箱线图 + 成功率 |
| 中心误差 | `/share/home/huangyy/DSDP/gnn_site/results/fig_center_error_hist.png` | 363 复合物盒子中心误差直方图 |
| 训练曲线 | `/share/home/huangyy/DSDP/gnn_site/results/fig_training_curve.png` | loss + val AUC 随 epoch |
| 口袋可视化 | `/share/home/huangyy/DSDP/gnn_site/results/viz/<id>_pocket.png` | 5 个代表蛋白（6g2o/6jao/6qqw 成功，6kqi/5zk7 失败） |
| PyMOL 脚本 | `/share/home/huangyy/DSDP/gnn_site/results/viz/<id>_pocket.pml` | 有 PyMOL 的环境可 `pymol -cq <id>_pocket.pml` 重渲染 |
