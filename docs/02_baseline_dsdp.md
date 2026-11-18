# C2 报告：DSDP（PUResNet 3D-CNN）残基级基线

时间：2026-11-18 20:15（Asia/Shanghai）
脚本：`gnn_site/baseline_dsdp_cnn.py`；作业 159729（GPU RTX 4090，cn1）；墙钟 77 s（含缓存复用）

## 1. 做法（与任务书 §4 Step 2 一致）

1. 直接复用仓库原版 `DSDP_blind_docking.py` 里的 `DrugSiteData` / `DrugSiteMind`
   （源码截到 `if __name__` 前 exec，特征工具还是 `protein_feature_tool` + `surface_tool`），
   对 `test_dataset/PDBBind_time_split_dataset` 的 **363 个**复合物推理。
2. 每个复合物得到 **36³ 概率体素**（`grid_resolution = 1/0.5 = 2 Å`，`max_distance = 35 Å`，
   `box_size = 36`），存 `gnn_site/c2_site/npy/<id>.npy`。
3. **投影到残基**：Cα 落点体素
   `i = round((ca - centroid + 35) / 2)`，截断到 [0,35]；残基分数两种口径：
   - `score_voxel` = Cα 所在体素概率（任务书指定口径）
   - `score_nbhd`  = 该体素 3×3×3 邻域最大概率（更宽松的替代口径）
4. 与 C4 生成的 5 Å 标签**逐残基对齐**（同一 `data/label/test/<id>.npz`，行数/顺序已校验）；
   Cα 坐标与仓库 pdbqt 完全一致（偏差 0.0000 Å），不存在坐标系错位。
5. 产出 `gnn_site/baseline_dsdp_residue.csv`（130,222 行）与 `gnn_site/reports/baseline_metrics.json`。

## 2. 结果（test 363，5 Å 标签，130,222 残基 / 7,290 正样本 = 5.60%）

| 投影口径 | ROC-AUC | PR-AUC | 概率均值 | 概率最大 |
|---|---|---|---|---|
| **Cα 体素（任务书口径）** | **0.8636** | **0.3204** | 0.00042 | 0.5017 |
| Cα 体素 3×3×3 邻域最大 | **0.9075** | **0.4925** | 0.01313 | 0.99998 |

- 用 CSV 独立复算：voxel ROC 0.8631 / PR 0.3193；nbhd ROC 0.9075 / PR 0.4923（CSV 保留 6 位小数，差异在舍入内）。
- **随机基线**：PR-AUC ≈ 0.056（正样本比例）。
- 说明：DSDP 的概率图非常"尖"（均值 0.0004、最大 0.50），单点投影会漏掉峰边缘，所以邻域最大口径明显更好（PR 0.32 → 0.49）。

## 3. 并与 GNN 的对比（同一批 test 363、同一套 5 Å 标签）

| 方法 | ROC-AUC | PR-AUC |
|---|---|---|
| DSDP 3D-CNN · Cα 体素 | 0.8636 | 0.3204 |
| DSDP 3D-CNN · 3×3×3 邻域 | **0.9075** | **0.4925** |
| **GNN（全量 16,379 训练）** | 0.8853 | 0.4049 |

**如实结论**：残基级 GNN 优于 DSDP 的"单点体素"投影（PR 0.405 vs 0.320，ROC 0.885 vs 0.864），
但**不如** DSDP 的"邻域最大"投影（PR 0.405 vs 0.493）。也就是说：
- 若对比表用任务书指定的"Cα 所在体素"口径，GNN 领先；
- 若用更宽松的邻域口径，DSDP 基线仍更强。
C10 出表时必须把两种口径都列出来，不能只挑对 GNN 有利的那个。

## 4. 文件

- `gnn_site/baseline_dsdp_cnn.py`：脚本
- `gnn_site/baseline_dsdp_residue.csv`：130,222 行（pdb_id,res_idx,chain,resseq,icode,resname,label,score_voxel,score_nbhd）
- `gnn_site/c2_site/npy/*.npy`：363 个 36³ 概率图
- `gnn_site/reports/baseline_metrics.json`、`reports/c2_slurm.out`
