# C10 报告：成品对比表 + 端到端成功率 + 可视化 + README

时间：2026-11-18 21:40（Asia/Shanghai）
产物目录：`$GNNSITE_HOME/results/`（PNG 需在 VS Code Remote 里打开）

## 1. 端到端成功率（40 个 test 复合物，两条管线）

抽样：test 363 按 **GNN top-20 命中率**分三层（高 ≥12、中 5–11、低 ≤4），各层按等距取，
共 40 个。两条管线用**完全相同的 DSDP 参数**（exhaustiveness 384 / search_depth 40 / top_n 1）：
- **GNN 线**：best_model → 概率 → 盒子（val 定的阈值 0.5 / 36 Å / 加权质心）→ `DSDP_redocking/DSDP --box_min/--box_max`
- **DSDP 线**：C2 已算好的 36³ 概率图 → `DSDP_blind_docking/DSDP --site_npy`（原版自己的聚类 + 出盒）

RMSD = 重原子、原子顺序一一对应、**不叠合**（同坐标系），<2 Å 记成功。

| 管线 | n | 成功 | 成功率 | RMSD 中位 | RMSD 均值 | 盒子平均覆盖率 | 失败分类（盒子没罩住 / 罩住没采样到） |
|---|---|---|---|---|---|---|---|
| **GNN 盒子** | 40 | 13 | **32.5%** | 5.72 Å | 11.00 Å | 0.855 | 6 / 21 |
| **DSDP 原版** | 40 | 16 | **40.0%** | 4.58 Å | 7.28 Å | 0.900 | 4 / 20 |

- 总墙钟 69.3 s（80 次对接 + GNN 盒子）。
- **DSDP 的盒子覆盖率按源码精确复算**（`main_SingleRun.cu`：top-200 体素 → 27 邻域聚类 → 每簇 30 Å 立方、
  protein_center 取全原子均值；任一簇覆盖即算覆盖），不是简单 bbox 近似。
- 失败模式：GNN 有 6 个 case 盒子完全没罩住配体；有 21 个罩住了但没采样到。**两种管线的主要瓶颈都是"口袋选对没有"，
  不是盒子不够大**（盒子覆盖率 0.855 vs 0.900，差距远小于成功率差距 32.5% vs 40.0%）。
- 互补案例：6kqi（GNN 32.94 Å / DSDP 0.80 Å）、6e4v（22.41 / 1.13）、6ffg（18.87 / 0.26）——DSDP 能救回 GNN 选错口袋的样本。

明细：`$GNNSITE_HOME/runs/e2e_benchmark.json`、`runs/e2e_benchmark.csv`（40 行）

## 2. 表 1：残基级（test 363，5 Å 标签）

| 方法 | 口径 | ROC-AUC | PR-AUC |
|---|---|---|---|
| DSDP 3D-CNN | Cα 所在体素（任务书口径） | 0.8636 | 0.3204 |
| DSDP 3D-CNN | 3×3×3 邻域最大 | **0.9075** | **0.4925** |
| GNN（全量 16,379） | 原始 | 0.8853 | 0.4049 |
| GNN | 1-hop 图邻域最大 | 0.8464 | 0.2487 |

## 3. 补充考场：DUD-E 102（已补做）

`test_dataset/DUD_E_dataset` 的 102 个靶点只有 pdbqt，先把 `<id>_protein.pdbqt` 清洗成标准 PDB
（补元素列、丢 AD4 类型列），再走正常特征/建图/推理；标签用 `<id>_ligand.pdbqt` 算 5 Å。

| 数据集 | 靶点 | 残基 | 正样本 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **DUD-E 102（补充列）** | 102 | 29,524 | 2,092 (7.1%) | **0.9141** | **0.5326** |

冒烟先行（aa2ar 287 残基、abl1 261 残基，特征/标签对齐 100%），全量 102 用时 17.7 s（8 进程）。
**不混进主表**：与 test 363 不同源、正样本比例不同（7.1% vs 5.60%）。

## 4. 图

| 文件 | 内容 |
|---|---|
| `$GNNSITE_HOME/results/fig_table1_auc.png` | 4 种口径的 ROC/PR 柱状图（表 1） |
| `$GNNSITE_HOME/results/fig_e2e_rmsd.png` | RMSD 箱线图（左）+ 成功率柱状图（右） |
| `$GNNSITE_HOME/results/fig_center_error_hist.png` | test 363 盒子中心误差直方图（中位 10.5 Å，覆盖≥90% 占 82.1%） |
| `$GNNSITE_HOME/results/fig_training_curve.png` | 训练曲线（附录，来自 c7b_curve.png） |
| `$GNNSITE_HOME/results/center_error_test363.csv` | 363 行逐复合物中心误差/覆盖率 |

## 5. 代表蛋白可视化（5 个）

环境没有 PyMOL（`import pymol` 失败），按任务书约定用 **matplotlib 3D** 出 PNG，
同时给每个蛋白写了 **可复现的 .pml 脚本**（有 PyMOL 的机器 `pymol -cq <id>_pocket.pml` 即可重渲染）。

| 蛋白 | GNN top-20 命中 | GNN RMSD | DSDP RMSD | PNG / PML |
|---|---|---|---|---|
| 6g2o | 4/13 | 0.20 Å ✅ | 0.21 Å | `results/viz/6g2o_pocket.png` / `.pml` |
| 6jao | 13/24 | 0.21 Å ✅ | 0.22 Å | `results/viz/6jao_pocket.png` / `.pml` |
| 6qqw | 12/20 | 0.22 Å ✅ | — (不在 40 抽样内) | `results/viz/6qqw_pocket.png` / `.pml` |
| 6kqi | **0/25** | 32.94 Å ❌ | **0.80 Å** | `results/viz/6kqi_pocket.png` / `.pml` |
| 5zk7 | **0/10** | 32.39 Å ❌ | 32.50 Å | `results/viz/5zk7_pocket.png` / `.pml` |

每张图：灰色 = 蛋白 Cα 骨架，绿色 = 晶体配体，红色球 = GNN top-20 残基，蓝色三角 = 对接 pose。

## 6. 文档

- `$GNNSITE_HOME/README.md`：一句话说明、环境、一键复现全流程、数据切分、
  两张表 + 图、7 条局限、文件清单、图索引。
- `docs/09_ppt_points.md`：一页 PPT 的 8 条要点 + 每条配哪张图。

## 7. 交付清单

```
$GNNSITE_HOME/
├── README.md                       # 主文档（含一键复现）
├── results/
│   ├── table_residue_test363.csv   # 表 1
│   ├── table_e2e.csv               # 表 2
│   ├── table_dude102.csv           # DUD-E 补充列
│   ├── table_all.json              # 三张表的机器可读版
│   ├── fig_table1_auc.png / fig_e2e_rmsd.png /
│   │   fig_center_error_hist.png / fig_training_curve.png
│   ├── center_error_test363.csv
│   └── viz/<id>_pocket.png|.pml    # 5 个蛋白
├── runs/e2e_benchmark.json / .csv        # 40 蛋白逐条结果
├── runs/e2e_benchmark.csv
├── benchmark_end2end.py / analyze_dsdp_boxes.py / make_figures.py / make_pocket_views.py / evaluate_dude.py
└── docs/09_ppt_points.md
```

## 8. 结论（demo 定位）

- 整条链路（数据 → 特征 → 图 → 训练 → 残基概率 → 盒子 → DSDP 对接 → RMSD → 图表）全部跑通、可复现。
- **GNN 没有超过 DSDP 基线**：残基级 3×3×3 口径 PR 0.4049 vs 0.4925；端到端 32.5% vs 40.0%（n=40）。
- 但 GNN 在「Cα 单点体素」口径上更强（PR 0.4049 vs 0.3204），且在 DUD-E 102 上 PR 0.5326 / ROC 0.9141，
  说明模型本身是有效的位点预测器。
- 下一步真正能提升的方向：换更强特征（PLM 嵌入 / EGNN）、把"选口袋"当独立分类任务、以及用 RMSD 而非 AUC 做模型选择。
