# C8 / C9 报告：残基→盒子 + 接回 DSDP 端到端

时间：2026-11-18 21:30（Asia/Shanghai）
设备：GPU RTX 4090（slurm 159804，cn1，排队 0 分钟）
结论：盒子规则**在 val 上定**（阈值 0.5 / 36 Å / 概率加权质心 / 单链接 8 Å）；6qqw 盒子覆盖真实配体 **100%**；端到端 **1a2b RMSD 1.66 Å / 6qqw 0.22 Å（成功）**，**6qxa 57.06 Å（失败但不崩）**。

## 0. 盒子参数在 val 968 上定（不用 test 调参）

`scripts/scan_box_params.py` 在 val 968 上扫：阈值 {0.3,0.4,0.5,0.6,0.7}、top-k {10,20,30} × 盒边长 {30,33,36,43} Å × 中心 {质心, 概率加权质心}，指标 = **真实配体重原子被盒子包住的百分比**（比 AUC 更贴近对接成败）。

决策规则（**看数据前就写死**）：盒边长 ≤ 36 Å，先最大化"覆盖 ≥90% 的复合物比例"，再比平均覆盖。val 结果（加权质心）：

| 规则 | 30 Å | 33 Å | 36 Å | 43 Å |
|---|---|---|---|---|
| thr 0.4 | 67.4% | 72.1% | 78.4% | 85.2% |
| **thr 0.5** | 68.1% | 73.7% | **79.0%** | 85.8% |
| thr 0.6 | 69.2% | 73.8% | 78.3% | 84.7% |
| thr 0.7 | 69.3% | 73.8% | 78.1% | 83.6% |
| top-20 | 68.9% | 72.2% | 76.3% | 82.5% |

（表内是"覆盖 ≥0.9 的复合物比例"；平均覆盖见 `reports/box_params_val.json`）

**选定：`threshold=0.5`、`box_size=36 Å`、`centre=概率加权质心`、`eps=8 Å`**
→ val：平均覆盖 **0.857**，**79.0%** 复合物覆盖 ≥90%，中位中心误差 **9.4 Å**。

为什么不选 43 Å：覆盖率能到 85.8%，但体积是 36 Å 的 1.7 倍，对接搜索代价明显上升；也不选 30 Å：覆盖率掉到 68%。36 Å 是性价比拐点。

## 1. C8：`gnn_site/residue_to_box.py`

流程（严格按任务书 §3）：
1. 复用 `scripts/predict.py` 的 `load_model` / `predict_one` 得每残基概率（没有重写模型代码）；
2. 选残基：`p ≥ 0.5`；若选中 <5 个则退回 top-20；`--topk` 可整体覆盖；
3. 单链接聚类（Cα ≤ 8 Å），主口袋 = `残基数 × 概率和` 最大的簇；
4. 中心 = 主簇的**概率加权 Cα 质心**；边长 36 Å；中心会平移以把盒子限制在蛋白 Cα 包围盒内（不会悬空）；
5. 输出 `box_min / box_max` + 诊断 json（选中数、簇数、主簇大小、边长、中心、蛋白 bbox 等）。

### 冒烟：6qqw（验收项）

```
[box] 6qqw (411 residues) model epoch=92
[box] selection=threshold n_selected=76 clusters=3 main=41
[box] box_size=36 centre=weighted -> box_min=[-37.635, -2.097, -26.046] box_max=[-1.635, 33.903, 9.954]
[box] ligand coverage=1.000  centre->ligand-centroid=6.02 A  centroid_inside=True
```
✅ 盒子**完全罩住真实配体**（coverage 1.0），配体质心在盒内，中心距配体质心 6.02 Å。

### 坐标一致性复核（任务书要求）

| 蛋白 | 对比 | 残基 | 平均偏差 | 最大偏差 |
|---|---|---|---|---|
| 6qqw | PDBBind `<id>_protein.pdb` vs 仓库 `_protein.pdbqt` | 411/411 CA | **0.0000 Å** | **0.0000 Å** |
| 1a2b | pdbqt→PDB 转换件 vs 原始 pdbqt | 178/178 CA | **0.0000 Å** | **0.0000 Å** |

### test 363 一次性覆盖数字（规则已在 val 定死，此处只报告，不回调）

| 指标 | 值 |
|---|---|
| 平均配体覆盖 | **0.889** |
| 覆盖 ≥90% 的复合物 | **82.1%** |
| 覆盖 ≥50% 的复合物 | 89.5% |
| 配体质心落在盒内 | **89.8%** |
| 中心到配体质心（中位 / 平均） | 10.5 Å / 11.5 Å |

## 2. C9：`gnn_site/run_gnn_dsdp.py` 端到端

```
python scripts/run_gnn_dsdp.py --protein <pdb|pdbqt|id> --ligand <pdbqt> --out runs/<id>_out.pdbqt
```
1. GNN → 盒子；2. 调现成 `./DSDP_redocking/DSDP --box_min/--box_max --exhaustiveness 384 --search_depth 40 --top_n 1`（GPU 节点）；3. 预测 pose vs 晶体配体（重原子、**不叠合**、同坐标系）算 RMSD，<2 Å 记成功。

| 蛋白 | 选择理由 | 盒内主簇 | 对接耗时 | **RMSD** | 成功(<2 Å) | 原子对应 |
|---|---|---|---|---|---|---|
| **1a2b** | 仓库自带例子（pdbqt→PDB 转换后喂 GNN） | 93/95 | 1.3 s | **1.662 Å** | ✅ | 32 vs 32，元素全对 |
| **6qqw** | GNN test 里表现好（top-20 命中 12/20） | 41/76 | 0.8 s | **0.223 Å** | ✅ | 22 vs 22，元素全对 |
| **6qxa** | GNN 失败案例（top-20 命中 **0/20**） | 69/116 | 1.5 s | **57.056 Å** | ❌（符合预期） | 23 vs 23，元素全对 |

- 三个都 `returncode=0`、都产出了 `*_out.pdbqt`——**失败案例不会崩**，只是盒子选错口袋。
- RMSD 用原子顺序一一对应（元素序列完全一致），未做叠合，符合"同一坐标系直接比"的定义。
- 1a2b 的 GNN 输入是用 `pdbqt_to_pdb` 现场转换的（1a2b 不在 PDBBind v2020 里），转换前后 CA 偏差 0.0000 Å。

## 3. 可选加分：GNN 邻域平滑口径（对比表第三行）

把每残基概率替换为"自身 + 10 Å 图邻居的最大值"，同一 test 363：

| 口径 | ROC-AUC | PR-AUC |
|---|---|---|
| GNN 原始 | **0.8853** | **0.4049** |
| GNN 1-hop 最大平滑 | 0.8464 | 0.2487 |
| DSDP 3D-CNN · Cα 体素 | 0.8636 | 0.3204 |
| DSDP 3D-CNN · 3×3×3 邻域 | 0.9075 | 0.4925 |

**平滑让 GNN 变差**（PR 0.405 → 0.249）。原因很直接：DSDP 的体素概率极稀疏（均值 0.0004），邻域最大值是在"补漏"；GNN 概率已经很密（均值 0.23），再做最大值只会把假阳性扩散到邻居。**这只是报告，没有据此回调任何参数**。

## 4. 交付物

| 文件 | 说明 |
|---|---|
| `gnn_site/residue_to_box.py` | C8：蛋白 → box_min/box_max + 诊断 |
| `gnn_site/run_gnn_dsdp.py` | C9：GNN → 盒子 → DSDP → RMSD 一条龙 |
| `gnn_site/scan_box_params.py` | val 上的盒子参数扫描 |
| `gnn_site/reports/box_params_val.json` | val 全表（64 组） + chosen |
| `gnn_site/reports/box_test_once.json` | test 363 一次性覆盖数字 |
| `gnn_site/reports/box_6qqw.json` | 6qqw 盒子诊断 |
| `gnn_site/runs/{1a2b,6qqw,6qxa}.json` / `_out.pdbqt` / `_out.log` | C9 三个蛋白的端到端产物 |
| `gnn_site/reports/c9_slurm.out` | C9 作业日志 |
| `gnn_site/reports/smoothing_check.json` | 平滑口径数字 |

## 5. 下一步

**C10**：对比表（363 主表 + 可选 DUD-E）+ 30~50 个蛋白的 RMSD 成功率 + 3~5 个可视化 + README。
