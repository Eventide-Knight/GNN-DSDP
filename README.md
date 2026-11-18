# GNN-DSDP：残基级 GNN 位点预测 + DSDP 对接

> **一句话**：把 DSDP 第一段"位点预测"从 **3D-CNN（体素）** 换成 **残基级 GCN**，
> 输出每个氨基酸残基的配体结合概率，再把高概率残基聚成一个对接盒子，接回 DSDP 原有的 CUDA 对接引擎，
> 在 PDBBind v2020 time split 上与原版 DSDP 做同一口径的对比。

这是一个**组内 demo**级别的实现：优先跑通、可复现、结论诚实，不追求 SOTA。

---

## 0. 仓库边界（先看这个，省 10 分钟）

| 包含 ✅ | 不包含 ❌ |
|---|---|
| 全部代码（19 个脚本）、slurm 提交脚本 | **任何结构/图数据**（PDBBind 许可禁止再分发） |
| 训练好的模型 `best_model.pt`（55 KB） | 中间产物（`data/graph` 4.1 GB、`c2_site` 等） |
| 所有结果表/图/日志（`results/`、`reports/`、`runs/`） | DSDP 的源码与二进制（需另装，见 §3.3） |
| 各阶段实验报告（`docs/`） | 别人集群的绝对路径（已全部改为 `scripts/paths.py` + 环境变量） |

**换机器能不能直接跑？** 代码层面可以（路径全部由 `scripts/paths.py` 解析、默认相对仓库自身），
但有 4 样外部东西必须先准备好：**PDBBind 结构、EquiBind 切分编号、DSDP 仓库、GPU 环境**。
四样都是公开的，§3 逐条写了去哪拿、放哪里。

> 验证状态：本仓库的代码与数字来自同一台集群上的实际运行；**未在第二台机器上做过端到端验证**。
> 第一次在新机器上跑，请先执行 `python scripts/paths.py`（会打印所有路径并告诉你哪个不存在），再按 §4 走。

---

## 1. 结果速览

### 表 1 · 残基级位点预测（test 363，5 Å 标签，130,222 残基 / 5.60% 正样本）

| 方法 | 口径 | ROC-AUC | PR-AUC |
|---|---|---|---|
| DSDP 3D-CNN | Cα 所在体素（任务书口径） | 0.8636 | 0.3204 |
| DSDP 3D-CNN | 3×3×3 邻域最大 | **0.9075** | **0.4925** |
| **GNN（全量 16,379 训练）** | 原始概率 | 0.8853 | 0.4049 |
| GNN | 1-hop 图邻域最大 | 0.8464 | 0.2487 |

明细：`results/table_residue_test363.csv`；图：`results/fig_table1_auc.png`

### 表 2 · 端到端对接成功率（test 363 分层抽 40 个，同一 DSDP 引擎，exhaustiveness 384 / search_depth 40 / top_n 1）

| 管线 | n | RMSD<2 Å 成功率 | RMSD 中位数 | 盒子平均覆盖率 | 失败（盒子没罩住 / 罩住没采样到） |
|---|---|---|---|---|---|
| GNN 盒子（阈值 0.5 / 36 Å） | 40 | **13/40 = 32.5%** | 5.72 Å | 0.855 | 6 / 21 |
| DSDP 原版 blind docking | 40 | **16/40 = 40.0%** | 4.58 Å | 0.900 | 4 / 20 |

明细：`results/table_e2e.csv`、`runs/e2e_benchmark.csv`；图：`results/fig_e2e_rmsd.png`

### 其它

- **val 968**（选模型用）：PR-AUC **0.4452** / ROC-AUC **0.9147**（best epoch 92）
- **test 363**：PR-AUC **0.4049** / ROC-AUC **0.8853**
- **DUD-E 102**（补充考场，不进主表）：PR-AUC **0.5326** / ROC-AUC **0.9141**
- **盒子中心误差**（test 363）：中位 **10.5 Å** / 平均 11.5 Å；覆盖 ≥90% 配体的复合物占 **82.1%**

---

## 2. 环境

Python 3.9 + PyTorch + PyTorch Geometric，详细版本见 `requirements.txt`。

```bash
conda create -n dsdp-gnn python=3.9
conda activate dsdp-gnn
pip install -r requirements.txt
conda install -c conda-forge dssp          # 提供 mkdssp（DSSP 二级结构）
```

要点：

- **mkdssp 必须在 PATH 里**（或 `export MKDSSP=/path/to/mkdssp`）。`scripts/features.py` 直接调它。
  找不到时脚本**不会崩**，但 DSSP 那 15 维会全变成 0、预测质量明显下降（会打印一次警告）；
  想让它直接报错，设 `export GNNSITE_STRICT=1`。先跑 `python scripts/paths.py` 就能看到 `MKDSSP` 是 YES 还是 NO。
- `freesasa` 是 pip 源码包，需要 C 编译器；装不上就用 `conda install -c conda-forge freesasa`。
- **GPU**：训练与对接需要 CUDA。登录节点通常没有 GPU，DSDP 的 CUDA 二进制在纯 CPU 节点会直接
  `Segmentation fault`，所以要用集群调度器（本仓库给了 `*.sbatch` 模板，改分区名即可）。
  没有 GPU 也能跑通数据构建与 CPU 训练，只是慢。
- 本仓库的脚本都在 `scripts/` 下，用 `python scripts/xxx.py` 直接运行，**不需要安装成包**
  （脚本之间通过同目录 `import paths` 解析路径，`paths.py` 会自动认出仓库根目录）。

---

## 3. 数据准备（4 样东西）

> **本仓库不含任何数据文件。** 下面每条都写了来源、目标位置、如何验证。

### 3.0 目录约定

默认布局（用环境变量可以改，见 §3.4）：

```
你的工作目录/
├── gnn-dsdp/                 # 本仓库（clone 下来就叫这个，任意名字都行）
│   ├── README.md  LICENSE  requirements.txt  .gitignore
│   ├── scripts/              # 全部 Python 脚本（含 paths.py 配置入口）
│   ├── slurm/                # slurm 作业模板（*.sbatch）
│   ├── docs/                 # 各阶段实验报告（编号 01–09）
│   ├── best_model.pt         # 训练好的模型（55 KB）
│   ├── pdbbind_v2020/        # ← §3.1 解包到这里（约 10 GB），或设 PDBBIND_DIR
│   ├── splits/               # ← §3.2 三个编号文件放这里，或设 SPLIT_DIR
│   └── data/  results/  reports/  runs/     # 生成物
└── DSDP/                     # ← §3.3 另一个仓库（默认找 ../DSDP），或设 DSDP_HOME
    ├── test_dataset/{PDBBind_time_split_dataset,DUD_E_dataset,DSDP_dataset}/
    ├── DSDP_redocking/DSDP
    ├── DSDP_blind_docking/DSDP
    └── binding_site_model.pth
```

### 3.1 PDBBind v2020（蛋白-配体结构，约 10 GB）

- **来源**：https://www.pdbbind-plus.org.cn/ → `Download Data Packages` → **PDBBind v2020** 页。
  需要注册账号（学术免费）。下载 **protein-ligand** 的两个包：
  - `Protein-ligand complexes: The general set minus refined set`（1.8 GB，14,127 个）
  - `Protein-ligand complexes: The refined set`（658 MB，5,316 个）
  - 外加 `Index files of PDBbind`（3 MB）
  - 合计 **19,443** 个复合物，正好覆盖本项目用到的 train/val/test。
  - （页面上另有 "Demo: PDBbind v2020.R1"（19,037 个）也能用，但会缺几百个，脚本会报告缺失数。）
- **放哪**：解包到 `$PDBBIND_DIR`（默认 `<repo>/pdbbind_v2020/`），结构应为：
  ```
  pdbbind_v2020/
  ├── refined-set/<pdbid>/<pdbid>_protein.pdb, _ligand.mol2, _ligand.sdf, _pocket.pdb
  └── v2020-other-PL/<pdbid>/...
  ```
- **验证**：`python scripts/collect_structures.py` 会打印 5,316 + 14,127 = 19,443，并核对切分覆盖率。
- ⚠️ **许可**：PDBBind 明确禁止再分发。请不要把这份数据（或其派生的 `data/`）提交到仓库；
  `.gitignore` 已经把它们挡住了。

### 3.2 EquiBind 时间切分编号（三个文本文件）

- **来源**：https://github.com/HannesStark/EquiBind 的 `data/` 目录，原始文件名：
  - `timesplit_no_lig_overlap_train`（16,379 行）
  - `timesplit_no_lig_overlap_val`（968 行）
  - `timesplit_test`（363 行）  ← **注意这个文件没有 `no_lig_overlap` 前缀**
- **放哪**：`$SPLIT_DIR`（默认 `<repo>/splits/`），保持上面的文件名不变。
- **验证**：`python scripts/paths.py` 会打印三个文件的行数；应为 16379 / 968 / 363。
- 说明：这是 PDBBind 的 **time split**（训练集年份早于测试集），且训练/测试**无配体重叠**，
  所以不存在泄漏，不需要自己再随机切分。

> EquiBind 原本打包的分数据集（Zenodo record 6408497）**已于 2025-01 因版权删除**，
> 现在只能像上面这样"自己去 PDBBind 官网下 + 用它的编号文件"，这也是 EquiBind README 现在的指引。

### 3.3 DSDP 仓库（对接引擎 + 测试集 pdbqt）

```bash
git clone https://github.com/PKUGaoGroup/DSDP.git DSDP
cd DSDP
# 测试集在 git 里但默认被 sparse-checkout 排除，恢复它：
git sparse-checkout add test_dataset        # 恢复后 test_dataset 共 1460 个复合物
```

- **放哪**：默认 `../DSDP`（与本仓库同级）；别的路径用 `export DSDP_HOME=...`。
- **要编译**（见 DSDP 自己的 README）：`DSDP_redocking/`、`DSDP_blind_docking/`、`surface_tool/`
  各 `make`，`protein_feature_tool/` 用 `g++` 编译。需要 CUDA toolkit 与匹配的 gcc。
- 本项目用到：
  - `test_dataset/PDBBind_time_split_dataset/`（363 个，主考场）
  - `test_dataset/DUD_E_dataset/`（102 个，补充考场）
  - `test_dataset/DSDP_dataset/`（995 个，其中 1a2b 用作冒烟例子）
  - `DSDP_redocking/DSDP`（按盒子对接）、`DSDP_blind_docking/DSDP`（基线盲对接）
  - `binding_site_model.pth`（C2 基线用的 3D-CNN 权重）
- **验证**：`python scripts/paths.py` 里 `DSDP_HOME` / `TEST_PDBBIND_DIR` / `REDOCK_BIN` 应为 `YES`。

### 3.4 全部环境变量（都可选）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GNNSITE_HOME` | 本仓库目录 | 代码与生成数据的根 |
| `DSDP_HOME` | `../DSDP` | DSDP 仓库根 |
| `PDBBIND_DIR` | `$GNNSITE_HOME/pdbbind_v2020` | PDBBind v2020 解包目录 |
| `SPLIT_DIR` | `$GNNSITE_HOME/splits` | 三个编号文件所在目录 |
| `MKDSSP` | 从 PATH 找 | mkdssp 可执行文件 |
| `REDOCK_BIN` / `BLIND_BIN` | `$DSDP_HOME/...` | 两个 DSDP 二进制 |

```bash
python scripts/paths.py       # 一次性看全部路径 + 是否存在 + 切分文件行数
```

---

## 4. 快速开始（按顺序，全部在仓库根目录执行）

```bash
cd gnn-dsdp
python scripts/paths.py            # 0. 自检：4 样数据是否就位

python scripts/collect_structures.py                    # 1. 结构索引 + 覆盖率核对（秒级）

python scripts/build_dataset.py --split test            # 2. 5 Å 残基标签（test 12 s / val 1.5 min / train 21 min）
python scripts/build_dataset.py --split val
python scripts/build_dataset.py --split train

python scripts/graph.py --batch --split test  --workers 16   # 3. 46 维特征 + 建图（train 约 19 min，输出 data/graph/*.pt）
python scripts/graph.py --batch --split val   --workers 16
python scripts/graph.py --batch --split train --workers 16

sbatch slurm/train_full.sbatch        # 4. 训练（GPU，100 epoch 约 12 min；CPU 会慢很多）
#   等价：python scripts/train_full.py --device cuda --epochs 100 --batch-size 32 --patience 15
#           --seed 0 --tag c7b --ckpt-name best_model.pt

python scripts/evaluate_model.py --model best_model.pt --device cpu   # 5. val/test 指标 + 阈值表

python scripts/predict.py --protein 6qqw --out predict_6qqw.csv # 6. 单蛋白：每残基概率表

python scripts/residue_to_box.py --protein 6qqw \               # 7. 残基 → 对接盒子
    --ligand "$DSDP_HOME/test_dataset/PDBBind_time_split_dataset/6qqw/6qqw_ligand.pdbqt"

sbatch slurm/baseline_dsdp_cnn.sbatch           # 8a. 基线：原版 DSDP 3D-CNN 在 test 363 上的残基级指标
sbatch slurm/run_docking_examples.sbatch           # 8b. 端到端：1a2b / 6qqw / 6qxa 出 pose + RMSD（GPU）
sbatch slurm/benchmark_end2end.sbatch         # 8c. 40 个蛋白、两条管线对比（GPU）

python scripts/make_figures.py                       # 9. 表与图 → results/
python scripts/make_pocket_views.py --ids 6g2o,6jao,6qqw,6kqi,5zk7   #    口袋可视化 → results/viz/
```

**预计总耗时**（16 核 + 单卡）：数据构建约 45 min，训练约 12 min，评估/盒子/图几分钟。
**磁盘**：PDBBind 原包 ~10 GB，`data/graph` 约 4.1 GB，其余 < 1 GB。

### 常见改动

| 想做什么 | 怎么做 |
|---|---|
| 换 GPU 分区 | 改 `*.sbatch` 里的 `#SBATCH -p` |
| 只用一小部分训练（冒烟） | `python scripts/train_full.py --train-limit 3000 --device cuda` |
| 已有训练好的模型只做推理 | 直接用 `best_model.pt`：`scripts/predict.py` / `scripts/residue_to_box.py` / `scripts/run_gnn_dsdp.py` |
| 自己的蛋白 | `predict.py --protein my_protein.pdb`；盒子 `residue_to_box.py --protein my_protein.pdb` |

---

## 5. 方法要点（两分钟看懂）

1. **节点 = 残基**，**边 = 两残基 Cα 距离 ≤ 10 Å**，边权重 `1/d²`。
2. **节点特征 46 维**：氨基酸 one-hot 20 + 序列前后邻居 2 + DSSP 二级结构 7 + 主链氢键 8
   + 二面角（φ/ψ/τ/θ）4 + 主链/侧链 SASA 2 + 到蛋白质心的欧氏/余弦距离 2 + 节点度 1。
   （pLDDT 维因晶体结构没有该量而省略；`features.py --with-plddt` 可加恒 1.0 的第 47 维。）
3. **模型**：4 层 GCN `46 → 38 → 28 → 18 → 8 → 2`，ELU，log_softmax，**仅 3,570 参数**。
4. **标签**：残基重原子与配体重原子最小距离 **≤ 5 Å** 记正样本
   （train 4.30% / val 4.01% / test 5.60% 正样本，故用 **PR-AUC** 为主指标）。
5. **损失**：`NLLLoss(weight=[1, neg/pos≈22])` 对付类别不平衡。
6. **残基 → 盒子**：概率 ≥ 0.5 的残基 → Cα 单链接聚类（≤8 Å）→ 取最大的簇 →
   概率加权质心为盒心 → **36 Å 立方** → 交给 `DSDP_redocking/DSDP --box_min/--box_max`。
   阈值/边长/聚类半径**全部在 val 968 上选定**（见 `scripts/scan_box_params.py`），test 只做一次性报告。

---

## 6. 关键结论与坑

- **GNN 赢「Cα 单点体素」口径，输「3×3×3 邻域」口径**（PR 0.4049 vs 0.4925），端到端也略输
  （32.5% vs 40.0%，n=40）。原因：DSDP 的体素概率极稀疏（均值 0.0004），邻域最大值相当于"补容差"；
  GNN 概率本身就密（均值 0.23），再做最大平滑只会扩散假阳性（平滑后 PR 掉到 0.2487）。
- 端到端失败的两种模式很清楚：GNN 40 个里 **6 个盒子没罩住**、21 个罩住了但采样没找到；
  DSDP 是 4 / 20。**瓶颈主要不在"盒子大小"，而在"口袋选对没有"**。
- 盒子参数**只在 val 上选**，test 只做一次性报告，未回流调参。
- 工程坑（都写进了 `docs/01_environment.md` 第 6 节）：
  1. 登录节点无 GPU，CUDA 二进制会 `Segmentation fault`，必须走调度器；
  2. DSDP 的 `--dataset_path` 会把目录下**每个子目录**都当成复合物 —— `site/out/log` 不能放进去；
  3. **DSSP 4.x 输出列位变了**（1–5 列是连续序号，6–10 列才是残基号），Biopython 1.83 会错位，
     `scripts/features.py` 改为按固定列解析并用 Biopython 交叉验证氢键（410/410 一致）；
  4. 少数残基缺 CA，用重原子质心兜底。

---

## 7. 诚实的局限（demo 定位，不是论文）

1. **没有超过 DSDP 基线**：残基级 3×3×3 口径 DSDP 更强（PR 0.4925 vs 0.4049），端到端也更强
   （40.0% vs 32.5%，n=40、样本小、未做显著性检验）。本项目的贡献是"把管线换成 GNN 并跑通"，
   不宣称效果更好。
2. **盒子精度有限**：中心到真实配体质心中位 **10.5 Å**；36 Å 盒子只保证 **82.1%** 的复合物
   覆盖 ≥90% 配体。残基级粒度天然比 2 Å 体素粗。
3. **概率未校准**：val 上平均预测概率约 0.23，而真实正样本只有 4.01%（`pos_weight≈22` 的副作用）。
   PR/ROC 不受影响，但"概率 0.5"不等于"50% 把握"，阈值是**排序阈值**。
4. **失败模式**：40 个样本里 6 个盒子完全没罩住配体（如 6kqi、5zk7、6qxa），
   DSDP 在部分蛋白上能救回来（6kqi：GNN 32.9 Å vs DSDP 0.8 Å）。
5. **模型很小**（3,570 参数、46 维手工特征），没有 PLM 嵌入 / EGNN；train loss 到 0.36 仍在降，
   瓶颈是特征表达而不是过拟合。
6. **DUD-E 102 是另一套考场**：只有 pdbqt、蛋白是本地清洗成 PDB 后特征化的，与 test 363 不同源、
   正样本比例更高（7.1%），只作补充列。
7. **RMSD 口径**：重原子、原子顺序一一对应、**不叠合**（同一坐标系直接比），<2 Å 记成功；
   未做对称原子校正，长链配体可能因此偏严。

**下一步可以往哪走**（README 里留个方向）：换更强特征（PLM 嵌入 / EGNN）、
把"选哪个口袋"当独立分类任务、用 RMSD 而不是 AUC 做模型选择。

---

## 8. 文件清单

| 文件 | 作用 |
|---|---|
| `scripts/paths.py` | **唯一路径配置**（环境变量 + 默认值），`python scripts/paths.py` 自检 |
| `scripts/collect_structures.py` | 扫 PDBBind 建 `structure_index.json`，核对切分覆盖率 |
| `scripts/build_dataset.py` | 5 Å 残基标签 → `data/label/<split>/*.npz` |
| `scripts/features.py` | 46 维残基特征（自写 DSSP 解析 + freesasa） |
| `scripts/graph.py` | Cα ≤10 Å 建边、权重 1/d²、PyG `Data`；支持批量多进程 |
| `scripts/model.py` | 4 层 GCN + 维度自检 |
| `scripts/train_smoke.py` | 冒烟训练（30 图 / 2 epoch，验证管线） |
| `scripts/train_full.py` | 正式训练（早停 / 存 best / 日志 / 曲线） |
| `scripts/predict.py` | 每残基概率 → CSV |
| `scripts/residue_to_box.py` | 概率 → 聚类 → 36 Å 盒子（`box_min/box_max` + 诊断 json） |
| `scripts/run_gnn_dsdp.py` | GNN → 盒子 → DSDP 对接 → RMSD 一条龙 |
| `scripts/scan_box_params.py` | 盒子参数在 val 上的扫描（选阈值/边长/中心规则） |
| `scripts/baseline_dsdp_cnn.py` | 原版 DSDP 3D-CNN 推理 + 体素→残基投影（基线） |
| `scripts/evaluate_model.py` / `scripts/analyze_box_coverage.py` | 最终指标 / 盒子覆盖分析 |
| `scripts/benchmark_end2end.py` | 40 蛋白两条管线端到端对比 |
| `scripts/analyze_dsdp_boxes.py` | 按 DSDP 源码复算其盒子覆盖率 |
| `scripts/evaluate_dude.py` | DUD-E 102 补充评估 |
| `scripts/make_figures.py` / `scripts/make_pocket_views.py` | 表与图 / 口袋 3D 可视化 |
| `slurm/*.sbatch` | slurm 模板：`train_full` / `train_3000` / `baseline_dsdp_cnn` / `run_docking_examples` / `benchmark_end2end`(+`_smoke`) / `smoke_dsdp` |
| `best_model.pt` | **最终模型**（epoch 92，3,570 参数） |
| `docs/` | 各阶段实验报告：`01_environment` … `08_benchmark_and_figures`，`09_ppt_points` |
| `results/` | 两张主表 CSV、4 张图、5 个蛋白可视化 |
| `reports/` | 训练日志 json、阈值表、盒子扫描、数据统计等 |

> `docs/` 与 `reports/*.json` 里的路径是从原始运行记录里保留下来的占位符
> （`$GNNSITE_HOME` / `$DSDP_HOME` / `$SPLIT_DIR`），含义见 §3.4，不影响阅读结论。

---

## 9. 图与可视化

| 图 | 路径 | 内容 |
|---|---|---|
| 表 1 主图 | `results/fig_table1_auc.png` | 4 种口径的 ROC/PR 柱状图 |
| 端到端 | `results/fig_e2e_rmsd.png` | RMSD 箱线图 + 成功率 |
| 中心误差 | `results/fig_center_error_hist.png` | 363 复合物盒子中心误差直方图 |
| 训练曲线 | `results/fig_training_curve.png` | loss + val AUC 随 epoch |
| 口袋可视化 | `results/viz/<id>_pocket.png` | 5 个代表蛋白（6g2o/6jao/6qqw 成功，6kqi/5zk7 失败） |
| PyMOL 脚本 | `results/viz/<id>_pocket.pml` | 有 PyMOL 的机器：`pymol -cq <id>_pocket.pml` 重渲染 |

---

## 10. 引用与许可

- 代码许可：见 `LICENSE`（MIT）。
- **数据**：PDBBind v2020（禁止再分发）、EquiBind 的 time split 编号文件、DSDP 仓库自带的测试集。
  本仓库不包含上述数据，使用时请遵守各自许可。
- 模型权重 `best_model.pt` 由 PDBBind v2020 训练而来，**仅限学术研究使用**。
- 如果用到本仓库的代码，请同时引用：

```bibtex
@article{dsdp2023,
  title  = {DSDP: A Blind Docking Strategy Accelerated by GPUs},
  author = {Gao Group, Peking University},
  year   = {2023},
  note   = {https://github.com/PKUGaoGroup/DSDP}
}

@inproceedings{equibind2022,
  title     = {EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction},
  author    = {St{\"a}rk, Hannes and Ganea, Octavian-Eugen and Pattanaik, Lagnajit and Barzilay, Regina and Jaakkola, Tommi},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2022}
}

@inproceedings{nodecoder2021,
  title  = {Residue characterization on AlphaFold2 protein structures using graph neural networks},
  author = {Nasim Abdollahi and others},
  booktitle = {NeurIPS MLSB Workshop},
  year   = {2021},
  note   = {https://github.com/NasimAbdollahi/Nodecoder}
}
```
