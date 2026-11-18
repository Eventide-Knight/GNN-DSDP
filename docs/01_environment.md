# C1 环境与数据盘点（Step 1）

生成时间：2026-11-18 17:57（Asia/Shanghai）
执行机：`mgt`（Ubuntu 22.04.5 LTS，kernel 5.15.0-153，Slurm 24.05.3）
工作根：`$DSDP_HOME`

**结论：C1 全部验收项通过 ✅（唯一注意点：登录节点无 GPU，必须走 slurm GPU 分区）。**

---

## 1. 计算环境

| 项 | 实测值 | 状态 |
|---|---|---|
| CPU / 内存（登录节点） | 96 核 / 251 GB | ✅ |
| GPU（登录节点） | **无**：`nvidia-smi` 报 "couldn't communicate with the NVIDIA driver" | ⚠️ 见 §6 |
| Slurm 分区 | `gpu1*`（默认，cn1/cn3/cn4，`gpu:geforce:8`）、`gpu2`（cn2，`gpu:rtx:8`）、`gpu3`（cn5-7，`gpu:geforce:8`） | ✅ |
| GPU 实机 | 冒烟作业拿到 **NVIDIA GeForce RTX 4090**，`torch.cuda.is_available()=True` | ✅ |
| gcc / g++ | 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.3) | ✅ |
| nvcc / CUDA | `/usr/local/cuda/bin/nvcc`，**CUDA 12.4 V12.4.131**（登录节点未把 nvcc 加进 PATH，需要时自行加） | ✅ |
| 磁盘 | `/share` 641T，已用 454T，可用 188T | ✅ |

## 2. conda 环境（DSDP.yml 已建好，本次未重建）

`conda` 不在非交互 PATH 里，需先 `source $CONDA_ROOT/etc/profile.d/conda.sh`。

解释器：`$PY` → **3.9.25**

| 包 | 版本 | 状态 |
|---|---|---|
| torch | **2.8.0+cu128** | ✅ |
| torch_geometric (PyG) | **2.6.1**（`GCNConv(46,38)` 前向实测通过） | ✅ |
| numpy / pandas / scipy | 1.26.3 / 2.2.0 / 1.12.0 | ✅ |
| scikit-learn | 1.4.0 | ✅ |
| biopython | 1.83 | ✅ |
| openbabel | 3.1.0（python 绑定） | ✅ |
| scikit-image | 0.24.0（`DSDP_blind_docking.py` 依赖） | ✅ |
| **freesasa** | **2.2.1**（本次 conda-forge 补装） | ✅ |
| **mkdssp（dssp 包）** | **4.5.3**，`envs/DSDP/bin/mkdssp`（本次补装；任务书写的 4.6.1，conda 解析到 4.5.3，功能等价） | ✅ |
| **loguru** | 0.8.0（本次 pip 补装） | ✅ |
| rdkit | 未安装 | ⚠️ 规格未要求，仅记录 |

CUDA 相关事实：`torch.version.cuda = 12.8`、`torch.cuda.is_available() = False`（登录节点）；**在 GPU 节点为 True**。

## 3. DSDP 编译产物（已存在，未重编译）

| 产物 | 路径 | 检查 |
|---|---|---|
| blind docking | `DSDP_blind_docking/DSDP` (3.1 MB, 11-11 19:32) | ✅ 可执行，无参数打印 usage；`ldd` 无 not found |
| redocking | `DSDP_redocking/DSDP` (3.1 MB) | ✅ 同上，支持 `--box_min/--box_max` |
| surface tool | `surface_tool/surface_tool` (2.7 MB) | ✅ 无 missing so |
| feature tool | `protein_feature_tool/protein_feature_tool` (24 KB) | ✅ 无 missing so |

Makefile 现状：三个 Makefile 有**历史本地改动**（git 中 `M`），`COMMON_COMMAND` 为 `-gencode arch=compute_70,code=compute_70 ... -std=c++11`，与 CUDA 12.4 + gcc 11 匹配。**本次没有重新编译**（产物可用，且任务书禁止动 CUDA 源码）。

`DSDP_blind_docking.py` 有历史改动（`.bak` 对比仅 1 行）：`torch.load(..., weights_only=False)`，为 torch≥2.6 必需，保留。

## 4. 数据盘点

### 4.1 测试集（仓库自带，已恢复，无需再 checkout）

| 目录 | 复合物数 | 内容 |
|---|---|---|
| `test_dataset/DSDP_dataset` | **995** | `<id>_protein.pdbqt` + `<id>_ligand.pdbqt` |
| `test_dataset/DUD_E_dataset` | **102** | 同上 |
| `test_dataset/PDBBind_time_split_dataset` | **363** | 同上 |
| **合计** | **1460** ✅ | 与任务书 995+102+363 完全一致 |

`train_example/train_dataset/` 也已恢复（1agw、1aht …）。

### 4.2 训练/验证/测试编号

| split | 文件 | 行数 |
|---|---|---|
| train | `DataInEquibind/timesplit_no_lig_overlap_train` | 16379 |
| val | `DataInEquibind/timesplit_no_lig_overlap_val` | 968 |
| test | `DataInEquibind/timesplit_test` | 363 |

⚠️ **坑**：任务书 §2.5 写的 `timesplit_no_lig_overlap_test` **在机器上不存在**，实际文件名是 `timesplit_test`。已验证它与仓库 `test_dataset/PDBBind_time_split_dataset` 的 363 个目录**集合完全相同**，即同一批。

互斥性实测：train∩val=0、train∩test=0、val∩test=0（官方 time split，无泄漏）。

### 4.3 PDBBind v2020 全量结构（本次解包）

原始包：`DataInEquibind/pdbbind_v2020/` 三个 `tar.gz`，解包前 `gzip -t` 三个全部 OK。
解包位置：`$GNNSITE_HOME/pdbbind_v2020/`（refined 2.7G + general 7.3G，合计约 10G）。

| 子集 | 4 字符 ID 目录 | protein / mol2 / sdf / pocket |
|---|---|---|
| `refined-set/` | **5316** | 5316 / 5316 / 5316 / 5316 ✅ |
| `v2020-other-PL/` | **14127** | 14127 / 14127 / 14127 / 14127 ✅ |
| **合计** | **19443** ✅ | 与任务书一致 |

索引与覆盖率（`gnn_site/structure_index.json`，由 `collect_structures.py` 生成）：

| split | 需要的 ID | 命中 | 缺失 |
|---|---|---|---|
| train | 16379 | **16379** | **0** |
| val | 968 | **968** | **0** |
| test | 363 | **363** | **0** |
| 索引缺 protein/ligand 的条目 | — | — | **0** |

`gnn_site/reports/skipped.txt`：**0 行**（即无任何 ID 需要走 RCSB 兜底）。

## 5. 冒烟结果（验收关键）

### 5.1 1a2b blind docking（DSDP 全链路）

登录节点直接跑 `DSDP_redocking/DSDP` → **Segmentation fault (exit 139)**，因为无 GPU 驱动。改走 slurm：

```bash
cd $GNNSITE_HOME
sbatch smoke_dsdp.sbatch     # -p gpu1 --gres=gpu:1 -t 40min
```

作业 159677 在 cn1（RTX 4090）成功，耗时约 **11 s**：

| 产物 | 大小 | 说明 |
|---|---|---|
| `gnn_site/smoke_1a2b/out/1a2b_out.pdbqt` | 4712 B | **验收物**，含 `REMARK DSDP RESULT: -9.627915`，41 个 ATOM |
| `gnn_site/smoke_1a2b/site/1a2b.npy` | 186 KB | 36³ 位点概率体素（DrugSiteMind 推理） |
| `gnn_site/smoke_1a2b/site/1a2b_feature_generate.txt` | 170 KB | protein_feature_tool 输出 |
| `gnn_site/smoke_1a2b/site/1a2b_surface.txt` | 2.8 KB | surface_tool 输出 |

注意 sbatch 里 `--dataset_path` 必须指向**只含复合物目录**的文件夹：脚本会把 `dataset_path` 下每个子目录当成一个复合物，若把 `site/`、`out/`、`log/` 放在里面会被误判成复合物（本次已踩并修正为 `smoke_1a2b/data/`）。

### 5.2 PyG 可用性

```bash
python -c "import torch_geometric; from torch_geometric.nn import GCNConv; ..."
# pyg 2.6.1 GCNConv(46,38) -> torch.Size([10, 38])   ✅
```

### 5.3 特征工具可用性

- `mkdssp 4.5.3` 在 PDBBind `6ayo_protein.pdb` 上 0.6 s 出 DSSP（H/E/G/I/B/S/T 均有赋值）✅
- ⚠️ 次要告警：`mkdssp` 会往 stderr 刷 `Unknown compound: GLY ...`，原因是 conda 包只带了 `share/libcifpp/{dssp-extension.dic,mmcif_pdbx.dic}`，缺 `components.cif`。**不影响输出**；若要消音，可 `export LIBCIFPP_DATA_DIR=...` 并下载 `https://files.wwpdb.org/pub/pdb/data/monomers/components.cif`（可选，未做）。
- `import freesasa` ✅

## 6. 已知坑与注意事项（给后续 chunk）

1. **登录节点没有 GPU**：DSDP 任何 CUDA 二进制在登录节点会 segfault；训练/对接都要 `sbatch`/`srun` 到 `gpu1|gpu2|gpu3`。
2. **GPU 资源紧张**：盘点上 3 个分区共 7 节点 × 8 卡**全部 Allocated**（别人长期占用），提交后可能排队；本次 1 卡作业碰巧秒进。C7b 全量训练要按"会排队"来预留时间。
3. `conda` 需先 source `conda.sh`；非交互 shell 里 `conda` 命令不存在。
4. `nvcc` 不在 PATH：`export PATH=/usr/local/cuda/bin:$PATH`。
5. test 的 split 文件名是 `timesplit_test`，不是 `timesplit_no_lig_overlap_test`。
6. PDBBind 的 `protein.pdb` 里 **HETATM 只有 HOH（水）**（6ayo 实测 349 个 HOH），**不含配体**，所以不会泄漏标签；但解析时必须只取 ATOM + 20 种标准残基。
7. PDBBind 的 mol2 **含氢**（6ayo：22 个 H / 42 原子），标签必须只取重原子。

## 7. 一键命令清单（可复现）

```bash
# --- 0. 进环境 ---
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate DSDP
cd $DSDP_HOME

# --- 1. 校验依赖 ---
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
python -c "import freesasa, loguru; print('freesasa/loguru ok')"
mkdir -p /tmp/dsspt && mkdssp --output-format dssp \
  gnn_site/pdbbind_v2020/refined-set/6ayo/6ayo_protein.pdb /tmp/dsspt/6ayo.dssp >/dev/null 2>&1 && echo "mkdssp ok"

# --- 2. 校验数据 ---
for d in test_dataset/DSDP_dataset test_dataset/DUD_E_dataset test_dataset/PDBBind_time_split_dataset; do
  echo "$d $(find $d -mindepth 1 -maxdepth 1 -type d | wc -l)"; done
# 期望 995 / 102 / 363
wc -l $SPLIT_DIR/timesplit_no_lig_overlap_train \
      $SPLIT_DIR/timesplit_no_lig_overlap_val \
      $SPLIT_DIR/timesplit_test    # 16379 / 968 / 363

# --- 3. 重建结构索引（19443，覆盖率 100%） ---
python gnn_site/collect_structures.py

# --- 4. 冒烟：1a2b blind docking（必须上 GPU 节点） ---
cd gnn_site && sbatch smoke_dsdp.sbatch
# 产出 smoke_1a2b/out/1a2b_out.pdbqt

# --- 5. 残基级 5 Å 标签（test 363 全量，约 12 s） ---
python gnn_site/build_dataset.py --split test      # -> data/label/test/*.npz + reports/label_stat_test.txt
python gnn_site/build_dataset.py --split val --limit 10
python gnn_site/build_dataset.py --split train --limit 3000   # 约 1~2 min
```

## 8. C1 验收对照

| 验收项 | 结果 | 证据 |
|---|---|---|
| `test_dataset` 恢复为 1460 个复合物 | ✅ | 995+102+363 |
| PDBBind v2020 解包 + 19443 ID 与 train/val/test 100% 对上 | ✅ | 19443；16379/968/363 全命中，missing=0 |
| DSDP 编译产物可用 | ✅ | 4 个产物可执行、ldd 干净、1a2b 全链路出 pdbqt |
| NodeCoder 在 `$NODECODER_DIR` | ✅ | featurizer/gcn/graph_generator 齐全 |
| 依赖 freesasa/mkdssp/loguru/PyG 齐 | ✅ | 2.2.1 / 4.5.3 / 0.8.0 / 2.6.1 |
| `python -c "import torch_geometric"` 不报错 | ✅ | 2.6.1 |
| 能跑出 `1a2b_out.pdbqt` | ✅ | 4712 B，score -9.63 |
