#!/bin/bash
#SBATCH -J gnn_c9
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 1:30:00
#SBATCH -o c9_%j.out
#SBATCH -e c9_%j.err
#
# C9：GNN -> 盒子 -> DSDP 对接 -> RMSD 的三个代表蛋白
#   1a2b（仓库自带例子） / 6qqw（成功案例） / 6qxa（失败案例）
# 依赖：DSDP 已编译（$DSDP_HOME/DSDP_redocking/DSDP），test 数据集在 $DSDP_HOME/test_dataset
# 提交前：  sbatch c9_gpu.sbatch

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p runs

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-DSDP}"

TEST="$(python -c "import paths;print(paths.TEST_PDBBIND_DIR)")"
DSDP_DS="$(python -c "import paths;print(paths.DSDP_DATASET_DIR)")"
echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
echo "test=$TEST"
nvidia-smi -L || true

echo "=== 1a2b (repo example, pdbqt protein converted to pdb for the GNN) ==="
python run_gnn_dsdp.py \
  --protein "$DSDP_DS/1a2b/1a2b_protein.pdbqt" \
  --ligand  "$DSDP_DS/1a2b/1a2b_ligand.pdbqt" \
  --out runs/1a2b_out.pdbqt --log runs/1a2b_out.log --json runs/1a2b.json \
  --device cuda

echo "=== 6qqw (GNN top-20 hits 12/20) ==="
python run_gnn_dsdp.py \
  --protein 6qqw \
  --ligand  "$TEST/6qqw/6qqw_ligand.pdbqt" \
  --out runs/6qqw_out.pdbqt --log runs/6qqw_out.log --json runs/6qqw.json \
  --device cuda

echo "=== 6qxa (GNN top-20 hits 0/20, failure case) ==="
python run_gnn_dsdp.py \
  --protein 6qxa \
  --ligand  "$TEST/6qxa/6qxa_ligand.pdbqt" \
  --out runs/6qxa_out.pdbqt --log runs/6qxa_out.log --json runs/6qxa.json \
  --device cuda

echo "END $(date) exit=$?"
ls -l runs/ | head
