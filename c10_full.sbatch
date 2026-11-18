#!/bin/bash
#SBATCH -J gnn_c10full
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o c10full_%j.out
#SBATCH -e c10full_%j.err
#
# C10：40 个 test 复合物、两条管线（GNN 盒子 vs DSDP 原版盲对接）的端到端对比
# 提交前：  sbatch c10_full.sbatch

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p runs/c10

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-DSDP}"

echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi -L || true
python c10_e2e.py --n 40 --device cuda \
  --json runs/c10_e2e.json --csv runs/c10_e2e.csv --outdir runs/c10
echo "END $(date) exit=$?"
