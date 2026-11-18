#!/bin/bash
#SBATCH -J gnn_c10smoke
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -o c10smoke_%j.out
#SBATCH -e c10smoke_%j.err
#
# C10 冒烟：只跑 2 个蛋白，验证两条管线都能出 RMSD
# 提交前：  sbatch c10_smoke.sbatch

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p runs/c10_smoke

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-DSDP}"

echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi -L || true
python c10_e2e.py --ids 6qqw,1a2b --device cuda \
  --json runs/c10_smoke.json --csv runs/c10_smoke.csv --outdir runs/c10_smoke
echo "END $(date) exit=$?"
