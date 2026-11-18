#!/bin/bash
#SBATCH -J gnn_c2
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 3:00:00
#SBATCH -o c2_%j.out
#SBATCH -e c2_%j.err
#
# C2 基线：用原版 DSDP 的 3D-CNN（binding_site_model.pth）跑 test 363 的位点预测，
# 再投影到残基级。需要 DSDP 仓库（默认与本仓库同级，可用 DSDP_HOME 覆盖）。
# 提交前：  sbatch baseline_dsdp_cnn.sbatch

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-DSDP}"

export DSDP_HOME="${DSDP_HOME:-$(python -c "import paths;print(paths.DSDP_HOME)")}"
echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME dsdp=$DSDP_HOME"
nvidia-smi -L || true

# baseline_dsdp_cnn.py 会 import DSDP_blind_docking.py 里的 DrugSiteMind，所以从 DSDP 根目录跑
cd "$DSDP_HOME"
python "$HERE/baseline_dsdp_cnn.py" --device cuda
echo "END $(date) exit=$?"
