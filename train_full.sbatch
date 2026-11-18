#!/bin/bash
#SBATCH -J gnn_c7b
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -o c7b_%j.out
#SBATCH -e c7b_%j.err
#
# 正式训练（全量 16,379）。提交前：  sbatch train_full.sbatch
# 需要事先 conda 环境（见 README「2. 环境」），或用 CONDA_SH / CONDA_ENV 覆盖。

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-DSDP}"

echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME dir=$HERE"
nvidia-smi -L || true
python -c "import torch;print('cuda:',torch.cuda.is_available())"

python train_full.py --device cuda --epochs 100 --batch-size 32 --patience 15 \
    --seed 0 --weight-mode linear --tag c7b --ckpt-name best_model.pt
echo "END $(date) exit=$?"
