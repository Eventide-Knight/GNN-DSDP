#!/bin/bash
#SBATCH -J gnn_c7a
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -o c7a_%j.out
#SBATCH -e c7a_%j.err
#
# 两段式训练的第一段：只用前 3,000 个 train 复合物（用来快速验证管线与超参）。
# 提交前：  sbatch c7a_gpu.sbatch

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-DSDP}"

echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME dir=$HERE"
nvidia-smi -L || true
python -c "import torch;print('cuda:',torch.cuda.is_available())"

python train_full.py --device cuda --epochs 100 --batch-size 32 --patience 15 \
    --seed 0 --weight-mode linear --tag c7a --train-limit 3000 --ckpt-name c7a_best.pt
echo "END $(date) exit=$?"
