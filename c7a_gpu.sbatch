#!/bin/bash
#SBATCH -J gnn_c7a
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -o /share/home/huangyy/DSDP/gnn_site/reports/c7a_gpu_slurm.out
#SBATCH -e /share/home/huangyy/DSDP/gnn_site/reports/c7a_gpu_slurm.err

set -x
source /share/home/huangyy/miniconda3/etc/profile.d/conda.sh
conda activate DSDP
cd /share/home/huangyy/DSDP/gnn_site || exit 1
echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi -L
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python train_full.py --device cuda --epochs 100 --batch-size 32 --patience 15 \
    --seed 0 --weight-mode linear --tag c7a
echo "END $(date) exit=$?"
