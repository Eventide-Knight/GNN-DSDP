#!/bin/bash
#SBATCH -J gnn_c2
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 3:00:00
#SBATCH -o /share/home/huangyy/DSDP/gnn_site/reports/c2_slurm.out
#SBATCH -e /share/home/huangyy/DSDP/gnn_site/reports/c2_slurm.err

set -x
source /share/home/huangyy/miniconda3/etc/profile.d/conda.sh
conda activate DSDP
cd /share/home/huangyy/DSDP || exit 1
echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi -L
python gnn_site/c2_baseline.py --device cuda
rc=$?
echo "END $(date) exit=$rc"
exit $rc
