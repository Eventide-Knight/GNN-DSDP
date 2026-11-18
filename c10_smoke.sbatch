#!/bin/bash
#SBATCH -J gnn_c10smoke
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -o /share/home/huangyy/DSDP/gnn_site/reports/c10_smoke_slurm.out
#SBATCH -e /share/home/huangyy/DSDP/gnn_site/reports/c10_smoke_slurm.err

set -x
source /share/home/huangyy/miniconda3/etc/profile.d/conda.sh
conda activate DSDP
cd /share/home/huangyy/DSDP/gnn_site || exit 1
mkdir -p runs/c10_smoke
nvidia-smi -L
python c10_e2e.py --ids 6qqw,1a2b --device cuda \
  --json runs/c10_smoke.json --csv runs/c10_smoke.csv --outdir runs/c10_smoke
rc=$?
echo "END $(date) exit=$rc"
exit $rc
