#!/bin/bash
#SBATCH -J gnn_smoke_1a2b
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 00:40:00
#SBATCH -o /share/home/huangyy/DSDP/gnn_site/smoke_1a2b_slurm.out
#SBATCH -e /share/home/huangyy/DSDP/gnn_site/smoke_1a2b_slurm.err

set -x
source /share/home/huangyy/miniconda3/etc/profile.d/conda.sh
conda activate DSDP
cd /share/home/huangyy/DSDP || exit 1
mkdir -p ./gnn_site/smoke_1a2b/site ./gnn_site/smoke_1a2b/out ./gnn_site/smoke_1a2b/log
nvidia-smi -L
python -c "import torch; print('cuda avail:', torch.cuda.is_available())"
python DSDP_blind_docking.py \
  --dataset_path ./gnn_site/smoke_1a2b/data/ \
  --dataset_name smoke_1a2b \
  --site_path ./gnn_site/smoke_1a2b/site/ \
  --exhaustiveness 384 --search_depth 40 --top_n 1 \
  --out ./gnn_site/smoke_1a2b/out/ \
  --log ./gnn_site/smoke_1a2b/log/
echo "PYTHON_EXIT=$?"
ls -l /share/home/huangyy/DSDP/gnn_site/smoke_1a2b/out/ /share/home/huangyy/DSDP/gnn_site/smoke_1a2b/site/
