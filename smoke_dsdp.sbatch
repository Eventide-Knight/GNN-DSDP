#!/bin/bash
#SBATCH -J gnn_smoke_1a2b
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 00:40:00
#SBATCH -o smoke1a2b_%j.out
#SBATCH -e smoke1a2b_%j.err
#
# 冒烟：用原版 DSDP blind docking 跑通一个复合物（验证 DSDP 编译产物 + GPU 环境）。
# 需要先把复合物放进一个"只含复合物子目录"的文件夹（DSDP 会把它下面每个子目录当成一个复合物）：
#   mkdir -p smoke_1a2b/data && cp -r $DSDP_HOME/test_dataset/DSDP_dataset/1a2b smoke_1a2b/data/
# 提交前：  sbatch smoke_dsdp.sbatch

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-DSDP}"

export DSDP_HOME="${DSDP_HOME:-$(python -c "import paths;print(paths.DSDP_HOME)")}"
DATA="${GNNSMOKE_DATA:-$HERE/smoke_1a2b/data}"
OUT="$HERE/smoke_1a2b"
mkdir -p "$OUT/site" "$OUT/out" "$OUT/log"

echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi -L || true
cd "$DSDP_HOME"
python DSDP_blind_docking.py \
  --dataset_path "$DATA/" \
  --dataset_name smoke_1a2b \
  --site_path "$OUT/site/" \
  --exhaustiveness 384 --search_depth 40 --top_n 1 \
  --out "$OUT/out/" \
  --log "$OUT/log/"
echo "PYTHON_EXIT=$?"
ls -l "$OUT/out/"
