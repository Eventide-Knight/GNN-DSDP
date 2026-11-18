#!/bin/bash
#SBATCH -J gnn_c9
#SBATCH -p gpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 1:30:00
#SBATCH -o /share/home/huangyy/DSDP/gnn_site/reports/c9_slurm.out
#SBATCH -e /share/home/huangyy/DSDP/gnn_site/reports/c9_slurm.err

set -x
source /share/home/huangyy/miniconda3/etc/profile.d/conda.sh
conda activate DSDP
cd /share/home/huangyy/DSDP/gnn_site || exit 1
mkdir -p runs
echo "START $(date) job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi -L

TEST=/share/home/huangyy/DSDP/test_dataset/PDBBind_time_split_dataset
DSDP_DS=/share/home/huangyy/DSDP/test_dataset/DSDP_dataset

echo "=== 1a2b (repo example, pdbqt protein -> converted pdb for GNN) ==="
python run_gnn_dsdp.py \
  --protein $DSDP_DS/1a2b/1a2b_protein.pdbqt \
  --ligand  $DSDP_DS/1a2b/1a2b_ligand.pdbqt \
  --out runs/1a2b_out.pdbqt --log runs/1a2b_out.log --json runs/1a2b.json \
  --device cuda
echo "=== 6qqw (GNN top-20 hits 12/20) ==="
python run_gnn_dsdp.py \
  --protein 6qqw \
  --ligand  $TEST/6qqw/6qqw_ligand.pdbqt \
  --out runs/6qqw_out.pdbqt --log runs/6qqw_out.log --json runs/6qqw.json \
  --device cuda
echo "=== 6qxa (GNN top-20 hits 0/20, failure case) ==="
python run_gnn_dsdp.py \
  --protein 6qxa \
  --ligand  $TEST/6qxa/6qxa_ligand.pdbqt \
  --out runs/6qxa_out.pdbqt --log runs/6qxa_out.log --json runs/6qxa.json \
  --device cuda

rc=$?
echo "END $(date) exit=$rc"
ls -l runs/
exit $rc
