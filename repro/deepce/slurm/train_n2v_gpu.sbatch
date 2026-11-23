#!/usr/bin/env bash
#SBATCH --job-name=deepce-n2v
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH --output=data/repro/deepce/logs/n2v-train-%A_%a.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
# RECOMB study E3: DeepCE with node2vec gene vectors from STRING functional >= 400 and its nulls
# (make_n2v_genes.py, job 296800). One array task per graph variant, seeds 0-2 in sequence, 60 epochs.
ENV=$UNDETECT_ROOT/.pixi/envs/cluster
KINDS=(real rewired labels er); K=${KINDS[${SLURM_ARRAY_TASK_ID:-0}]}
GV=$UNDETECT_DATA/repro/deepce/gene_variants
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
cd $UNDETECT_ROOT/repro/deepce
for s in 0 1 2; do
  "$ENV/bin/python" run_deepce.py --gene-variant file --gene-file "$GV/n2v_$K.csv" --label "n2v_$K" --seed $s --epochs 60
done
date --iso-8601=seconds
