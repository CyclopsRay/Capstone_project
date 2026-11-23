#!/usr/bin/env bash
#SBATCH --job-name=deepce
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=data/repro/deepce/logs/train-%A_%a.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
# DeepCE reproduction + gene-feature ablation (repro/deepce, gitignored). Array index -> (variant, seed).
ENV=$UNDETECT_ROOT/.pixi/envs/cluster
VARIANTS=(ppi shuffled random onehot)
i=${SLURM_ARRAY_TASK_ID:-0}
VARIANT=${VARIANT:-${VARIANTS[$((i % 4))]}}
SEED=${SEED:-$((i / 4))}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$ENV/bin/python" -c "import torch, rdkit, numpy; print('torch', torch.__version__, torch.version.cuda, 'rdkit', rdkit.__version__, 'numpy', numpy.__version__)"
cd $UNDETECT_ROOT/repro/deepce
"$ENV/bin/python" run_deepce.py --gene-variant "$VARIANT" --seed "$SEED" --epochs "${EPOCHS:-100}" ${OUT:+--out "$OUT"}
date --iso-8601=seconds
