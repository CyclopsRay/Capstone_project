#!/usr/bin/env bash
#SBATCH --job-name=dpb-train
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=data/logs/dpb-train-%j.out
# Bai et al. 2026 main_benchmark reproduction: MODELS / FOLDS / EPOCHS (optional) / RESULTS via --export.
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
W=$UNDETECT_DATA/repro/dpb_bench; ENV=$UNDETECT_DATA/envs/dpb-benchmark
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=8 HF_HOME=$UNDETECT_DATA/hf_cache PATH="$ENV/bin:$PATH"
cd "$W"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
sha256sum data/XPert/processed_data/l1000_sdst_78453.h5ad | cut -c1-16
python experiments/main_benchmark/train_all.py --models ${MODELS} --folds ${FOLDS:-0 1 2 3 4} \
  --results_dir "${RESULTS:-$W/results/main_benchmark}" ${EPOCHS:+--epochs $EPOCHS} --continue_on_fail
date --iso-8601=seconds
