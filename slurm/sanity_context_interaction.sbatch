#!/usr/bin/env bash
#SBATCH --job-name=ctx-sanity
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=data/logs/ctx-sanity-%j.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
# Context x compound sanity checks on the drug-blind benchmark (no network training): line-similarity structure of
# compound responses, oracle headroom from v3 saved predictions, linear drug-free backbones. Deterministic (seed 0).
ENV=$UNDETECT_ROOT/.pixi/envs/cluster
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date --iso-8601=seconds; hostname
"$ENV/bin/python" -c "import numpy, pandas; print('numpy', numpy.__version__, 'pandas', pandas.__version__)"
cat $UNDETECT_DATA/derived/xpert_sdst/v1/manifest.json
SCRIPT=${SCRIPT:-sanity_context_interaction.py}  # or sanity_response_line_kernel.py
sha256sum $UNDETECT_ROOT/scripts/sanity_context_interaction.py $UNDETECT_ROOT/scripts/$SCRIPT
"$ENV/bin/python" -I $UNDETECT_ROOT/scripts/$SCRIPT
date --iso-8601=seconds
