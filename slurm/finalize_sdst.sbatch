#!/usr/bin/env bash
#SBATCH --job-name=finalize-sdst
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/finalize-sdst-%j.out
# Final drug-blind numbers on the CPU backbone run (scripts/finalize_sdst.py; no training).
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
ENV=$UNDETECT_ROOT/.pixi/envs/cluster
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas; print('numpy', numpy.__version__, 'pandas', pandas.__version__)"
sha256sum $UNDETECT_ROOT/scripts/finalize_sdst.py
"$ENV/bin/python" -I -u $UNDETECT_ROOT/scripts/finalize_sdst.py
date --iso-8601=seconds
