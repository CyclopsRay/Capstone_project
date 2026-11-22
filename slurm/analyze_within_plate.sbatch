#!/usr/bin/env bash
#SBATCH --job-name=within-plate
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=data/logs/within-plate-%j.out
# Within-plate contrast metric PCC_within (scripts/analyze_within_plate.py; no training; record 2026-11-03 07:00).
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
ENV=$UNDETECT_DATA/envs/undetect
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas; print("numpy", numpy.__version__, "pandas", pandas.__version__)"
sha256sum $UNDETECT_ROOT/scripts/analyze_within_plate.py $UNDETECT_DATA/derived/xpert_sdst/v1/sdst.npz
"$ENV/bin/python" -I -u $UNDETECT_ROOT/scripts/analyze_within_plate.py
date --iso-8601=seconds
