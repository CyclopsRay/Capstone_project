#!/usr/bin/env bash
#SBATCH --job-name=oracle-final
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=data/logs/oracle-final-%j.out
# Uniform oracle on the final drug-blind run, same rows as UnDetect / XPert / ensemble (scripts/oracle_final.py; no
# training; record 2026-11-03). Deterministic: no random numbers are drawn.
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
ENV=$UNDETECT_ROOT/.pixi/envs/cluster
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas; print('numpy', numpy.__version__, 'pandas', pandas.__version__)"
cat $UNDETECT_DATA/derived/xpert_sdst/v1/manifest.json
sha256sum $UNDETECT_ROOT/scripts/oracle_final.py $UNDETECT_DATA/results/sdst_hybrid/final_cpu/final.json
"$ENV/bin/python" -I -u $UNDETECT_ROOT/scripts/oracle_final.py
date --iso-8601=seconds
