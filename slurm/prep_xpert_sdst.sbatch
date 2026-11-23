#!/usr/bin/env bash
#SBATCH --job-name=prep-sdst
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=data/logs/prep-sdst-%j.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
ENV=$UNDETECT_ROOT/.pixi/envs/prep
date --iso-8601=seconds; hostname
"$ENV/bin/python" -c "import numpy, pandas, h5py; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'h5py', h5py.__version__)"
"$ENV/bin/python" -I $UNDETECT_ROOT/scripts/prep_xpert_sdst.py ${EXTRA:-}
date --iso-8601=seconds
