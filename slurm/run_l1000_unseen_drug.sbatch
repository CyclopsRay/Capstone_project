#!/usr/bin/env bash
#SBATCH --job-name=l1000-unseen
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=data/logs/l1000-unseen-%j.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
# Unseen-drug benchmark on L1000 (pre-registered 2026-10-03, report/10_09_record.md).
# SLICE=P1-24|P1-6|P2-24, SPLIT=ik14|scaffold, EXTRA= further args.
ENV_ROOT=$UNDETECT_DATA/envs/undetect-prep
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" -c 'import numpy, scipy, pandas, sklearn; print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "sklearn", sklearn.__version__)'
"${ENV_ROOT}/bin/python" $UNDETECT_ROOT/scripts/run_l1000_unseen_drug.py \
  --slice "${SLICE:-P1-24}" --split "${SPLIT:-ik14}" ${EXTRA:-}
date --iso-8601=seconds
