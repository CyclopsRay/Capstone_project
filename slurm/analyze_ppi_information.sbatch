#!/usr/bin/env bash
#SBATCH --job-name=ppi-info
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=data/logs/ppi-info-%j.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
# Exploratory, model-free: does STRING wiring carry response information beyond degree? (report/10_09_record.md)
ENV_ROOT=$UNDETECT_DATA/envs/undetect-prep
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
cd $UNDETECT_ROOT/scripts
for s in ${SLICES:-P2-24 P1-24}; do
  "${ENV_ROOT}/bin/python" analyze_ppi_information.py --slice "$s"
done
date --iso-8601=seconds
