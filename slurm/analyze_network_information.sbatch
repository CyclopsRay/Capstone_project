#!/usr/bin/env bash
#SBATCH --job-name=net-info
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-23%4
#SBATCH --output=data/logs/net-info-%A_%a.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
# RECOMB study E1/E7 (PLAN.md Phase 6): control-ladder information test, one (slice, network) per task.
ENV_ROOT=$UNDETECT_ROOT/.pixi/envs/prep
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
SLICES=(P2-24 P1-24 T-5)
NETS=(string_physical400 string_functional700 biogrid huri primekg omnipath collectri signaling_tf)
i=${SLURM_ARRAY_TASK_ID:-0}
SLICE=${SLICES[$((i / 8))]}; NET=${NETS[$((i % 8))]}
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
cd $UNDETECT_ROOT/scripts
"${ENV_ROOT}/bin/python" analyze_network_information.py --slice "$SLICE" --networks "$NET"
date --iso-8601=seconds
