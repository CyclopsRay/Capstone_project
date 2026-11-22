#!/usr/bin/env bash
#SBATCH --job-name=l1000-slices
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/l1000-slices-%j.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
# Pre-registered L1000 unseen-drug slices P1-24, P1-6, P2-24 (report/10_09_record.md).
ENV_ROOT=$UNDETECT_DATA/envs/undetect-prep
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" $UNDETECT_ROOT/scripts/build_l1000_unseen_drug.py
date --iso-8601=seconds
