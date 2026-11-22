#!/usr/bin/env bash
#SBATCH --job-name=drug-features
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/drug-features-%j.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
# RDKit standardization + Morgan/MACCS/RDKit-2D/scaffolds + target/MOA sets (unseen-drug branch).
ENV_ROOT=$UNDETECT_DATA/envs/undetect-prep
date --iso-8601=seconds; hostname; "${ENV_ROOT}/bin/python" --version
"${ENV_ROOT}/bin/python" $UNDETECT_ROOT/scripts/build_drug_features.py
date --iso-8601=seconds
