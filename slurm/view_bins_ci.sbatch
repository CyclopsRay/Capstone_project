#!/usr/bin/env bash
#SBATCH --job-name=view-ci
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/view-ci-%j.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
# Fig. 3b intervals: compound-bootstrap 95% CIs of response similarity per view bin (scripts/view_bins_ci.py;
# no training; record 2026-11-04). Deterministic (seed 0 in the script).
ENV=$UNDETECT_ROOT/.pixi/envs/cluster
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date --iso-8601=seconds; hostname
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__, 'torch', torch.__version__)"
cat $UNDETECT_DATA/derived/xpert_sdst/v1/manifest.json
sha256sum $UNDETECT_ROOT/scripts/view_bins_ci.py $UNDETECT_ROOT/scripts/analyze_sdst_compounds.py $UNDETECT_ROOT/scripts/run_sdst_hybrid.py
"$ENV/bin/python" -I $UNDETECT_ROOT/scripts/view_bins_ci.py
date --iso-8601=seconds
