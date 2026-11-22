#!/usr/bin/env bash
#SBATCH --job-name=sdst-xfit
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=data/logs/sdst-xfit-%j.out
# Interaction on fully cross-fitted training inputs (scripts/sdst_crossfit_interaction.py; record 2026-11-02 E4);
# one fold per job: FOLD=<f>. Backbone cache_v1 (seed 1729); cross-fit seed 0, Q = 5.
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
ENV=$UNDETECT_DATA/envs/undetect
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
date --iso-8601=seconds; hostname; nproc; echo "fold ${FOLD:?}"
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__)"
cat $UNDETECT_DATA/derived/xpert_sdst/v1/manifest.json
sha256sum $UNDETECT_ROOT/scripts/{sdst_crossfit_interaction,sdst_view_ablation,cache_undetect,run_sdst_hybrid,run_undetect_v2,stage2_context}.py
"$ENV/bin/python" -I -u $UNDETECT_ROOT/scripts/sdst_crossfit_interaction.py --folds "$FOLD" ${EXTRA:-}
date --iso-8601=seconds
