#!/usr/bin/env bash
#SBATCH --job-name=sdst-hybrid-cpu
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=data/logs/sdst-hybrid-cpu-%j.out
# CPU copy of run_sdst_hybrid.sbatch (e.g. the ablation arms on the CPU backbone cache: --backbone-cache cache_v1); arguments via EXTRA.
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
ENV=$UNDETECT_ROOT/.pixi/envs/cluster
export OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 MKL_NUM_THREADS=48
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__)"
cat $UNDETECT_DATA/derived/xpert_sdst/v1/manifest.json
sha256sum $UNDETECT_ROOT/scripts/run_sdst_hybrid.py $UNDETECT_DATA/results/sdst_hybrid/cache_v1/fold_*/cache.npz
"$ENV/bin/python" -I $UNDETECT_ROOT/scripts/run_sdst_hybrid.py ${EXTRA:-}
date --iso-8601=seconds
