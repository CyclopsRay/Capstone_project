#!/usr/bin/env bash
#SBATCH --job-name=stage2
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=data/logs/stage2-%j.out
# Second stage (scripts/stage2_context.py --v2) on cache_gpu_s<SEED>, tag s2gpu_s<SEED>; then PCC_plate contrasts vs pool_kta.
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
ENV=$UNDETECT_DATA/envs/undetect
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
date --iso-8601=seconds; hostname; nproc; echo "seed ${SEED:?}"
"$ENV/bin/python" -c "import numpy, pandas, torch; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'torch', torch.__version__)"
cat $UNDETECT_DATA/results/sdst_hybrid/cache_gpu_s$SEED/fold_*/summary.json | grep -h '"seconds"\|"mlp_epoch"'
sha256sum $UNDETECT_ROOT/scripts/stage2_context.py $UNDETECT_ROOT/scripts/analyze_sdst_centred.py
"$ENV/bin/python" -I $UNDETECT_ROOT/scripts/stage2_context.py --v2 --cache "cache_gpu_s$SEED" --tag "s2gpu_s$SEED"
"$ENV/bin/python" -I $UNDETECT_ROOT/scripts/analyze_sdst_centred.py --tag "s2gpu_s$SEED" --ref pool_kta
date --iso-8601=seconds
