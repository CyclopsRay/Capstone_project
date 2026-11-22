#!/usr/bin/env bash
#SBATCH --job-name=undetect-cache
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=data/logs/undetect-cache-%j.out
# UnDetect part cache for second-stage experiments (scripts/cache_undetect.py); arguments via EXTRA.
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
ENV=$UNDETECT_DATA/envs/undetect
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__)"
cat $UNDETECT_DATA/derived/xpert_sdst/v1/manifest.json
sha256sum $UNDETECT_ROOT/scripts/run_sdst_hybrid.py $UNDETECT_ROOT/scripts/run_undetect_v2.py $UNDETECT_ROOT/scripts/cache_undetect.py
"$ENV/bin/python" -I $UNDETECT_ROOT/scripts/cache_undetect.py ${EXTRA:-}
date --iso-8601=seconds
