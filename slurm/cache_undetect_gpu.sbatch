#!/usr/bin/env bash
#SBATCH --job-name=undetect-cache-gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/undetect-cache-gpu-%j.out
# GPU copy of cache_undetect_cpu.sbatch (same backbone hardware as v3): one cache per seed in SEEDS, tag cache_gpu_s<seed>.
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
ENV=$UNDETECT_DATA/envs/undetect
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__)"
cat $UNDETECT_DATA/derived/xpert_sdst/v1/manifest.json
sha256sum $UNDETECT_ROOT/scripts/run_sdst_hybrid.py $UNDETECT_ROOT/scripts/run_undetect_v2.py $UNDETECT_ROOT/scripts/cache_undetect.py
for S in ${SEEDS:-1729}; do
  echo "== seed $S"; "$ENV/bin/python" -I $UNDETECT_ROOT/scripts/cache_undetect.py --seed "$S" --tag "cache_gpu_s$S" ${EXTRA:-}
done
date --iso-8601=seconds
