#!/usr/bin/env bash
#SBATCH --job-name=undetect-v2c
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=data/logs/undetect-v2c-%j.out
# CPU copy of run_undetect_v2.sbatch (GPU queue ~1.5 days on 2026-10-31); pass a separate --tag via EXTRA so GPU and CPU
# outputs never share a directory. The backbone MLP is not bit-identical to the GPU runs: compare arms within a run.
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
ENV=$UNDETECT_ROOT/.pixi/envs/cluster
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import numpy, pandas, rdkit, torch; print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'rdkit', rdkit.__version__)"
cat $UNDETECT_DATA/derived/xpert_sdst/v1/manifest.json
sha256sum $UNDETECT_ROOT/scripts/run_sdst_hybrid.py $UNDETECT_ROOT/scripts/run_undetect_v2.py
"$ENV/bin/python" -I $UNDETECT_ROOT/scripts/run_undetect_v2.py ${EXTRA:-}
date --iso-8601=seconds
