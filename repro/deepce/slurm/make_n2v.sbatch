#!/usr/bin/env bash
#SBATCH --job-name=deepce-n2v
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=data/repro/deepce/logs/n2v-%j.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
ENV=$UNDETECT_ROOT/.pixi/envs/cluster
date --iso-8601=seconds; hostname; "$ENV/bin/python" -c "import pecanpy, gensim, numba; print('gensim', gensim.__version__, 'numba', numba.__version__)"
cd $UNDETECT_ROOT/repro/deepce && "$ENV/bin/python" make_n2v_genes.py
date --iso-8601=seconds
