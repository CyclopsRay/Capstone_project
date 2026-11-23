#!/usr/bin/env bash
#SBATCH --job-name=deepce-n2v-cpu
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=data/repro/deepce/logs/n2v-cpu-%j.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
# RECOMB study E3 on CPU (GPU node gpu-05 broken, GPU QOS full): DeepCE with node2vec gene vectors from
# STRING functional >= 400 and its nulls; the four variants of one seed in parallel, 8 threads each.
ENV=$UNDETECT_ROOT/.pixi/envs/cluster
GV=$UNDETECT_DATA/repro/deepce/gene_variants
SEED=${SEED:-0}
date --iso-8601=seconds; hostname; nproc
cd $UNDETECT_ROOT/repro/deepce
for k in real rewired labels er; do
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$ENV/bin/python" run_deepce.py --gene-variant file --gene-file "$GV/n2v_$k.csv" \
    --label "n2v_$k" --seed "$SEED" --epochs 60 > "$UNDETECT_DATA/repro/deepce/logs/run-${SLURM_JOB_ID}-n2v_$k-seed$SEED.out" 2>&1 &
done
wait
date --iso-8601=seconds
