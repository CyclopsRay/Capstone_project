#!/usr/bin/env bash
#SBATCH --job-name=deepce-cpu
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=data/repro/deepce/logs/train-cpu-%j.out
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
# DeepCE reproduction + gene-feature ablation on CPU (GPU QOS full): the four gene variants of one
# seed run in parallel, 8 threads each. repro/deepce is gitignored.
ENV=$UNDETECT_ROOT/.pixi/envs/cluster
SEED=${SEED:-0}; EPOCHS=${EPOCHS:-100}
date --iso-8601=seconds; hostname; nproc
"$ENV/bin/python" -c "import torch, rdkit, numpy; print('torch', torch.__version__, 'rdkit', rdkit.__version__, 'numpy', numpy.__version__)"
cd $UNDETECT_ROOT/repro/deepce
for v in ${VARIANTS:-ppi shuffled random onehot}; do
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$ENV/bin/python" run_deepce.py --gene-variant "$v" --seed "$SEED" \
    --epochs "$EPOCHS" ${OUT:+--out "$OUT"} > "$UNDETECT_DATA/repro/deepce/logs/run-${SLURM_JOB_ID}-${v}-seed${SEED}.out" 2>&1 &
done
wait
grep -h "test_pearson_best_epoch_optimistic\|\"pearson\"" $UNDETECT_DATA/repro/deepce/logs/run-${SLURM_JOB_ID}-*.out || true
date --iso-8601=seconds
