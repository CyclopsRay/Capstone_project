#!/usr/bin/env bash
#SBATCH --job-name=fetch-xpert
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=data/repro/dpb/logs/fetch-xpert-%j.out
# XPert L1000 SDST release used by Bai et al. 2026 (drug-perturbation-benchmark): figshare 28955141 file 58181296
# (= Zenodo 17182939 processed_data.zip, md5 4a58bc01b42c7909e7adc2ddd40b30ac). Raw, immutable.
ROOT="${UNDETECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root (README)
export UNDETECT_ROOT="$ROOT" UNDETECT_DATA="${UNDETECT_DATA:-$ROOT/data}"
set -euo pipefail
date --iso-8601=seconds; hostname; aria2c --version | head -1
D=$UNDETECT_DATA/raw/xpert_sdst
cd $D
aria2c -c -x 8 -s 8 -k 64M --file-allocation=none --max-tries=0 --retry-wait=20 --timeout=60 \
  --lowest-speed-limit=100K --summary-interval=300 --console-log-level=warn \
  -o processed_data.zip https://ndownloader.figshare.com/files/58181296
echo "4a58bc01b42c7909e7adc2ddd40b30ac  processed_data.zip" | md5sum -c
printf '%s\t%s\t%s\n' processed_data.zip https://ndownloader.figshare.com/files/58181296 "$(date -u --iso-8601=seconds)" >> SOURCE
sha256sum processed_data.zip >> SHA256SUMS
unzip -o -q processed_data.zip && find . -maxdepth 2 -type f -exec ls -la {} \;
date --iso-8601=seconds
