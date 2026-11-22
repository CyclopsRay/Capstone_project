# UnDetect

Code for **"Predicting Transcriptional Responses to Unseen Compounds through Context Modeling and Multi-View
Transfer"** (Yuqi Lei, Hanzhe Liang, Keqin Peng, Jun Wen; RECOMB 2027 submission).

UnDetect (UNseen Drug dEcomposition for Transcriptional rEsponse prediCTion) predicts the transcriptional response to
a compound that was never profiled in any cell line. It fits the response in two parts:

1. **Context.** A backbone network on the paired DMSO control profile predicts the response that any compound would
   show in that sample (cell line and assay plate). It sees no compound information.
2. **Compound effect.** A multi-view kernel over chemical structure (Morgan fingerprints), annotated protein targets,
   mechanism of action and Cell Painting morphology is weighted by its alignment with the training residuals and
   normalised so that compounds with missing views are not shrunk. Kernel ridge regression carries the residual
   responses of profiled compounds to unprofiled ones, shared across cell lines and per line. A gene-wise
   interaction with the principal components of the control rescales the compound effect.

Every step after the backbone has a closed form.

## Repository layout

```
scripts/            UnDetect, the benchmarks, analyses and data preparation (Python)
scripts/figures/    paper figures
slurm/              the Slurm job scripts used for the paper (one per experiment)
repro/deepce/       DeepCE benchmark: our DeepCE reproduction, baselines and UnDetect on DeepCE's split
repro/dpb/slurm/    download of the drug-blind benchmark release
envs/               environment specifications
tests/              unit tests (network null models)
```

## Setup

All scripts read and write under one data directory, `$UNDETECT_DATA` (default `data/` in the repository root), and
locate code through `$UNDETECT_ROOT` (default: the repository root). Run commands from the repository root:

```bash
export UNDETECT_ROOT=$PWD UNDETECT_DATA=$PWD/data      # or a large storage location
mkdir -p "$UNDETECT_DATA/logs"
python3.12 -m venv "$UNDETECT_DATA/envs/undetect"      && "$UNDETECT_DATA/envs/undetect/bin/pip" install -r envs/undetect.txt
python3.12 -m venv "$UNDETECT_DATA/envs/undetect-prep" && "$UNDETECT_DATA/envs/undetect-prep/bin/pip" install -r envs/undetect-prep.txt
```

The job scripts in `slurm/` use these environment paths. Their `--partition`/`--qos` lines name the cluster we used;
change them for yours, or run the Python command inside a job script directly. Log files go to
`$UNDETECT_DATA/logs`. The published models of the drug-blind benchmark have their own environment
(`dpb-benchmark`), built by `slurm/dpb_setup.sbatch`.

Hardware: the backbone trains on one GPU or on CPU (the paper's final numbers use the CPU run). On one A100, fitting
UnDetect on one fold of the drug-blind benchmark takes about 5 minutes and 5 GB of GPU memory.

## Data

| data | source | used by |
|---|---|---|
| drug-blind benchmark (Bai et al. 2026; XPert L1000 SDST release) | Zenodo 17182939 / figshare 28955141, `processed_data.zip` (MD5 `4a58bc01…`) | `repro/dpb/slurm/fetch_xpert.sbatch`, `scripts/prep_xpert_sdst.py` |
| released predictions of the seven published models | Zenodo 22095049 (v2.0.0) | `slurm/dpb_fetch_released.sbatch`, `scripts/prep_published_preds.py` |
| drug-perturbation-benchmark code | `github.com/baijinming97/drug-perturbation-benchmark`, commit `55edbfe` | clone to `repro/dpb/upstream` (prediction index) and to `$UNDETECT_DATA/repro/dpb_bench` (retraining) |
| LINCS L1000 Level 5 | GEO GSE92742 (Phase I), GSE70138 (Phase II) | `scripts/build_l1000_unseen_drug.py` |
| Tahoe-100M pseudobulk deltas | `tahoebio/tahoe-de-rhaister`, rev. `c7963cf` | `scripts/build_tahoe_unseen_drug.py` |
| DeepCE benchmark | `github.com/pth1993/DeepCE`, commit `9b0d04f` | clone to `repro/deepce/upstream` |
| targets, mechanism of action | CLUE LINCS2020 `compoundinfo_beta`; Tahoe drug table | `scripts/build_drug_features.py` |
| Cell Painting | `broadinstitute/lincs-cell-painting`, A549 48 h spherized consensus profiles | `scripts/build_cell_painting_features.py` |
| protein interactions | STRING v12.0, human physical links | `scripts/netnull.py`, `scripts/run_l1000_mkl.py` |

Raw downloads go to `$UNDETECT_DATA/raw`, derived arrays to `$UNDETECT_DATA/derived`, results to
`$UNDETECT_DATA/results`. Each script's docstring names its inputs and outputs.

## Reproducing the paper

Each step lists the job script; the Python script it runs is named inside it. Arguments that differ from the
defaults are given.

### 1. Drug-blind benchmark (Table 2, Table 4, Fig. 2a–b, Fig. 3)

```bash
sbatch repro/dpb/slurm/fetch_xpert.sbatch            # benchmark release
sbatch slurm/prep_xpert_sdst.sbatch                  # arrays, folds (5-fold, 3:1:1 by compound)
sbatch slurm/build_drug_features.sbatch              # Morgan, targets, mechanism of action
sbatch slurm/build_cell_painting_features.sbatch     # Cell Painting profiles
sbatch slurm/dpb_fetch_released.sbatch && python scripts/prep_published_preds.py   # published models' predictions

# UnDetect (final run, CPU backbone, seed 1729)
EXTRA="--tag cache_v1" sbatch slurm/cache_undetect_cpu.sbatch           # backbone, kernel terms, leave-one-compound-out values
python scripts/stage2_context.py --v2 --cache cache_v1 --tag stage2_v2  # compound–context interaction (UnDetect = ctx_pc)
EXTRA="--backbone-cache cache_v1 --viewsets chem multi_norm --plate-control --save-pred --input-mlp \
       --input-arms nnf_multi --tag v8_cpu_abl" sbatch slurm/run_sdst_hybrid_cpu.sbatch   # ablation arms (Table 4)
sbatch slurm/finalize_sdst.sbatch                    # all arms scored together: PCC_DEG, PCC_plate, bootstrap contrasts
```

| result | script |
|---|---|
| plate-response baseline, cell-line mean | `scripts/sdst_plate_baseline.py` |
| PCC_plate (plate-centred score) | `scripts/analyze_sdst_centred.py` |
| PCC_within (within-plate score) | `scripts/analyze_within_plate.py` |
| reference that uses the compound's own responses | `scripts/oracle_final.py` |
| UnDetect + XPert ensemble, weight chosen without the test fold | `scripts/blend_crossfit.py`, `scripts/blend_clean_weight.py` |
| view ablation, joint permutation (Fig. 3a) | `scripts/sdst_view_ablation.py`, `scripts/analyze_permjoint.py` |
| cross-fitted interaction | `scripts/sdst_crossfit_interaction.py` |
| gain by view coverage | `scripts/analyze_sdst_coverage.py` |
| PCC_plate by number of plate-mates | `scripts/analyze_plate_mate_count.py` |
| response similarity by view (Fig. 3b) | `scripts/analyze_sdst_compounds.py`, `scripts/view_bins_ci.py` |
| compound case study (Fig. 3c–d) | `scripts/case_study_compounds.py`, `scripts/case_study_scan.py` |
| retraining the published models (optional) | `slurm/dpb_setup.sbatch`, `slurm/dpb_prepare.sbatch`, `slurm/dpb_train.sbatch` |

### 2. DeepCE benchmark (Fig. 2c)

```bash
git clone https://github.com/pth1993/DeepCE repro/deepce/upstream && git -C repro/deepce/upstream checkout 9b0d04f
sbatch repro/deepce/slurm/xline_kernel.sbatch        # UnDetect with the dose-resolved coregionalised kernel
sbatch repro/deepce/slurm/train.sbatch               # DeepCE reproduction (VARIANT, SEED as environment variables)
python repro/deepce/baselines.py                     # mean, nearest-neighbour baselines
python repro/deepce/mlp_multimodal.py                # neural network with the same compound views
```

### 3. LINCS and Tahoe-100M (Table 3) and network nulls

```bash
sbatch slurm/build_l1000_unseen_drug.sbatch          # LINCS slices (P1-24, P2-24) and splits (ik14, plate map)
python scripts/build_tahoe_unseen_drug.py            # Tahoe-100M slices (0.05, 0.5, 5 uM)
SLICE=P2-24 SPLIT=ik14 sbatch slurm/run_l1000_xline.sbatch   # per-line and shared kernel models; repeat per slice/split
SLICE=P2-24 SPLIT=platemap sbatch slurm/run_l1000_mkl.sbatch # per-line kernels, permuted targets, rewired STRING
```

`run_l1000_xline.py --keep-lines k` reproduces the sparse-coverage experiment and `--gate` the cell-line interaction.
The network null models are in `scripts/netnull.py` (degree-preserving rewiring) with `analyze_ppi_information.py`
and `analyze_network_information.py`.

### 4. Figures

```bash
pip install -r envs/figures.txt
python scripts/figures/fig1_cards_data.py && python scripts/figures/fig1_cp_profiles.py && python scripts/figures/fig1_cards.py
python scripts/figures/fig2_benchmark.py figures
python scripts/figures/fig_views.py figures
python scripts/figures/fig_appendix.py figures
```

Set `FIG_FONT` to an Arial `.ttf` to reproduce the paper's typography; without it the figures are drawn in DejaVu
Sans and saved with a `_preview` suffix. Outputs go to `figures/`.

## Notes

- Evaluation holds out whole compounds; every cell line is seen in training. Hyperparameters are chosen on
  validation compounds and each test fold is scored once.
- No response of a test compound enters any feature or setting.
- Comments in the scripts refer to the project's internal experiment log (`report/...`); they are kept as provenance.

## Citation

```bibtex
@inproceedings{lei2027undetect,
  title     = {Predicting Transcriptional Responses to Unseen Compounds through Context Modeling and Multi-View Transfer},
  author    = {Lei, Yuqi and Liang, Hanzhe and Peng, Keqin and Wen, Jun},
  booktitle = {RECOMB 2027 (submitted)},
  year      = {2027}
}
```

## License

The code is released under the MIT License (`LICENSE`). The datasets keep their own licenses and terms of use; see
their sources in [Data](#data).
