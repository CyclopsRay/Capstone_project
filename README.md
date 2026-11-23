# UnDetect, made reproducible

[![Reproduce](https://github.com/CyclopsRay/Capstone_project/actions/workflows/reproduce.yml/badge.svg)](https://github.com/CyclopsRay/Capstone_project/actions/workflows/reproduce.yml)

Code for **"Predicting Transcriptional Responses to Unseen Compounds through Context Modeling and Multi-View
Transfer"** (Yuqi Lei, Hanzhe Liang, Keqin Peng, Jun Wen; RECOMB 2027 submission), set up so that anyone can install
the software and run the method in about ten minutes. This is the capstone project of the course Reproducible and
Open Science (CBIO7501): the research code, with Pixi environments, a Snakemake workflow, a Quarto report and a
GitHub Action that reruns everything from a clean clone.

## Quick start

You need [Pixi](https://pixi.sh) (one command to install, no administrator rights) and an internet connection.
Linux (x86-64) and macOS (Apple silicon) are supported; on Windows, use WSL2.

```bash
git clone https://github.com/CyclopsRay/Capstone_project.git
cd Capstone_project
pixi run reproduce
```

`pixi run reproduce` installs the locked environment, downloads 4.6 MB of data, runs the pipeline and writes the
report `reports/undetect-demo.html`. The [Reproduce](.github/workflows/reproduce.yml) GitHub Action does the same on
a fresh machine after every push, without any cache, so its run time is the time a new user needs.

## 1. What is this project about?

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

## 2. Is there a publication?

The paper is submitted to RECOMB 2027; see [How to cite](#how-to-cite).

## 3. What is in this repository?

| Folder | What goes in it | Rule |
|--------|-----------------|------|
| `scripts/` | UnDetect, the benchmarks, analyses and data preparation (the paper's code) | Committed. |
| `scripts/demo/` | Simulation and scoring for the demo run | Committed. |
| `scripts/figures/` | The paper's figures | Committed. |
| `workflow/` | `Snakefile`: the demo pipeline; `profiles/slurm/`: run its steps as Slurm jobs | Committed. |
| `config/` | `config.yaml`: every setting of the demo run | Committed. |
| `slurm/`, `repro/` | The job scripts of the paper's runs; the DeepCE and drug-blind benchmarks | Committed. |
| `reports/` | Quarto report of the demo run, bibliography | Source committed, HTML not. |
| `metadata/` | Data sources with licenses and checksums | Small text files, committed. |
| `data/raw/` | Downloaded data, exactly as received | Read-only, not committed. |
| `data/demo/` | Simulated data and model outputs of the demo run | Regenerated, not committed. |
| `results/` | Tables and logs of the demo run | Regenerated, not committed. |
| `tests/` | Unit tests (network null models) | Committed. |
| `docs/` | Design decisions | Committed. |
| `envs/` | Conda environment files exported from `pixi.toml`, for people without Pixi | Generated. |
| `containers/` | Dockerfile and Apptainer recipe built from `pixi.lock` | Committed. |

The software environments are defined in `pixi.toml` and locked in `pixi.lock` (both committed):

| Environment | Platforms | Used for |
|-------------|-----------|----------|
| `default` | linux-64, osx-arm64 | the demo, the report, figures, CPU runs (PyTorch 2.4.1, CPU) |
| `cluster` | linux-64 with an NVIDIA driver (CUDA 11.8) | the GPU jobs of the paper (PyTorch 2.4.1, CUDA 11.8) |
| `prep` | linux-64 | data preparation, LINCS and Tahoe-100M kernels, network nulls |

They replace the three pip requirement files of the original repository with the same versions;
[docs/decisions.md](docs/decisions.md) lists every difference. `pixi task list` shows the tasks.

## 4. What hardware does it need?

**Demo:** any recent laptop. Measured on a MacBook (Apple silicon): the pipeline takes about 2 minutes with the
environment installed, at most 1 GB of memory per fold (up to about 3 GB with folds in parallel), and the
environment needs about 1.6 GB of disk (a 350 MB download on macOS, 450 MB on Linux). On the GitHub Action
(4 CPU cores, nothing cached), installing and running everything takes about 6 minutes.

**Paper:** a Slurm cluster. The job scripts ask for 8 to 32 CPU cores and 8 to 96 GB of memory, and one GPU for the
GPU runs. On one A100, fitting UnDetect on one fold of the drug-blind benchmark takes about 5 minutes and 5 GB of
GPU memory; the paper's final numbers use the CPU backbone run. The raw data take tens of GB; the two largest
downloads are 9.3 GB and 1.6 GB.

## 5. How do I reproduce the analyses?

### The demo (any computer, minutes)

```bash
pixi run reproduce     # download -> simulate -> UnDetect (5 folds) -> scores -> report
pixi run dry-run       # what would run now, and why (nothing, after a complete run)
pixi run test          # unit tests
```

The paper's data are a 1.6 GB download and its runs take hours on a cluster, so the demo runs the paper's scripts,
unchanged, on a **simulated benchmark of the same shape**: 978 genes, cell lines, assay plates and five folds that
hold out whole compounds, in the files and formats that `scripts/prep_xpert_sdst.py` writes for the real data. The
compounds are real (240 compounds from 24 mechanism-of-action classes of the CLUE compound table, with their
structures, targets and mechanisms); their Cell Painting profiles and all responses are simulated, with a known
answer. Snakemake runs `scripts/cache_undetect.py` and `scripts/stage2_context.py` on each fold, scores them against
baselines and the noise ceiling, and renders the report. Only UnDetect is trained; the baselines are means of the
training responses, computed while scoring. The numbers of the demo describe the simulation, not the paper.

Every setting is in `config/config.yaml`. Change one and run `pixi run reproduce` again: Snakemake reruns only the
steps that depend on it.

### The paper (Slurm cluster, hours)

All scripts read and write under one data directory, `$UNDETECT_DATA` (default `data/` in the repository root), and
locate code through `$UNDETECT_ROOT` (default: the repository root). On the cluster's login node, from the repository
root:

```bash
export UNDETECT_ROOT=$PWD UNDETECT_DATA=$PWD/data      # or a large storage location
mkdir -p "$UNDETECT_DATA/logs"
pixi install                                          # default environment
pixi install -e prep
CONDA_OVERRIDE_CUDA=11.8 pixi install -e cluster      # GPU environment; the login node may have no GPU
```

The job scripts in `slurm/` run Python from these environments (`$UNDETECT_ROOT/.pixi/envs/...`). Their
`--partition`/`--qos` lines name the cluster we used; change them for yours, or run the Python command inside a job
script directly (`pixi run -e cluster python ...`). Log files go to `$UNDETECT_DATA/logs`. The published models of the
drug-blind benchmark have their own environment (`dpb-benchmark`), built by `slurm/dpb_setup.sbatch`.

Each step lists the job script; the Python script it runs is named inside it. Arguments that differ from the
defaults are given.

#### 1. Drug-blind benchmark (Table 2, Table 4, Fig. 2a–b, Fig. 3)

```bash
sbatch repro/dpb/slurm/fetch_xpert.sbatch            # benchmark release
sbatch slurm/prep_xpert_sdst.sbatch                  # arrays, folds (5-fold, 3:1:1 by compound)
sbatch slurm/build_drug_features.sbatch              # Morgan, targets, mechanism of action
sbatch slurm/build_cell_painting_features.sbatch     # Cell Painting profiles
sbatch slurm/dpb_fetch_released.sbatch && pixi run python scripts/prep_published_preds.py   # published models' predictions

# UnDetect (final run, CPU backbone, seed 1729)
EXTRA="--tag cache_v1" sbatch slurm/cache_undetect_cpu.sbatch           # backbone, kernel terms, leave-one-compound-out values
pixi run python scripts/stage2_context.py --v2 --cache cache_v1 --tag stage2_v2  # compound–context interaction (UnDetect = ctx_pc)
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

#### 2. DeepCE benchmark (Fig. 2c)

```bash
git clone https://github.com/pth1993/DeepCE repro/deepce/upstream && git -C repro/deepce/upstream checkout 9b0d04f
sbatch repro/deepce/slurm/xline_kernel.sbatch        # UnDetect with the dose-resolved coregionalised kernel
sbatch repro/deepce/slurm/train.sbatch               # DeepCE reproduction (VARIANT, SEED as environment variables)
pixi run -e cluster python repro/deepce/baselines.py         # mean, nearest-neighbour baselines
pixi run -e cluster python repro/deepce/mlp_multimodal.py    # neural network with the same compound views
```

#### 3. LINCS and Tahoe-100M (Table 3) and network nulls

```bash
sbatch slurm/build_l1000_unseen_drug.sbatch          # LINCS slices (P1-24, P2-24) and splits (ik14, plate map)
pixi run -e prep python scripts/build_tahoe_unseen_drug.py   # Tahoe-100M slices (0.05, 0.5, 5 uM)
SLICE=P2-24 SPLIT=ik14 sbatch slurm/run_l1000_xline.sbatch   # per-line and shared kernel models; repeat per slice/split
SLICE=P2-24 SPLIT=platemap sbatch slurm/run_l1000_mkl.sbatch # per-line kernels, permuted targets, rewired STRING
```

`run_l1000_xline.py --keep-lines k` reproduces the sparse-coverage experiment and `--gate` the cell-line interaction.
The network null models are in `scripts/netnull.py` (degree-preserving rewiring) with `analyze_ppi_information.py`
and `analyze_network_information.py`.

#### 4. Figures

```bash
pixi run paper-figures       # all paper figures, from $UNDETECT_DATA/results (copy it from the cluster)
```

Set `FIG_FONT` to an Arial `.ttf` to reproduce the paper's typography; without it the figures are drawn in DejaVu
Sans and saved with a `_preview` suffix. Outputs go to `figures/`.

#### Notes

- Evaluation holds out whole compounds; every cell line is seen in training. Hyperparameters are chosen on
  validation compounds and each test fold is scored once.
- No response of a test compound enters any feature or setting.
- Comments in the scripts refer to the project's internal experiment log (`report/...`); they are kept as provenance.

## 6. What outputs should I expect?

From `pixi run reproduce`:

- `reports/undetect-demo.html`: the report. Every number, table and figure in it is computed from the results when
  it renders; it also lists the software versions it ran with.
- `results/demo/scores.csv`: PCC_DEG and PCC_plate of each method on each test fold; `weights.csv`: the view weights;
  `summary.json`: the five-fold means; `logs/`: one log per step.
- `data/demo/`: the simulated benchmark and the outputs of the paper's scripts, in the same layout as the full run.

On the simulated benchmark UnDetect scores well above its drug-free backbone and below the noise ceiling; the exact
values differ slightly between machines (CPU threads and math libraries), so the report reads them from the results
rather than quoting fixed numbers.

The full run writes its results under `$UNDETECT_DATA/results` and the figures to `figures/`.

## 7. How can I contact the authors?

Open an issue on [GitHub](https://github.com/CyclopsRay/Capstone_project/issues).

## Data

| Data | Source | Used by |
|---|---|---|
| compound annotations (structure, targets, MOA) | CLUE LINCS2020 `compoundinfo_beta.txt` | demo; `scripts/build_drug_features.py` |
| drug-blind benchmark (XPert L1000 SDST release) | Zenodo [10.5281/zenodo.17182939](https://doi.org/10.5281/zenodo.17182939) / figshare 28955141, `processed_data.zip` (MD5 `4a58bc01…`) | `repro/dpb/slurm/fetch_xpert.sbatch`, `scripts/prep_xpert_sdst.py` |
| released predictions and metrics of the seven published models | Zenodo [10.5281/zenodo.22095049](https://doi.org/10.5281/zenodo.22095049) (v2.0.0) | `slurm/dpb_fetch_released.sbatch`, `scripts/prep_published_preds.py` |
| drug-perturbation-benchmark code | `github.com/baijinming97/drug-perturbation-benchmark`, commit `55edbfe` | clone to `repro/dpb/upstream` (prediction index) and to `$UNDETECT_DATA/repro/dpb_bench` (retraining) |
| LINCS L1000 Level 5 | GEO GSE92742 (Phase I), GSE70138 (Phase II) | `scripts/build_l1000_unseen_drug.py` |
| Tahoe-100M pseudobulk deltas | `tahoebio/tahoe-de-rhaister`, rev. `c7963cf` | `scripts/build_tahoe_unseen_drug.py` |
| DeepCE benchmark | `github.com/pth1993/DeepCE`, commit `9b0d04f` | clone to `repro/deepce/upstream` |
| Cell Painting | `broadinstitute/lincs-cell-painting`, A549 48 h spherized consensus profiles | `scripts/build_cell_painting_features.py` |
| protein interactions | STRING v12.0, human physical links | `scripts/netnull.py`, `scripts/run_l1000_mkl.py` |

`metadata/sources.tsv` lists every source with its URL, license and the provider's checksum;
`metadata/checksums.sha256` has the SHA-256 of the demo's downloads (`shasum -a 256 -c metadata/checksums.sha256`).
Raw downloads go to `$UNDETECT_DATA/raw` and are never edited, derived arrays to `$UNDETECT_DATA/derived`, results
to `$UNDETECT_DATA/results`. Each script's docstring names its inputs and outputs.

## Data availability

All data used here are public; none are redistributed in this repository. The demo downloads the CLUE compound
table from the source above and simulates everything else with `scripts/demo/simulate_benchmark.py`. The
paper's data are available from the sources in the table above, under their own licenses and terms of use
(`metadata/sources.tsv`).

## How to cite

Cite the paper:

```bibtex
@inproceedings{lei2027undetect,
  title     = {Predicting Transcriptional Responses to Unseen Compounds through Context Modeling and Multi-View Transfer},
  author    = {Lei, Yuqi and Liang, Hanzhe and Peng, Keqin and Wen, Jun},
  booktitle = {RECOMB 2027 (submitted)},
  year      = {2027}
}
```

`CITATION.cff` describes this software; GitHub shows a "Cite this repository" button for it.

## License

The code is released under the MIT License (`LICENSE`). The text of the report and the documentation is released under
CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`). The datasets keep their own licenses and terms of use; see `metadata/sources.tsv`.
The simulated data of the demo are generated by the code and carry no further restrictions.

---
Project layout follows the CBIO7501 project template
(<https://github.com/khan-lab/cbio7501-f26/tree/main/templates/cookiecutter-repro-project>).
