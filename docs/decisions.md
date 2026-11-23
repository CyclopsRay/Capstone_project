# Design decisions

Why this repository looks the way it does. The research code is the code of the
paper (CyclopsRay/undetect at commit c7e8359, imported without its Git history);
the decisions below are about how it is installed, run, checked and documented.

## Environments: Pixi instead of three pip virtual environments

The original repository installed three venvs from `envs/undetect.txt`,
`envs/undetect-prep.txt` and `envs/figures.txt` with `pip`. They pinned the
top-level packages only, needed a system Python 3.12, and could not be checked
for drift. `pixi.toml` now defines three environments and `pixi.lock` locks every
package, for every platform:

| Original | Pixi environment | Same versions | Differences |
|----------|------------------|---------------|-------------|
| `undetect.txt` (CUDA 11.8 build of PyTorch) | `cluster` | Python 3.12, torch 2.4.1, numpy 1.26.4, scipy 1.13.1, pandas 2.2.3, scikit-learn 1.5.2, rdkit 2024.3.6, gensim 4.4.0, numba 0.68 | packages from conda-forge instead of PyPI and the PyTorch wheel index; `pecanpy` still from PyPI (not on conda-forge) |
| `undetect.txt` with a CPU build of PyTorch (the README said it runs every script) | `default` | as above | PyTorch's `cpu_generic` build from conda-forge (no MKL, a smaller download); adds Matplotlib, Quarto, Jupyter kernel and Snakemake |
| `undetect-prep.txt` | `prep` | Python 3.12, torch 2.14.1 (CPU), numpy 2.2.6, scipy 1.16.3, pandas 2.3.3, pyarrow 21.0.0, h5py 3.14.0, scikit-learn 1.7.2, rdkit 2026.3.6 | conda-forge builds; Linux only, because it processes the full raw data, which lives on the cluster (its RDKit build would also need macOS 13.3) |
| `figures.txt` (numpy, pandas, matplotlib) | `default` | | the figure scripts also need RDKit (`fig1_cards_data.py`), which the default environment has |

The `cluster` environment declares CUDA 11.8 on a separate platform entry
(`linux-64-cuda`, Pixi 0.82 or newer), so the lock file records the CUDA build
for the GPU nodes and the CPU build for everything else. On a login node without
a GPU, install it with `CONDA_OVERRIDE_CUDA=11.8 pixi install -e cluster`.

The job scripts in `slurm/` and `repro/*/slurm/` changed in one line each: the
Python they run moved from `$UNDETECT_DATA/envs/<venv>` to the Pixi environment
in `$UNDETECT_ROOT/.pixi/envs/<environment>`. They were not rerun on the cluster
after this change. The benchmark's own environment (`dpb-benchmark`, built by
`slurm/dpb_setup.sbatch` for the published models) is unchanged.

## A demo run on simulated data of the same shape

`pixi run reproduce` must run on a fresh machine in minutes. The paper's data do
not allow that: the benchmark alone is a 1.6 GB download and the runs take hours
on a cluster. The course allows public or simulated data of the same shape for
the reproducible version, so:

- `scripts/demo/simulate_benchmark.py` writes a benchmark in exactly the files and
  formats of the real one (`sdst.npz`, `drugs.tsv`, `cell_painting_features.npz`),
  with 978 genes, cell lines, assay plates and the benchmark's compound split rule.
- The compounds are real (CLUE compound table: structures, targets, mechanisms),
  so the structure, target and mechanism kernels are those of real compounds.
  Cell Painting profiles and all responses are simulated, from a model whose parts
  match UnDetect's parts (context, plate, compound effect by view, line-specific
  effect, context gate, noise), so the answer is known.
- The paper's scripts run on it **unchanged** (`cache_undetect.py`,
  `stage2_context.py`), through `UNDETECT_DATA=data/demo`, the same mechanism the
  job scripts use for the real data. The demo therefore tests the paper's code
  path, not a copy of it.
- The backbone trains for at most 100 epochs (the paper: 500, patience 50) so the
  whole run takes about two minutes. Everything else uses the paper's settings.
- Only UnDetect is trained (Yuki's decision, 2026-11-23). Its scores sit next
  to two reference predictions that are not models (the mean training response
  of the sample's cell line and of its plate) and the noise ceiling.

Committing the paper's own result summaries, so that the demo redraws the paper's
figures, was considered and not done: the paper is under review, and redrawing
figures from stored numbers would not run the method.

## Workflow: Snakemake for the demo, job scripts for the paper

The demo is a Snakemake workflow (`workflow/Snakefile`): one rule per step, each
fold a separate job, settings from `config/config.yaml` as parameters, so changing
a setting reruns exactly what depends on it. The paper's runs stay as the job
scripts they were run with: rewriting thirty of them as an untested workflow would
make the record of what was run less certain, not more. The Slurm profile in
`workflow/profiles/slurm/` runs the demo's steps as Slurm jobs; it has not been
tested on a cluster.

## Report: computed when it renders

`reports/undetect-demo.qmd` reads the results and computes every number, table
and figure when it renders (inline `{python}` expressions); nothing is typed by
hand except facts about the paper's data. It uses the paper's figure style
(`scripts/figures/style.py`). Figures have captions and alt text; the software
versions and the hash of `pixi.lock` are read at render time.

## Checks

- `.github/workflows/reproduce.yml`: a fresh clone on a fresh Linux machine,
  `pixi install --locked`, the unit tests, `pixi run reproduce`, and a check that
  no committed file changed. The Pixi cache is off, so the run time is a new
  user's time.
- `.github/workflows/container.yml`: builds the container image from `pixi.lock`
  and checks that its tools run.

## Licenses

Code: MIT, as in the original repository. Text of the report and documentation:
CC BY 4.0. Downloaded data keep their own licenses (`metadata/sources.tsv`);
nothing downloaded is committed.
