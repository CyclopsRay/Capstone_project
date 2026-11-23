# envs

The software environments of this project are defined in `pixi.toml` and
locked, package by package, in `pixi.lock`. Those two files are the source of
truth; use them with [Pixi](https://pixi.sh):

    pixi install                 # the default environment (CPU, laptop and CI)
    pixi install -e prep         # data preparation (Linux)
    CONDA_OVERRIDE_CUDA=11.8 pixi install -e cluster   # GPU jobs (Linux + NVIDIA driver)

The files here are exports for people who use conda or mamba instead of Pixi.
They list the same package specifications, not the locked builds, so a conda
solve can pick newer builds. Regenerate them after changing `pixi.toml`:

    pixi run export-envs

| File | Pixi environment | Replaces (original repository) |
|------|------------------|--------------------------------|
| `default.yml` | `default` | `envs/undetect.txt` (CPU build of PyTorch) and `envs/figures.txt` |
| `prep.yml` | `prep` | `envs/undetect-prep.txt` |

The `cluster` environment (`envs/undetect.txt` with the CUDA 11.8 build of
PyTorch) is not exported: it is tied to the CUDA platform entry in `pixi.toml`,
which conda environment files cannot express. It is the default environment
plus `pytorch 2.4.1 cuda118*`, `gensim 4.4.0`, `numba 0.68.0` and, from PyPI,
`pecanpy==2.0.9`.
