# scripts

- The paper's code (`*.py` here): UnDetect, the benchmarks, analyses and data
  preparation. Each script's docstring names its inputs and outputs. They read
  and write under `$UNDETECT_DATA` (default `data/`) and run from the repository
  root, for example `pixi run python scripts/stage2_context.py --v2`.
- `figures/`: the paper's figures (`pixi run paper-figures`), with the shared style
  in `figures/style.py`.
- `demo/`: the demo run. `simulate_benchmark.py` writes a simulated benchmark in
  the formats of the real one; `score_demo.py` scores the outputs of the paper's
  scripts on it.
- `download_data.py`: downloads the demo's raw data and checks the checksums.
