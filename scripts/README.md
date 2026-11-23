# scripts

- `undetect/`: the model, importable as a package
  - `kernels.py`: compound kernels (structure, targets, mechanism, Cell Painting), their weights and fusion
  - `backbone.py`: the drug-free network that predicts a response from the control profile
  - `model.py`: UnDetect on one fold: backbone, compound terms, context gate
  - `metrics.py`: PCC_DEG
- One script per pipeline step, run from the repository root by `workflow/Snakefile`:
  `download_data.py`, `simulate_data.py`, `train.py --fold F`, `evaluate.py`
