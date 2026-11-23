# data

Nothing in this folder is committed.

- `raw/`: downloaded data, exactly as received, read-only. For the demo:
  `pixi run download` (`scripts/download_data.py`) fetches the files listed in
  `metadata/sources.tsv` and checks them against `metadata/checksums.sha256`.
- `demo/`: the data root (`UNDETECT_DATA`) of the demo run: the simulated
  benchmark in `derived/` and the outputs of the paper's scripts in `results/`,
  in the same layout as the full run.

For the full paper run `UNDETECT_DATA` defaults to this folder, so it also holds
`raw/` (the paper's downloads), `derived/`, `results/` and `logs/`; on a cluster,
point `UNDETECT_DATA` at a large storage location instead.
