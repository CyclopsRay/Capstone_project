# workflow

`Snakefile`: the pipeline, run by `pixi run reproduce` from the repository root.

    download -> simulate -> train (5 folds, in parallel) -> evaluate -> report

Each rule runs one script from `scripts/`; settings come from `config/config.yaml`.

    pixi run dry-run                            # what would run, and why
    pixi run snakemake --cores all --forceall   # run everything again
