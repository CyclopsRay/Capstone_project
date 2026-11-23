# workflow

`Snakefile`: the demo pipeline, run by `pixi run reproduce` from the repository
root:

    download -> simulate -> undetect_backbone (x5 folds) -> undetect_context (x5)
             -> score -> report

Each rule runs one script; the two UnDetect rules run the paper's scripts
unchanged, with `UNDETECT_DATA` pointing at the demo's data root. Settings come
from `config/config.yaml`.

    pixi run snakemake --cores all --dry-run   # what would run, and why
    pixi run snakemake --cores all --forceall  # rerun everything

`profiles/slurm/config.yaml` runs the same workflow with each step as a Slurm job
(`pixi run snakemake --profile workflow/profiles/slurm`); set the partition and
QOS of your cluster first. Not tested on a cluster yet.
