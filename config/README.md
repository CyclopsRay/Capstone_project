# config

`config.yaml` holds every setting of the demo run: the size and structure of the
simulated benchmark, the seeds, and UnDetect's training settings. Change a setting
here, not in the code, then run `pixi run reproduce`: Snakemake reruns the steps
whose settings changed, and everything after them.

The paper's runs take their settings from the command lines in `slurm/` and from
the defaults in each script (README.md, "The paper").
