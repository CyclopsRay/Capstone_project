# Design decisions

This repository shows how to build a standard, reproducible pipeline around a
model; UnDetect is the example. It started from the code of CyclopsRay/undetect
(commit c7e8359, imported without its Git history) and keeps only what the demo
needs.

**The model, rewritten small.** `scripts/undetect/` holds the model in four short
modules (kernels, backbone, model, metrics) instead of the research scripts it
came from. Before the research scripts were removed, both versions ran on the same
simulated data: on all five folds they chose the same settings and gave the same
scores to six decimals.

**Simulated data with real compounds.** The real training data are gigabytes and
take hours on a cluster; the demo must run in minutes on any computer. The
compounds and their annotations are real (CLUE); their responses are simulated
from a model whose parts match UnDetect's, so the answer is known and the report
can show how close the model gets to the noise ceiling.

**Train, validation and test.** Each fold holds out whole compounds: a fifth for
testing and a fifth for choosing every setting. The report shows each stage of
the model on all three splits, so overfitting to the settings would be visible.
On training compounds the compound terms are leave-one-compound-out values.

**One environment.** `pixi.toml` and `pixi.lock` pin every package for Linux and
macOS; the versions of the core packages are those of the original project. PyTorch
is the CPU build without MKL and Matplotlib comes without Qt, to keep the download
small. The research project's GPU and data-preparation environments were dropped
with the research scripts.

**Snakemake.** One rule per step, each fold a separate job, the settings as rule
parameters: changing a setting reruns exactly what depends on it.

**Checks.** On every push, `.github/workflows/reproduce.yml` installs the locked
environment on a fresh Linux machine without any cache, runs the unit tests and
`pixi run reproduce`, and fails if a committed file changed.
`.github/workflows/container.yml` builds the container image from `pixi.lock` and
runs the pipeline inside it.

**Licenses.** Code: MIT, as in the original project. Text of the report and the
documentation: CC BY 4.0. The CLUE compound table keeps its own terms and is not
redistributed here.
