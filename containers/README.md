# containers

Recipes for a container image of this project's `default` environment:

- `Dockerfile`: installs the environment from `pixi.lock` with Pixi, then copies
  only the environment into a small Ubuntu image, so the container and your
  laptop run exactly the same packages. The GitHub Action `container.yml` builds
  it and reruns the demo inside it whenever the environment changes.
- `apptainer.def`: the same image for clusters that use Apptainer.

Build and run (Docker, from the repository root):

    docker build --platform linux/amd64 -f containers/Dockerfile --build-arg ENV_NAME=default -t undetect:0.1.0 .
    docker run --rm -v "$PWD":/work -w /work undetect:0.1.0 snakemake --cores all

`--build-arg ENV_NAME=prep` builds the data-preparation environment instead.
