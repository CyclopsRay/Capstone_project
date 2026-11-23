# containers

`Dockerfile`: builds an image of the Pixi environment from `pixi.lock`, so the container and your laptop run the
same packages. The GitHub Action `container.yml` builds it and runs the pipeline inside it.

    docker build --platform linux/amd64 -f containers/Dockerfile -t undetect-demo:0.1.0 .
    docker run --rm -v "$PWD":/work -w /work undetect-demo:0.1.0 snakemake --cores all
