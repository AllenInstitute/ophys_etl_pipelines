FROM continuumio/miniconda3:4.8.2

LABEL maintainer="nicholas.mei@alleninstitute.org"
LABEL version=1.0
LABEL description="This dockerfile provides a working environment for \
                   Allen Institute for Brain Science optical physiology data \
                   processing pipelines."

ARG SUITE2P_TAG=0.7.5
ARG OPHYS_ETL_TAG=main
ARG OPHYS_ETL_COMMIT_SHA="unknown build"

ENV OPHYS_ETL_COMMIT_SHA=${OPHYS_ETL_COMMIT_SHA}

# Get Suite2P general dependencies
RUN apt-get -y update \
    && apt-get -y install \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/*

# NOTE: To install into conda environments during docker build we need to
# use "conda run -n <my_env> subsequent commands". For details see:
# https://pythonspeed.com/articles/activate-conda-dockerfile/

# Install Suite2P (into conda base environment)
WORKDIR /suite2p
RUN git clone -b ${SUITE2P_TAG} https://github.com/MouseLand/suite2p . \
    && conda env update --name base --file environment.yml \
    && conda clean --all \
    && conda run --name base pip install --no-cache . \
    && rm -rf /suite2p

# Install ophys_etl_pipelines
WORKDIR /ophys_etl_pipelines
RUN git clone -b ${OPHYS_ETL_TAG} https://github.com/AllenInstitute/ophys_etl_pipelines . \
    && conda run --name base pip install --no-cache .

# Running with default entrypoint example:
# docker run --rm alleninstitutepika/ophys_etl_pipelines python -m ophys_etl.transforms.suite2p_wrapper -h

# If you need to troubleshoot by running interacticely inside the container:
# docker run --rm -it --entrypoint "/bin/bash" alleninstitutepika/ophys_etl_pipelines

# Uses the bash $@ special parameter to consume all docker args after
# image_name:tag as args to container bash shell
ENTRYPOINT ["/bin/bash", "-c", "$@", "--"]
