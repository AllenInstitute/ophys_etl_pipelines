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
ENV CONDA_ENVS=/envs
ENV SUITE2P_ENV=${CONDA_ENVS}/suite2p
ENV OPHYS_ETL_ENV=${CONDA_ENVS}/ophys_etl
ENV NUMBA_CACHE_DIR=/tmp

RUN mkdir ${CONDA_ENVS}

# Get Suite2P general dependencies
RUN apt-get -y update \
    && apt-get -y install \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/*

# NOTE: To install into conda environments during docker build we need to
# use "conda run -n <my_env> subsequent commands". For details see:
# https://pythonspeed.com/articles/activate-conda-dockerfile/

# Install Suite2P (into conda environment named suite2p)
WORKDIR /suite2p
RUN git clone -b ${SUITE2P_TAG} https://github.com/MouseLand/suite2p . \
    && conda env update --prefix ${SUITE2P_ENV} --file environment.yml \
    && conda clean --all \
    && conda run --prefix ${SUITE2P_ENV} pip install --no-cache . \
    && rm -rf /suite2p

# Install ophys_etl_pipelines into suite2p environment
WORKDIR /ophys_etl_pipelines
RUN git clone -b ${OPHYS_ETL_TAG} https://github.com/AllenInstitute/ophys_etl_pipelines . \
    && conda run --prefix ${SUITE2P_ENV} pip install --no-cache . \
    && echo "use for suite2p "$(conda run --prefix ${SUITE2P_ENV} which python)

# create conda env named ophys_etl and install this repo into it
RUN conda create --prefix ${OPHYS_ETL_ENV} python=3.8 \
    && conda run --prefix ${OPHYS_ETL_ENV} pip install --no-cache . \
    && echo "use for ophys_etl "$(conda run --prefix ${OPHYS_ETL_ENV} which python)

# The base image has the default entrypoint activate the base conda env. We are
# creating additional environments that need to be specified at run time.
# docker run --rm --entrypoint /bin/bash alleninstitutepika/ophys_etl_pipelines -c "conda run --name suite2p python -m ophys_etl.transforms.suite2p_wrapper -h"

# If you need to troubleshoot by running interacticely inside the container:
# docker run --rm -it --entrypoint "/bin/bash" alleninstitutepika/ophys_etl_pipelines

# Uses the bash $@ special parameter to consume all docker args after
# image_name:tag as args to container bash shell
ENTRYPOINT ["/bin/bash", "-c", "$@", "--"]
