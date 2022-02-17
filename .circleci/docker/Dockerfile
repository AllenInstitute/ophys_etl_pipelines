FROM continuumio/miniconda3:4.8.2

LABEL maintainer="scott.daniel@alleninstitute.org"
LABEL version=1.0
LABEL description="This dockerfile provides an environment in which to run \
                   the unit tests for the Allen Institue for Brain Science \
                   optical physiology data processing pipelines."

ARG OPHYS_ETL_TAG
ARG OPHYS_ETL_COMMIT_SHA="unknown build"
ARG TEST_PYTHON_VERSION

ENV OPHYS_ETL_COMMIT_SHA=${OPHYS_ETL_COMMIT_SHA}
ENV CONDA_ENVS=/envs
ENV OPHYS_ETL_ENV=${CONDA_ENVS}/ophys_etl
ENV NUMBA_CACHE_DIR=/tmp

RUN mkdir ${CONDA_ENVS}

# NOTE: To install into conda environments during docker build we need to
# use "conda run -n <my_env> subsequent commands". For details see:
# https://pythonspeed.com/articles/activate-conda-dockerfile/

# Install Suite2P (into conda environment named suite2p)
WORKDIR /repos
RUN apt-get -y update --allow-releaseinfo-change \
    && git clone -b ${OPHYS_ETL_TAG} https://github.com/AllenInstitute/ophys_etl_pipelines ./ophys_etl \
    && conda create --prefix ${OPHYS_ETL_ENV} python=${TEST_PYTHON_VERSION} \
    # the following installs scipy/numpy with MKL backend,
    # if requirements.txt specifies a different version, these will get overwritten
    # and some other BLAS backend will be used - speed will decrease
    && conda run --prefix ${OPHYS_ETL_ENV} conda install scipy \
    && conda run --prefix ${OPHYS_ETL_ENV} pip install --no-cache ./ophys_etl \
    && echo "use for ophys_etl "$(conda run --prefix ${OPHYS_ETL_ENV} which python) \
    && conda clean --all

# leave /repos/ophys_etl so we can run tests

ENTRYPOINT ["/bin/bash", "-c", "$@", "--"]
