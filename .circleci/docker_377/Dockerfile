FROM continuumio/miniconda3:4.8.2

LABEL maintainer="nicholas.mei@alleninstitute.org"
LABEL version=1.0
LABEL description="This dockerfile provides a working environment for \
                   Allen Institute for Brain Science optical physiology data \
                   processing pipelines."

ARG OPHYS_ETL_TAG=main
ARG OPHYS_ETL_COMMIT_SHA="unknown build"

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

# The base image has the default entrypoint activate the base conda env.
# We are creating 2 separate envs.
# The easiest way to run them is 
# docker run --read-only --tmpfs /tmp alleninstitutepika/ophys_etl_pipelines:<tag> /envs/suite2p/bin/python -m ophys_etl.transforms.suite2p_wrapper -h"
# or
# docker run --read-only --tmpfs /tmp alleninstitutepika/ophys_etl_pipelines:<tag> /envs/ophys_etl/bin/python -m ophys_etl.transforms.postprocess_rois -h"

# If you need to troubleshoot by running interacticely inside the container:
# docker run --rm -it --entrypoint "/bin/bash" alleninstitutepika/ophys_etl_pipelines

# If you need to pass args that contain quotes:
# docker run --entrypoint /bin/bash --read-only --tmpfs /tmp alleninstitutepika/ophys_etl_pipelines:<tag> -c "/envs/ophys_etl/bin/python -m pytest -m 'not suite2p_only'"

# Uses the bash $@ special parameter to consume all docker args after
# image_name:tag as args to container bash shell
ENTRYPOINT ["/bin/bash", "-c", "$@", "--"]
