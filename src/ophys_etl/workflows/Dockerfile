FROM apache/airflow:2.6.3

USER root
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  git \
  libgl1 libglib2.0-0  # needed for some dependencies to load properly
USER airflow

# TODO remove workflows_v2 tag when merged into main branch
RUN git clone -b workflow_v2 https://github.com/AllenInstitute/ophys_etl_pipelines

RUN pip install ./ophys_etl_pipelines[workflow,pytorch_deps,deepinterpolation]

# installing airflow again as best practice according to https://airflow.apache.org/docs/docker-stack/build.html#adding-packages-from-requirements-txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}"

# Copy dags into expected place
COPY --chown=airflow:root on_prem/dags/ /opt/airflow/dags

# copying the app config into the container. Expects this file to be
# downloaded from s3 into the cwd
COPY app_config.yml /
ENV OPHYS_WORKFLOW_APP_CONFIG_PATH=/app_config.yml