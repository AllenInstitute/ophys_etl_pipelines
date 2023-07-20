FROM apache/airflow:2.6.3

# TODO remove workflows_v2 tag when merged into main branch
RUN git clone -b workflows_v2 https://github.com/AllenInstitute/ophys_etl_pipelines

RUN pip install ./ophys_etl_pipelines[workflow]

# installing airflow again as best practice according to https://airflow.apache.org/docs/docker-stack/build.html#adding-packages-from-requirements-txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}"

# Copy dags into expected place
COPY --chown=airflow:root ophys_etl_pipelines/src/ophys_etl/workflows/on_prem/dags/ /opt/airflow/dags

# copying the app config into the container. Expects this file to be
# downloaded from s3 into the cwd
COPY app_config.yml /
ENV OPHYS_WORKFLOW_APP_CONFIG_PATH=/app_config.yml
