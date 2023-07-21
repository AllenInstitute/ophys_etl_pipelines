FROM apache/airflow:2.6.3

ARG aws_access_key_id
ARG aws_secret_access_key

ENV AWS_ACCESS_KEY_ID=$aws_access_key_id
ENV AWS_SECRET_ACCESS_KEY=$aws_secret_access_key

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

RUN pip install awscli

# Copy dags into expected place
COPY --chown=airflow:root on_prem/dags/ /opt/airflow/dags

# copying the app config into the container
RUN aws s3 cp s3://ophys-processing-airflow.alleninstitute.org/prod/app_config.yml .
ENV OPHYS_WORKFLOW_APP_CONFIG_PATH=/app_config.yml

# copying deepinterpolation model into the container
RUN aws s3 cp s3://ophys-processing-airflow.alleninstitute.org/prod/ensemble_ssf_model_even_smaller_validation_mean_squared_error-0120-0.9622.h5 .
