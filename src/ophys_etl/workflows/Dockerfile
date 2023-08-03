FROM apache/airflow:2.6.3

ARG aws_access_key_id
ARG aws_secret_access_key

ARG ophys_etl_branch=main

ENV AWS_ACCESS_KEY_ID=$aws_access_key_id
ENV AWS_SECRET_ACCESS_KEY=$aws_secret_access_key
ENV AWS_DEFAULT_REGION=us-west-2

USER root
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    unzip \
    git \
    libgl1 libglib2.0-0  # needed for some dependencies to load properly

# install awscli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

# Adding the host user to this container
RUN groupadd -g 0 -o svc_ophysprd
RUN useradd -m -u 19220 -g 0 -o -s /bin/bash svc_ophysprd

USER svc_ophysprd:root

ENV PATH "$PATH:/home/svc_ophysprd/.local/bin"

RUN git clone -b $ophys_etl_branch https://github.com/AllenInstitute/ophys_etl_pipelines

RUN pip install ./ophys_etl_pipelines[workflow,pytorch_deps,deepinterpolation]

# installing airflow again as best practice according to https://airflow.apache.org/docs/docker-stack/build.html#adding-packages-from-requirements-txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}"

RUN pip install psycopg2-binary

# Copy dags into expected place
RUN cp -r ./ophys_etl_pipelines/src/ophys_etl/workflows/on_prem/dags/ /opt/airflow/dags

# copying the app config into the container
RUN aws s3 cp s3://ophys-processing-airflow.alleninstitute.org/prod/app_config.yml /opt/airflow/
ENV OPHYS_WORKFLOW_APP_CONFIG_PATH=/opt/airflow/app_config.yml