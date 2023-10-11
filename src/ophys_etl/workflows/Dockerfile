FROM apache/airflow:2.6.3-python3.8

# dev or prod
ARG env

ARG host_username
# id -u
ARG host_userid

# Only set to true if the app database does not yet exist
ARG create_app_db=false

RUN if [ $env != 'dev' ] && [ $env != 'prod' ] && [ $env != 'staging' ]; then \
        echo "env must be set to 'dev', 'prod' or 'staging'"; \
        exit 1; \
    fi

USER root
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    git \
    libgl1 libglib2.0-0  # needed for some dependencies to load properly

# Adding the host user to this container
RUN groupadd -g 0 -o $host_username
RUN useradd -m -u $host_userid -g 0 -o -s /bin/bash $host_username

USER $host_username:root

ENV PATH "$PATH:/home/$host_username/.local/bin"

COPY --chown=$host_username:root . /opt/airflow/ophys_etl_pipelines

RUN if [ $env = "prod" ]; then \
        pip install /opt/airflow/ophys_etl_pipelines[workflow,pytorch_deps,deepinterpolation]; \
    else \
        pip install -e /opt/airflow/ophys_etl_pipelines[workflow,pytorch_deps,deepinterpolation]; \
    fi

# installing airflow again as best practice according to https://airflow.apache.org/docs/docker-stack/build.html#adding-packages-from-requirements-txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}"
RUN pip install "apache-airflow-providers-amazon"

RUN pip install psycopg2-binary

RUN if [ $create_app_db = "true" ]; then \
      python -m ophys_etl.workflows.db.initialize_db --add_roi_classifier_training_run True \
    fi

# Copy dags into expected place
RUN cp -r /opt/airflow/ophys_etl_pipelines/src/ophys_etl/workflows/on_prem/dags/* /opt/airflow/dags

RUN cp /opt/airflow/ophys_etl_pipelines/src/ophys_etl/workflows/app_config/$env.app_config.yml /opt/airflow/app_config.yml

ENV OPHYS_WORKFLOW_APP_CONFIG_PATH=/opt/airflow/app_config.yml