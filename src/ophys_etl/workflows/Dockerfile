FROM apache/airflow:2.6.3

ARG aws_access_key_id
ARG aws_secret_access_key

# Set this to true to initialize app db. Only needed if it does not exist
ARG initialize_app_db=false

# Optional args only needed if initialize_app_db is true
ARG app_db_username=null
ARG app_db_password=null
ARG app_db_host=null
ARG app_db_port=null
ARG app_db_name=null
ARG cell_classifier_mlflow_parent_run_name=null

RUN if [ $initialize_app_db = true ] && [[ $app_db_username = null || $app_db_password = null || $app_db_host = null || $app_db_port = null || $app_db_name = null || $cell_classifier_mlflow_parent_run_name = null ]] ; then \
    echo "if initialize_app_db is true then pass required arguments"; \
    exit 1; \
    fi

ENV AWS_ACCESS_KEY_ID=$aws_access_key_id
ENV AWS_SECRET_ACCESS_KEY=$aws_secret_access_key
ENV AWS_DEFAULT_REGION=us-west-2

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
RUN aws s3 cp s3://ophys-processing-airflow.alleninstitute.org/prod/app_config.yml /opt/airflow/
ENV OPHYS_WORKFLOW_APP_CONFIG_PATH=/opt/airflow/app_config.yml

# copying deepinterpolation model into the container
RUN aws s3 cp s3://ophys-processing-airflow.alleninstitute.org/prod/ensemble_ssf_model_even_smaller_validation_mean_squared_error-0120-0.9622.h5 /opt/airflow/

# initialize app database
RUN if [ "$initialize_app_db" = true ] ; then \
    python -m ophys_etl.workflows.initialize_db \
        --db_url postgresql+psycopg2://$app_db_username:$app_db_password@$app_db_host:$app_db_port/$app_db_name \
        --add_roi_classifier_training_run True \
        --roi_classifier_args.trained_model_dest /opt/airflow/cell_classifier_model/ \
        --roi_classifier_args.mlflow_parent_run_name $cell_classifier_mlflow_parent_run_name; \
    fi