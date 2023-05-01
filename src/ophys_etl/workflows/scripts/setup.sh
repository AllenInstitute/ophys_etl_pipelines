#!/bin/bash

##############
# overriding default airflow configs
##############

# disable loading example DAGs
export AIRFLOW__CORE__LOAD_EXAMPLES=false

# Allow deserialization of OutputFile
export AIRFLOW__CORE__ALLOWED_DESERIALIZATION_CLASSES="airflow\..* ophys_etl.workflows.output_file.OutputFile"

# Set REST API auth to username/password auth
export AIRFLOW__API__AUTH_BACKENDS="airflow.api.auth.backend.basic_auth"

##############

nohup airflow scheduler &
nohup airflow webserver --port 8080 &