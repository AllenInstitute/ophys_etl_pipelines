#!/bin/bash

##############
# overriding default airflow configs
##############

##############
# disable loading example DAGs
export AIRFLOW__CORE__LOAD_EXAMPLES=false

# Allow deserialization of OutputFile
export AIRFLOW__CORE__ALLOWED_DESERIALIZATION_CLASSES="airflow\..* ophys_etl.workflows.output_file.OutputFile"

export AIRFLOW__CORE__EXECUTOR="LocalExecutor"

# Increasing these since had issues with DAG import taking longer than default
export AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=120.0
export AIRFLOW__CORE__DAG_FILE_PROCESSOR_TIMEOUT=180

export AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG=1000
export AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG=1000

# allow for retrying a failed task up to 5 times
export AIRFLOW__CORE__DEFAULT_TASK_RETRIES=5
##############


##############
# parse DAGs every 3 hours instead of 30 seconds
export AIRFLOW__SCHEDULER__MIN_FILE_PROCESS_INTERVAL=10800
export AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL=10800

# increase number of processing for parsing dags
export AIRFLOW__SCHEDULER__PARSING_PROCESSES=16

##############

# Set REST API auth to username/password auth
export AIRFLOW__API__AUTH_BACKENDS="airflow.api.auth.backend.basic_auth"

##############

nohup airflow scheduler &
nohup airflow webserver --port 8080 &
