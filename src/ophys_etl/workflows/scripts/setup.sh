#!/bin/bash

# overriding default airflow configs
export AIRFLOW__CORE__LOAD_EXAMPLES=false
export AIRFLOW__CORE_ALLOWED_DESERIALIZATION_CLASSES="airflow\..*
ophys_etl.workflows.pipeline_module.OutputFile"

nohup airflow scheduler &
nohup airflow webserver --port 8080 &