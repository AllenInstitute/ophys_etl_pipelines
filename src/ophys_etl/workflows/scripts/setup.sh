#!/bin/bash

# overriding default airflow configs
export AIRFLOW__CORE__LOAD_EXAMPLES=false
export AIRFLOW__CORE__ALLOWED_DESERIALIZATION_CLASSES="airflow\..* ophys_etl.workflows.output_file.OutputFile"

nohup airflow scheduler &
nohup airflow webserver --port 8080 &