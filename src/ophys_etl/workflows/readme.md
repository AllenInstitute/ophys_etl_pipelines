# Ophys processing workflows

## Installation

If airflow not already installed:

`./scripts/install_airflow.sh`

### Setup environment variables
    
Required environment variables:

| Env var                           | Description                                     |
|-----------------------------------|-------------------------------------------------|
| AIRFLOW_HOME                      | Should point to root directory containing dags. |
| OPHYS_WORKFLOW_APP_CONFIG_PATH    | Path to app config                              |

### If from scratch:
Initialize db using `airflow db init`. This populates the airflow database

Create users for GUI using `airflow users create`

Initialize ophys workflow db using `python -m ophys_etl.workflows.db.initialize_db`

### start airflow

`./scripts/setup.sh`