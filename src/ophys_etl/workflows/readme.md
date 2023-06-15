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
1. Using sqlite db

    Initialize db using `airflow db init`. This populates the airflow database

    Create users using `airflow users create`

2. Using postgres

   1. If For development use the connection details in section 1.2.1 of [this guide](http://confluence.corp.alleninstitute.org/pages/viewpage.action?pageId=60855687) to setup a dev db
   > **_NOTE:_**  Use port 5412 for the dev postgres server, as the version at the default port 5432 does not work

   2. Create a user using the instructions [here](https://airflow.apache.org/docs/apache-airflow/stable/howto/set-up-database.html#setting-up-a-postgresql-database)
   3. Modify the variable `sql_alchemy_conn` in `airflow.cfg` found within `AIRFLOW_HOME`
   4. Run `airflow db init`
   5. Create users using `airflow users create`
   
Initialize ophys workflow db using `python -m ophys_etl.workflows.db.initialize_db`

### start airflow

`./scripts/setup.sh`