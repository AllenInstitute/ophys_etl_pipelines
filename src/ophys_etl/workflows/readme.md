# Ophys processing workflows

## Installation

### Set up app DB

Use the connection details in section 1.2.1 of [this guide](http://confluence.corp.alleninstitute.org/pages/viewpage.action?pageId=60855687) to setup a db

Initialize ophys workflow db using `python -m ophys_etl.workflows.db.initialize_db`

### Launch app using docker compose

In same dir as `workflows/docker-compose-yaml`, run
```bash
BASE_DIR=<base directory to store logs in>
AIRFLOW_WEBSERVER_HOST_PORT=<port to map webserver to on the host>

echo -e "AIRFLOW_UID=$(id -u)" >> .env
echo -e BASE_DIR=$BASE_DIR >> .env
echo -e AIRFLOW_WEBSERVER_HOST_PORT=$AIRFLOW_WEBSERVER_HOST_PORT >> .env
```

Place app_config_dev.yml in same directory as docker-compose.yaml. An example can be found at `s3://ophys-processing-airflow.alleninstitute.org/dev/app_config_dev.yml`. The schema is `ophys_etl.workflows.app_config.app_config.AppConfig`

> **Note**: webserver.hostname should be "ophys-processing-dev-airflow-webserver-${UID}" (replace UID with the result of id -u) which is the container_name for the airflow webserver

Run `docker compose build`. Make sure to pass in the arguments for the `args` as defined in `build.args` in the `docker-compose.yaml` by passing `--build-arg <arg>=...`

Set the directory containing the root dir of the codebase to group read/write/execute with 

```chmod ../../../../ g+rwx```

Run `docker compose --project-name <something unique> up airflow-init`

Run `docker compose --project-name <something unique> up`

> **Note**: --project-name is important and allows for running the same docker-compose file at the same time on the same machine by multiple users

### Environment variables
    
Required environment variables (only when not running using docker-compose.yaml or in production):

| Env var                           | Description                   |
|-----------------------------------|-------------------------------|
| OPHYS_WORKFLOW_APP_CONFIG_PATH    | Path to app config .yaml file |

## Testing

A script that can be used for testing is `ophys_etl/workflows/scrips/run_pipeline_end_to_end_test.py`

## Deployment

The app is deployed using [airflow helm chart](https://airflow.apache.org/docs/helm-chart/stable/index.html) 

### Prerequisites

docker
```bash
sudo dnf -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

[kind](https://kind.sigs.k8s.io/)

[helm](https://helm.sh/docs/intro/install/)

There need to be 2 databases already created:
1. airflow db - connection details need to be added to `values.yaml`
2. ophys app db - connection details need to be added to `app_config.yml`

***
1. build image 
```bash
docker build -t alleninstitutepika/ophys_processing_airflow:<tag> ...
```
2. push image
```bash
docker push alleninstitutepika/ophys_processing_airflow:<tag>
```

3. upload and/or download [values.yaml](https://helm.sh/docs/chart_template_guide/values_files/) from s3://ophys-processing-airflow.alleninstitute.org/prod/

4. deploy app
```bash
python ophys_etl_pipelines/src/ophys_etl/workflows/deploy.py
```
For more details: [airflow kind quickstart guide](https://airflow.apache.org/docs/helm-chart/stable/quick-start.html)

### Testing the deployment

Run end to end test.

1. Open shell in arbitrary pod
    ```bash
    kubectl exec --namespace $KUBERNETES_NAMESPACE --stdin --tty ophys-processing-scheduler-0 -- /bin/bash
    ```
2. Run end to end test
    ```bash
   python ophys_etl_pipelines/src/ophys_etl/workflows/scripts/run_end_to_end_test.py
   ```

### Debugging deployment

Some useful commands
```bash
kubectl get pods
```
```bash
kubectl describe
```
```bash
kubectl logs
```
```bash
kubectl get events
```

Sometimes docker might get full and deployments will fail because of this. To clean it up:
```bash
docker system prune -a -f
```