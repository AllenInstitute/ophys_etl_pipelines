# Ophys processing workflows

## Installation

### Set up app DB

Use the connection details in section 1.2.1 of [this guide](http://confluence.corp.alleninstitute.org/pages/viewpage.action?pageId=60855687) to setup a db

Initialize ophys workflow db using `python -m ophys_etl.workflows.db.initialize_db`

### Launch app using docker compose

In same dir as `workflows/docker-compose-yaml`, run
```bash
AIRFLOW_PROJ_DIR=<directory where you want airflow outputs to be output>
LOG_DIR=<directory to output logs>
AIRFLOW_WEBSERVER_HOST_PORT=<port to map webserver to on the host>
mkdir -p $AIRFLOW_PROJ_DIR/logs $AIRFLOW_PROJ_DIR/config $AIRFLOW_PROJ_DIR/plugins

echo -e "AIRFLOW_UID=$(id -u)" >> .env
echo -e AIRFLOW_PROJ_DIR=$AIRFLOW_PROJ_DIR >> .env
echo -e AIRFLOW_WEBSERVER_HOST_PORT=$AIRFLOW_WEBSERVER_HOST_PORT >> .env
echo -e LOG_DIR=$LOG_DIR >> .env

```
   > **_IMPORTANT:_**  Use /ifs instead of /allen for `AIRFLOW_PROJ_DIR`

Place app_config_dev.yml in same directory as docker-compose.yaml

Run `docker compose build`. Make sure to pass in the arguments for the `args` as defined in `build.args` in the `docker-compose.yaml` by passing `build-arg <arg>=...`

Run `docker compose up airflow-init`

Run `docker compose up`

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

4. Follow [airflow kind quickstart guide](https://airflow.apache.org/docs/helm-chart/stable/quick-start.html)

    Make sure to pass `-f values.yaml` to `helm install`/`helm upgrade` commands

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