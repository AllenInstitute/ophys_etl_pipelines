from moto import mock_dynamodb2
import boto3
import pytest

from ophys_etl.modules.classifier_inference.utils import RegistryConnection


@pytest.fixture
def connection(scope="function"):
    mock_dynamo = mock_dynamodb2()
    mock_dynamo.start()
    table_name = "test-table"
    client = boto3.client("dynamodb", region_name="us-west-2")
    client.create_table(
        TableName=table_name,
        KeySchema=[
            {
                "AttributeName": "model_name",
                "KeyType": "HASH"
            },
            {
                "AttributeName": "timestamp",
                "KeyType": "RANGE"
            }
        ],
        AttributeDefinitions=[
            {
                "AttributeName": "model_name",
                "AttributeType": "S"
            },
            {
                "AttributeName": "timestamp",
                "AttributeType": "N"
            }
        ],
        ProvisionedThroughput={
            "ReadCapacityUnits": 123,
            "WriteCapacityUnits": 123
        }
    )
    yield RegistryConnection(table_name)
    mock_dynamo.stop()


def test_register_active_model_roundtrip(connection, monkeypatch):
    """Register an active model and query for it to check."""
    monkeypatch.setattr(
            "ophys_etl.modules.classifier_inference.utils.time.time",
            lambda: 123)
    connection.register_active_model("twiggy", "prod", "abc999", "s3://path")
    record = connection._client.get_item(
        TableName=connection._table_name,
        Key={"model_name": {"S": "twiggy"},
             "timestamp": {"N": "123"}},
        ProjectionExpression="model_name,timestamp,artifact_location,mlflow_run_id"    # noqa
    )
    # Roundtrip get the records that were put
    assert record["Item"] == {
        "model_name": {"S": "twiggy"}, "timestamp": {"N": "123"},
        "mlflow_run_id": {"S": "abc999"},
        "artifact_location": {"S": "s3://path"}}
    active = connection._client.get_item(
        TableName=connection._table_name,
        Key={"model_name": {"S": "prod"}, "timestamp": {"N": "0"}},
        ProjectionExpression="artifact_location,mlflow_run_id"
    )
    assert active["Item"] == {
        "mlflow_run_id": {"S": "abc999"},
        "artifact_location": {"S": "s3://path"}}


@pytest.mark.parametrize(
    "env,path", [
        ("prod", "s3://path/prod"),
        ("stage", "s3://path/stage"),
        ("dev", "/allen/aibs/")
    ]
)
def test_register_get_model_integration(env, path, connection):
    connection.register_active_model("twiggy", env, "abc999", path)
    assert path == connection.get_active_model(env)


@pytest.mark.parametrize(
    "name", ["prod", "PROD", "stage", "dev"]
)
def test_register_active_model_raises_error_protected_name(name, connection):
    with pytest.raises(ValueError):
        connection.register_active_model(name, "prod", 123, "s3://model/path")


def test_register_active_model_raises_error_wrong_env(connection):
    with pytest.raises(ValueError):
        connection.register_active_model("tyra", "staging", 123, "path")


@pytest.mark.parametrize(
    "env", ["prod", "stage", "dev"]
)
def test_get_active_fails_http_status(env, monkeypatch, connection):
    """Force non-200 status."""
    monkeypatch.setattr(
        connection._client, "get_item",
        lambda **x: {"ResponseMetadata": {"HTTPStatusCode": 0}})
    with pytest.raises(RuntimeError):
        connection.get_active_model(env)


def test_get_active_fails_not_exist(connection):
    with pytest.raises(KeyError):
        connection.get_active_model("prod")


def test_register_active_fails_http_status(connection, monkeypatch):
    monkeypatch.setattr(
        connection._client, "put_item",
        lambda **x: {"ResponseMetadata": {"HTTPStatusCode": 0}})
    with pytest.raises(RuntimeError):
        connection.register_active_model(
            "tyra", "stage", "aaa999", "s3://path")


def test_activate_model_fails_http_status_get(connection, monkeypatch):
    monkeypatch.setattr(
        connection._client, "get_item",
        lambda **x: {"ResponseMetadata": {"HTTPStatusCode": 0}})
    with pytest.raises(RuntimeError):
        connection.activate_model("tyra", "stage", "123")


def test_activate_model_fails_http_status_put(connection, monkeypatch):
    connection._client.put_item(
        TableName=connection._table_name,
        Item={"model_name": {"S": "tyra"}, "timestamp": {"N": "123"},
              "artifact_location": {"S": "s3://path"},
              "mlflow_run_id": {"S": "aaa000"}}
    )
    monkeypatch.setattr(
        connection._client, "put_item",
        lambda **x: {"ResponseMetadata": {"HTTPStatusCode": 0}})
    with pytest.raises(RuntimeError):
        connection.activate_model("tyra", "stage", "123")


def test_activate_model_fails_no_model_exists(connection, monkeypatch):
    with pytest.raises(KeyError):
        connection.activate_model("tyra", "stage", "123")


@pytest.mark.parametrize(
    "env", ["dev", "stage", "prod"]
)
def test_activate_model_integration(env, connection):
    connection._client.put_item(
        TableName=connection._table_name,
        Item={"model_name": {"S": "tyra"}, "timestamp": {"N": "123"},
              "artifact_location": {"S": "s3://path/tyra"},
              "mlflow_run_id": {"S": "aaa000"}}
    )
    response = connection.activate_model("tyra", env, "123")
    assert "s3://path/tyra" == connection.get_active_model(env)
    # Check boto3 response has metadata
    assert "ResponseMetadata" in response.keys()
