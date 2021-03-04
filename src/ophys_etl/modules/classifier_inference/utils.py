import boto3
from typing import Optional, Union
import time


class RegistryConnection(object):
    def __init__(
            self,
            table_name: str,
            region: Optional[str] = "us-west-2",
            aws_access_key_id: Optional[str] = None,
            aws_secret_access_key: Optional[str] = None,
            aws_session_token: Optional[str] = None):
        """
        Instantiate a connection to the dynamoDB table used as a model
        registry.
        
        Parameters
        ----------
        table_name: str
            The name of the DynamoDB table containing the registry
        aws_access_key_id: Optional[str]
            Access key id for AWS account. If not used, will discover
            credentials according to the discovery order in boto3 (e.g. 
            environment variables, ~/.aws/credentials file)
            https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html    # noqa
        aws_secret_access_key: Optional[str]
            Secret access key for AWS account. See `aws_access_key_id`.
        aws_session_token: Optional[str]
            Session key for AWS account. Only needed when using temporary
            credentials.

        Methods
        -------
        * get_active_model
        * register_active_model
        * activate_model

        """
        self._client = boto3.client(
            "dynamodb",
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        self._table_name = table_name

    def get_active_model(self, env: str = "prod") -> str:
        """
        Get the active model's artifact location for a particularly
        environment ('dev', 'stage', or 'prod'). This is the entry
        point that the classification/segmentation module should use
        to get the location of the trained model for classifying ROIs.

        Parameters
        ----------
        env: str ("dev|stage|prod")
            The environment to retrieve the model from.

        Returns
        -------
        str
            The location of the model artifact (either an S3 URI or a
            filepath on the isilon)

        Raises
        ------
        KeyError
            If there is not an active model for the environment `env`.
        """
        response = self._client.get_item(
            TableName=self._table_name,
            Key={"model_name": {"S": env},
                 "timestamp": {"N": "0"}},
            ProjectionExpression="artifact_location",
            ConsistentRead=True
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise RuntimeError(
                "Accessing record in DynamoDB was unsuccessful. "
                "Most likely your credentials are incorrect or do not "
                "allow access to the table. Please try again. "
                f"Response: \n {response}")
        elif response.get("Item") is None:
            raise KeyError(f"No active model was found in {env} environment.")
        return response["Item"]["artifact_location"]["S"]

    def register_active_model(
            self, model_name: str, env: str, mlflow_run_id: str,
            artifact_location: str):
        """
        Create a record for a new active model in the registry for
        a specific environment.

        Parameters
        ----------
        model_name: str
            The name of the model to register as active in the database.
        env: str ("dev|stage|prod")
            The environment to register the active model
        artifact_location: str
            The location of the model artifact. Can be an s3 URI or a
            filepath on the isilon.
        mlflow_run_id: str
            The ID of the mlflow run used to train the model, for
            linking with the mlflow database

        Returns
        -------
        List of put_item responses from DynamoDB.

        Raises
        ------
        ValueError
            If a user tries to register a model with the
            `model_name` in the list: ["dev", "stage", "prod"].
            These values must be reserved to point to the currently
             active model.
        ValueError
            If `env` is not one of "dev", "stage", or "prod".
        RuntimeError
            If a non-200 (OK) status code is returned when
            trying to put items in the table.
        """
        if model_name.lower() in ["dev", "stage", "prod"]:
            raise ValueError(f"Cannot use protected name '{model_name}'.")
        if env not in ["dev", "stage", "prod"]:
            raise ValueError(f"Value for `env` must be one of 'dev', 'stage', "
                             f"or 'prod' (got '{env}'.")
        # Add the new record first
        response = self._client.put_item(
            TableName=self._table_name,
            Item={
                "model_name": {"S": model_name},
                "timestamp": {"N": str(int(time.time()))},
                "artifact_location": {"S": artifact_location},
                "mlflow_run_id": {"S": mlflow_run_id}
            }
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:    # error
            raise RuntimeError(
                "Putting new record in DynamoDB was unsuccessful. "
                "Most likely your credentials are incorrect or do not "
                "allow access to the table. Please try again. "
                f"Response: \n {response}")
        # Replace the active record
        active_response = self._client.put_item(
            TableName=self._table_name,
            Item={
                "model_name": {"S": env},
                "timestamp": {"N": "0"},
                "artifact_location": {"S": artifact_location},
                "mlflow_run_id": {"S": mlflow_run_id}
            }
        )
        # if status not OK
        if active_response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise RuntimeError(
                "Putting new active model in DynamoDB was unsuccessful. "
                "Most likely your credentials are incorrect or do not "
                "allow access to the table. Please try again. "
                f"Response: \n {active_response}")
        return [response, active_response]

    def activate_model(
            self, model_name: str, env: str, timestamp: Union[str, int]):
        """
        Reactivate a model in the registry by name and timestamp.
        Only one model can be active per environment, so the previously
         active model is replaced.

        Parameters
        ----------
        model_name: str
            The model's name
        env: str ("dev|stage|prod")
            The environment for which to activate the model
        timestamp: int
            The model's timestamp in seconds from epoch (either str
            or int). This is the sort key in the table.

        Returns
        -------
        Response from DynamoDB put operation.

        Raises
        ------
        KeyError
            If no value is returned from the table (did not exist)
        RuntimeError
            If the DynamoDB response was not 200 (OK)
        """
        model_response = self._client.get_item(
            TableName=self._table_name,
            Key={"model_name": {"S": model_name},
                 "timestamp": {"N": str(timestamp)}},
            ProjectionExpression="artifact_location,mlflow_run_id"
        )
        if model_response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise RuntimeError(
                "Unable to get response from DynamoDB table. "
                "Most likely your credentials are incorrect or do not "
                "allow access to the table. Please try again. "
                f"Response: \n {model_response}")
        elif model_response.get("Item") is None:
            raise KeyError(f"No model with name {model_name} and timestamp "
                           f"{timestamp} was found in the table.")

        model = model_response["Item"]
        active_response = self._client.put_item(
            TableName=self._table_name,
            Item={"model_name": {"S": env},
                  "timestamp": {"N": "0"},
                  "artifact_location": model["artifact_location"],
                  "mlflow_run_id": model["mlflow_run_id"]})
        if active_response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise RuntimeError(
                "Putting new active model in DynamoDB was unsuccessful. "
                "Most likely your credentials are incorrect or do not "
                "allow access to the table. Please try again. "
                f"Response: \n {active_response}")
        return active_response
