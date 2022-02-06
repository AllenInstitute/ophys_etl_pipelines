import argschema


class DockerSchema(argschema.ArgSchema):
    repository_name = argschema.fields.Str(
        default='train-deepinterpolation',
        description='Docker repository name that will be created, and also '
                    'the base name for the sagemaker training job (training '
                    'job will be <repository name>-<timestamp>)'
    )

    image_tag = argschema.fields.Str(
        default='latest',
        description='Image tag'
    )


class S3ParamsSchema(argschema.ArgSchema):
    bucket_name = argschema.fields.Str(
        default='deepinterpolation',
        description='Bucket on S3 to use for storing input data, '
                    'model outputs, logs'
    )


class CloudDenoisingTrainerSchema(argschema.ArgSchema):
    local_mode = argschema.fields.Bool(
        default=False,
        description='Whether to run estimator in local mode. Useful for '
                    'debugging'
    )

    profile_name = argschema.fields.Str(
        default='default',
        description='AWS profile name. Useful for debugging to use a sandbox '
                    'account'
    )

    instance_type = argschema.fields.Str(
        required=False,
        default=None,
        allow_none=True,
        description='EC2 instance type. Required if not in local mode'
    )

    instance_count = argschema.fields.Int(
        default=1,
        description='Number of EC2 instances to use'
    )

    sagemaker_execution_role = argschema.fields.Str(
        required=True,
        description='The role id with AmazonSageMakerFullAccess permissions. '
                    'This role should already be created in AWS IAM. '
                    'Unfortunately still required to exist in AWS even in '
                    'local mode'
    )

    docker_params = argschema.fields.Nested(
        DockerSchema,
        default={}
    )

    s3_params = argschema.fields.Nested(
        S3ParamsSchema,
        default={}
    )

    pretrained_model_path = argschema.fields.InputFile(
        required=False,
        description='Path to pretrained model to finetune'
    )

    input_json_path = argschema.fields.InputFile(
        required=True,
        description='The input json to pass along to the deepinterpolation '
                    'CLI'
    )