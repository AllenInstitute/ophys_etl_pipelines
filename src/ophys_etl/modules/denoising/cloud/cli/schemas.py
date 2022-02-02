import argschema


class DockerSchema(argschema.ArgSchema):
    repository_name = argschema.fields.Str(
        default='train_deepinterpolation',
        description='Docker repository name that will be created'
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
        required=False,
        default=None,
        allow_none=True,
        description='The role id with AmazonSageMakerFullAccess permissions. '
                    'This role should already be created in AWS IAM'
    )

    local_data_dir = argschema.fields.InputDir(
        default=None,
        description='Directory containing local input data. The data here '
                    'will either be used locally for training if in local '
                    'mode or will be uploaded to S3 for cloud training'
    )

    docker_params = argschema.fields.Nested(
        DockerSchema,
        default={}
    )

    s3_params = argschema.fields.Nested(
        S3ParamsSchema,
        default={}
    )
