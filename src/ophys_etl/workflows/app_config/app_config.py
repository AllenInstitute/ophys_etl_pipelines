"""App config"""
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import StrictStr, SecretStr, FilePath, Field

from ophys_etl.workflows.utils.pydantic_model_utils import ImmutableBaseModel


class _AppDB(ImmutableBaseModel):
    """
    App DB (not airflow) config
    """
    conn_string: StrictStr = Field(
        description='Conn string to db. See '
                    'https://airflow.apache.org/docs/apache-airflow/stable/howto/connection.html'   # noqa E501
    )


class _LimsDB(ImmutableBaseModel):
    """LIMS DB config"""
    username: SecretStr = Field(
        description='username'
    )
    password: SecretStr = Field(
        description='password'
    )


class _Singularity(ImmutableBaseModel):
    """Singularity config"""
    username: SecretStr = Field(
        description='username'
    )
    password: SecretStr = Field(
        description='password'
    )


class _PipelineStep(ImmutableBaseModel):
    """A pipeline step config"""
    docker_tag: StrictStr = Field(
        description='Docker tag to use to run pipeline step'
    )


class _Denoising(_PipelineStep):
    """Deepinterpolation config"""
    base_model_path: FilePath = Field(
        description='Path to base deepinterpolation model to use for '
                    'finetuning'
    )
    downsample_frac: Optional[float] = Field(
        default=None,
        description='Amount to downsample the training data by. '
                    'I.e. a downsample frac of 0.1 would randomly sample 10% '
                    'of the data'
    )


class _MotionCorrection(_PipelineStep):
    pass


class _Segmentation(_PipelineStep):
    pass


class _PipelineSteps(ImmutableBaseModel):
    """All pipeline steps configs"""
    denoising: _Denoising
    motion_correction: _MotionCorrection
    segmentation: _Segmentation


class AppConfig(ImmutableBaseModel):
    """Workflow config"""
    app_db: _AppDB
    output_dir: Path = Field(
        description='Root dir to output files'
    )
    job_timeout: float = Field(
        description='Job timeout in seconds',
        default=24 * 60 * 60 * 3    # 3 days
    )
    lims_db: _LimsDB
    singularity: _Singularity
    pipeline_steps: _PipelineSteps


def load_config() -> AppConfig:
    config_path = os.environ['OPHYS_WORKFLOW_APP_CONFIG_PATH']

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # replace value with env var when testing
    if os.environ.get('TEST_DI_BASE_MODEL_PATH', None) is not None:
        config['pipeline_steps']['denoising']['base_model_path'] = \
            os.environ['TEST_DI_BASE_MODEL_PATH']

    # serialize to AppConfig
    config = AppConfig(**config)

    return config


def _set_environment_variables(config: AppConfig):
    """Sets environment variables"""
    os.environ['AIRFLOW_CONN_OPHYS_WORKFLOW_DB'] = config.app_db.conn_string


app_config = load_config()
_set_environment_variables(config=app_config)
