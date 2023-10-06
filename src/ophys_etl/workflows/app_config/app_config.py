"""App config"""
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import yaml
from airflow.models import Variable, Connection
from deepcell.cli.modules.create_dataset import VoteTallyingStrategy
from deepcell.datasets.channel import Channel
from pydantic import (
    Field,
    FilePath,
    SecretStr,
    StrictFloat,
    StrictInt,
    StrictStr, validator
)

from ophys_etl.workflows.app_config._ophys_processing_trigger import \
    OphysProcessingTrigger
from ophys_etl.workflows.app_config.slurm import SlurmSettings
from ophys_etl.workflows.utils.pydantic_model_utils import ImmutableBaseModel


class Environment(Enum):
    """What environment we are running in"""
    DEV = 'dev'
    STAGING = 'staging'
    PROD = 'prod'


class _AirflowWebserverConfig(ImmutableBaseModel):
    """
    Config for airflow webserver
    """
    host_name: StrictStr = Field(
        description='Host name for web server',
        default='0.0.0.0'
    )
    username: SecretStr = Field(
        description='Username to authenticate with REST API. Generate by '
                    'creating user using the Airflow UI')
    password: SecretStr = Field(
        description='Password to authenticate with REST API. Generate by '
                    'creating user using the Airflow UI')


class _AppDB(ImmutableBaseModel):
    """
    App DB (not airflow) config
    """

    conn_string: StrictStr = Field(
        description="Conn string to db. See "
        "https://airflow.apache.org/docs/apache-airflow/stable/howto/connection.html"  # noqa E501
    )


class _LimsDB(ImmutableBaseModel):
    """LIMS DB config"""

    username: SecretStr = Field(description="username")
    password: SecretStr = Field(description="password")


class _Singularity(ImmutableBaseModel):
    """Singularity config"""

    username: SecretStr = Field(description="username")
    password: SecretStr = Field(description="password")


class _Slurm(ImmutableBaseModel):
    """Slurm config"""

    username: StrictStr = Field(description="Username to run jobs under")
    api_token: SecretStr = Field(
        description="api token, generated using "
                    "scontrol token lifespan=94610000"
    )
    partition: StrictStr = 'braintv'


##################
# Pipeline steps
##################


class _PipelineStep(ImmutableBaseModel):
    """A pipeline step config"""

    docker_tag: StrictStr = Field(
        default='main',
        description="Docker tag to use to run pipeline step",
    )
    slurm_settings: SlurmSettings = Field(
        description='Settings to use when running pipeline step on SLURM',
        default=SlurmSettings()
    )
    _default_slurm_settings = slurm_settings

    @validator('slurm_settings', pre=True, always=True)
    @classmethod
    def set_slurm_settings(cls, v):
        """Overrides the default settings with the ones passed by the user.
        Allows to only override some of the settings and takes the rest
        from the defaults defined in the `_default_slurm_settings`"""
        if isinstance(v, SlurmSettings):
            v = v.dict()
        default = cls._default_slurm_settings
        if getattr(default, 'default', None) is not None:
            default = getattr(default, 'default')
        updated = {**default.dict(), **v}
        return updated


class _DenoisingFineTuning(_PipelineStep):
    base_model_path: FilePath = Field(
        description="Path to base deepinterpolation model to use for "
        "finetuning"
    )
    downsample_frac: Optional[float] = Field(
        default=None,
        description="Amount to downsample the training data by. "
        "I.e. a downsample frac of 0.1 would randomly sample 10% "
        "of the data",
    )
    slurm_settings = SlurmSettings(
        cpus_per_task=17,
        mem=85,
        time=12 * 60,
        gpus=1
    )
    _default_slurm_settings = slurm_settings


class _DenoisingInference(_PipelineStep):
    normalize_cache: bool = Field(
        default=False,
        description="Whether to normalize the cache data."
        "Generally disable if RAM is less than 128GB"
    )
    gpu_cache_full: bool = Field(
        default=False,
        description="Whether to use GPU for caching. Enable if GPU"
        "has more RAM than the size of the movie, e.g. A100 48GB"
    )
    slurm_settings = SlurmSettings(
        cpus_per_task=17,
        mem=250,
        time=480,
        gpus=1
    )
    _default_slurm_settings = slurm_settings


class _Denoising(_PipelineStep):
    """Deepinterpolation config"""
    batch_size: StrictInt = Field(
        default=8,
        description="Batch size of the generator for fine tuning"
        "and inference. Set equal to number of workers for multiprocessing"
    )
    finetuning: _DenoisingFineTuning
    inference: _DenoisingInference = Field(
        default=_DenoisingInference()
    )


class _MotionCorrection(_PipelineStep):
    nonrigid: bool = Field(
        default=False,
        description='Whether to turn on nonrigid motion correction')
    slurm_settings = SlurmSettings(
        cpus_per_task=32,
        mem=250,
        time=300,
        request_additional_tmp_storage=True
    )
    _default_slurm_settings = slurm_settings


class _Segmentation(_PipelineStep):
    slurm_settings = SlurmSettings(
        cpus_per_task=8,
        mem=80,
        time=240
    )
    _default_slurm_settings = slurm_settings


class _TraceExtraction(_PipelineStep):
    pass


class _Decrostalk(_PipelineStep):
    pass


class _DemixTraces(_PipelineStep):
    slurm_settings = SlurmSettings(
        cpus_per_task=32,
        mem=96,
        time=960
    )
    _default_slurm_settings = slurm_settings


class _NeuropilCorrection(_PipelineStep):
    slurm_settings = SlurmSettings(
        cpus_per_task=1,
        mem=4,
        time=600
    )
    _default_slurm_settings = slurm_settings


class _DffCalculationModule(_PipelineStep):
    slurm_settings = SlurmSettings(
        cpus_per_task=24,
        mem=140,
        time=120
    )
    _default_slurm_settings = slurm_settings


class _EventDetectionModule(_PipelineStep):
    slurm_settings = SlurmSettings(
        cpus_per_task=24,
        mem=90,
        time=90
    )
    _default_slurm_settings = slurm_settings


class _GenerateCorrelationProjection(_PipelineStep):
    n_workers: int = 4
    slurm_settings = SlurmSettings(
        cpus_per_task=4,
        mem=128,
        time=480
    )
    _default_slurm_settings = slurm_settings


class _GenerateThumbnails(_PipelineStep):
    pass


class _NwayCellMatching(_PipelineStep):
    pass


class _ROIClassifierTraining(_PipelineStep):
    class TrainTestSplit(ImmutableBaseModel):
        test_size: StrictFloat = Field(
            description="Fraction of the experiments to reserve for the test "
            "sample.",
            default=0.3,
        )

    class ModelParams(ImmutableBaseModel):
        freeze_to_layer: Optional[StrictInt] = Field(
            description="See deepcell.models.classifier.Classifier."
            "freeze_up_to_layer",
            default=None,
        )
        truncate_to_layer: Optional[StrictInt] = Field(
            description="See deepcell.models.classifier.Classifier."
            "truncate_to_layer",
            default=None,
        )

    class DockerParams(ImmutableBaseModel):
        image_uri: Optional[StrictStr] = Field(
            default=None,
            description="URI to a prebuilt docker container on AWS ECR to use "
            "for training",
        )

    class S3Params(ImmutableBaseModel):
        bucket_name: StrictStr = Field(
            description="The bucket to upload data to"
        )
        data_key: Optional[StrictStr] = Field(
            description="If provided, will pull data from this key on s3, "
            "rather than uploading it. Should be what comes after "
            "input_data/. i.e. s3://<bucket>/input_data/foo would "
            'be "foo',
            default=None,
        )

    class TrackingParams(ImmutableBaseModel):
        mlflow_server_uri: StrictStr = Field(
            description="MLFlow server uri to use for tracking"
        )
        mlflow_experiment_name: StrictStr = Field(
            description="MLFlow experiment name to use to track training run",
            default="deepcell-train",
        )

    train_test_split: Optional[TrainTestSplit]
    model: Optional[ModelParams]
    docker: Optional[DockerParams]
    s3: S3Params
    n_folds: StrictInt = Field(
        default=5, description="Number of folds for cross validation"
    )
    tracking: TrackingParams
    voting_strategy: VoteTallyingStrategy = VoteTallyingStrategy.MAJORITY


class _ROIClassifierInference(_PipelineStep):
    classification_threshold: float = Field(
        default=0.5, description='classification threshold'
    )


class _ROIClassification(ImmutableBaseModel):
    input_channels: List[Channel]
    cell_labeling_app_host: StrictStr
    generate_correlation_projection: _GenerateCorrelationProjection = Field(
        default=_GenerateCorrelationProjection()
    )
    generate_thumbnails: _GenerateThumbnails = Field(
        default=_GenerateThumbnails()
    )
    training: _ROIClassifierTraining
    inference: _ROIClassifierInference = Field(
        default=_ROIClassifierInference()
    )


class _PipelineSteps(ImmutableBaseModel):
    """All pipeline steps configs"""
    docker_tag: StrictStr = Field(
        default="main",
        description="Docker tag to use to run pipeline step. Defaults to "
        '"main"',
    )
    denoising: _Denoising
    motion_correction: _MotionCorrection = Field(
        default=_MotionCorrection()
    )
    segmentation: _Segmentation = Field(
        default=_Segmentation()
    )
    trace_extraction: _TraceExtraction = Field(default=_TraceExtraction())
    demix_traces: _DemixTraces = Field(
        default=_DemixTraces()
    )
    roi_classification: _ROIClassification
    decrosstalk: _Decrostalk = Field(default=_Decrostalk())
    neuropil_correction: _NeuropilCorrection = Field(
        default=_NeuropilCorrection()
    )
    dff: _DffCalculationModule = Field(
        default=_DffCalculationModule()
    )
    event_detection: _EventDetectionModule = Field(
        default=_EventDetectionModule()
    )
    nway_cell_matching: _NwayCellMatching = Field(default=_NwayCellMatching())


##################


class AppConfig(ImmutableBaseModel):
    """Workflow config"""

    env: Environment = Environment.DEV
    webserver: _AirflowWebserverConfig
    is_debug: bool = Field(
        default=False,
        description="If True, will not actually run the modules, but "
        "will run a dummy command instead to test the "
        "workflow",
    )
    app_db: _AppDB
    output_dir: Path = Field(description="Root dir to output files")
    job_timeout: float = Field(
        description="Job timeout in seconds",
        default=24 * 60 * 60 * 3,  # 3 days
    )
    lims_db: _LimsDB
    singularity: _Singularity
    pipeline_steps: _PipelineSteps
    slurm: _Slurm
    ophys_processing_trigger: OphysProcessingTrigger = Field(
        default=OphysProcessingTrigger()
    )
    fov_shape: Tuple[int, int] = (512, 512)


def _set_sensitive_configs(config: Dict):
    """
    Sets sensitive configs from Variable store. For staging or production,
    we are using AWS secrets manager. See
    https://airflow.apache.org/docs/apache-airflow-providers-amazon/stable/secrets-backends/aws-secrets-manager.html

    Parameters
    ----------
    config:
        The config parsed from yaml

    Returns
    -------
    Dict
        The config with sensitive configs set
    """
    webserver = Variable.get('webserver', deserialize_json=True)
    lims = Variable.get('lims', deserialize_json=True)
    slurm = Variable.get('slurm', deserialize_json=True)
    singularity = Variable.get('singularity', deserialize_json=True)

    config['webserver'] = webserver
    config['lims_db'] = lims
    config['slurm'] = slurm
    config['singularity'] = singularity

    config['output_dir'] = Variable.get('output_dir')

    config['pipeline_steps']['roi_classification']['cell_labeling_app_host'] = (    # noqa E402
        Variable.get('cell_labeling_app_host'))
    config['pipeline_steps']['roi_classification']['training']['tracking']['mlflow_server_uri'] = ( # noqa E402
        Variable.get('mlflow_server_uri'))
    config['pipeline_steps']['denoising']['finetuning']['base_model_path'] = (
        Variable.get('base_deepinterpolation_model_path'))

    config['app_db'] = {
        'conn_string': Connection.get_connection_from_secrets(
            conn_id='app_db').get_uri()
    }

    return config


def load_config() -> AppConfig:
    config_path = os.environ["OPHYS_WORKFLOW_APP_CONFIG_PATH"]

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # replace value with env var when testing
    if os.environ.get("TEST_DI_BASE_MODEL_PATH", None) is not None:
        denoising_conf = config["pipeline_steps"]["denoising"]
        denoising_conf['finetuning']["base_model_path"] = os.environ[
            "TEST_DI_BASE_MODEL_PATH"
        ]

    if 'env' in config and config['env'] in (
            Environment.STAGING.value, Environment.PROD.value):
        config = _set_sensitive_configs(config=config)

    # serialize to AppConfig
    config = AppConfig(**config)

    return config


app_config = load_config()
