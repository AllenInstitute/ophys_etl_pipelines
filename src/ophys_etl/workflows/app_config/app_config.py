"""App config"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from deepcell.cli.modules.create_dataset import VoteTallyingStrategy
from deepcell.datasets.channel import Channel
from pydantic import (
    Field,
    FilePath,
    SecretStr,
    StrictFloat,
    StrictInt,
    StrictStr
)

from ophys_etl.workflows.app_config._ophys_processing_trigger import \
    OphysProcessingTrigger
from ophys_etl.workflows.utils.pydantic_model_utils import ImmutableBaseModel


class _AirflowRESTAPIConfig(ImmutableBaseModel):
    """
    Config to use Airflow REST API
    """
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
        description="api token, generated using scontrol token"
    )


##################
# Pipeline steps
##################


class _PipelineStep(ImmutableBaseModel):
    """A pipeline step config"""

    docker_tag: StrictStr = Field(
        default='main',
        description="Docker tag to use to run pipeline step",
    )


class _Denoising(_PipelineStep):
    """Deepinterpolation config"""

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
    batch_size: StrictInt = Field(
        default=8,
        description="Batch size of the generator for fine tuning"
        "and inference. Set equal to number of workers for multiprocessing"
    )
    normalize_cache: bool = Field(
        default=True,
        description="Whether to normalize the cache data."
        "Generally disable if RAM is less than 128GB"
    )
    gpu_cache_full: bool = Field(
        default=False,
        description="Whether to use GPU for caching. Enable if GPU"
        "has more RAM than the size of the movie, e.g. A100 48GB"
    )


class _MotionCorrection(_PipelineStep):
    pass


class _Segmentation(_PipelineStep):
    pass


class _TraceExtraction(_PipelineStep):
    pass


class _Decrostalk(_PipelineStep):
    pass


class _DemixTraces(_PipelineStep):
    pass


class _NeuropilCorrection(_PipelineStep):
    pass


class _DFOverFCalculation(_PipelineStep):
    pass


class _EventDetection(_PipelineStep):
    pass


class _GenerateCorrelationProjection(_PipelineStep):
    n_workers: int = 4


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
    motion_correction: _MotionCorrection = Field(default=_MotionCorrection())
    segmentation: _Segmentation = Field(default=_Segmentation())
    trace_extraction: _TraceExtraction = Field(default=_TraceExtraction())
    demix_traces: _DemixTraces = Field(default=_DemixTraces())
    roi_classification: _ROIClassification
    decrosstalk: _Decrostalk = Field(default=_Decrostalk())
    neuropil_correction: _NeuropilCorrection = Field(
        default=_NeuropilCorrection())
    dff: _DFOverFCalculation = Field(default=_DFOverFCalculation())
    event_detection: _EventDetection = Field(default=_EventDetection())
    nway_cell_matching: _NwayCellMatching = Field(default=_NwayCellMatching())


##################


class AppConfig(ImmutableBaseModel):
    """Workflow config"""

    airflow_rest_api_credentials: _AirflowRESTAPIConfig
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


def load_config() -> AppConfig:
    config_path = os.environ["OPHYS_WORKFLOW_APP_CONFIG_PATH"]

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # replace value with env var when testing
    if os.environ.get("TEST_DI_BASE_MODEL_PATH", None) is not None:
        config["pipeline_steps"]["denoising"]["base_model_path"] = os.environ[
            "TEST_DI_BASE_MODEL_PATH"
        ]

    # serialize to AppConfig
    config = AppConfig(**config)

    return config


app_config = load_config()
