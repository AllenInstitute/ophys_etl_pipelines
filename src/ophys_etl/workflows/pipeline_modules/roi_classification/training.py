from types import ModuleType
from typing import Dict, List

from deepcell.cli.schemas.train import TrainSchema
from deepcell.cli.modules.cloud import train
from sqlmodel import Session

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.db.schemas import (
    ROIClassifierEnsemble,
    ROIClassifierTrainingRun,
)
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.pipeline_modules.roi_classification.utils.mlflow_utils import ( # noqa E501
    MLFlowRun,
)
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class TrainingModule(PipelineModule):
    def __init__(self, prevent_file_overwrites: bool = True, **kwargs):

        self._model_inputs_path: OutputFile = kwargs["train_set_path"]
        self._mlflow_run_name = kwargs["mlflow_run_name"]
        super().__init__(
            ophys_experiment=None,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs
        )

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.ROI_CLASSIFICATION_TRAINING

    @property
    def module_schema(self) -> TrainSchema:
        return TrainSchema()

    @property
    def inputs(self) -> Dict:
        return {
            "train_params": {
                "model_inputs_path": self._model_inputs_path,
                "model_params": {
                    "freeze_to_layer": (
                        app_config.pipeline_steps.roi_classification.training.model.freeze_to_layer # noqa E501
                    ),
                    "truncate_to_layer": (
                        app_config.pipeline_steps.roi_classification.training.model.truncate_to_layer # noqa E501
                    ),
                },
                "tracking_params": {
                    "mlflow_server_uri": (
                        app_config.pipeline_steps.roi_classification.training.tracking.mlflow_server_uri # noqa E501
                    ),
                    "mlflow_run_name": self._mlflow_run_name,
                },
            },
            "docker_params": {
                "image_uri": (
                    app_config.pipeline_steps.roi_classification.training.docker.image_uri # noqa E501
                )
            },
            "s3_params": {
                "bucket_name": (
                    app_config.pipeline_steps.roi_classification.training.s3.bucket_name # noqa E501
                ),
                "data_key": (
                    app_config.pipeline_steps.roi_classification.training.s3.data_key # noqa E501
                ),
            },
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_CLASSIFICATION_TRAINED_MODEL
                ),
                path=self.output_path / "model",
            )
        ]

    @property
    def executable(self) -> ModuleType:
        return train

    @staticmethod
    def save_trained_model_to_db(
        output_files: Dict[str, OutputFile],
        session: Session,
        run_id: int,
        mlflow_parent_run_name: str,
        **kwargs
    ):
        mlflow_run = MLFlowRun(
            mlflow_experiment_name=(
                app_config.pipeline_steps.roi_classification.training.tracking.mlflow_experiment_name # noqa E501
            ),
            run_name=mlflow_parent_run_name,
        )

        ensemble = ROIClassifierEnsemble(
            workflow_step_run_id=run_id,
            mlflow_run_id=mlflow_run.run.info.run_id,
            classification_threshold=(
                app_config.pipeline_steps.roi_classification.inference.
                classification_threshold)
        )
        session.add(ensemble)

        # flush to get ensemble id of just added ensemble
        session.flush()

        for child_run in mlflow_run.child_runs:
            training_run = ROIClassifierTrainingRun(
                ensemble_id=ensemble.id,
                mlflow_run_id=child_run.run.info.run_id,
                sagemaker_job_id=child_run.sagemaker_job_id,
            )
            session.add(training_run)
