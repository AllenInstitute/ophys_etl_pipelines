"""Denoising finetuning pipeline module"""
from types import ModuleType
from typing import List, Dict

from ophys_etl.modules.denoising import fine_tuning
from ophys_etl.modules.denoising.fine_tuning.schemas import FineTuningInputSchemaPreDataSplit  # noqa: E501
from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_modules.denoising._denoising import (
    _DenoisingModule,
)
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class DenoisingFinetuningModule(_DenoisingModule):
    """Denoising Finetuning module"""

    @property
    def executable(self) -> ModuleType:
        return fine_tuning

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.DENOISING_FINETUNING

    @property
    def module_schema(self) -> FineTuningInputSchemaPreDataSplit:
        return FineTuningInputSchemaPreDataSplit()

    @property
    def inputs(self) -> Dict:
        return {
            "data_split_params": {
                "ophys_experiment_id": str(self.ophys_experiment.id),
                "dataset_output_dir": str(self.output_path),
                "movie_path": self._motion_corrected_path,
                "downsample_frac": (
                    app_config.pipeline_steps.denoising.finetuning.
                    downsample_frac
                ),
            },
            "finetuning_params": {
                "apply_learning_decay": False,
                "caching_validation": False,
                "epochs_drop": 5,
                "learning_rate": 0.0001,
                "loss": "mean_squared_error",
                "model_source": {
                    "local_path": (
                        app_config.pipeline_steps.denoising.finetuning.
                        base_model_path
                    )
                },
                "model_string": "",
                "multi_gpus": False,
                "name": "transfer_trainer",
                "nb_times_through_data": 1,
                "nb_workers": 15,
                "output_dir": str(self.output_path),
                "period_save": 1,
                "steps_per_epoch": 20,
            },
            "generator_params": {
                "batch_size": app_config.pipeline_steps.denoising.batch_size,
                "name": "MovieJSONGenerator",
                "post_frame": 30,
                "pre_frame": 30,
                "pre_post_omission": 0,
                "seed": 1234
            },
            "log_level": "INFO",
            "output_full_args": True,
            "test_generator_params": {
                "batch_size": 5,
                "name": "MovieJSONGenerator",
                "post_frame": 30,
                "pre_frame": 30,
                "pre_post_omission": 0,
                "seed": 1234
            },
            "run_uid": str(self.ophys_experiment.id),
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.DEEPINTERPOLATION_FINETUNED_MODEL
                ),
                path=(
                    self.output_path / f"{self.ophys_experiment.id}_"
                    f"mean_squared_error_transfer_model.h5"
                ),
            )
        ]
