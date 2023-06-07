"""Denoising inference pipeline module"""
from types import ModuleType
from typing import Dict, List

from ophys_etl.modules.denoising import inference
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_modules.denoising._denoising import (
    _DenoisingModule,
)
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class DenoisingInferenceModule(_DenoisingModule):
    """Denoising inference module"""

    def __init__(
        self,
        ophys_experiment: OphysExperiment,
        prevent_file_overwrites: bool = True,
        **kwargs,
    ):
        super().__init__(
            ophys_experiment=ophys_experiment,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs,
        )

        trained_model_file: OutputFile = kwargs["trained_denoising_model_file"]
        self._trained_model_path = str(trained_model_file.path)

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.DENOISING_INFERENCE

    @property
    def inputs(self) -> Dict:
        return {
            "generator_params": {
                "batch_size": app_config.pipeline_steps.denoising.batch_size,
                "name": "InferenceOphysGenerator",
                "start_frame": 0,
                "cache_data": True,
                "normalize_cache": app_config.pipeline_steps.denoising.normalize_cache, # noqa E501
                "gpu_cache_full": app_config.pipeline_steps.denoising.gpu_cache_full, # noqa E501
                "data_path": self._motion_corrected_path,
                "seed": 1234
            },
            "inference_params": {
                "model_source": {"local_path": self._trained_model_path},
                "rescale": True,
                "save_raw": False,
                "output_padding": True,
                "output_file": (
                    str(
                        self.output_path
                        / f"{self.ophys_experiment.id}_denoised_video.h5"
                    )
                ),
                "steps_per_epoch": 0
            },
            "run_uid": self.ophys_experiment.id,
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.DEEPINTERPOLATION_DENOISED_MOVIE
                ),
                path=(
                    self.output_path
                    / f"{self.ophys_experiment.id}_denoised_video.h5"
                ),
            )
        ]

    @property
    def executable(self) -> ModuleType:
        return inference
