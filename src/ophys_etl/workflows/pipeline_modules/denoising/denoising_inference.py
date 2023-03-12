"""Denoisig inference pipeline module"""
from typing import List, Dict

from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.pipeline_module import OutputFile
from ophys_etl.workflows.pipeline_modules.denoising._denoising import \
    _DenoisingModule
from ophys_etl.workflows.well_known_file_types import WellKnownFileType


class DenoisingInferenceModule(_DenoisingModule):
    """Denoising inference module"""
    def __init__(
        self,
        ophys_experiment: OphysExperiment,
        prevent_file_overwrites: bool = True,
        **kwargs
    ):
        super().__init__(
            ophys_experiment=ophys_experiment,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs
        )

        trained_model_file: OutputFile = \
            kwargs['trained_denoising_model_file']
        self._trained_model_path = str(trained_model_file.path)

    @property
    def queue_name(self) -> str:
        return 'DEEPINTERPOLATION_INFERENCE'

    @property
    def inputs(self) -> Dict:
        return {
            "generator_params": {
                "batch_size": 5,
                "name": "OphysGenerator",
                "start_frame": 0,
                "data_path": self._motion_corrected_path
            },
            "inference_params": {
                "model_source": {
                    "local_path": self._trained_model_path
                },
                "n_parallel_workers": 85,
                "rescale": True,
                "save_raw": False,
                "output_file": (
                    str(self.output_path /
                        f'{self.ophys_experiment.id}_denoised_video.h5'))
            },
            "run_uid": self.ophys_experiment.id
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileType.DEEPINTERPOLATION_DENOISED_MOVIE),
                path=(self.output_path /
                      f'{self.ophys_experiment.id}_denoised_video.h5')
            )
        ]

    @property
    def _executable(self) -> str:
        return 'ophys_etl.modules.denoising.inference'
