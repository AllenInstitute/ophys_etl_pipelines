from typing import List

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.pipeline_module import OutputFile
from ophys_etl.workflows.pipeline_modules.denoising._denoising import \
    _DenoisingModule


class DenoisingFinetuningModule(_DenoisingModule):
    """Denoising Finetuning module"""
    @property
    def _executable(self) -> str:
        return 'ophys_etl.modules.denoising.fine_tuning'

    @property
    def queue_name(self) -> str:
        return 'DEEPINTERPOLATION_FINETUNING'

    @property
    def inputs(self):
        return {
            "data_split_params": {
                "ophys_experiment_id": self.ophys_experiment.id,
                "dataset_output_dir": str(self.output_path),
                "movie_path": self._motion_corrected_path,
                "downsample_frac": (
                    app_config.pipeline_steps.denoising.downsample_frac)
            },
            "finetuning_params": {
                "apply_learning_decay": False,
                "caching_validation": False,
                "epochs_drop": 5,
                "learning_rate": 0.0001,
                "loss": "mean_squared_error",
                "model_source": {
                    "local_path": (
                        app_config.pipeline_steps.denoising.base_model_path)
                },
                "model_string": "",
                "multi_gpus": False,
                "name": "transfer_trainer",
                "nb_times_through_data": 1,
                "nb_workers": 15,
                "output_dir": str(self.output_path),
                "period_save": 1,
                "steps_per_epoch": 20
            },
            "generator_params": {
                "batch_size": 5,
                "name": "MovieJSONGenerator",
                "post_frame": 30,
                "pre_frame": 30,
                "pre_post_omission": 0
            },
            "log_level": "INFO",
            "output_full_args": True,
            "test_generator_params": {
                "batch_size": 5,
                "name": "MovieJSONGenerator",
                "post_frame": 30,
                "pre_frame": 30,
                "pre_post_omission": 0
            },
            "run_uid": self.ophys_experiment.id
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type='DeepInterpolationFinetunedModel',
                path=(self.output_path /
                      f'{self.ophys_experiment.id}_'
                      f'mean_squared_error_transfer_model.h5')
            )
        ]
