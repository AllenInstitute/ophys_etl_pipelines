import os
from pathlib import Path

import yaml

from ophys_etl.test_utils.workflow_utils import setup_app_config
from ophys_etl.workflows.app_config.app_config import AppConfig, \
    _DenoisingInference
from ophys_etl.workflows.app_config.slurm import SlurmSettings


class TestAppConfig:
    def setup_method(self):
        setup_app_config(
            ophys_workflow_app_config_path=(
                    Path(__file__).parent.parent / "resources" / "config.yml"
            ),
            test_di_base_model_path=(
                    Path(__file__).parent.parent / "resources" / "di_model.h5"
            ),
        )
        config_path = os.environ["OPHYS_WORKFLOW_APP_CONFIG_PATH"]
        with open(config_path) as f:
            config = yaml.safe_load(f)

        denoising_conf = config["pipeline_steps"]["denoising"]
        denoising_conf['finetuning']["base_model_path"] = os.environ[
            "TEST_DI_BASE_MODEL_PATH"
        ]
        self._config = config

    def test_set_slurm_settings_some_overridden(self):
        config = self._config
        denoising_inference = \
            config['pipeline_steps']['denoising']['inference']
        denoising_inference['slurm_settings'] = {'time': 720}
        app_config = AppConfig(**config)
        assert (app_config.pipeline_steps.denoising.inference.slurm_settings.
                time == 720)
        assert (app_config.pipeline_steps.denoising.inference.slurm_settings.
                mem == _DenoisingInference._default_slurm_settings.mem)

    def test_set_slurm_settings_none_overridden(self):
        config = self._config
        denoising_inference = \
            config['pipeline_steps']['denoising']['inference']
        del denoising_inference['slurm_settings']
        app_config = AppConfig(**config)
        assert (app_config.pipeline_steps.denoising.inference.slurm_settings.
                time == _DenoisingInference._default_slurm_settings.time)
        assert (app_config.pipeline_steps.denoising.inference.slurm_settings.
                mem == _DenoisingInference._default_slurm_settings.mem)

    def test_set_slurm_settings_gets_defaults(self):
        config = self._config
        default_slurm_settings = SlurmSettings()
        app_config = AppConfig(**config)
        # trace_extraction doesn't override defaults
        assert (app_config.pipeline_steps.trace_extraction.slurm_settings.
                time == default_slurm_settings.time)
        assert (app_config.pipeline_steps.trace_extraction.slurm_settings.
                mem == default_slurm_settings.mem)
