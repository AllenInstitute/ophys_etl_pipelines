from pathlib import Path
from typing import List, Dict

import json

from deepcell.datasets.model_input import ModelInput
from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.workflows.ophys_experiment import OphysExperiment

from ophys_etl.workflows.pipeline_module import PipelineModule, OutputFile
from ophys_etl.workflows.pipeline_modules.roi_classification.utils\
    .mlflow_utils \
    import \
    MLFlowRun


class InferenceModule(PipelineModule):
    """Uses trained ROI classifier to classify ROIs"""
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

        thumbnails_dir: OutputFile = kwargs['thumbnails_dir']
        rois_file: OutputFile = kwargs['rois_file']

        self._model_inputs_path = self._write_model_inputs_to_disk(
            rois_path=rois_file.path,
            thumbnails_dir=thumbnails_dir.path
        )

    @property
    def queue_name(self) -> str:
        return 'ROI_CLASSIFICATION_INFERENCE'

    @property
    def inputs(self) -> Dict:
        model_params = self._get_mlflow_model_params()
        return {
            'model_inputs_path': self._model_inputs_path,
            'model_params': {
                'use_pretrained_model': model_params['use_pretrained_model'],
                'model_architecture': model_params['model_architecture'],
                'truncate_to_layer': model_params['truncate_to_layer']
            }
        }

    @property
    def outputs(self) -> List[OutputFile]:
        pass

    @property
    def _executable(self) -> str:
        pass

    def _write_model_inputs_to_disk(
        self,
        rois_path: Path,
        thumbnails_dir: Path
    ) -> Path:
        """Creates and writes model inputs to disk

        Parameters
        ----------
        rois_path
            Path to rois
        thumbnails_dir
            Path to classifier thumbnail images directory

        Returns
        -------
        Path
            Path where model inputs file is saved
        """
        with open(rois_path) as f:
            rois = json.load(f)

        model_inputs = [
            ModelInput.from_data_dir(
                data_dir=thumbnails_dir,
                experiment_id=self.ophys_experiment.id,
                roi_id=roi['id'],
                channels=(
                    app_config.pipeline_steps.roi_classification.
                    input_channels)
            )
            for roi in rois]

        model_inputs = [x.to_dict() for x in model_inputs]

        out_path = \
            self.output_path / f'{self.ophys_experiment.id}_model_inputs.json'
        with open(out_path, 'w') as f:
            f.write(json.dumps(model_inputs, indent=2))

        return out_path

    @staticmethod
    def _get_mlflow_model_params() -> Dict:
        """Pulls the mlflow run for `run_id` and fetches the params used
        for that run

        Returns
        -------
        Dict
            The params used to train the model
        """
        run = MLFlowRun(
            run_name=(app_config.pipeline_steps.roi_classification.tracking.
                      mlflow_run_name)
        )
        params = run.run.data.params

        model_params = {
            param['key'].replace('model_params_', ''): param['value']
            for param in params
            if param['key'].startswith('model_params')
        }
        return model_params
