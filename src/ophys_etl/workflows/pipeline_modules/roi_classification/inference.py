import json
from ast import literal_eval
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple

import pandas as pd
from deepcell.cli.modules import inference
from deepcell.datasets.model_input import ModelInput
from sqlmodel import Session, select

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.db import engine
from ophys_etl.workflows.db.db_utils import get_well_known_file_type
from ophys_etl.workflows.db.schemas import (
    OphysROI,
    ROIClassifierEnsemble,
    ROIClassifierInferenceResults,
    WellKnownFile,
)
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.pipeline_modules.roi_classification.utils.mlflow_utils import ( # noqa E501
    MLFlowRun
)
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import get_latest_workflow_step_run
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class InferenceModule(PipelineModule):
    """Uses trained ROI classifier to classify ROIs"""

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

        thumbnails_dir: OutputFile = kwargs["thumbnails_dir"]

        self._ensemble = self._get_model_ensemble(
            ensemble_id=kwargs["ensemble_id"]
        )

        self._model_inputs_path = self._write_model_inputs_to_disk(
            thumbnails_dir=thumbnails_dir.path
        )

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.ROI_CLASSIFICATION_INFERENCE

    @property
    def inputs(self) -> Dict:
        model_params = self._get_mlflow_model_params()
        return {
            "model_inputs_paths": [self._model_inputs_path],
            "model_params": {
                "use_pretrained_model": model_params["use_pretrained_model"],
                "model_architecture": model_params["model_architecture"],
                "truncate_to_layer": model_params["truncate_to_layer"],
            },
            "model_load_path": self._ensemble[1],
            "save_path": self.output_path,
            "mode": "production",
            "experiment_id": self.ophys_experiment.id,
            "classification_threshold": (
                self._ensemble[0].classification_threshold)
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.ROI_CLASSIFICATION_EXPERIMENT_PREDICTIONS # noqa E501
                ),
                path=(
                    self.output_path
                    / f"{self.ophys_experiment.id}_inference.csv"
                ),
            )
        ]

    @property
    def executable(self) -> ModuleType:
        return inference

    def _write_model_inputs_to_disk(self, thumbnails_dir: Path) -> Path:
        """Creates and writes model inputs to disk

        Parameters
        ----------
        thumbnails_dir
            Path to classifier thumbnail images directory

        Returns
        -------
        Path
            Path where model inputs file is saved
        """
        rois = self._get_rois()
        model_inputs = [
            ModelInput.from_data_dir(
                data_dir=thumbnails_dir,
                experiment_id=self.ophys_experiment.id,
                roi_id=str(roi.id),
                channels=(
                    app_config.pipeline_steps.roi_classification.input_channels
                ),
            )
            for roi in rois
        ]

        model_inputs = [x.to_dict() for x in model_inputs]

        out_path = (
            self.output_path / f"{self.ophys_experiment.id}_model_inputs.json"
        )
        with open(out_path, "w") as f:
            f.write(json.dumps(model_inputs, indent=2))

        return out_path

    @staticmethod
    def save_predictions_to_db(
        output_files: Dict[str, OutputFile],
        session: Session,
        run_id: int,
        ensemble_id: int,
        **kwargs
    ):
        ensemble = InferenceModule._get_model_ensemble(ensemble_id=ensemble_id)

        preds_file = output_files[
            WellKnownFileTypeEnum.ROI_CLASSIFICATION_EXPERIMENT_PREDICTIONS.value # noqa E501
        ]
        preds = pd.read_csv(preds_file.path)

        # renaming so that hyphen doesn't cause problems
        preds.rename(columns={"roi-id": "roi_id"}, inplace=True)

        for pred in preds.itertuples(index=False):
            inference_res = ROIClassifierInferenceResults(
                roi_id=pred.roi_id,
                ensemble_id=ensemble_id,
                score=pred.y_score,
                is_cell=pred.y_score > ensemble[0].classification_threshold
            )
            session.add(inference_res)

    @staticmethod
    def _get_model_ensemble(
            ensemble_id: int
    ) -> Tuple[ROIClassifierEnsemble, str]:
        """

        Parameters
        ----------
        ensemble_id

        Returns
        -------
        Tuple[ROIClassifierEnsemble, str]
        ROIClassifierEnsemble and path to ensemble

        """
        with Session(engine) as session:
            model_file = get_well_known_file_type(
                session=session,
                name=WellKnownFileTypeEnum.ROI_CLASSIFICATION_TRAINED_MODEL,
                workflow=WorkflowNameEnum.ROI_CLASSIFIER_TRAINING,
                workflow_step_name=WorkflowStepEnum.ROI_CLASSIFICATION_TRAINING, # noqa E501
            )
            statement = select(
                ROIClassifierEnsemble, WellKnownFile.path
            ).join(
                WellKnownFile,
                onclause=(WellKnownFile.workflow_step_run_id ==
                          ROIClassifierEnsemble.workflow_step_run_id)
            ).where(
                ROIClassifierEnsemble.id == ensemble_id,
                WellKnownFile.workflow_step_run_id
                == ROIClassifierEnsemble.workflow_step_run_id,
                WellKnownFile.well_known_file_type_id == model_file.id,
            )
            res = session.exec(statement=statement).one()
        return res

    def _get_mlflow_model_params(self) -> Dict:
        """Pulls the mlflow run for `run_id` and fetches the params used
        for that run

        Returns
        -------
        Dict
            The params used to train the model
        """
        run = MLFlowRun(
            mlflow_experiment_name=(
                app_config.pipeline_steps.roi_classification.training.tracking.mlflow_experiment_name # noqa E501
            ),
            run_id=self._ensemble[0].mlflow_run_id
        )
        params = run.run.data.params

        model_params = {
            key.replace("model_params_", ""): value
            for key, value in params.items()
            if key.startswith("model_params")
        }

        # mlflow returns all `params` values as strings
        # Convert to python type
        for k, v in model_params.items():
            try:
                model_params[k] = literal_eval(v)
            except ValueError:
                # It's a string that can't be converted to another type
                pass

        return model_params

    def _get_rois(self) -> List[OphysROI]:
        """
        Returns
        -------
        ROIs from most recent segmentation run for `self.ophys_experiment.id`
        """
        with Session(engine) as session:
            segmentation_run_id = get_latest_workflow_step_run(
                session=session,
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                workflow_step=WorkflowStepEnum.SEGMENTATION,
                ophys_experiment_id=self.ophys_experiment.id,
            )

            rois = session.exec(
                select(OphysROI).where(
                    OphysROI.workflow_step_run_id == segmentation_run_id
                )
            ).all()
            return rois
