from pathlib import Path
from unittest.mock import patch

from sqlmodel import Session

from ophys_etl.workflows.db.schemas import (
    MotionCorrectionRun,
    OphysROI,
    OphysROIMaskValue,
    WorkflowStepRun,
)
from ophys_etl.workflows.ophys_experiment import (
    OphysExperiment,
    OphysSession,
    Specimen,
)
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import get_workflow_step_by_name
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from tests.workflows.conftest import MockSQLiteDB


class TestOphysExperiment(MockSQLiteDB):
    def _create_mock_data(self):
        ophys_experiment_id = "1"
        with Session(self._engine) as session:

            # add motion correction run
            motion_correction_workflow_step = get_workflow_step_by_name(
                session=session,
                workflow=WorkflowNameEnum.OPHYS_PROCESSING.value,
                name=WorkflowStepEnum.MOTION_CORRECTION.value,
            )
            motion_correction_run = WorkflowStepRun(
                ophys_experiment_id=ophys_experiment_id,
                workflow_step_id=motion_correction_workflow_step.id,
                log_path="log_path",
                storage_directory="storage_directory",
                start="2023-04-01 00:00:00",
                end="2023-04-01 01:00:00",
            )
            session.add(motion_correction_run)

            # add segmentation run
            segmentation_workflow_step = get_workflow_step_by_name(
                session=session,
                workflow=WorkflowNameEnum.OPHYS_PROCESSING.value,
                name=WorkflowStepEnum.SEGMENTATION.value,
            )
            segmentation_run = WorkflowStepRun(
                ophys_experiment_id=ophys_experiment_id,
                workflow_step_id=segmentation_workflow_step.id,
                log_path="log_path",
                storage_directory="storage_directory",
                start="2023-04-01 00:00:00",
                end="2023-04-01 01:00:00",
            )
            session.add(segmentation_run)

            session.flush()
            motion_correction = MotionCorrectionRun(
                workflow_step_run_id=motion_correction_run.id,
                max_correction_left=10,
                max_correction_right=20,
                max_correction_up=30,
                max_correction_down=40,
            )
            session.add(motion_correction)

            for i in range(2):
                ophys_roi = OphysROI(
                    workflow_step_run_id=segmentation_run.id,
                    x=10,
                    y=20,
                    width=30,
                    height=40,
                    is_in_motion_border=False,
                    is_small_size=False
                )
                session.add(ophys_roi)
                session.flush()

                ophys_roi_mask_value = OphysROIMaskValue(
                    ophys_roi_id=ophys_roi.id,
                    row_index=5,
                    col_index=6,
                )
                session.add(ophys_roi_mask_value)

                ophys_roi_mask_value = OphysROIMaskValue(
                    ophys_roi_id=ophys_roi.id,
                    row_index=6,
                    col_index=8,
                )
                session.add(ophys_roi_mask_value)
            session.commit()

    def setup(self):
        self._initializeDB()
        self._create_mock_data()
        self.ophys_experiment = OphysExperiment(
            id="1",
            session=OphysSession(id="2", specimen=Specimen("1")),
            specimen=Specimen(id="3"),
            storage_directory=Path("/storage_dir"),
            raw_movie_filename=Path("mov.h5"),
            movie_frame_rate_hz=11.0,
        )

    def test__roi_metadata(self):
        with patch(
            "ophys_etl.workflows.ophys_experiment.engine", self._engine
        ):
            roi_metadata = [x.to_dict() for x in
                            self.ophys_experiment.rois]
            assert len(roi_metadata) == 2
            for i in range(2):
                assert roi_metadata[i]["x"] == 10
                assert roi_metadata[i]["y"] == 20
                assert roi_metadata[i]["width"] == 30
                assert roi_metadata[i]["height"] == 40
                assert roi_metadata[i]["mask"][5][6] == 1
                assert len(roi_metadata[i]["mask"]) == 40
                assert len(roi_metadata[i]["mask"][0]) == 30

    def test_motion_border(self):
        with patch(
            "ophys_etl.workflows.ophys_experiment.engine", self._engine
        ):
            motion_border = self.ophys_experiment.motion_border.to_dict()
            assert motion_border["x0"] == 10
            assert motion_border["x1"] == 20
            assert motion_border["y0"] == 30
            assert motion_border["y1"] == 40
