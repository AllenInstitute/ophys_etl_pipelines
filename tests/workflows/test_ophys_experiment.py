import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
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
    OphysContainer
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

            motion_correction_run1 = WorkflowStepRun(
                ophys_experiment_id=ophys_experiment_id,
                workflow_step_id=motion_correction_workflow_step.id,
                log_path="log_path",
                storage_directory="storage_directory",
                start="2023-04-01 00:00:00",
                end="2023-04-01 01:00:00",
            )
            session.add(motion_correction_run1)

            motion_correction_run2 = WorkflowStepRun(
                ophys_experiment_id="2",
                workflow_step_id=motion_correction_workflow_step.id,
                log_path="log_path",
                storage_directory="storage_directory",
                start="2023-05-01 00:00:00",
                end="2023-05-01 01:00:00",
            )
            session.add(motion_correction_run2)

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
                workflow_step_run_id=motion_correction_run1.id,
                max_correction_left=10,
                max_correction_right=20,
                max_correction_up=30,
                max_correction_down=40,
            )
            session.add(motion_correction)

            motion_correction = MotionCorrectionRun(
                workflow_step_run_id=motion_correction_run2.id,
                max_correction_left=100,
                max_correction_right=200,
                max_correction_up=300,
                max_correction_down=400,
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
            id=1,
            session=OphysSession(id=2, specimen=Specimen("1")),
            container=OphysContainer(id=1, specimen=Specimen("1")),
            specimen=Specimen(id="3"),
            storage_directory=Path("/storage_dir"),
            raw_movie_filename=Path("mov.h5"),
            movie_frame_rate_hz=11.0,
            equipment_name='MESO.1',
            full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt"
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


class TestOphysExperimentGroup(MockSQLiteDB):
    def setup(self):
        super().setup()

        with Session(self._engine) as session:
            mock_completed_segmentation = [1, 2, 3]
            for oe_id in mock_completed_segmentation:
                save_job_run_to_db(
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[],
                    sqlalchemy_session=session,
                    ophys_experiment_id=oe_id,
                    storage_directory="foo",
                    log_path="foo",
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    workflow_step_name=WorkflowStepEnum.SEGMENTATION
                )

        self.ophys_experiment = OphysExperiment(
            id=1,
            session=OphysSession(id=2, specimen=Specimen("1")),
            container=OphysContainer(id=1, specimen=Specimen("1")),
            specimen=Specimen(id="3"),
            storage_directory=Path("/storage_dir"),
            raw_movie_filename=Path("mov.h5"),
            movie_frame_rate_hz=11.0,
            equipment_name='MESO.1',
            full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt"
        )

    @patch.object(OphysSession, 'get_ophys_experiment_ids')
    @patch.object(OphysContainer, 'get_ophys_experiment_ids')
    @pytest.mark.parametrize('experiment_group_name', ('session', 'container'))
    def test_has_completed_workflow_step(
            self,
            mock_container_exp_ids,
            mock_session_exp_ids,
            experiment_group_name
    ):
        mock_container_exp_ids.return_value = [1, 2]
        mock_session_exp_ids.return_value = [1, 2]

        with patch('ophys_etl.workflows.ophys_experiment.engine',
                   new=self._engine):
            if experiment_group_name == 'session':
                experiment_group = self.ophys_experiment.session
            else:
                experiment_group = self.ophys_experiment.container
            is_complete = \
                experiment_group.has_completed_workflow_step(
                    workflow_step=WorkflowStepEnum.SEGMENTATION
                )
        assert is_complete

    @patch.object(OphysSession, 'get_ophys_experiment_ids')
    @patch.object(OphysContainer, 'get_ophys_experiment_ids')
    @pytest.mark.parametrize('experiment_group_name', ('session', 'container'))
    def test_has_not_completed_workflow_step(
            self,
            mock_container_exp_ids,
            mock_session_exp_ids,
            experiment_group_name
    ):
        mock_container_exp_ids.return_value = [3, 4]
        mock_session_exp_ids.return_value = [3, 4]

        with patch('ophys_etl.workflows.ophys_experiment.engine',
                   new=self._engine):
            if experiment_group_name == 'session':
                experiment_group = self.ophys_experiment.session
            else:
                experiment_group = self.ophys_experiment.container
            is_complete = \
                experiment_group.has_completed_workflow_step(
                    workflow_step=WorkflowStepEnum.SEGMENTATION
                )
        assert not is_complete
