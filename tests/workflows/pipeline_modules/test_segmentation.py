import datetime
import os
import shutil
from pathlib import Path

import tempfile

from ophys_etl.test_utils.workflow_utils import setup_app_config
from ophys_etl.workflows.workflow_names import WorkflowNameEnum

setup_app_config(
    ophys_workflow_app_config_path=(
            Path(__file__).parent.parent / 'resources' / 'config.yml'),
    test_di_base_model_path=Path(__file__).parent.parent / 'resources' /
    'di_model.h5'
)

from ophys_etl.workflows.db.schemas import OphysROI, OphysROIMaskValue  # noqa E402
from ophys_etl.workflows.output_file import OutputFile  # noqa E402

from ophys_etl.workflows.db.db_utils import save_job_run_to_db  # noqa E402
from sqlmodel import create_engine, Session, select # noqa E402

from ophys_etl.workflows.db.initialize_db import InitializeDBRunner  # noqa E402
from ophys_etl.workflows.pipeline_modules.segmentation import \
    SegmentationModule  # noqa E402
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum # noqa E402
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum # noqa E402
from ophys_etl.test_utils.db_base import MockSQLiteDB

class TestSegmentation(MockSQLiteDB):

    def test_save_metadata_to_db(self):
        _rois_path = \
            Path(__file__).parent / 'resources' / 'rois.json'
        with Session(self._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.SEGMENTATION,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.OPHYS_ROIS),
                        path=_rois_path
                    )
                ],
                ophys_experiment_id='1',
                sqlalchemy_session=session,
                storage_directory='/foo',
                log_path='/foo',
                additional_steps=SegmentationModule.save_rois_to_db,
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING
            )
        with Session(self._engine) as session:
            rois = session.exec(select(OphysROI)).all()
            masks = session.exec(select(OphysROIMaskValue)).all()

        assert len(rois) == 2
        assert rois[0].x == 19
        assert rois[0].y == 326
        assert rois[0].workflow_step_run_id == 1
        assert rois[0].width == 14
        assert rois[0].height == 15

        assert len(masks) == 2
        mask1 = [x for x in masks if x.ophys_roi_id == 1]
        mask2 = [x for x in masks if x.ophys_roi_id == 2]
        assert {(x.row_index, x.col_index) for x in mask1} == {(0, 0)}
        assert {(x.row_index, x.col_index) for x in mask2} == {(0, 1)}
