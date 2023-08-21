import datetime
import tempfile
from pathlib import Path
from unittest.mock import patch, PropertyMock

import json

from ophys_etl.workflows.db.schemas import NwayCellMatch
from ophys_etl.workflows.pipeline_modules.nway_cell_matching import \
    NwayCellMatchingModule

from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    OphysSession, Specimen, OphysContainer

from ophys_etl.workflows.pipeline_modules.segmentation import \
    SegmentationModule

from ophys_etl.workflows.pipeline_modules.motion_correction import \
    MotionCorrectionModule

from ophys_etl.workflows.workflow_names import WorkflowNameEnum

from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum

from ophys_etl.workflows.output_file import OutputFile

from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from sqlmodel import Session, select

from tests.workflows.conftest import MockSQLiteDB


class TestNwayCellMatchingModule(MockSQLiteDB):
    def setup(self):
        super().setup()
        self._experiment_ids = [1, 2]

        xy_offset_path = (
            Path(__file__).parent / "resources" / "rigid_motion_transform.csv"
        )
        rois_path = Path(__file__).parent / "resources" / "rois.json"

        with open(rois_path) as f:
            rois = json.load(f)

        self._dummy_matches = {
            'nway_matches': [
                [x['id'] for x in rois]
            ]
        }
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)
        with Session(self._engine) as session:
            for oe_id in self._experiment_ids:
                # create dummy file
                with open(self._tmp_dir / f'{oe_id}_avg_proj.png', 'w') as f:
                    f.write('')

                save_job_run_to_db(
                    workflow_step_name=WorkflowStepEnum.MOTION_CORRECTION,
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[
                        OutputFile(
                            well_known_file_type=(
                                WellKnownFileTypeEnum.
                                MOTION_CORRECTED_IMAGE_STACK
                            ),
                            path=Path(f'{oe_id}_motion_correction.h5'),
                        ),
                        OutputFile(
                            well_known_file_type=(
                                WellKnownFileTypeEnum.
                                MAX_INTENSITY_PROJECTION_IMAGE
                            ),
                            path=Path(f'{oe_id}_max_proj.png'),
                        ),
                        OutputFile(
                            well_known_file_type=(
                                WellKnownFileTypeEnum.
                                AVG_INTENSITY_PROJECTION_IMAGE
                            ),
                            path=self._tmp_dir / f'{oe_id}_avg_proj.png',
                        ),
                        OutputFile(
                            well_known_file_type=(
                                WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA
                            ),
                            path=xy_offset_path
                        )
                    ],
                    ophys_experiment_id=str(oe_id),
                    sqlalchemy_session=session,
                    storage_directory="/foo",
                    log_path="/foo",
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    validate_files_exist=False,
                    additional_steps=MotionCorrectionModule.save_metadata_to_db
                )

                with patch.object(OphysExperiment,
                                  'from_id') as mock_oe_from_id:
                    mock_oe_from_id.return_value = OphysExperiment(
                        id=oe_id,
                        movie_frame_rate_hz=11.0,
                        raw_movie_filename=Path('foo'),
                        session=OphysSession(id=1,
                                             specimen=Specimen(id='1')),
                        container=OphysContainer(
                            id=1, specimen=Specimen(id='1')),
                        specimen=Specimen(id='1'),
                        storage_directory=Path('foo'),
                        full_genotype="abcd",
                        equipment_name='MESO.1'
                    )
                    with patch('ophys_etl.workflows.ophys_experiment.engine',
                               new=self._engine):
                        save_job_run_to_db(
                            workflow_step_name=WorkflowStepEnum.SEGMENTATION,
                            start=datetime.datetime.now(),
                            end=datetime.datetime.now(),
                            module_outputs=[
                                OutputFile(
                                    well_known_file_type=(
                                        WellKnownFileTypeEnum.OPHYS_ROIS
                                    ),
                                    path=rois_path,
                                )
                            ],
                            ophys_experiment_id=str(oe_id),
                            sqlalchemy_session=session,
                            storage_directory="/foo",
                            log_path="/foo",
                            workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                            validate_files_exist=False,
                            additional_steps=SegmentationModule.save_rois_to_db
                        )

    @patch.object(OphysExperiment, 'from_id')
    @patch.object(OphysSession, 'output_dir',
                  new_callable=PropertyMock)
    @patch.object(NwayCellMatchingModule, 'output_path',
                  new_callable=PropertyMock)
    def test_input(
        self,
        mock_output_path,
        mock_output_dir,
        mock_ophys_experiment_from_id
    ):
        """Smoke test that we can construct the inputs"""
        mock_ophys_experiment_from_id.side_effect = \
            lambda id: OphysExperiment(
                id=id,
                movie_frame_rate_hz=1,
                raw_movie_filename=Path('foo'),
                session=OphysSession(id=1, specimen=Specimen(id='1')),
                container=OphysContainer(id=1, specimen=Specimen(id='1')),
                specimen=Specimen(id='1'),
                storage_directory=Path('foo'),
                full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt",
                equipment_name='MESO.1'
            )
        mock_output_dir.return_value = self.temp_dir
        mock_output_path.return_value = self.temp_dir

        with patch('ophys_etl.workflows.pipeline_modules.nway_cell_matching'
                   '.engine', new=self._engine):
            with patch(
                    'ophys_etl.workflows.ophys_experiment.engine',
                    new=self._engine):
                with patch.object(OphysContainer, 'get_ophys_experiment_ids',
                                  return_value=self._experiment_ids):
                    mod = NwayCellMatchingModule(
                        docker_tag='main',
                        ophys_container=OphysContainer(id='1', specimen=Specimen(id='1'))
                    )
                    mod.inputs

    def test_save_matches_to_db(self):
        matches_path = self._tmp_dir / 'matches.json'
        with open(matches_path, 'w') as f:
            f.write(json.dumps(self._dummy_matches))

        with Session(self._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.NWAY_CELL_MATCHING,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[
                    OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.NWAY_CELL_MATCHING_METADATA),
                        path=matches_path
                    )
                ],
                ophys_container_id='1',
                sqlalchemy_session=session,
                storage_directory="/foo",
                log_path="/foo",
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                validate_files_exist=False,
                additional_steps=NwayCellMatchingModule.save_matches_to_db
            )

        with Session(self._engine) as session:
            matches = session.exec(
                select(NwayCellMatch)
            ).all()

        assert len(matches) == 2
        assert all([
            match.match_id == f'{matches[0].nway_cell_matching_run_id}_0'
            for match in matches
        ])
    
    def teardown(self):
        self.temp_dir_obj.cleanup()
