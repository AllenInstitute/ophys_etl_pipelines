import datetime
from pathlib import Path
from unittest.mock import patch, PropertyMock

from ophys_etl.workflows.pipeline_modules.motion_correction import \
    MotionCorrectionModule

from ophys_etl.workflows.db.schemas import OphysROI, OphysROIMaskValue, \
    MotionCorrectionRun

from ophys_etl.workflows.pipeline_modules.segmentation import \
    SegmentationModule

from ophys_etl.workflows.output_file import OutputFile

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from sqlmodel import Session, select

from ophys_etl.workflows.ophys_experiment import OphysSession, Specimen, \
    OphysExperiment, ImagingPlaneGroup

from ophys_etl.workflows.pipeline_modules.decrosstalk import \
    DecrosstalkModule, DECROSSTALK_FLAGS
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from tests.workflows.conftest import MockSQLiteDB


class TestDecrosstalk(MockSQLiteDB):
    @classmethod
    def setup_class(cls):
        cls._experiment_ids = ['oe_1', 'oe_2']

    def setup(self):
        super().setup()

        xy_offset_path = (
            Path(__file__).parent / "resources" / "rigid_motion_transform.csv"
        )
        with Session(self._engine) as session:
            for oe_id in self._experiment_ids:
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
                                WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA
                            ),
                            path=xy_offset_path
                        )
                    ],
                    ophys_experiment_id=oe_id,
                    sqlalchemy_session=session,
                    storage_directory="/foo",
                    log_path="/foo",
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    validate_files_exist=False,
                    additional_steps=MotionCorrectionModule.save_metadata_to_db
                )

                save_job_run_to_db(
                    workflow_step_name=WorkflowStepEnum.TRACE_EXTRACTION,
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[
                        OutputFile(
                            well_known_file_type=(
                                WellKnownFileTypeEnum.ROI_TRACE
                            ),
                            path=Path(f'{oe_id}_roi_traces.h5'),
                        )
                    ],
                    ophys_experiment_id=oe_id,
                    sqlalchemy_session=session,
                    storage_directory="/foo",
                    log_path="/foo",
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    validate_files_exist=False
                )

    @patch.object(OphysExperiment, 'from_id')
    @patch.object(OphysSession, 'ophys_experiment_ids',
                  new_callable=PropertyMock)
    @patch.object(OphysExperiment, 'motion_border',
                  new_callable=PropertyMock)
    @patch.object(OphysExperiment, 'rois',
                  new_callable=PropertyMock)
    def test_inputs(self,
                    mock_rois,
                    mock_motion_border,
                    mock_ophys_session_oe_ids,
                    mock_ophys_experiment_from_id
                    ):
        ophys_session = OphysSession(
            id='session_1',
            specimen=Specimen(id='specimen_1')
        )

        mock_roi = OphysROI(
                    id=1,
                    x=0,
                    y=0,
                    width=2,
                    height=1,
                    is_in_motion_border=False
                )
        mock_roi._mask_values = [
            OphysROIMaskValue(
                id=1,
                ophys_roi_id=1,
                row_index=0,
                col_index=0
            )
        ]
        mock_rois.return_value = [mock_roi]
        mock_motion_border.return_value = MotionCorrectionRun(
            max_correction_left=1,
            max_correction_right=3,
            max_correction_up=2,
            max_correction_down=4
        )
        mock_ophys_session_oe_ids.return_value = self._experiment_ids
        mock_ophys_experiment_from_id.side_effect = \
            lambda id: OphysExperiment(
                id=id,
                movie_frame_rate_hz=1,
                raw_movie_filename=Path('foo'),
                session=ophys_session,
                specimen=ophys_session.specimen,
                storage_directory=Path('foo'),
                imaging_plane_group=ImagingPlaneGroup(
                    id=0 if id == 'oe_1' else 1,
                    group_order=0 if id == 'oe_1' else 1
                ),
                full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt",
                equipment_name='MESO.1'
            )

        mod = DecrosstalkModule(
            docker_tag='main',
            ophys_session=ophys_session
        )

        with patch('ophys_etl.workflows.pipeline_modules.decrosstalk.engine',
                   new=self._engine):
            obtained_inputs = mod.inputs

        expected_inputs = {
            'ophys_session_id': ophys_session.id,
            'qc_output_dir': (
                    ophys_session.output_dir / 'DECROSSTALK' / mod.now_str),
            'coupled_planes': [
                {
                    'ophys_imaging_plane_group_id': (
                        0 if self._experiment_ids[i] == 'oe_1' else 1),
                    'group_order': (
                        0 if self._experiment_ids[i] == 'oe_1' else 1
                    ),
                    'planes': [
                        {
                            'ophys_experiment_id': self._experiment_ids[i],
                            'motion_corrected_stack': (
                                f'{self._experiment_ids[i]}_'
                                f'motion_correction.h5'),
                            'maximum_projection_image_file': (
                                f'{self._experiment_ids[i]}_max_proj.png'
                            ),
                            'output_roi_trace_file': (
                                mod.output_path /
                                f'ophys_experiment_{self._experiment_ids[i]}' /
                                'roi_traces.h5'
                            ),
                            'output_neuropil_trace_file': (
                                mod.output_path /
                                f'ophys_experiment_{self._experiment_ids[i]}' /
                                'neuropil_traces.h5'
                            ),
                            'motion_border': (
                                mock_motion_border.return_value.to_dict()),
                            'rois': [
                                x.to_dict() for x in mock_rois.return_value]
                        }
                    ]
                }
                for i in range(len(self._experiment_ids))]
        }
        assert obtained_inputs == expected_inputs

    @patch.object(OphysExperiment, 'from_id')
    def test_save_decrosstalk_flags_to_db(
            self,
            mock_ophys_experiment_from_id
    ):
        ophys_session = OphysSession(
            id='session_1',
            specimen=Specimen(id='specimen_1')
        )

        mock_ophys_experiment_from_id.side_effect = \
            lambda id: OphysExperiment(
                id=id,
                movie_frame_rate_hz=1,
                raw_movie_filename=Path('foo'),
                session=ophys_session,
                specimen=ophys_session.specimen,
                storage_directory=Path('foo'),
                full_genotype="Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt",
                equipment_name='MESO.1'
            )

        # 1. Save segmentation run
        _rois_path = Path(__file__).parent / "resources" / "rois.json"

        for oe_id in self._experiment_ids:
            with Session(self._engine) as session:
                with patch(
                        'ophys_etl.workflows.ophys_experiment.engine',
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
                                path=_rois_path,
                            )
                        ],
                        ophys_experiment_id=oe_id,
                        sqlalchemy_session=session,
                        storage_directory="/foo",
                        log_path="/foo",
                        additional_steps=SegmentationModule.save_rois_to_db,
                        workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    )

        # 2. Save decrosstalk run
        with patch('ophys_etl.workflows.ophys_experiment.engine',
                   new=self._engine):
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.DECROSSTALK,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[
                    OutputFile(
                        well_known_file_type=WellKnownFileTypeEnum
                        .DECROSSTALK_FLAGS,
                        path=(Path(__file__).parent / "resources" /
                              "decrosstalk_output.json")
                    )
                ],
                ophys_session_id=ophys_session.id,
                sqlalchemy_session=session,
                storage_directory="/foo",
                log_path="/foo",
                additional_steps=(
                    DecrosstalkModule.save_decrosstalk_flags_to_db),
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
            )

        # 3. Try fetch decrosstalk flags
        with Session(self._engine) as session:
            rois = session.exec(select(OphysROI)).all()

        expected_flags = {
            3: ['decrosstalk_invalid_raw',
                'decrosstalk_invalid_unmixed',
                'decrosstalk_ghost'],
            4: ['decrosstalk_invalid_raw',
                'decrosstalk_invalid_unmixed']
        }

        for roi in rois:
            flags = expected_flags.get(roi.id, [])
            for flag in DECROSSTALK_FLAGS:
                if flag in flags:
                    assert getattr(roi, f'is_{flag}')
                else:
                    assert getattr(roi, f'is_{flag}') is False
