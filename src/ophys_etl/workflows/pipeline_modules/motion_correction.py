from typing import List

from ophys_etl.workflows.pipeline_module import PipelineModule, OutputFile


class MotionCorrectionModule(PipelineModule):
    """Wrapper around motion correction module"""

    @property
    def _executable(self) -> str:
        return 'ophys_etl.modules.suite2p_registration'

    @property
    def queue_name(self):
        return 'SUITE2P_MOTION_CORRECTION_QUEUE'

    @property
    def inputs(self):
        module_args = {
            'movie_frame_rate_hz': self._ophys_experiment.movie_frame_rate_hz,
            'suite2p_args': {
                'h5py': str(self._ophys_experiment.storage_directory /
                            self._ophys_experiment.raw_movie_filename)
            },
            'motion_corrected_output': (
                str(self.output_path /
                    f'{self._ophys_experiment.id}_suite2p_motion_output.h5')),
            'motion_diagnostics_output': (
                str(self.output_path /
                    f'{self._ophys_experiment.id}_'
                    f'suite2p_rigid_motion_transform.csv')),
            'max_projection_output': (
                str(self.output_path / f'{self._ophys_experiment.id}_'
                                       f'suite2p_maximum_projection.png')),
            'avg_projection_output': (
                str(self.output_path / f'{self._ophys_experiment.id}_'
                                       f'suite2p_average_projection.png')),
            'registration_summary_output': (
                str(self.output_path / f'{self._ophys_experiment.id}_'
                                       f'suite2p_registration_summary.png')),
            'motion_correction_preview_output': (
                str(self.output_path / f'{self._ophys_experiment.id}_'
                                       f'suite2p_motion_preview.webm'))
        }
        return module_args

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type='MotionCorrectedImageStack',
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_suite2p_motion_output.h5')
            ),
            OutputFile(
                well_known_file_type='OphysMaxIntImage',
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_'
                      f'suite2p_maximum_projection.png')
            ),
            OutputFile(
                well_known_file_type='OphysMotionXyOffsetData',
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_'
                      f'suite2p_rigid_motion_transform.csv')
            ),
            OutputFile(
                well_known_file_type='OphysAverageIntensityProjectionImage',
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_'
                      f'suite2p_average_projection.png')
            ),
            OutputFile(
                well_known_file_type='OphysRegistrationSummaryImage',
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_'
                      f'suite2p_registration_summary.png')
            ),
            OutputFile(
                well_known_file_type='OphysMotionPreview',
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_'
                      f'suite2p_motion_preview.webm')
            ),
        ]
