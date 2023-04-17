from abc import ABC

from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule


class _DenoisingModule(PipelineModule, ABC):
    """Denoising Module. Abstract base class for denoising modules"""
    def __init__(
        self,
        ophys_experiment: OphysExperiment,
        prevent_file_overwrites: bool = True,
        docker_tag: str = 'main',
        **kwargs
    ):
        super().__init__(
            ophys_experiment=ophys_experiment,
            prevent_file_overwrites=prevent_file_overwrites,
            docker_tag=docker_tag
        )

        motion_corrected_ophys_movie: OutputFile = \
            kwargs['motion_corrected_ophys_movie_file']
        self._motion_corrected_path = str(motion_corrected_ophys_movie.path)
