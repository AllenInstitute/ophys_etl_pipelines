from ophys_etl.workflows.pipeline_module import PipelineModule
from tests.workflows.conftest import MockSQLiteDB
from ophys_etl.workflows.ophys_experiment import (
    OphysExperiment,  # noqa #402
    OphysSession,
    Specimen, OphysContainer,
)
from pathlib import Path

class TestPipelineModuleBase(MockSQLiteDB):

    __test__ = False
    MODULE_CLASS = None # Set this in the subclass
    
    @classmethod
    def setup_class(cls):

        if cls.MODULE_CLASS is None:
            raise ValueError("Must set MODULE_CLASS in subclass")

        if not issubclass(cls.MODULE_CLASS, PipelineModule):
            raise ValueError("MODULE_CLASS must be a subclass of PipelineModule")
        cls.ophys_experiment = OphysExperiment(
                    id=1,
                    session=OphysSession(id=2, specimen=Specimen("1")),
                    container=OphysContainer(id=1, specimen=Specimen("1")),
                    specimen=Specimen(id="3"),
                    storage_directory=Path("/storage_dir"),
                    raw_movie_filename=Path("mov.h5"),
                    movie_frame_rate_hz=11.0,
                    equipment_name='MESO.1',
                    full_genotype="abcd",
                )
        

    def test_init(self):
        mod = self.MODULE_CLASS(ophys_experiment = self.ophys_experiment,
                                docker_tag="main")
        assert isinstance(mod, self.MODULE_CLASS)