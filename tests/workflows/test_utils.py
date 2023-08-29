from ophys_etl.workflows.pipeline_module import PipelineModule

class TestPipelineMOduleBase:

    MODULE_CLASS = None # Set this in the subclass
    
    @classmethod
    def setup_class(cls):
        if cls.MODULE_CLASS is None:
            raise ValueError("Must set MODULE_CLASS in subclass")

        if not issubclass(cls.MODULE_CLASS, PipelineModule):
            raise ValueError("MODULE_CLASS must be a subclass of PipelineModule")
        
    def test_init(self):
        mod = self.MODULE_CLASS()
        assert isinstance(mod, self.MODULE_CLASS)
    
        

        