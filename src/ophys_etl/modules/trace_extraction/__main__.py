import argschema

from ophys_etl.modules.trace_extraction.utils import extract_traces
from ophys_etl.modules.trace_extraction.schemas import (
        TraceExtractionInputSchema, TraceExtractionOutputSchema)


class TraceExtraction(argschema.ArgSchemaParser):
    default_schema = TraceExtractionInputSchema
    default_output_schema = TraceExtractionOutputSchema

    def run(self):
        self.logger.name = type(self).__name__
        output = extract_traces(
                self.args['motion_corrected_stack'],
                self.args['motion_border'],
                self.args['storage_directory'],
                self.args['rois'])
        self.output(output, indent=2)


if __name__ == '__main__':  # pragma: nocover
    te = TraceExtraction()
    te.run()
