import argschema
import json
from marshmallow import ValidationError

from ophys_etl.schemas import DenseROISchema
from ophys_etl.utils.motion_border import MaxFrameShift
from ophys_etl.utils.rois import dense_to_extract
from ophys_etl.modules.binarize_extract_bridge.schemas import (
        BridgeInputSchema, BridgeOutputSchema)


class BinarizeToExtractBridge(argschema.ArgSchemaParser):
    """
    This module can bridge the output of
    'ophys_etl_pipelines/src/ophys_etl/transforms/convert_rois.py'
    to the input of
    https://github.com/AllenInstitute/AllenSDK/blob/7e60bc5a811f76750d22a507f449621a0784e6bd/allensdk/brain_observatory/ophys/trace_extraction/_schemas.py#L33-L41  # noqa
    In production, this purpose is served by a LIMS strategy. This python
    bridge is here as a helper for running the pipeline manually outside
    of production
    """
    default_schema = BridgeInputSchema
    default_output_schema = BridgeOutputSchema

    def run(self):
        self.logger.name = type(self).__name__

        with open(self.args['input_file'], "r") as f:
            compatible_rois = json.load(f)

        # validate ROIs
        errors = DenseROISchema(many=True).validate(compatible_rois)
        if any(errors):
            raise ValidationError(f"Schema validation errors: {errors}")

        # read the motion border and check they are all the same
        for i, roi in enumerate(compatible_rois):
            ishift = MaxFrameShift(
                up=roi['max_correction_up'],
                down=roi['max_correction_down'],
                left=roi['max_correction_left'],
                right=roi['max_correction_right'])
            if i == 0:
                frame_shift = ishift
            else:
                assert ishift == frame_shift

        frame_shift_dict = {
                'y1': frame_shift.up,
                'y0': frame_shift.down,
                'x0': frame_shift.right,
                'x1': frame_shift.left
                }

        converted_rois = [dense_to_extract(roi) for roi in compatible_rois]

        output = {
                'log_0': self.args['motion_correction_values'],
                'motion_corrected_stack': self.args['motion_corrected_video'],
                'storage_directory': self.args['storage_directory'],
                'rois': converted_rois,
                'motion_border': frame_shift_dict
                }

        self.output(output, indent=2)
        self.logger.info(f"transformed {self.args['input_file']} to "
                         f"{self.args['output_json']}")


if __name__ == "__main__":  # pragma: nocover
    bridge = BinarizeToExtractBridge()
    bridge.run()
