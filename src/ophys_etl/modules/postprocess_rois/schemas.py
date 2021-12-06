from pathlib import Path
from marshmallow.validate import Range
from argschema import ArgSchema
from argschema.fields import Float, InputFile, Int, OutputFile, Str, Bool


class PostProcessROIsInputSchema(ArgSchema):
    suite2p_stat_path = Str(
        required=True,
        validate=lambda x: Path(x).exists(),
        description=("Path to s2p output stat file containing ROIs generated "
                     "during source extraction"))
    motion_corrected_video = Str(
        required=True,
        validate=lambda x: Path(x).exists(),
        description=("Path to motion corrected video file *.h5"))
    motion_correction_values = InputFile(
        default=None,
        allow_none=True,
        description=("Path to motion correction values for each frame "
                     "stored in .csv format. This .csv file is expected to"
                     "have a header row of either:\n"
                     "['framenumber','x','y','correlation','kalman_x',"
                     "'kalman_y']\n['framenumber','x','y','correlation',"
                     "'input_x','input_y','kalman_x',"
                     "'kalman_y','algorithm','type']\n"
                     "If not specified, an empty motion border will be used "
                     "(i.e. pixels all the way up to the edge of the field "
                     "of view will be considered valid)"))
    output_json = OutputFile(
        required=True,
        description=("Path to a file to write output data."))
    maximum_motion_shift = Float(
        missing=30.0,
        required=False,
        allow_none=False,
        description=("The maximum allowable motion shift for a frame in pixels"
                     " before it is considered an anomaly and thrown out of "
                     "processing"))
    abs_threshold = Float(
        missing=None,
        required=False,
        allow_none=True,
        description=("The absolute threshold to binarize ROI masks against. "
                     "If not provided will use quantile to generate "
                     "threshold."))
    binary_quantile = Float(
        missing=0.1,
        validate=Range(min=0, max=1),
        description=("The quantile against which an ROI is binarized. If not "
                     "provided will use default function value of 0.1."))
    npixel_threshold = Int(
        default=50,
        required=False,
        description=("ROIs with fewer pixels than this will be labeled as "
                     "invalid and small size."))
    aspect_ratio_threshold = Float(
        default=0.2,
        required=False,
        description=("ROIs whose aspect ratio is <= this value are "
                     "not recorded. This captures a large majority of "
                     "Suite2P-created artifacts from motion border"))
    morphological_ops = Bool(
        default=True,
        required=False,
        description=("whether to perform morphological operations after "
                     "binarization. ROIs that are washed away to empty "
                     "after this operation are eliminated from the record. "
                     "This can apply to ROIs that were previously labeled "
                     "as small size, for example."))
