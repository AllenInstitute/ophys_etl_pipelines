import h5py
import boto3
import json
import math
import joblib
import numpy as np
from urllib.parse import urlparse
from scipy.sparse import coo_matrix
from argschema import ArgSchema, ArgSchemaParser, fields
from ophys_etl.schemas.fields import H5InputFile
from croissant.features import FeatureExtractor
from botocore.errorfactory import ClientError
from marshmallow import post_load, ValidationError, Schema, validates
import marshmallow.fields as mm_fields
import warnings
import os.path


NOT_CELL_EXCLUSION_LABEL = -99999


class RoiJsonSchema(Schema):
    """ Schema of individual ROI records """
    x = mm_fields.Int(required=True)
    y = mm_fields.Int(required=True)
    width = mm_fields.Int(required=True)
    height = mm_fields.Int(required=True)
    exclusion_labels = mm_fields.List(mm_fields.Int(), required=True)
    mask_matrix = mm_fields.List(
        mm_fields.List(mm_fields.Boolean()), required=True)
    max_correction_up = mm_fields.Int(required=True)
    max_correction_down = mm_fields.Int(required=True)
    max_correction_left = mm_fields.Int(required=True)
    max_correction_right = mm_fields.Int(required=True)
    valid_roi = mm_fields.Boolean(required=True)
    coo_roi = mm_fields.Field(required=False, load_only=True)

    @post_load
    def add_coo_data(self, data, **kwargs):
        """Convert ROIs to coo format, which is used by the croissant
        FeatureExtractor. Input includes 'x' and 'y' fields
        which designate the cartesian coordinates of the top right corner,
        the width and height of the bounding box, and boolean values for
        whether the mask pixel is contained.
        The sparse matrix should technically include the shape of the input
        movie (the values for 'x' and 'y' correspond to the position on
        the input movie), but that's not important for our purposes since
        we don't actually need to align with any movie data.
        """
        shape = (data["height"], data["width"])
        arr = np.array(data["mask_matrix"]).astype("int")
        if data["height"] + data["width"] == 0:
            warnings.warn("Input data contains empty ROI. "
                          "This may cause problems.")
        elif arr.shape != shape:
            raise ValidationError("Data in mask matrix did not correspond to "
                                  "the (height, width) dimensions. Please "
                                  "check the input data.")
        mat = coo_matrix(arr)
        data.update({"coo_roi": mat})
        return data


class InferenceInputSchema(ArgSchema):
    """ Argschema parser for module as a script """
    neuropil_traces_path = H5InputFile(
        required=True,
        description=(
            "Path to neuropil traces from an experiment (h5 format). "
            "The order of the traces in the dataset should correspond to "
            "the order of masks in `roi_masks_path`.")
    )
    neuropil_traces_data_key = fields.Str(
        required=False,
        missing="data",
        default="data",
        description=("Key in `neuropil_traces_path` h5 file where data array "
                     "is stored.")
    )
    traces_path = H5InputFile(
        required=True,
        description=(
            "Path to traces extracted from an experiment (h5 format). "
            "The order of the traces in the dataset should correspond to "
            "the order of masks in `roi_masks_path`.")
    )
    traces_data_key = fields.Str(
        required=False,
        missing="data",
        default="data",
        description=("Key in `traces_path` h5 file where data array is "
                     "stored.")
    )
    roi_masks_path = fields.InputFile(
        required=True,
        description=("Path to json file of segmented ROI masks. The file "
                     "records must conform to the schema "
                     "`RoiJsonRecordSchema`")
    )
    rig = fields.Str(
        required=True,
        description=("Name of the ophys rig used for the experiment.")
    )
    depth = fields.Int(
        required=True,
        description=("Imaging depth for the experiment.")
    )
    full_genotype = fields.Str(
        required=True,
        description=("Genotype of the experimental subject.")
    )
    targeted_structure = fields.Str(
        required=True,
        description=("Name of the brain structure targeted by imaging.")
    )
    classifier_model_path = fields.Str(
        required=True,
        description=("Path to model. Can either be an s3 location or a "
                     "path on the local file system. The output of the model "
                     "should be 0 if the ROI is classified as not a cell, "
                     "and 1 if the ROI is classified as a cell.")
    )
    trace_sampling_fps = fields.Int(
        required=True,
        description="Sampling rate of trace (frames per second)."
    )
    downsample_to = fields.Int(
        required=False,
        default=4,
        validate=lambda x: x > 0,
        description=("Target rate to downsample trace data (frames per "
                     "second). Will use average bin values for downsampling.")
    )
    output_json = fields.OutputFile(
        required=True,
        description="Filepath to dump json output."
    )

    @validates("classifier_model_path")
    def validate_classifier_model_path(self, uri: str, **kwargs):
        """ Check to see if file exists (either s3 or local file) """
        if uri.startswith("s3://"):
            s3 = boto3.client("s3")
            parsed = urlparse(uri, allow_fragments=False)
            try:
                s3.head_object(
                    Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
            except ClientError as e:
                if e.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
                    raise ValidationError(
                        f"Object at URI {uri} does not exist.")
                else:
                    raise e from None
        else:
            if not os.path.exists(uri):
                raise ValidationError(f"File at '{uri}' does not exist.")

    @post_load
    def check_keys_exist(self, data: dict, **kwargs):
        """ For h5 files, check that the passed key exists in the data. """
        pairs = [("neuropil_traces_path", "neuropil_traces_data_key"),
                 ("traces_path", "traces_data_key")]
        for h5file, key in pairs:
            with h5py.File(data[h5file], "r") as f:
                if not data[key] in f.keys():
                    raise ValidationError(
                        f"Key '{data[key]}' ({key}) was missing in h5 file "
                        f"{data[h5file]} ({h5file}.")
        return data


class InferenceOutputSchema(Schema):
    """ Schema for output json (result of main module script) """
    classified_rois = fields.Nested(
        RoiJsonSchema,
        many=True,
        required=True
    )
    classifier_model_path = fields.Str(
        required=True,
        description=("Path to model. Can either be an s3 location or a "
                     "path on the local file system.")
    )


class InferenceParser(ArgSchemaParser):
    """ Argschema entry point """
    default_schema = InferenceInputSchema
    default_output_schema = InferenceOutputSchema


def _munge_data(parser: InferenceParser, roi_data: list):
    """
    Format the input data for downstream processing.
    Params
    ------
    parser: InferenceParser
        An instance of InferenceParser
    roi_data: list
        List of objects conforming to RoiJsonSchema
    Returns
    -------
    tuple
        rois (list of coo_matrices), metadata dictionary,
        traces data (np.1darray) and neuropil traces data (np.1darray)
    """
    # Format metadata and multiply for input (all same)
    metadata = [{
        "depth": parser.args["depth"],
        "rig": parser.args["rig"],
        "targeted_structure": parser.args["targeted_structure"],
        "full_genotype": parser.args["full_genotype"]
        }] * len(roi_data)
    rois = [r["coo_roi"] for r in roi_data]
    traces = []
    np_traces = []

    traces_file = h5py.File(parser.args["traces_path"], "r")
    np_traces_file = h5py.File(parser.args["neuropil_traces_path"], "r")
    traces_data = traces_file[parser.args["traces_data_key"]]
    np_traces_data = np_traces_file[parser.args["neuropil_traces_data_key"]]
    for n in range(len(roi_data)):
        # Downsample traces by accessing on axis that the h5 file should be
        # more performant on
        trace = downsample(
            traces_data[n, :], parser.args["trace_sampling_fps"],
            parser.args["downsample_to"])
        np_trace = downsample(
            np_traces_data[n, :], parser.args["trace_sampling_fps"],
            parser.args["downsample_to"])
        traces.append(trace)
        np_traces.append(np_trace)
    traces_file.close()
    np_traces_file.close()
    return rois, metadata, traces, np_traces


def downsample(trace: np.ndarray, input_fps: int, output_fps: int):
    """Downsample 1d array using an averaging strategy.
    If a full bin cannot be populated, the data will be truncated
    to approximate required output_fps.
    Example: downsampling from 5 fps to 2 fps
    for a trace with 13 points would need to drop the last data point.

    Parameters
    ----------
    trace: np.ndarray
        1d array of values to downsample
    input_fps: int
        The FPS that the trace data was captured at
    output_fps: int
        The desired FPS of the trace
    Returns
    -------
    np.ndarray
        1d array of values, downsampled to output_fps (may be approximate)
    """
    if input_fps == output_fps:
        return trace
    points = trace.shape[0]
    output_size = math.floor(points * output_fps / input_fps)
    bin_size = math.floor(points / output_size)
    if points / output_size != bin_size:    # if the bin needed to be rounded
        warnings.warn(f"Can't evenly downsample {points} points at "
                      f"{input_fps} FPS to {output_fps} FPS. Output fps is "
                      "approximate.")
    truncate = points % output_size
    if truncate != 0:
        downsampled = (trace[:-truncate].reshape(output_size, bin_size)
                       .mean(axis=1))
    else:
        downsampled = trace.reshape(output_size, bin_size).mean(axis=1)
    return downsampled


def main(parser):
    with open(parser.args["roi_masks_path"], "r") as f:
        raw_roi_data = json.load(f)
    roi_input_schema = RoiJsonSchema(many=True)
    roi_data = roi_input_schema.load(raw_roi_data)
    rois, metadata, traces, _ = _munge_data(parser, roi_data)
    # TODO: add neuropil traces later
    features = FeatureExtractor(rois, traces, metadata).run()
    model = joblib.load(parser.args["classifier_model_path"])
    predictions = model.predict(features)
    if len(predictions) != len(roi_data):
        raise ValueError(
            f"Expected the number of predictions ({len(predictions)}) to  "
            f"equal the number of input ROIs ({len(roi_data)}), but they "
            "are not the same.")
    for obj, prediction in zip(roi_data, predictions):
        if prediction == 0:
            obj["exclusion_labels"].append(NOT_CELL_EXCLUSION_LABEL)
            obj["valid_roi"] = False
    output_data = {
        "classified_rois": roi_data,
        "classifier_model_path": parser.args["classifier_model_path"]
    }
    parser.output(output_data)


if __name__ == "__main__":
    parser = InferenceParser()
    main(parser)
