import json
import math
import os.path
import logging
import tempfile
import warnings
from typing import Any, List, Tuple
from urllib.parse import urlparse

import boto3
import h5py
import joblib
import marshmallow.fields as mm_fields
import numpy as np
from argschema import ArgSchema, ArgSchemaParser, fields
from botocore.errorfactory import ClientError
from marshmallow import Schema, ValidationError, post_load, pre_load, validates
from marshmallow.validate import OneOf
from scipy.signal import resample_poly
from scipy.sparse import coo_matrix

from croissant.features import FeatureExtractor
from ophys_etl.types import DenseROI
from ophys_etl.schemas import DenseROISchema
from ophys_etl.schemas.fields import H5InputFile
from ophys_etl.transforms.registry import RegistryConnection

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


NOT_CELL_EXCLUSION_LABEL = "classified_as_not_cell"


class SparseAndDenseROI(DenseROI):
    coo_roi: coo_matrix


class SparseAndDenseROISchema(DenseROISchema):
    """Version of DenseROISchema which also includes ROIs in sparse format."""
    coo_roi = mm_fields.Field(required=False, load_only=True)

    @post_load
    def add_coo_data(self, data, **kwargs):
        """Convert ROIs to coo format, which is used by the croissant
        FeatureExtractor. Input includes 'x' and 'y' fields
        which designate the cartesian coordinates of the top right corner,
        the width and height of the bounding box, and boolean values for
        whether the mask pixel is contained. The returned coo_matrix
        will contain all the data in the mask in the proper shape,
        but essentially discards the 'x' and 'y' information (the
        cartesian position of the masks is not important for the
        below methods). Represented as a dense array, the mask data
        would be "cropped" to the bounding box.

        Note: If the methods were updated such that the position of
        the mask relative to the input data *were*
        important (say, if necessary to align the masks to the movie
        from which they were created), then this function would require
        the dimensions of the source movie.
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
        description=("Key in `neuropil_traces_path` h5 file where data array "
                     "is stored.")
    )
    neuropil_trace_names_key = fields.Str(
        required=False,
        missing="roi_names",
        description=("Key in `neuropil_traces_path` h5 file which describes"
                     "the roi name (id) associated with each trace.")
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
        description=("Key in `traces_path` h5 file where data array is "
                     "stored.")
    )
    trace_names_key = fields.Str(
        required=False,
        missing="roi_names",
        description=("Key in `traces_path` h5 file which describes"
                     "the roi name (id) associated with each trace.")
    )
    roi_masks_path = fields.InputFile(
        required=True,
        description=("Path to json file of segmented ROI masks. The file "
                     "records must conform to the schema "
                     "`DenseROISchema`")
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
                     "and 1 if the ROI is classified as a cell. If this "
                     "field is not provided, the classifier model registry "
                     "DynamoDB will be queried.")
    )
    trace_sampling_rate = fields.Int(
        required=False,
        missing=31,
        description=("Sampling rate of trace (frames per second). By default "
                     "trace sampling rates are assumed to be 31 Hz (inherited "
                     "from the source motion_corrected.h5 movie).")
    )
    desired_trace_sampling_rate = fields.Int(
        required=False,
        missing=4,
        validate=lambda x: x > 0,
        description=("Target rate to downsample trace data (frames per "
                     "second). Will use average bin values for downsampling.")
    )
    output_json = fields.OutputFile(
        required=True,
        description="Filepath to dump json output."
    )
    model_registry_table_name = fields.Str(
        required=False,
        missing="ROIClassifierRegistry",
        description=("The name of the DynamoDB table containing "
                     "the ROI classifier model registry.")
    )
    model_registry_env = fields.Str(
        required=False,
        validate=OneOf({'dev', 'stage', 'prod'},
                       error=("'{input}' is not a valid value for the "
                              "'model_registry_env' field. Possible "
                              "valid options are: {choices}")),
        missing="prod",
        description=("Which environment to query when searching for a "
                     "classifier model path from the classifier model "
                     "registry. Possible options are: ['dev', 'stage', 'prod]")
    )

    @pre_load
    def determine_classifier_model_path(self, data: dict, **kwargs) -> dict:
        if "classifier_model_path" not in data:
            # Can't rely on field `missing` param as it doesn't get filled in
            # until deserialization/validation. The get defaults should match
            # the 'missing' param for the model_registry_table_name and
            # model_registry_env fields.
            table_name = data.get("model_registry_table_name",
                                  "ROIClassifierRegistry")
            model_env = data.get("model_registry_env", "prod")
            model_registry = RegistryConnection(table_name=table_name)
            model_path = model_registry.get_active_model(env=model_env)
            data["classifier_model_path"] = model_path
        return data

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
    def check_keys_exist(self, data: dict, **kwargs) -> dict:
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
        SparseAndDenseROISchema,
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


def downsample(trace: np.ndarray, input_fps: int, output_fps: int):
    """Downsample 1d array using scipy resample_poly.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly    # noqa
    for more information.

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
        1d array of values, downsampled to output_fps
    """
    if input_fps == output_fps:
        return trace
    elif output_fps > input_fps:
        raise ValueError("Output FPS can't be greater than input FPS.")
    gcd = math.gcd(input_fps, output_fps)
    up = output_fps / gcd
    down = input_fps / gcd
    downsample = resample_poly(trace, up, down, axis=0, padtype="median")
    return downsample


def filter_excluded_rois(rois: List[SparseAndDenseROI]
                         ) -> Tuple[List[SparseAndDenseROI],
                                    List[SparseAndDenseROI]]:
    """Filter ROIs by presence or absence of "exclusion_labels".

    Parameters
    ----------
    rois : List[SparseAndDenseROI]
        A list of ROIs.

    Returns
    -------
    Tuple[List[SparseAndDenseROI], List[SparseAndDenseROI]]
        A tuple of: (included_rois, excluded_rois)
    """
    included, excluded = [], []
    for r in rois:
        excluded.append(r) if r["exclusion_labels"] else included.append(r)
    return included, excluded


def _munge_traces(roi_data: List[SparseAndDenseROI], trace_file_path: str,
                  trace_data_key: str, trace_names_key: str,
                  trace_sampling_rate: int, desired_trace_sampling_rate: int
                  ) -> List[np.ndarray]:
    """Read trace data from h5 file (in proper order) and downsample it.

    Parameters
    ----------
    roi_data : List[SparseAndDenseROI]
        A list of ROIs that conform to the SparseAndDenseROISchema.
    trace_file_path : str
        Path to h5 file containing trace data.
    trace_data_key : str
        Key used to access trace data.
    trace_names_key : str
        Key used to access the ROI name (id) corresponding to each trace.
    trace_sampling_rate : int
        Sampling rate (in Hz) of trace data.
    desired_trace_sampling_rate : int
        Desired sampling rate (in Hz) that trace data should be
        downsampled to.

    Returns
    -------
    List[np.1darray]
        A list of downsampled traces with each element corresponding to
        an element in roi_data.
    """
    traces_file = h5py.File(trace_file_path, "r")
    traces_data = traces_file[trace_data_key]

    # An array of str(int) describing roi names (id) associated with each trace
    # Example: ['0', '1', '10', '100', ..., '2', '20', '200', ...]
    traces_id_order = traces_file[trace_names_key][:].astype(int)
    traces_id_mapping = np.argsort(traces_id_order)

    downsampled_traces = []
    for roi in roi_data:
        assert roi['id'] == traces_id_order[traces_id_mapping[roi['id']]]
        trace_indx = traces_id_mapping[roi["id"]]
        ds_trace = downsample(traces_data[trace_indx, :],
                              trace_sampling_rate,
                              desired_trace_sampling_rate)
        downsampled_traces.append(ds_trace)

    traces_file.close()
    return downsampled_traces


MungedData = Tuple[List[coo_matrix], List[dict],
                   List[np.ndarray], List[np.ndarray]]


def _munge_data(parser: InferenceParser,
                roi_data: List[SparseAndDenseROI]) -> MungedData:
    """
    Format the input data for downstream processing.
    Params
    ------
    parser: InferenceParser
        An instance of InferenceParser
    roi_data: List[SparseAndDenseROI]
        List of objects conforming to SparseAndDenseROISchema
    Returns
    -------
    tuple: MungedData
        rois (list of coo_matrices), metadata dictionary,
        traces data (np.1darray) and neuropil traces data (np.1darray)
    """
    # Format metadata and multiply for input (all same)
    metadata = [{
        "depth": parser.args["depth"],
        "full_genotype": parser.args["full_genotype"],
        "targeted_structure": parser.args["targeted_structure"],
        "rig": parser.args["rig"]
        }] * len(roi_data)
    rois = [r["coo_roi"] for r in roi_data]

    traces = _munge_traces(roi_data, parser.args["traces_path"],
                           parser.args["traces_data_key"],
                           parser.args["trace_names_key"],
                           parser.args["trace_sampling_rate"],
                           parser.args["desired_trace_sampling_rate"])

    np_traces = _munge_traces(roi_data, parser.args["neuropil_traces_path"],
                              parser.args["neuropil_traces_data_key"],
                              parser.args["neuropil_trace_names_key"],
                              parser.args["trace_sampling_rate"],
                              parser.args["desired_trace_sampling_rate"])

    return rois, metadata, traces, np_traces


def load_model(classifier_model_uri: str) -> Any:
    """Load a classifier model given a valid URI.

    Parameters
    ----------
    classifier_model_uri : str
        A valid URI that points to either an AWS S3 resource or a
        local filepath. URI validity is only guaranteed by the
        'InferenceInputSchema'.

    Returns
    -------
    Any
        A loaded ROI classifier model.
    """
    if classifier_model_uri.startswith("s3://"):
        s3 = boto3.client("s3")
        parsed = urlparse(classifier_model_uri, allow_fragments=False)

        with tempfile.TemporaryFile() as fp:
            s3.download_fileobj(Bucket=parsed.netloc,
                                Key=parsed.path.lstrip("/"),
                                Fileobj=fp)
            fp.seek(0)
            model = joblib.load(fp)
    else:
        model = joblib.load(classifier_model_uri)
    return model


def main(parser):

    with open(parser.args["roi_masks_path"], "r") as f:
        raw_roi_data = json.load(f)

    roi_data = SparseAndDenseROISchema(many=True).load(raw_roi_data)
    roi_data, excluded_rois = filter_excluded_rois(roi_data)
    rois, metadata, traces, _ = _munge_data(parser, roi_data)
    # TODO: add neuropil traces later
    logger.info(f"Extracting features from '{len(rois)}' ROIs")
    features = FeatureExtractor(rois, traces, metadata).run()
    logger.info(f"Using the following classifier model: "
                f"{parser.args['classifier_model_path']}")
    model = load_model(parser.args["classifier_model_path"])
    logger.info("Classifying ROIs with features")
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

    roi_data.extend(excluded_rois)

    output_data = {
        "classified_rois": roi_data,
        "classifier_model_path": parser.args["classifier_model_path"]
    }
    parser.output(output_data)
    logger.info("ROI classification successfully completed!")


if __name__ == "__main__":
    parser = InferenceParser()
    main(parser)
