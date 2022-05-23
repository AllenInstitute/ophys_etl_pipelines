import boto3
import h5py
import os
import numpy as np
from argschema import ArgSchema, fields
import marshmallow.fields as mm_fields
from marshmallow import Schema, ValidationError, post_load, pre_load, validates
from marshmallow.validate import OneOf
from botocore.errorfactory import ClientError
from urllib.parse import urlparse
from scipy.sparse import coo_matrix
import warnings

from ophys_etl.types import DenseROI
from ophys_etl.schemas import DenseROISchema
from ophys_etl.modules.classifier_inference import utils
from ophys_etl.schemas.fields import H5InputFile


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
    # The options below are set by the LIMS queue but are not necessary to run
    # the code.
    motion_corrected_movie_path = fields.InputFile(
            required=False,
            default=None,
            allow_none=True,
            description=("Path to motion corrected video."))
    movie_frame_rate_hz = fields.Float(
            required=False,
            default=None,
            allow_none=True,
            description=("The frame rate (in Hz) of the optical physiology "
                         "movie to be Suite2P segmented. Used in conjunction "
                         "with 'bin_duration' to derive an 'nbinned' "
                         "Suite2P value."))

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
            model_registry = utils.RegistryConnection(table_name=table_name)
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
