import json
import math
import logging
import tempfile
from typing import Any, List, Tuple
from urllib.parse import urlparse

import boto3
import h5py
import joblib
import numpy as np
from argschema import ArgSchemaParser
from scipy.signal import resample_poly
from scipy.sparse import coo_matrix

from croissant.features import FeatureExtractor
from ophys_etl.modules.classifier_inference.schemas import (
        SparseAndDenseROISchema, SparseAndDenseROI,
        InferenceOutputSchema, InferenceInputSchema)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


NOT_CELL_EXCLUSION_LABEL = "classified_as_not_cell"


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
    # Example: ['10', '100', ..., '2', '20', '200', ..., '3']
    traces_id_order = traces_file[trace_names_key][:].astype(int)
    traces_id_mapping = {val: ind for ind, val in enumerate(traces_id_order)}

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
