# The ruby code that runs the LIMS strategies reads in the output.json
# produced by each module and uses it to set attributes in the LIMS
# database. The JSON parser in that code does not handle NaNs or Infs.
# Unfortunately, the ScanImage metadata prepended to the Mesoscope
# TIFF files includes several values that are set to Inf. This module
# provides some methods to sanitize the output data before it is written
# to the output.json.

from typing import Union
import numbers
import numpy as np


def _sanitize_element(element: numbers.Number) -> Union[str, numbers.Number]:
    """
    If element is NaN or +/-inf, convert to a
    string; otherwise, return element as-si
    """
    if np.isfinite(element):
        return element
    if np.isnan(element):
        return "_NaN_"
    if element > 0:
        return "_Inf_"
    return "_-Inf_"


def _sanitize_data(
        data: Union[list, dict, numbers.Number, str]
        ) -> Union[list, dict, numbers.Number, str]:
    """
    Iteratively sanitize a data structure so that there are no NaNs
    or infinities in the output.json returned by this module

    Return the sanitized version of data.
    """
    if isinstance(data, numbers.Number):
        return _sanitize_element(element=data)
    elif isinstance(data, list):
        for ii in range(len(data)):
            data[ii] = _sanitize_data(data=data[ii])
    elif isinstance(data, dict):
        key_list = list(data.keys())
        for key in key_list:
            data[key] = _sanitize_data(data=data[key])
    return data


def get_sanitized_json_data(
        output_json_data: Union[dict, list]) -> Union[list, dict]:
    """
    Take the data that is supposed to be written to the output.json
    and sanitize it so that there are no NaNs or infs.

    Parameters
    ----------
    output_json_data: Union[dict, list]
        The data this meant to be written to the output.json file

    Returns
    -------
    sanitized_output_json_data: Union[dict, list]
        A version of output_json_data that is fit to be written to
        the output.json and ingested by LIMS
    """
    return _sanitize_data(data=output_json_data)
