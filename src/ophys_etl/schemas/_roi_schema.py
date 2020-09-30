import numpy as np
import warnings
from scipy.sparse import coo_matrix
from marshmallow import Schema, post_load, ValidationError
from marshmallow.fields import List, Str, Float, Int, Bool, Field


class ExtractROISchema(Schema):
    """This ROI format is the expected input of AllenSDK's extract_traces
    'rois' field
    """

    id = Int(required=True,
             description=("Unique ID of the ROI, get's overwritten writting "
                          "to LIMS"))
    x = Int(required=True,
            description="X location of top left corner of ROI in pixels")
    y = Int(required=True,
            description="Y location of top left corner of ROI in pixels")
    width = Int(required=True,
                description="Width of the ROI in pixels")
    height = Int(required=True,
                 description="Height of the ROI in pixels")
    valid = Bool(required=True,
                 description=("Boolean indicating if the ROI is a valid "
                              "cell or not"))
    mask = List(List(Bool), required=True,
                description=("Bool nested list describing which pixels "
                             "in the ROI area are part of the cell"))


class DenseROISchema(Schema):
    """This ROI format is the expected output of Segmentation/Binarization
    and the expected input of Feature_extraction/Classification.
    """

    id = Int(required=True,
             description=("Unique ID of the ROI, get's overwritten writting "
                          "to LIMS"))
    x = Int(required=True,
            description="X location of top left corner of ROI in pixels")
    y = Int(required=True,
            description="Y location of top left corner of ROI in pixels")
    width = Int(required=True,
                description="Width of the ROI in pixels")
    height = Int(required=True,
                 description="Height of the ROI in pixels")
    valid_roi = Bool(required=True,
                     description=("Boolean indicating if the ROI is a valid "
                                  "cell or not"))
    mask_matrix = List(List(Bool), required=True,
                       description=("Bool nested list describing which pixels "
                                    "in the ROI area are part of the cell"))
    max_correction_up = Float(required=True,
                              description=("Max correction in pixels in the "
                                           "up direction"))
    max_correction_down = Float(required=True,
                                description=("Max correction in pixels in the "
                                             "down direction"))
    max_correction_left = Float(required=True,
                                description=("Max correction in pixels in the "
                                             "left direction"))
    max_correction_right = Float(required=True,
                                 description="Max correction in the pixels in "
                                             "the right direction")
    mask_image_plane = Int(required=True,
                           description=("The old segmentation pipeline stored "
                                        "overlapping ROIs on separate image "
                                        "planes. For compatibility purposes, "
                                        "this field must be kept, but will "
                                        "always be set to zero for the new "
                                        "updated pipeline"))
    exclusion_labels = List(Str, required=True,
                            description=("LIMS ExclusionLabel names used to "
                                         "track why a given ROI is not "
                                         "considered a valid_roi. (examples: "
                                         "motion_border, "
                                         "classified_as_not_cell)"))


class SparseAndDenseROISchema(DenseROISchema):
    """Version of DenseROISchema which also includes ROIs in sparse format."""
    coo_roi = Field(required=False, load_only=True)

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
