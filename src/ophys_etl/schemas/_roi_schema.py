from marshmallow import Schema
from marshmallow.fields import (List, Str, Float, Int, Bool)


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
