from argschema.schemas import DefaultSchema
from argschema.fields import (Nested, Int, Float, Str, DateTime, Constant)
from .base import BaseSchema

class BaseRegistrationItem(DefaultSchema):
    x_offset = Int(
        required=True,
        description=("X offset from the top-left of the image to the center "
                     "of the gold reticle in image coordinates (pixel units)"))
    y_offset = Int(
        required=True,
        description=("Y offset from the top-left of the image to the center "
                     "of the gold reticle in image coordinates (pixel units)"))
    rotation = Float(
        required=True,
        description=("Rotation of the image relative to gold reticle in "
                     "degrees"))
    pixel_size_um = Float(
        required=True,
        description="Linear dimension of a pixel in microns")
    acquired_at = DateTime(
        required=True,
        description="Acquisition timestamp")
    stage_image_rotation = Float(
        description="Rotation of stage relative to image")
    stage_x = Float(
        description="X position of stage when image was acquired")
    stage_y = Float(
        description="Y position of stage when image was acquired")
    stage_z = Float(
        description="Z position of stage when image was acquired")
    home_offset_x = Int(
        description="X offset between reticle home and vasculature home")
    home_offset_y = Int(
        description="Y offset between reticle home and vasculature home")


class ImageRegistrationItem(BaseRegistrationItem):
    filename = Str(
        required=True,
        description=("Base filename (without path) for image as it appears in "
                     "upload directory"))


class PlatformV2(BaseSchema):
    schema_version = Constant(
        2,
        required=True,
        description="Version of schema file.")
    rig_id = Str(
        required=True,
        description="Rig name. Should be a name in http://lims2/equipment")
    stimulus_pkl = Str(
        required=True,
        description="Base filename (without path) for stimulus pickle file")
    sync_file = Str(
        required=True,
        description="Base filename (without path) for sync h5 file")
    eye_tracking_video = Str(
        description="Base filename (without path) for eye tracking avi")
    behavior_video = Str(
        description="Base filename (without path) for behavior avi")
    foraging_id = Int(
        description="ID of associated foraging session")


class DeepscopeSessionRegistration(DefaultSchema):
    surface_vasculature = Nested(
        ImageRegistrationItem,
        required=True,
        description="Registration info for epifluorescent vasculature image")
    reticle_image = Nested(
        ImageRegistrationItem,
        required=True,
        description="Registration info for reticle image")
    z_stack_column = Nested(
        ImageRegistrationItem,
        description="Registration info for column z_stack hdf5 file")


class DeepscopeExperimentRegistration(DefaultSchema):
    timeseries = Nested(
        ImageRegistrationItem,
        required=True,
        description="Registration info for timeseries hdf5 file")
    z_stack_local = Nested(
        ImageRegistrationItem,
        description="Registration info for local z_stack hdf5 file")
    depth_2p = Nested(
        ImageRegistrationItem,
        description="Registration info for depth_2p image")
    surface_2p = Nested(
        ImageRegistrationItem,
        description="Registration info for surface_2p image")


class ImagingPlane(DefaultSchema):
    targeted_structure_id = Int(
        required=True,
        description="LIMS ID of targeted structure")
    targeted_depth = Float(
        required=True,
        description="Depth of imaging plane from surface in microns")
    targeted_x = Float(
        required=True,
        description="Targeted x coordinate (microns) in gold reticle space")
    targeted_y = Float(
        required=True,
        description="Targeted y coordinate (microns) in gold reticle space")
    registration = Nested(
        DeepscopeExperimentRegistration,
        required=True)


class DeepscopePlane(ImagingPlane):
    slm_pattern_file = Str(
        required=True,
        description=("base filename (without path) of slm pattern for this "
                     "plane"))


class DeepscopeSchema(PlatformV2):
    schema_type = Constant(
        "Deepscope",
        required=True,
        description="Constant indicating this is a deepscope experiment")
    registration = Nested(
        DeepscopeSessionRegistration,
        required=True)
    imaging_planes = Nested(
        DeepscopePlane,
        many=True)
