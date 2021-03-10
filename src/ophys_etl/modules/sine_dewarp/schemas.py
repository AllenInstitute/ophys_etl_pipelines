import argschema

from ophys_etl.schemas.fields import H5InputFile


class SineDewarpInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.Str(default="INFO")
    input_h5 = H5InputFile(
        required=True,
        description="h5 movie file to be dewarped")
    output_h5 = argschema.fields.OutputFile(
        required=True,
        description="destination path for dewarped movie")
    aL = argschema.fields.Float(
        required=True,
        description=("How far into the image (from the left side inward, "
                     "in pixels) the warping occurs. This is specific to "
                     "the experiment, and is known at the time of data "
                     "collection."))
    aR = argschema.fields.Float(
        required=True,
        description=("How far into the image (from the right side inward, "
                     "in pixels) the warping occurs. This is specific to "
                     "the experiment, and is known at the time of data "
                     "collection."))
    bL = argschema.fields.Float(
        required=True,
        description=("Roughly, a measurement of how strong the warping effect "
                     "was on the left side for the given experiment."))
    bR = argschema.fields.Float(
        required=True,
        description=("Roughly, a measurement of how strong the warping effect "
                     "was on the right side for the given experiment."))
    n_parallel_workers = argschema.fields.Int(
        required=False,
        default=4,
        description="how many multiprocessing workers")
    FOV_width = argschema.fields.Int(
        required=False,
        default=0,
        description=("Field of View width. Will be the width of the output "
                     "image, unless 0, then it will be the width of the input "
                     "image."))
    noise_reduction = argschema.fields.Int(
        required=False,
        default=0,
        description=("1: Reduce the data value by 2 * sigma. "
                     "2: Reduce the data value by sigma and normalize it. "
                     "3: Normalize the data value."
                     "else: No noise reduction"))
    equipment_name = argschema.fields.Str(
        required=True,
        description=("as far as I can tell, this is not used, but we need "
                     "this in the schema to match the current LIMS strategy."))
    chunk_size = argschema.fields.Int(
        required=False,
        default=1000,
        description=("size of chunks, in frames, for breaking up processing "
                     "into smaller jobs"))
