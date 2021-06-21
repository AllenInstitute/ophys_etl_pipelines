import argschema


class SeederBaseSchema(argschema.schemas.DefaultSchema):
    exclusion_buffer = argschema.fields.Int(
        required=False,
        default=1,
        description=("when exlucding pixels, an additional dilated boundary "
                     "of this thickness (in pixels) will be added to the "
                     "excluded pixels. I.e. one might exclude an ROI and "
                     "also the extra pixels that result from a binary "
                     "dilation of that ROI."))


class ImageMetricSeederSchema(SeederBaseSchema):
    keep_fraction = argschema.fields.Float(
        required=False,
        default=0.2,
        description=("fraction of selected seeds to keep as candidates. "
                     "Fraction kept is applied after sorting by image value. "
                     "Seeds not kept are marked as 'excluded_by_fraction'."))
    seeder_grid_size = argschema.fields.Int(
        required=False,
        default=None,
        allow_none=True,
        description=("Seeds will be the coordinates of the max values in "
                     "tiling square blocks of this size. This is the HNCcorr "
                     "method for selecting seeds. If left as None, all "
                     "pixels are considered and ranked."))


class BatchImageMetricSeederSchema(ImageMetricSeederSchema):
    n_samples = argschema.fields.Int(
        required=False,
        default=10,
        description=("the batch seeder will provide a list of up to this "
                     "many seeds per iteration."))
    minimum_distance = argschema.fields.Float(
        required=False,
        default=20.0,
        description=("Seeds provided in a single iteration will be spaced "
                     "at least this distance (in pixels) from other seeds "
                     "in the same batch."))
