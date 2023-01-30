import argschema
from deepinterpolation.cli.schemas import FineTuningInputSchema, \
    GeneratorSchema

from ophys_etl.schemas.fields import H5InputFile


class DataSplitterInputSchema(argschema.ArgSchema):
    movie_path = H5InputFile(
        required=True,
        description='Path to ophys movie'
    )
    train_frac = argschema.fields.Float(
        default=0.7,
        description='Amount of data to set aside for training. The rest will '
                    'be used for validation'
    )
    seed = argschema.fields.Int(
        default=1234,
        allow_none=True,
        description='Seed for train/val split reproducibility'
    )


class DataSplitterOutputSchema(argschema.ArgSchema):
    mean = argschema.fields.Float(
        required=True,
        description='Mean data value, used for normalization'
    )
    std = argschema.fields.Float(
        required=True,
        description='Std deviation of data. Used for normalization'
    )
    path = argschema.fields.InputFile(
        required=True,
        description='Path to movie'
    )
    frames = argschema.fields.List(
        argschema.fields.Int,
        required=True,
        description='List of center frames to use for train/validation'
    )


class GeneratorSchemaPreDataSplit(GeneratorSchema):
    """Same as `GeneratorSchema` except making `data_path` optional since
    we need to split the data into train/val first
    """
    data_path = argschema.fields.String(
        allow_none=True,
        default=None,
        description="Dummy field"
    )


class FineTuningInputSchemaPreDataSplit(FineTuningInputSchema):
    """Same as `FineTuningInputSchema` except the data hasn't been split
    into train/val yet"""
    data_split_params = argschema.fields.Nested(
        DataSplitterInputSchema, default={}
    )
    generator_params = argschema.fields.Nested(
        GeneratorSchemaPreDataSplit, default={})
    test_generator_params = argschema.fields.Nested(
        GeneratorSchemaPreDataSplit, default={})
