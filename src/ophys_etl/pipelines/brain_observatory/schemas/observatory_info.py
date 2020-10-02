from argschema.fields import Nested, Int, Str
from .base import BaseSchema


class ProjectSchema(BaseSchema):
    _MANY = True
    id = Int(
        required=True,
        description="Project ID")
    code = Str(
        required=True,
        description="Project code")
    trigger_dir = Str(
        required=True,
        description="Project trigger directory")


class ExperimentSchema(BaseSchema):
    id = Int(
        required=True,
        description="ID")
    name = Str(
        description="Name")
    storage_directory = Str(
        description="Storage directory")
    workflow_state = Str(
        description="Workflow state")


class SpecimenSchema(BaseSchema):
    id = Int(
        required=True,
        description="Specimen ID")
    name = Str(
        description="Specimen Name")
    project = Nested(
        ProjectSchema,
        required=True,
        description="Project info")
    isi_experiments = Nested(
        ExperimentSchema,
        many=True)
    ophys_sessions = Nested(
        ExperimentSchema,
        many=True)


class DonorInfoSchema(BaseSchema):
    _MANY = True
    id = Int(
        required=True,
        description="Donor ID")
    name = Str(
        required=True,
        description="Donor Name")
    external_donor_name = Str(
        description="External donor name (expected to be labtracks ID)")
    specimens = Nested(
        SpecimenSchema,
        many=True)


class StructureSchema(BaseSchema):
    id = Int(
        required=True,
        description="Structure ID")
    acronym = Str(
        required=True,
        description="Short structure name (example: VISp)")
    name = Str(
        description="Structure name (example: Primary visual area)")


class StructureSetSchema(BaseSchema):
    id = Int(
        required=True,
        description="Structure set ID")
    name = Str(
        description="Structure set name")
    structures = Nested(
        StructureSchema,
        many=True)
