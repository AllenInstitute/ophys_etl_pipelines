from argschema.schemas import DefaultSchema
from argschema.utils import load


class BaseSchema(DefaultSchema):
    _MANY = False

    @classmethod
    def load_validated(cls, data):
        return load(cls(many=cls._MANY), data)