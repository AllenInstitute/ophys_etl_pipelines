"""Pydantic utilities"""
from pydantic import BaseModel, Extra


class ImmutableBaseModel(BaseModel, frozen=True, extra=Extra.forbid):
    """Immutable base model"""
    pass
