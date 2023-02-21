import dataclasses
import json
from enum import Enum
from pathlib import Path


def _dataclass_dict_factory(data):
    """Supports converting Enum to dict representation key: value
    rather than key: <class>.<name>: <value> for dataclass"""
    def convert_value(obj):
        if isinstance(obj, Enum):
            return obj.value
        return obj

    return dict((k, convert_value(v)) for k, v in data)


class EnhancedJSONEncoder(json.JSONEncoder):
    """Same as json.dumps except works for other objects also"""
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o, dict_factory=_dataclass_dict_factory)
        elif isinstance(o, Enum):
            return o.value
        elif isinstance(o, Path):
            return str(o)
        return super().default(o)
