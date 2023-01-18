import dataclasses
import json
from pathlib import Path


class EnhancedJSONEncoder(json.JSONEncoder):
    """Same as json.dumps except works for other objects also"""
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)
