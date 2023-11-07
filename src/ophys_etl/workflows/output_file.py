from dataclasses import dataclass
from pathlib import Path

from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum


@dataclass
class OutputFile:
    """File output by module"""

    path: Path
    well_known_file_type: WellKnownFileTypeEnum

    @classmethod
    def from_dict(cls, x) -> "OutputFile":
        return OutputFile(
            path=Path(x['path']),
            well_known_file_type=WellKnownFileTypeEnum(
                x['well_known_file_type'])
        )
