from dataclasses import dataclass
from pathlib import Path

from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum


@dataclass
class OutputFile:
    """File output by module"""

    path: Path
    well_known_file_type: WellKnownFileTypeEnum
