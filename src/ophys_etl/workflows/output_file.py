from dataclasses import dataclass
from pathlib import Path

from ophys_etl.workflows.well_known_file_types import WellKnownFileType


@dataclass
class OutputFile:
    """File output by module"""
    path: Path
    well_known_file_type: WellKnownFileType
