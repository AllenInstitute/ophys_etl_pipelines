"""Ophys experiment"""
import os
from dataclasses import dataclass
from pathlib import Path

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.utils.lims_utils import LIMSDB


@dataclass
class OphysSession:
    """Container for an ophys session"""
    id: str


@dataclass
class Specimen:
    """Container for a specimen"""
    id: str


@dataclass
class OphysExperiment:
    """Container for an ophys experiment"""
    id: str
    session: OphysSession
    specimen: Specimen
    storage_directory: Path
    raw_movie_filename: Path
    movie_frame_rate_hz: float

    @property
    def output_dir(self) -> Path:
        """Where to output files to for this experiment"""
        base_dir = app_config.output_dir

        output_dir = Path(base_dir) / f'specimen_{self.specimen.id}' / \
            f'session_{self.session.id}' / f'experiment_{self.id}'
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @classmethod
    def from_id(
            cls,
            id: str
    ) -> "OphysExperiment":
        """Returns an `OphysExperiment` given a LIMS id for an
        ophys experiment

        Parameters
        ----------
        id
            LIMS ID for the ophys experiment

        """
        query = f'''
            SELECT
                oe.storage_directory,
                oe.ophys_session_id as session_id,
                os.specimen_id,
                oe.movie_frame_rate_hz,
                images.jp2 as raw_movie_filename
            FROM ophys_experiments oe
            JOIN images on images.id = oe.ophys_primary_image_id
            JOIN ophys_sessions os on os.id = oe.ophys_session_id
            WHERE oe.id = {id}
        '''
        lims_db = LIMSDB()
        res = lims_db.query(query=query)

        if len(res) == 0:
            raise ValueError(f'Could not fetch OphysExperiment '
                             f'for ophys experiment id '
                             f'{id}')
        res = res[0]

        session = OphysSession(id=res['session_id'])
        specimen = Specimen(id=res['specimen_id'])

        return cls(
            id=id,
            storage_directory=Path(res['storage_directory']),
            movie_frame_rate_hz=res['movie_frame_rate_hz'],
            raw_movie_filename=res['raw_movie_filename'],
            session=session,
            specimen=specimen
        )
