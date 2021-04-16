import os
from ophys_etl.modules.decrosstalk.ophys_plane import DecrosstalkingOphysPlane
from ophys_etl.modules.decrosstalk.decrosstalk import run_decrosstalk

from .utils import create_data


def test_run_decrosstalk(tmpdir):
    """
    Test that run_decrosstalk() can run
    as expected in production.

    Note: this is just a smoke test; not validation
    is done on the outputs of run_decrosstalk
    """

    session = create_data(tmpdir)

    pair = session['coupled_planes'][0]['planes']
    plane0 = DecrosstalkingOphysPlane.from_schema_dict(pair[0])
    plane1 = DecrosstalkingOphysPlane.from_schema_dict(pair[1])
    run_decrosstalk(plane0, plane1,
                    cache_dir=session['qc_output_dir'],
                    clobber=True)
    run_decrosstalk(plane1, plane0,
                    cache_dir=session['qc_output_dir'],
                    clobber=True)
