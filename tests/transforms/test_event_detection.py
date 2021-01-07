import pytest
import h5py
from unittest.mock import Mock
import sys
sys.modules['FastLZeroSpikeInference'] = Mock()
import ophys_etl.transforms.event_detection as event
from ophys_etl.resources import event_decay_lookup_dict as decay_lookup

#@pytest.mark.event_detect_only
#@pytest.fixture(scope="function")
#def traces_hdf5_fixture(tmp_path, request):
#    sigma = request.param.get("sigma")
#    nframes = request.param.get("nframes")
#    timestamps = request.param.get("timestamps")

def test_EventDetectionSchema(tmp_path):
    fpath = tmp_path / "junk_input.h5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("data", data=[1, 2, 3.14])

    dtime = 1.234

    # this works
    args = {
            'movie_frame_rate_hz': 31.0,
            'ophysdfftracefile': str(fpath),
            'valid_roi_ids': [4, 5, 6],
            'output_event_file': str(tmp_path / "junk_output.hdf5"),
            'decay_time': dtime
            }
    parser = event.EventDetection(input_data=args, args=[])
    assert parser.args['decay_time'] == dtime

    # this also works
    _ = args.pop('decay_time')
    key = "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt"
    args['full_genotype'] = key
    parser = event.EventDetection(input_data=args, args=[])
    assert 'decay_time' in parser.args
    assert parser.args['decay_time'] == decay_lookup[key]

    args['full_genotype'] = 'non-existent-genotype'
    with pytest.raises(event.EventDetectionException, match=r".*not available.*"):
        parser = event.EventDetection(input_data=args, args=[])

    _ = args.pop('full_genotype')
    with pytest.raises(event.EventDetectionException, match=r"Must provide either.*"):
        parser = event.EventDetection(input_data=args, args=[])
