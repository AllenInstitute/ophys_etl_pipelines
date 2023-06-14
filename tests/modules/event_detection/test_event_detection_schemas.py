import h5py
import contextlib
import pytest
from unittest.mock import Mock
import sys


from ophys_etl.modules.event_detection.exceptions import \
    EventDetectionException
from ophys_etl.modules.event_detection.resources.event_decay_time_lookup \
        import event_decay_lookup_dict as decay_lookup
sys.modules['FastLZeroSpikeInference'] = Mock()
import ophys_etl.modules.event_detection.__main__ as emod  # noqa: E402


@pytest.mark.event_detect_only
@pytest.mark.parametrize(
        ("rate", "specify_multiplier", "multiplier", "expected"),
        [
            (11.0, True, 12, 12),
            (11.0, False, None, 2.6),
            (31.0, True, 2.4, 2.4),
            (31.0, False, None, 2.0),
            ])
def test_EventDetectionSchema_multiplier(tmp_path, rate, expected,
                                         specify_multiplier, multiplier):
    fpath = tmp_path / "junk_input.h5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("data", data=[1, 2, 3.14])
        f.create_dataset("roi_names", data=[5, 6, 7, 8])

    args = {
            'movie_frame_rate_hz': rate,
            'ophysdfftracefile': str(fpath),
            'valid_roi_ids': [4, 5, 6],
            'output_event_file': str(tmp_path / "junk_output.hdf5"),
            'decay_time': 1.234
            }
    if specify_multiplier:
        args['noise_multiplier'] = multiplier
    parser = emod.EventDetection(input_data=args, args=[])
    assert parser.args['noise_multiplier'] == expected


@pytest.mark.event_detect_only
@pytest.mark.parametrize(
        "missing_field, context",
        [
            (False, contextlib.nullcontext()),
            (True, pytest.raises(
                EventDetectionException,
                match=r".*does not have the key 'roi_names'.*"))])
def test_EventDetectionSchema_missing_name(tmp_path, missing_field, context):
    fpath = tmp_path / "junk_input.h5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("data", data=[1, 2, 3.14])
        if not missing_field:
            f.create_dataset("roi_names", data=[5, 6, 7, 8])

    args = {
            'movie_frame_rate_hz': 31.0,
            'ophysdfftracefile': str(fpath),
            'valid_roi_ids': [4, 5, 6],
            'output_event_file': str(tmp_path / "junk_output.hdf5"),
            'decay_time': 1.234
            }
    with context:
        parser = emod.EventDetection(input_data=args, args=[])
        assert 'halflife' in parser.args


@pytest.mark.event_detect_only
def test_EventDetectionSchema_decay_time(tmp_path):
    fpath = tmp_path / "junk_input.h5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("data", data=[1, 2, 3.14])
        f.create_dataset("roi_names", data=[5, 6, 7, 8])

    # specify decay_time explicitly
    dtime = 1.234
    args = {
            'movie_frame_rate_hz': 31.0,
            'ophysdfftracefile': str(fpath),
            'valid_roi_ids': [4, 5, 6],
            'output_event_file': str(tmp_path / "junk_output.hdf5"),
            'decay_time': dtime
            }
    parser = emod.EventDetection(input_data=args, args=[])
    assert parser.args['decay_time'] == dtime

    # specifying valid genotype rather than decay time
    args.pop('decay_time')
    key = "Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt"
    args['full_genotype'] = key
    parser = emod.EventDetection(input_data=args, args=[])
    assert 'decay_time' in parser.args
    assert parser.args['decay_time'] == decay_lookup[key]
    assert 'halflife' in parser.args

    # non-existent genotype exception
    args['full_genotype'] = 'non-existent-genotype'
    with pytest.raises(EventDetectionException,
                       match=r".*not available.*"):
        parser = emod.EventDetection(input_data=args, args=[])

    # neither arg supplied
    args.pop('full_genotype')
    with pytest.raises(EventDetectionException,
                       match=r"Must provide either.*"):
        parser = emod.EventDetection(input_data=args, args=[])
