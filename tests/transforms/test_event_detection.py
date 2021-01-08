import pytest
import h5py
import numpy as np
from collections import namedtuple
try:
    import ophys_etl.transforms.event_detection as emod
except ModuleNotFoundError:
    # even though we might skip tests, pytest tries these imports
    from unittest.mock import Mock
    import sys
    sys.modules['FastLZeroSpikeInference'] = Mock()
    import ophys_etl.transforms.event_detection as emod
from ophys_etl.resources import event_decay_lookup_dict as decay_lookup


Events = namedtuple('Events', ['id', 'timestamps', 'magnitudes'])


def make_event(length, index, magnitude, decay_time, rate):
    timestamps = np.arange(length) / rate
    t0 = timestamps[index]
    z = np.zeros(length)
    z[index:] = magnitude * np.exp(-(timestamps[index:] - t0) / decay_time)
    return z


@pytest.fixture(scope="function")
def dff_hdf5(tmp_path, request):
    sigma = request.param.get("sigma")
    offset = request.param.get("offset")
    decay_time = request.param.get("decay_time")
    nframes = request.param.get("nframes")
    events = request.param.get("events")
    rate = request.param.get("rate")

    rng = np.random.default_rng(42)
    data = rng.normal(loc=offset, scale=sigma, size=(len(events), nframes))
    for i, event in enumerate(events):
        for ts, mag in zip(event.timestamps, event.magnitudes):
            data[i] += make_event(nframes, ts, mag, decay_time, rate)

    names = [i.id for i in events]

    h5path = tmp_path / "fake_dff.h5"
    with h5py.File(h5path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("roi_names", data=names)

    with h5py.File(h5path, "r") as f:
        yield h5path, decay_time, rate, events


@pytest.mark.event_detect_only
def test_EventDetectionSchema(tmp_path):
    fpath = tmp_path / "junk_input.h5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("data", data=[1, 2, 3.14])

    dtime = 1.234

    # specifying decay time directly
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

    # non-existent genotype exception
    args['full_genotype'] = 'non-existent-genotype'
    with pytest.raises(
            emod.EventDetectionException,
            match=r".*not available.*"):
        parser = emod.EventDetection(input_data=args, args=[])

    # neither arg supplied
    args.pop('full_genotype')
    with pytest.raises(
            emod.EventDetectionException,
            match=r"Must provide either.*"):
        parser = emod.EventDetection(input_data=args, args=[])


@pytest.mark.event_detect_only
@pytest.mark.parametrize(
        "dff_hdf5",
        [
            {
                "sigma": 0.1,
                "decay_time": 0.415,
                "offset": 0.0,
                "nframes": 1000,
                "rate": 31.0,
                "events": [
                    Events(
                        id=123,
                        timestamps=[45, 112, 232, 410, 490, 700, 850],
                        magnitudes=[4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0]),
                    Events(
                        id=124,
                        timestamps=[145, 212, 280, 310, 430, 600, 810],
                        magnitudes=[4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0])]
                    }], indirect=True)
def test_EventDetection(dff_hdf5, tmp_path):
    """This test runs the actual spike inference on fake data. The fake
    data is constructed by the dff_hdf5 fixture. This is an
    easy test, in that the SNR for the spikes is high, there is neither offset
    nor low frequency background, the fake spikes are not too close together,
    and, some false spikes are ignored at the end.
    """

    dff_path, decay_time, rate, expected_events = dff_hdf5

    args = {
            'movie_frame_rate_hz': rate,
            'ophysdfftracefile': str(dff_path),
            'valid_roi_ids': [123, 124],
            'output_event_file': str(tmp_path / "junk_output.npz"),
            'decay_time': decay_time
            }
    ed = emod.EventDetection(input_data=args, args=[])
    ed.run()

    with np.load(args['output_event_file'], allow_pickle=True) as f:
        events = f['events']

    # empirically, this test case has very small events at the end
    # truncate before comparing to expectations
    events = events[:, 0:950]
    for result, expected in zip(events, expected_events):
        nresult = np.count_nonzero(result)
        # check that the number of events match the expectation:
        assert nresult == len(expected.timestamps)

        # check that they are in the right place:
        result_index = np.argwhere(result != 0).flatten()
        np.testing.assert_allclose(result_index, expected.timestamps, atol=1)
