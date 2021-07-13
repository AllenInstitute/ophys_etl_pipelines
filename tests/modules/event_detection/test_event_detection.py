import pytest
import h5py
import numpy as np
from collections import namedtuple

from ophys_etl.modules.event_detection import validation

try:
    import ophys_etl.modules.event_detection.__main__ as emod
except ModuleNotFoundError:
    # even though we might skip tests, pytest tries these imports
    from unittest.mock import Mock
    import sys
    sys.modules['FastLZeroSpikeInference'] = Mock()
    import ophys_etl.modules.event_detection.__main__ as emod


Events = namedtuple('Events', ['id', 'timestamps', 'magnitudes'])


@pytest.fixture(scope="function")
def dff_hdf5(tmp_path, request):
    sigma = request.param.get("sigma")
    offset = request.param.get("offset")
    decay_time = request.param.get("decay_time")
    nframes = request.param.get("nframes")
    events = request.param.get("events")
    rate = request.param.get("rate")
    noise_mult = request.param.get("noise_multiplier")

    rng = np.random.default_rng(42)
    data = rng.normal(loc=offset, scale=sigma, size=(len(events), nframes))
    for i, event in enumerate(events):
        data[i] += validation.sum_events(nframes, event.timestamps,
                                         event.magnitudes,
                                         decay_time, rate)

    names = [i.id for i in events]

    h5path = tmp_path / "fake_dff.h5"
    with h5py.File(h5path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("roi_names", data=names)

    with h5py.File(h5path, "r") as f:
        yield h5path, decay_time, rate, events, noise_mult


@pytest.mark.event_detect_only
@pytest.mark.parametrize(
        "dff_hdf5",
        [
            {
                "sigma": 1.0,
                "decay_time": 0.415,
                "offset": 0.0,
                "nframes": 1000,
                "rate": 31.0,
                "noise_multiplier": 1.0,
                "events": [
                    Events(
                        id=123,
                        timestamps=[145, 212, 280, 310, 430, 600, 890],
                        magnitudes=[4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0]),
                    Events(
                        id=124,
                        timestamps=[45, 112, 232, 410, 490, 650, 850],
                        magnitudes=[4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0])]
                    },
            {
                "sigma": 1.0,
                "decay_time": 0.415,
                "offset": 0.0,
                "nframes": 1000,
                "rate": 11.0,
                "noise_multiplier": 3.0,
                "events": []},
            {
                "sigma": 1.0,
                "decay_time": 0.415,
                "offset": 0.0,
                "nframes": 1000,
                "rate": 11.0,
                "noise_multiplier": 3.0,
                "events": [
                    Events(
                        id=123,
                        timestamps=[145, 212, 280, 310, 430, 600, 890],
                        magnitudes=[4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0]),
                    Events(
                        id=124,
                        timestamps=[45, 112, 232, 410, 490, 650, 850],
                        magnitudes=[4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0])]
                    },
            ], indirect=True)
def test_EventDetection(dff_hdf5, tmp_path):
    """This test runs the actual spike inference on fake data. The fake
    data is constructed by the dff_hdf5 fixture. This is an
    easy test, in that the SNR for the spikes is high, there is neither offset
    nor low frequency background, the fake spikes are not too close together,
    and, some false spikes are ignored at the end.
    """

    dff_path, decay_time, rate, expected_events, noise_mult = dff_hdf5

    args = {
            'movie_frame_rate_hz': rate,
            'ophysdfftracefile': str(dff_path),
            'valid_roi_ids': [123, 124],
            'output_event_file': str(tmp_path / "junk_output.h5"),
            'decay_time': decay_time,
            'noise_multiplier': noise_mult
            }
    ed = emod.EventDetection(input_data=args, args=[])
    ed.run()

    with h5py.File(args['output_event_file'], "r") as f:
        keys = list(f.keys())
        for k in ['events', 'roi_names', 'noise_stds', 'lambdas']:
            assert k in keys
        events = f['events'][()]
        if expected_events == []:
            assert "warning" in keys
            assert "No valid ROIs in" in str(f['warning'][()])

    for result, expected in zip(events, expected_events):
        nresult = np.count_nonzero(result)
        result_index = np.argwhere(result != 0).flatten()
        # check that the number of events match the expectation:
        print(result_index)
        print(expected.timestamps)
        print(ed.args['noise_multiplier'])
        assert nresult == len(expected.timestamps)

        # check that they are in the right place:
        np.testing.assert_array_equal(result_index, expected.timestamps)
