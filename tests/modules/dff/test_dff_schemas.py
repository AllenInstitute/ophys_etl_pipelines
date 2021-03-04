import pytest
from ophys_etl.modules.dff.schemas import DffJobSchema


@pytest.mark.parametrize(
    "frame_rate, short_filter_s, long_filter_s, expected_short, expected_long",
    [
        (30., 3.3, 600., 99, 18001),
        (11.1, 10.0, 1001., 111, 11111)
    ]
)
def test_dff_schema_post_load(tmp_path, trace_h5, frame_rate, short_filter_s,
                              long_filter_s, expected_short, expected_long):
    args = {
        "input_file": str(tmp_path / "input_trace.h5"),
        "output_file": str(tmp_path / "output_dff.h5"),
        "long_baseline_filter_s": long_filter_s,
        "short_filter_s": short_filter_s,
        "movie_frame_rate_hz": frame_rate
    }
    data = DffJobSchema().load(args)
    assert data["short_filter_frames"] == expected_short
    assert isinstance(data["short_filter_frames"], int)
    assert data["long_filter_frames"] == expected_long
    assert isinstance(data["long_filter_frames"], int)
