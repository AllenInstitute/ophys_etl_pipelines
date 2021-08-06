import pytest
import contextlib
from marshmallow import ValidationError

from ophys_etl.modules.segmentation.modules import schemas


@pytest.fixture
def video_path_fixture(tmpdir):
    video_path = tmpdir / "video.h5"
    with open(video_path, "w") as f:
        f.write("dummy content")
    yield str(video_path)


@pytest.fixture
def graph_path_fixture(tmpdir):
    graph_path = tmpdir / "graph.pkl"
    with open(graph_path, "w") as f:
        f.write("dummy content")
    yield str(graph_path)


@pytest.mark.parametrize(
        "overwrite_log, file_exists, context",
        [
            (True, True,
             pytest.warns(UserWarning, match=".*deleting contents.*")),
            (True, False,
             contextlib.nullcontext()),
            (False, True,
             pytest.raises(ValidationError, match="already exists")),
            (False, False,
             contextlib.nullcontext())
            ])
def test_segmentation_log_overwrite(tmpdir, overwrite_log, file_exists,
                                    context, video_path_fixture,
                                    graph_path_fixture):
    log_path = tmpdir / "log_test.h5"

    if file_exists:
        with open(log_path, "w") as f:
            f.write("dummy content")

    args = {
            "log_path": str(log_path),
            "overwrite_log": overwrite_log,
            "video_input": video_path_fixture,
            "graph_input": graph_path_fixture
            }

    with context:
        schema = schemas.SharedSegmentationInputSchema()
        schema.load(data=args)
