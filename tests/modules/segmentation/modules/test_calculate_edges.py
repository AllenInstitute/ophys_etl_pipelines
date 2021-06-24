import pytest
import h5py
import numpy as np
from pathlib import Path
from marshmallow import ValidationError

from ophys_etl.modules.segmentation.modules import calculate_edges as ce


@pytest.fixture
def video_path(tmpdir, request):
    vpath = Path(tmpdir / "video.h5")
    data = np.random.randint(0, 2**15,
                             size=request.param["video_shape"],
                             dtype='uint16')
    with h5py.File(vpath, "w") as f:
        f.create_dataset("data", data=data)
    yield vpath


@pytest.mark.parametrize(
        "video_path",
        [
            {"video_shape": (10, 8, 40)}
        ], indirect=["video_path"])
@pytest.mark.parametrize(
        "attribute_name",
        ["Pearson", "filtered_Pearson",
         "hnc_Gaussian", "filtered_hnc_Gaussian"])
def test_CalculateEdges(video_path, attribute_name, tmpdir):
    graph_output = Path(tmpdir / "graph.pkl")
    plot_output = Path(tmpdir / "plot.png")
    args = {
            "video_path": str(video_path),
            "plot_output": str(plot_output),
            "graph_output": str(graph_output),
            "attribute_name": attribute_name,
            "n_parallel_workers": 1}
    ecalc = ce.CalculateEdges(input_data=args, args=[])
    ecalc.run()
    assert graph_output.exists()
    assert plot_output.exists()


@pytest.mark.parametrize(
        "video_path",
        [
            {"video_shape": (10, 8, 40)}
        ], indirect=["video_path"])
def test_CalculateEdgesMultiprocessing(video_path, tmpdir):
    graph_output = Path(tmpdir / "graph.pkl")
    plot_output = Path(tmpdir / "plot.png")
    args = {
            "video_path": str(video_path),
            "plot_output": str(plot_output),
            "graph_output": str(graph_output),
            "attribute_name": "Pearson",
            "n_parallel_workers": 2}
    ecalc = ce.CalculateEdges(input_data=args, args=[])
    ecalc.run()
    assert graph_output.exists()
    assert plot_output.exists()


@pytest.mark.parametrize(
        "video_path",
        [
            {"video_shape": (10, 8, 40)}
        ], indirect=["video_path"])
@pytest.mark.parametrize("filter_fraction", [0.0, 1.0001])
def test_CalculateEdgesRangeValidate(video_path, tmpdir, filter_fraction):
    graph_output = Path(tmpdir / "graph.pkl")
    plot_output = Path(tmpdir / "plot.png")
    args = {
            "video_path": str(video_path),
            "plot_output": str(plot_output),
            "graph_output": str(graph_output),
            "attribute_name": "Pearson",
            "n_parallel_workers": 2,
            "filter_fraction": filter_fraction}
    with pytest.raises(ValidationError):
        ce.CalculateEdges(input_data=args, args=[])
