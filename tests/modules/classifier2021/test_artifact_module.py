import pytest
import tempfile
import h5py
import json
import pathlib
import numpy as np
import PIL.Image
from itertools import product

from ophys_etl.modules.segmentation.utils.roi_utils import (
    sanitize_extract_roi_list,
    extract_roi_to_ophys_roi)

from ophys_etl.modules.classifier2021.compute_artifacts import (
    ArtifactGenerator)

from ophys_etl.modules.classifier2021.utils import (
    scale_img_to_uint8,
    get_traces)

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)


@pytest.mark.parametrize(
        "video_lower_quantile,video_upper_quantile,"
        "projection_lower_quantile,projection_upper_quantile, use_graph",
        product((0.1, 0.2), (0.7, 0.8), (0.1, 0.2), (0.7, 0.8),
                (True, False)))
def test_with_graph(
        classifier2021_video_fixture,
        classifier2021_video_hash_fixture,
        suite2p_roi_fixture,
        suite2p_roi_hash_fixture,
        classifier2021_corr_graph_fixture,
        classifier2021_corr_graph_hash_fixture,
        classifier2021_corr_png_fixture,
        classifier2021_corr_png_hash_fixture,
        tmpdir,
        video_lower_quantile,
        video_upper_quantile,
        projection_lower_quantile,
        projection_upper_quantile,
        use_graph):

    if use_graph:
        corr_fixture = classifier2021_corr_graph_fixture
        corr_hash = classifier2021_corr_graph_hash_fixture
    else:
        corr_fixture = classifier2021_corr_png_fixture
        corr_hash = classifier2021_corr_png_hash_fixture

    output_path = tempfile.mkstemp(dir=tmpdir,
                                   prefix='artifact_file_',
                                   suffix='.h5')[1]

    output_path = pathlib.Path(output_path)

    # because tempfile.mkstemp actually creates the file
    output_path.unlink()
    #assert not output_path.exists()

    input_data = dict()
    input_data['video_path'] = str(classifier2021_video_fixture)
    input_data['roi_path'] = str(suite2p_roi_fixture)
    input_data['correlation_path'] = str(corr_fixture)
    input_data['artifact_path'] = str(output_path)
    input_data['clobber'] = False
    input_data['video_lower_quantile'] = video_lower_quantile
    input_data['video_upper_quantile'] = video_upper_quantile
    input_data['projection_lower_quantile'] = projection_lower_quantile
    input_data['projection_upper_quantile'] = projection_upper_quantile

    generator = ArtifactGenerator(input_data=input_data, args=[])
    generator.run()

    assert output_path.is_file()

    with h5py.File(output_path, 'r') as artifact_file:

        # test that ROIs were written correctly
        with open(suite2p_roi_fixture, 'rb') as in_file:
            expected_rois = json.load(in_file)
        expected_rois = sanitize_extract_roi_list(expected_rois)

        artifact_rois = json.loads(
                           artifact_file['rois'][()].decode('utf-8'))

        assert expected_rois == artifact_rois

        # test that all ROIs appear in color map
        color_map = json.loads(
                        artifact_file['roi_color_map'][()].decode('utf-8'))
        assert len(color_map) == len(expected_rois)
        for roi in expected_rois:
            assert str(roi['id']) in color_map

        # test that traces were written correctly
        ophys_rois = [extract_roi_to_ophys_roi(roi)
                      for roi in expected_rois]
        expected_traces = get_traces(classifier2021_video_fixture,
                                     ophys_rois)

        for roi_id in expected_traces:
            np.testing.assert_array_equal(
                    expected_traces[roi_id],
                    artifact_file[f'traces/{roi_id}'][()])

        # test that the scaled video data was written correctly
        scaled_video = artifact_file['video_data'][()]

        with h5py.File(classifier2021_video_fixture, 'r') as raw_file:
            raw_video = raw_file['data'][()]
        raw_max = np.max(raw_video, axis=0)
        raw_avg = np.mean(raw_video, axis=0)

        mn, mx = np.quantile(raw_video, (video_lower_quantile,
                                         video_upper_quantile))

        raw_video = np.where(raw_video>mn, raw_video, mn)
        raw_video = np.where(raw_video<mx, raw_video, mx)
        delta = mx-mn
        raw_video = raw_video-mn
        raw_video = raw_video.astype(float)
        raw_video = np.round(255.0*raw_video/delta).astype(np.uint8)
        np.testing.assert_array_equal(raw_video, scaled_video)
        del raw_video
        del scaled_video

        # test that max and avg projection images wer written correctly
        for raw_img, img_key in zip((raw_max, raw_avg),
                                    ('max_projection', 'avg_projection')):
            artifact_img = artifact_file[img_key][()]
            mn, mx = np.quantile(raw_img,
                                 (projection_lower_quantile,
                                  projection_upper_quantile))
            raw_img = np.where(raw_img>mn, raw_img, mn)
            raw_img = np.where(raw_img<mx, raw_img, mx)
            delta = mx-mn
            raw_img = raw_img-mn
            raw_img = raw_img.astype(float)
            raw_img = np.round(255.0*raw_img/delta).astype(np.uint8)
            np.testing.assert_array_equal(raw_img, artifact_img)

        artifact_corr = artifact_file['correlation_projection'][()]
        if use_graph:
            expected_corr = scale_img_to_uint8(
                              graph_to_img(
                                    corr_fixture,
                                    attribute_name='filtered_hnc_Gaussian'))
        else:
            expected_corr = scale_img_to_uint8(
                                np.array(PIL.Image.open(corr_fixture, 'r')))

        np.testing.assert_array_equal(artifact_corr, expected_corr)

        metadata = json.loads(artifact_file['metadata'][()].decode('utf-8'))

    # test that metadata has the right contents
    assert metadata['video']['path'] == str(classifier2021_video_fixture)
    assert metadata['video']['hash'] == classifier2021_video_hash_fixture

    assert metadata['rois']['path'] == str(suite2p_roi_fixture)
    assert metadata['rois']['hash'] == suite2p_roi_hash_fixture

    assert metadata['correlation']['path'] == str(corr_fixture)
    assert metadata['correlation']['hash'] == corr_hash

    assert metadata['generator_args'] == input_data



def test_clobber_error(
        classifier2021_video_fixture,
        suite2p_roi_fixture,
        classifier2021_corr_graph_fixture,
        tmpdir):
    """
    Test that the artifact generator will not let you over write an
    existing file unless you specify clobber=True
    """

    output_path = tempfile.mkstemp(dir=tmpdir,
                                   prefix='artifact_file_',
                                   suffix='.h5')[1]

    output_path = pathlib.Path(output_path)
    assert output_path.exists()

    input_data = dict()
    input_data['video_path'] = str(classifier2021_video_fixture)
    input_data['roi_path'] = str(suite2p_roi_fixture)
    input_data['correlation_path'] = str(classifier2021_corr_graph_fixture)
    input_data['artifact_path'] = str(output_path)
    input_data['clobber'] = False


    with pytest.raises(RuntimeError, match='--clobber=True'):
        ArtifactGenerator(input_data=input_data, args=[])

    input_data['clobber'] = True
    ArtifactGenerator(input_data=input_data, args=[])
