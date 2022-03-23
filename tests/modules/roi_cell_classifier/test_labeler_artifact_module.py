import pytest
import tempfile
import h5py
import json
import pathlib
import copy
import numpy as np
import os
import PIL.Image
from itertools import product

from ophys_etl.utils.array_utils import normalize_array

from ophys_etl.utils.rois import (
    sanitize_extract_roi_list,
    extract_roi_to_ophys_roi)

from ophys_etl.modules.roi_cell_classifier.compute_labeler_artifacts import (
    LabelerArtifactGenerator)

from ophys_etl.modules.roi_cell_classifier.utils import (
    get_traces)

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)


@pytest.mark.parametrize(
        "video_lower_quantile, video_upper_quantile,"
        "projection_lower_quantile, projection_upper_quantile, use_graph, "
        "with_motion_border",
        product((0.1, 0.2), (0.7, 0.8), (0.1, 0.2), (0.7, 0.8),
                (True, False), (True, False)))
def test_labeler_artifact_generator(
        tmp_path_factory,
        classifier2021_video_fixture,
        classifier2021_video_hash_fixture,
        suite2p_roi_fixture,
        suite2p_roi_hash_fixture,
        classifier2021_corr_graph_fixture,
        classifier2021_corr_graph_hash_fixture,
        classifier2021_corr_png_fixture,
        classifier2021_corr_png_hash_fixture,
        video_lower_quantile,
        video_upper_quantile,
        projection_lower_quantile,
        projection_upper_quantile,
        use_graph,
        with_motion_border):
    """
    Test that LabelerArtifactGenerator runs and produces expected output
    """

    tmpdir = tmp_path_factory.mktemp('full_artifact_generation')
    if with_motion_border:
        motion_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir,
                                                    suffix='.csv')[1])
        with open(motion_path, 'w') as out_file:
            out_file.write('x,y\n')
            out_file.write('5,6\n')
            out_file.write('14,-3\n')
        expected_motion_border = {'bottom': 6.0,
                                  'top': 3.0,
                                  'right_side': 14.0,
                                  'left_side': 0.0}

        motion_path = str(motion_path.resolve().absolute())

    else:
        motion_path = None
        expected_motion_border = {'top': 0,
                                  'bottom': 0,
                                  'left_side': 0,
                                  'right_side': 0}

    if use_graph:
        corr_fixture = classifier2021_corr_graph_fixture
        corr_hash = classifier2021_corr_graph_hash_fixture
    else:
        corr_fixture = classifier2021_corr_png_fixture
        corr_hash = classifier2021_corr_png_hash_fixture

    output_tuple = tempfile.mkstemp(dir=tmpdir,
                                    prefix='artifact_file_',
                                    suffix='.h5')

    # without this, got a "too many files open" error
    os.close(output_tuple[0])

    output_path = pathlib.Path(output_tuple[1])

    # because tempfile.mkstemp actually creates the file
    output_path.unlink()

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
    input_data['motion_border_path'] = motion_path

    generator = LabelerArtifactGenerator(input_data=input_data, args=[])
    generator.run()

    assert output_path.is_file()

    with h5py.File(output_path, 'r') as artifact_file:

        motion_border = json.loads(
                          artifact_file['motion_border'][()].decode('utf-8'))
        assert motion_border == expected_motion_border
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
        assert artifact_file['video_data'].chunks is not None
        scaled_video = artifact_file['video_data'][()]

        with h5py.File(classifier2021_video_fixture, 'r') as raw_file:
            raw_video = raw_file['data'][()]
        raw_max = np.max(raw_video, axis=0)
        raw_avg = np.mean(raw_video, axis=0)

        mn, mx = np.quantile(raw_video, (video_lower_quantile,
                                         video_upper_quantile))

        raw_video = np.where(raw_video > mn, raw_video, mn)
        raw_video = np.where(raw_video < mx, raw_video, mx)
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
            raw_img = np.where(raw_img > mn, raw_img, mn)
            raw_img = np.where(raw_img < mx, raw_img, mx)
            raw_img = raw_img.astype(float)
            np.testing.assert_array_equal(raw_img, artifact_img)

        artifact_corr = artifact_file['correlation_projection'][()]
        if use_graph:
            expected_corr = normalize_array(
                              graph_to_img(
                                    corr_fixture,
                                    attribute_name='filtered_hnc_Gaussian'))
        else:
            expected_corr = normalize_array(
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
    if with_motion_border:
        assert 'motion_csv' in metadata
    else:
        assert 'motion_csv' not in metadata

    tmpdir = pathlib.Path(tmpdir)
    path_list = [n for n in tmpdir.rglob('*')]
    for this_path in path_list:
        if this_path.is_file():
            try:
                this_path.unlink()
            except Exception:
                pass


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
        LabelerArtifactGenerator(input_data=input_data, args=[])

    input_data['clobber'] = True
    LabelerArtifactGenerator(input_data=input_data, args=[])


@pytest.fixture(scope='session')
def well_made_config_fixture(
        classifier2021_video_fixture,
        suite2p_roi_fixture,
        tmp_path_factory):
    """
    A dict representing the input_json for LabelerArtifactGenerator.
    This one will pass validation.
    """

    tmpdir = tmp_path_factory.mktemp('for_config')
    corr_path = tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1]
    output_path = tempfile.mkstemp(dir=tmpdir, suffix='.h5')[1]

    input_data = dict()
    input_data['video_path'] = str(classifier2021_video_fixture)
    input_data['roi_path'] = str(suite2p_roi_fixture)
    input_data['correlation_path'] = str(corr_path)
    input_data['artifact_path'] = str(output_path)
    input_data['clobber'] = True

    yield input_data


@pytest.mark.parametrize(
        'bad_key',
        ['video_path', 'roi_path',
         'correlation_path', 'artifact_path',
         None])
def test_sufix_validation(
        well_made_config_fixture,
        tmp_path_factory,
        bad_key):
    """
    Test that if you specify a file with the wrong suffix as an input,
    you get an error
    """

    tmpdir = tmp_path_factory.mktemp('to_test_config')
    bad_file = tempfile.mkstemp(dir=tmpdir, suffix='.txt')[1]

    input_data = copy.deepcopy(well_made_config_fixture)
    if bad_key is None:
        LabelerArtifactGenerator(input_data=input_data, args=[])
    else:
        input_data.pop(bad_key)
        input_data[bad_key] = bad_file
        with pytest.raises(ValueError, match='must have suffix'):
            LabelerArtifactGenerator(input_data=input_data, args=[])
