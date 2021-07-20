import pytest
import pathlib

from ophys_etl.modules.segmentation.modules.roi_comparison import (
    ROIComparisonSchema)


def test_roi_comparison_schema(tmpdir):
    tmpdir_path = pathlib.Path(tmpdir)
    roi_paths = [str(tmpdir_path/'a.json'),
                 str(tmpdir_path/'b.json')]
    bckgd_paths = [str(tmpdir_path/'c.png'),
                   str(tmpdir_path/'d.pkl')]

    for file_path in bckgd_paths+roi_paths:
        with open(file_path, 'w') as out_file:
            out_file.write('junk')

    roi_schema = ROIComparisonSchema()

    data = {'background_paths': bckgd_paths,
            'background_names': ['a', 'b'],
            'roi_paths': roi_paths,
            'roi_names': ['aa', 'bb'],
            'plot_output': str(tmpdir_path/'junk.png')}

    roi_schema.load(data)

    data = {'background_paths': bckgd_paths,
            'roi_paths': roi_paths,
            'roi_names': ['aa', 'bb'],
            'plot_output': str(tmpdir_path/'junk.png')}

    result = roi_schema.load(data)
    assert result['background_names'] == ['c.png', 'd.pkl']

    data = {'background_paths': bckgd_paths,
            'background_names': ['a', 'b'],
            'roi_paths': roi_paths,
            'plot_output': str(tmpdir_path/'junk.png')}

    result = roi_schema.load(data)
    assert result['roi_names'] == ['a.json', 'b.json']

    data = {'background_paths': bckgd_paths,
            'roi_paths': roi_paths,
            'plot_output': str(tmpdir_path/'junk.png')}

    result = roi_schema.load(data)
    assert result['roi_names'] == ['a.json', 'b.json']
    assert result['background_names'] == ['c.png', 'd.pkl']

    with pytest.raises(RuntimeError, match='should be the same length'):

        data = {'background_paths': bckgd_paths,
                'background_names': ['a', 'b', 'c'],
                'roi_paths': roi_paths,
                'roi_names': ['aa', 'bb'],
                'plot_output': str(tmpdir_path/'junk.png')}

        roi_schema.load(data)

    with pytest.raises(RuntimeError, match='should be the same length'):

        data = {'background_paths': bckgd_paths,
                'background_names': ['a', 'b'],
                'roi_paths': roi_paths,
                'roi_names': ['aa'],
                'plot_output': str(tmpdir_path/'junk.png')}

        roi_schema.load(data)
