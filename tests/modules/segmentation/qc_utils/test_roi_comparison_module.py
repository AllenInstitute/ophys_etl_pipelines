import pytest
import pathlib

from ophys_etl.modules.segmentation.modules.roi_comparison import (
    ROIComparisonSchema,
    ROIComparisonEngine)


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


@pytest.mark.parametrize('roi_names, background_names',
                         [(None, None), (['ROIs'], ['pkl', 'png'])])
def test_roi_comparison_engine(tmpdir,
                               roi_file,
                               background_pkl,
                               background_png,
                               roi_names,
                               background_names):

    plot_path = pathlib.Path(tmpdir)/'output_plot.png'
    data = {'roi_paths': [str(roi_file)],
            'roi_names': roi_names,
            'background_paths': [str(background_pkl), str(background_png)],
            'background_names': background_names,
            'attribute_name': 'dummy_value',
            'figsize_per': 3,
            'plot_output': str(plot_path)}

    engine = ROIComparisonEngine(input_data=data, args=[])
    engine.run()
    assert plot_path.is_file()
