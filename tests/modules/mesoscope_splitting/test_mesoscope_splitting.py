import h5py
import numpy as np
import pytest
from unittest.mock import (
    MagicMock, Mock, patch, create_autospec)

from ophys_etl.modules.mesoscope_splitting import __main__ as \
    run_mesoscope_splitting
from ophys_etl.modules.mesoscope_splitting.tiff import MesoscopeTiff
from ophys_etl.modules.mesoscope_splitting.conversion_utils import (
    volume_to_tif)


@pytest.fixture
def mock_volume():
    volume = MagicMock()
    volume.plane_shape = (100, 100)

    return volume


@pytest.fixture
def exp_info():
    experiment_info = {"resolution": 0.5,
                       "offset_x": 20,
                       "offset_y": 50,
                       "rotation": 0.1}

    return experiment_info


def test_conversion_output(mock_volume, exp_info):
    run_mesoscope_splitting.conversion_output(
        mock_volume, "test.out", exp_info)


@pytest.fixture
def experiments(tmpdir):
    """ These values are from real experiments where we noticed
    a potential problem and needed to investigate. Only the
    experiment_id and storage_directory have been changed.
    """
    exps = [
        {
            "experiment_id": 0,
            "storage_directory": tmpdir,
            "roi_index": 0,
            "scanfield_z": 85,
            "resolution": 0,
            "offset_x": 0,
            "offset_y": 0,
            "rotation": -108.1414
        },
        {
            "experiment_id": 1,
            "storage_directory": tmpdir,
            "roi_index": 0,
            "scanfield_z": -11,
            "resolution": 0,
            "offset_x": 0,
            "offset_y": 0,
            "rotation": -108.1414
        },
        {
            "experiment_id": 2,
            "storage_directory": tmpdir,
            "roi_index": 0,
            "scanfield_z": 155,
            "resolution": 0,
            "offset_x": 0,
            "offset_y": 0,
            "rotation": -108.1414
        },
        {
            "experiment_id": 3,
            "storage_directory": tmpdir,
            "roi_index": 0,
            "scanfield_z": -111,
            "resolution": 0,
            "offset_x": 0,
            "offset_y": 0,
            "rotation": -108.1414
        },
        {
            "experiment_id": 4,
            "storage_directory": tmpdir,
            "roi_index": 1,
            "scanfield_z": 165,
            "resolution": 0,
            "offset_x": 0,
            "offset_y": 0,
            "rotation": -108.1414
        },
        {
            "experiment_id": 5,
            "storage_directory": tmpdir,
            "roi_index": 1,
            "scanfield_z": 69,
            "resolution": 0,
            "offset_x": 0,
            "offset_y": 0,
            "rotation": -108.1414
        },
        {
            "experiment_id": 6,
            "storage_directory": tmpdir,
            "roi_index": 1,
            "scanfield_z": 245,
            "resolution": 0,
            "offset_x": 0,
            "offset_y": 0,
            "rotation": -108.1414
        },
        {
            "experiment_id": 7,
            "storage_directory": tmpdir,
            "roi_index": 1,
            "scanfield_z": -31,
            "resolution": 0,
            "offset_x": 0,
            "offset_y": 0,
            "rotation": -108.1414
        }
    ]

    return exps


@pytest.fixture
def timeseries_roi_metadata():
    roi_list = [
        {
            'ver': 1,
            'classname': 'scanimage.mroi.Roi',
            'name': 'V1',
            'zs': [-111, -11, 85, 155],
            'scanfields': [
                {
                    'ver': 1,
                    'classname': 'scanimage.mroi.scanfield.'
                                 'fields.RotatedRectangle',
                    'name': '',
                    'centerXY': [2.283966727, -7.963886189],
                    'sizeXY': [2.366863905, 2.366863905],
                    'rotationDegrees': 0,
                    'enable': 1,
                    'pixelResolutionXY': [512, 512],
                    'pixelToRefTransform': [
                        [0.004622781065, 0, 1.098223383],
                        [0, 0.004622781065, -9.149629532],
                        [0, 0, 1]
                    ],
                    'affine': [
                        [2.366863905, 0, 1.100534774],
                        [0, 2.366863905, -9.147318141],
                        [0, 0, 1]
                    ]
                },
                {
                    'ver': 1,
                    'classname': 'scanimage.mroi.scanfield.'
                                 'fields.RotatedRectangle',
                    'name': '',
                    'centerXY': [2.283966727, -7.963886189],
                    'sizeXY': [2.366863905, 2.366863905],
                    'rotationDegrees': 0,
                    'enable': 1,
                    'pixelResolutionXY': [512, 512],
                    'pixelToRefTransform': [
                        [0.004622781065, 0, 1.098223383],
                        [0, 0.004622781065, -9.149629532],
                        [0, 0, 1]
                    ],
                    'affine': [
                        [2.366863905, 0, 1.100534774],
                        [0, 2.366863905, -9.147318141],
                        [0, 0, 1]
                    ]
                },
                {
                    'ver': 1,
                    'classname': 'scanimage.mroi.scanfield.'
                                 'fields.RotatedRectangle',
                    'name': '',
                    'centerXY': [2.283966727, -7.963886189],
                    'sizeXY': [2.366863905, 2.366863905],
                    'rotationDegrees': 0,
                    'enable': 1,
                    'pixelResolutionXY': [512, 512],
                    'pixelToRefTransform': [
                        [0.004622781065, 0, 1.098223383],
                        [0, 0.004622781065, -9.149629532],
                        [0, 0, 1]
                    ],
                    'affine': [
                        [2.366863905, 0, 1.100534774],
                        [0, 2.366863905, -9.147318141],
                        [0, 0, 1]
                    ]
                },
                {
                    'ver': 1,
                    'classname': 'scanimage.mroi.scanfield.'
                                 'fields.RotatedRectangle',
                    'name': '',
                    'centerXY': [2.283966727, -7.963886189],
                    'sizeXY': [2.366863905, 2.366863905],
                    'rotationDegrees': 0,
                    'enable': 1,
                    'pixelResolutionXY': [512, 512],
                    'pixelToRefTransform': [
                        [0.004622781065, 0, 1.098223383],
                        [0, 0.004622781065, -9.149629532],
                        [0, 0, 1]
                    ],
                    'affine': [
                        [2.366863905, 0, 1.100534774],
                        [0, 2.366863905, -9.147318141],
                        [0, 0, 1]
                    ]
                }
            ],
            'discretePlaneMode': 1,
            'powers': None,
            'pzAdjust': None,
            'Lzs': None,
            'interlaceDecimation': None,
            'interlaceOffset': None
        },
        {
            'ver': 1,
            'classname': 'scanimage.mroi.Roi',
            'name': 'LM',
            'zs': [-31, 69, 165, 245],
            'scanfields': [
                {
                    'ver': 1,
                    'classname': 'scanimage.mroi.scanfield.'
                                 'fields.RotatedRectangle',
                    'name': '',
                    'centerXY': [-0.587744573, 1.268341762],
                    'sizeXY': [2.366863905, 2.366863905],
                    'rotationDegrees': 0,
                    'enable': 1,
                    'pixelResolutionXY': [512, 512],
                    'pixelToRefTransform': [
                        [0.004622781065, 0, -1.773487916],
                        [0, 0.004622781065, 0.08259841914],
                        [0, 0, 1]
                    ],
                    'affine': [
                        [2.366863905, 0, -1.771176526],
                        [0, 2.366863905, 0.08490980967],
                        [0, 0, 1]
                    ]
                },
                {
                    'ver': 1,
                    'classname': 'scanimage.mroi.scanfield.'
                                 'fields.RotatedRectangle',
                    'name': '',
                    'centerXY': [-0.587744573, 1.268341762],
                    'sizeXY': [2.366863905, 2.366863905],
                    'rotationDegrees': 0,
                    'enable': 1,
                    'pixelResolutionXY': [512, 512],
                    'pixelToRefTransform': [
                        [0.004622781065, 0, -1.773487916],
                        [0, 0.004622781065, 0.08259841914],
                        [0, 0, 1]
                    ],
                    'affine': [
                        [2.366863905, 0, -1.771176526],
                        [0, 2.366863905, 0.08490980967],
                        [0, 0, 1]
                    ]
                },
                {
                    'ver': 1,
                    'classname': 'scanimage.mroi.scanfield.'
                                 'fields.RotatedRectangle',
                    'name': '',
                    'centerXY': [-0.587744573, 1.268341762],
                    'sizeXY': [2.366863905, 2.366863905],
                    'rotationDegrees': 0,
                    'enable': 1,
                    'pixelResolutionXY': [512, 512],
                    'pixelToRefTransform': [
                        [0.004622781065, 0, -1.773487916],
                        [0, 0.004622781065, 0.08259841914],
                        [0, 0, 1]
                    ],
                    'affine': [
                        [2.366863905, 0, -1.771176526],
                        [0, 2.366863905, 0.08490980967],
                        [0, 0, 1]
                    ]
                },
                {
                    'ver': 1,
                    'classname': 'scanimage.mroi.scanfield.'
                                 'fields.RotatedRectangle',
                    'name': '',
                    'centerXY': [-0.587744573, 1.268341762],
                    'sizeXY': [2.366863905, 2.366863905],
                    'rotationDegrees': 0,
                    'enable': 1,
                    'pixelResolutionXY': [512, 512],
                    'pixelToRefTransform': [
                        [0.004622781065, 0, -1.773487916],
                        [0, 0.004622781065, 0.08259841914],
                        [0, 0, 1]
                    ],
                    'affine': [
                        [2.366863905, 0, -1.771176526],
                        [0, 2.366863905, 0.08490980967],
                        [0, 0, 1]
                    ]
                }
            ],
            'discretePlaneMode': 1,
            'powers': None,
            'pzAdjust': None,
            'Lzs': None,
            'interlaceDecimation': None,
            'interlaceOffset': None
        }
    ]

    return {
        "RoiGroups": {
            "imagingRoiGroup": {
                "rois": roi_list
            }
        }
    }


@pytest.fixture
def surface_image_roi_metadata():
    roi_list = [
        {
            'Lzs': None,
            'classname': 'scanimage.mroi.Roi',
            'discretePlaneMode': 1,
            'interlaceDecimation': None,
            'interlaceOffset': None,
            'name': 'V1',
            'powers': None,
            'pzAdjust': None,
            'scanfields': {
                'affine': [
                    [2.366863905, 0, 1.100534774],
                    [0, 2.366863905, -9.147318141],
                    [0, 0, 1]
                ],
                'centerXY': [2.283966727, -7.963886189],
                'classname': 'scanimage.mroi.scanfield.'
                             'fields.RotatedRectangle',
                'enable': 1,
                'name': '',
                'pixelResolutionXY': [512, 512],
                'pixelToRefTransform': [
                    [0.004622781065, 0, 1.098223383],
                    [0, 0.004622781065, -9.149629532],
                    [0, 0, 1]
                ],
                'rotationDegrees': 0,
                'sizeXY': [2.366863905, 2.366863905],
                'ver': 1
            },
            'ver': 1,
            'zs': -190
        },
        {
            'Lzs': None,
            'classname': 'scanimage.mroi.Roi',
            'discretePlaneMode': 1,
            'interlaceDecimation': None,
            'interlaceOffset': None,
            'name': 'LM',
            'powers': None,
            'pzAdjust': None,
            'scanfields': {
                'affine': [
                    [2.366863905, 0, -1.771176526],
                    [0, 2.366863905, 0.08490980967],
                    [0, 0, 1]
                ],
                'centerXY': [-0.587744573, 1.268341762],
                'classname': 'scanimage.mroi.scanfield'
                             '.fields.RotatedRectangle',
                'enable': 1,
                'name': '',
                'pixelResolutionXY': [512, 512],
                'pixelToRefTransform': [
                    [0.004622781065, 0, -1.773487916],
                    [0, 0.004622781065, 0.08259841914],
                    [0, 0, 1]
                ],
                'rotationDegrees': 0,
                'sizeXY': [2.366863905, 2.366863905],
                'ver': 1
            },
            'ver': 1,
            'zs': -110
        }
    ]

    return {
        "RoiGroups": {
            "imagingRoiGroup": {
                "rois": roi_list
            }
        }
    }


@pytest.fixture
def z_stack_roi_metadata():
    roi_list = [
        {
            'Lzs': None,
            'classname': 'scanimage.mroi.Roi',
            'discretePlaneMode': 0,
            'interlaceDecimation': None,
            'interlaceOffset': None,
            'name': 'V1',
            'powers': None,
            'pzAdjust': None,
            'scanfields': {
                'affine': [
                    [2.366863905, 0, 1.100534775],
                    [0, 2.366863905, -9.147318141],
                    [0, 0, 1]
                ],
                'centerXY': [2.283966727, -7.963886189],
                'classname': 'scanimage.mroi.scanfield.'
                             'fields.RotatedRectangle',
                'enable': 1,
                'name': '',
                'pixelResolutionXY': [512, 512],
                'pixelToRefTransform': [
                    [0.004622781064, 0, 1.098223384],
                    [0, 0.004622781064, -9.149629532],
                    [0, 0, 1]
                ],
                'rotationDegrees': 0,
                'sizeXY': [2.366863905, 2.366863905],
                'ver': 1
            },
            'ver': 1,
            'zs': -190
        },
        {
            'Lzs': None,
            'classname': 'scanimage.mroi.Roi',
            'discretePlaneMode': 1,
            'interlaceDecimation': None,
            'interlaceOffset': None,
            'name': 'LM',
            'powers': None,
            'pzAdjust': None,
            'scanfields': {
                'affine': [
                    [2.366863905, 0, -1.771176526],
                    [0, 2.366863905, 0.0849098095],
                    [0, 0, 1]],
                'centerXY': [-0.587744573, 1.268341762],
                'classname': 'scanimage.mroi.scanfield.'
                             'fields.RotatedRectangle',
                'enable': 1,
                'name': '',
                'pixelResolutionXY': [512, 512],
                'pixelToRefTransform': [
                    [0.004622781064, 0, -1.773487916],
                    [0, 0.004622781064, 0.08259841897],
                    [0, 0, 1]],
                'rotationDegrees': 0,
                'sizeXY': [2.366863905, 2.366863905],
                'ver': 1
            },
            'ver': 1,
            'zs': -110
        }
    ]

    return {
        "RoiGroups": {
            "imagingRoiGroup": {
                "rois": roi_list
            }
        }
    }


@pytest.fixture
def timeseries_frame_metadata():
    ts_m = {
        'SI': {
            "hChannels": {
                "channelsActive": [[1], [2]]
            },
            "hFastZ": {
                "numFramesPerVolume": 4,
                "numVolumes": 200000,
                "userZs": [[85, -11], [155, -111], [165, 69], [245, -31]]
            },
            "hScan2D": {
                "uniformSampling": False,
                "fillFractionTemporal": 0.712867,
                "sampleRate": 80000000.0,
                "linePhase": 6.25e-08,
                "bidirectional": True,
                "mask": [
                    [8], [8], [9], [8], [8], [8], [7], [8], [7], [8], [7],
                    [7], [8], [7], [7], [7], [7], [6], [7], [7], [6], [7],
                    [7], [6], [6], [7], [6], [6], [6], [7], [6], [6], [6],
                    [6], [6], [5], [6], [6], [6], [6], [5], [6], [5], [6],
                    [6], [5], [6], [5], [6], [5], [5], [6], [5], [5], [5],
                    [6], [5], [5], [5], [5], [5], [6], [5], [5], [5], [5],
                    [5], [5], [5], [4], [5], [5], [5], [5], [5], [5], [4],
                    [5], [5], [5], [4], [5], [5], [4], [5], [5], [4], [5],
                    [4], [5], [5], [4], [5], [4], [5], [4], [5], [4], [5],
                    [4], [5], [4], [4], [5], [4], [5], [4], [4], [5], [4],
                    [4], [5], [4], [4], [4], [5], [4], [4], [4], [5], [4],
                    [4], [4], [4], [5], [4], [4], [4], [4], [4], [5], [4],
                    [4], [4], [4], [4], [4], [4], [4], [5], [4], [4], [4],
                    [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4],
                    [4], [4], [4], [4], [4], [4], [3], [4], [4], [4], [4],
                    [4], [4], [4], [4], [4], [4], [3], [4], [4], [4], [4],
                    [4], [4], [4], [3], [4], [4], [4], [4], [4], [3], [4],
                    [4], [4], [4], [4], [3], [4], [4], [4], [4], [3], [4],
                    [4], [4], [4], [3], [4], [4], [4], [3], [4], [4], [4],
                    [4], [3], [4], [4], [4], [3], [4], [4], [4], [3], [4],
                    [4], [3], [4], [4], [4], [3], [4], [4], [4], [3], [4],
                    [4], [4], [3], [4], [4], [3], [4], [4], [4], [3], [4],
                    [4], [3], [4], [4], [4], [3], [4], [4], [3], [4], [4],
                    [4], [3], [4], [4], [3], [4], [4], [4], [3], [4], [4],
                    [3], [4], [4], [4], [3], [4], [4], [3], [4], [4], [4],
                    [3], [4], [4], [3], [4], [4], [4], [3], [4], [4], [4],
                    [3], [4], [4], [4], [3], [4], [4], [3], [4], [4], [4],
                    [3], [4], [4], [4], [3], [4], [4], [4], [4], [3], [4],
                    [4], [4], [3], [4], [4], [4], [4], [3], [4], [4], [4],
                    [4], [3], [4], [4], [4], [4], [4], [3], [4], [4], [4],
                    [4], [4], [3], [4], [4], [4], [4], [4], [4], [4], [3],
                    [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [3],
                    [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4],
                    [4], [4], [4], [4], [4], [4], [4], [4], [4], [5], [4],
                    [4], [4], [4], [4], [4], [4], [4], [5], [4], [4], [4],
                    [4], [4], [5], [4], [4], [4], [4], [5], [4], [4], [4],
                    [5], [4], [4], [4], [5], [4], [4], [5], [4], [4], [5],
                    [4], [5], [4], [4], [5], [4], [5], [4], [5], [4], [5],
                    [4], [5], [4], [5], [5], [4], [5], [4], [5], [5], [4],
                    [5], [5], [4], [5], [5], [5], [4], [5], [5], [5], [5],
                    [5], [5], [4], [5], [5], [5], [5], [5], [5], [5], [6],
                    [5], [5], [5], [5], [5], [6], [5], [5], [5], [6], [5],
                    [5], [6], [5], [6], [5], [6], [6], [5], [6], [5], [6],
                    [6], [6], [6], [5], [6], [6], [6], [6], [6], [7], [6],
                    [6], [6], [7], [6], [6], [7], [7], [6], [7], [7], [6],
                    [7], [7], [7], [7], [8], [7], [7], [8], [7], [8], [7],
                    [8], [8], [8], [9], [8], [8]
                ]
            },
            "hStackManager": {
                "numSlices": 1,
                "zs_v3_v4": [[85, -11], [155, -111], [165, 69], [245, -31]]
            }
        }
    }

    return ts_m


@pytest.fixture
def image_frame_metadata():
    im_m = {
        'SI': {
            "hChannels": {
                "channelsActive": [[1], [2]]
            },
            "hFastZ": {
                "numFramesPerVolume": 2,
                "numVolumes": 16,
                "userZs": [[-190, 0], [-110, 0]]
            },
            "hScan2D": {
                "uniformSampling": False,
                "fillFractionTemporal": 0.712867,
                "sampleRate": 80000000.0,
                "linePhase": 6.25e-08,
                "bidirectional": True,
                "mask": [
                    [8], [8], [9], [8], [8], [8], [7], [8], [7], [8], [7],
                    [7], [8], [7], [7], [7], [7], [6], [7], [7], [6], [7],
                    [7], [6], [6], [7], [6], [6], [6], [7], [6], [6], [6],
                    [6], [6], [5], [6], [6], [6], [6], [5], [6], [5], [6],
                    [6], [5], [6], [5], [6], [5], [5], [6], [5], [5], [5],
                    [6], [5], [5], [5], [5], [5], [6], [5], [5], [5], [5],
                    [5], [5], [5], [4], [5], [5], [5], [5], [5], [5], [4],
                    [5], [5], [5], [4], [5], [5], [4], [5], [5], [4], [5],
                    [4], [5], [5], [4], [5], [4], [5], [4], [5], [4], [5],
                    [4], [5], [4], [4], [5], [4], [5], [4], [4], [5], [4],
                    [4], [5], [4], [4], [4], [5], [4], [4], [4], [5], [4],
                    [4], [4], [4], [5], [4], [4], [4], [4], [4], [5], [4],
                    [4], [4], [4], [4], [4], [4], [4], [5], [4], [4], [4],
                    [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4],
                    [4], [4], [4], [4], [4], [4], [3], [4], [4], [4], [4],
                    [4], [4], [4], [4], [4], [4], [3], [4], [4], [4], [4],
                    [4], [4], [4], [3], [4], [4], [4], [4], [4], [3], [4],
                    [4], [4], [4], [4], [3], [4], [4], [4], [4], [3], [4],
                    [4], [4], [4], [3], [4], [4], [4], [3], [4], [4], [4],
                    [4], [3], [4], [4], [4], [3], [4], [4], [4], [3], [4],
                    [4], [3], [4], [4], [4], [3], [4], [4], [4], [3], [4],
                    [4], [4], [3], [4], [4], [3], [4], [4], [4], [3], [4],
                    [4], [3], [4], [4], [4], [3], [4], [4], [3], [4], [4],
                    [4], [3], [4], [4], [3], [4], [4], [4], [3], [4], [4],
                    [3], [4], [4], [4], [3], [4], [4], [3], [4], [4], [4],
                    [3], [4], [4], [3], [4], [4], [4], [3], [4], [4], [4],
                    [3], [4], [4], [4], [3], [4], [4], [3], [4], [4], [4],
                    [3], [4], [4], [4], [3], [4], [4], [4], [4], [3], [4],
                    [4], [4], [3], [4], [4], [4], [4], [3], [4], [4], [4],
                    [4], [3], [4], [4], [4], [4], [4], [3], [4], [4], [4],
                    [4], [4], [3], [4], [4], [4], [4], [4], [4], [4], [3],
                    [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [3],
                    [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4],
                    [4], [4], [4], [4], [4], [4], [4], [4], [4], [5], [4],
                    [4], [4], [4], [4], [4], [4], [4], [5], [4], [4], [4],
                    [4], [4], [5], [4], [4], [4], [4], [5], [4], [4], [4],
                    [5], [4], [4], [4], [5], [4], [4], [5], [4], [4], [5],
                    [4], [5], [4], [4], [5], [4], [5], [4], [5], [4], [5],
                    [4], [5], [4], [5], [5], [4], [5], [4], [5], [5], [4],
                    [5], [5], [4], [5], [5], [5], [4], [5], [5], [5], [5],
                    [5], [5], [4], [5], [5], [5], [5], [5], [5], [5], [6],
                    [5], [5], [5], [5], [5], [6], [5], [5], [5], [6], [5],
                    [5], [6], [5], [6], [5], [6], [6], [5], [6], [5], [6],
                    [6], [6], [6], [5], [6], [6], [6], [6], [6], [7], [6],
                    [6], [6], [7], [6], [6], [7], [7], [6], [7], [7], [6],
                    [7], [7], [7], [7], [8], [7], [7], [8], [7], [8], [7],
                    [8], [8], [8], [9], [8], [8]
                ]
            },
            "hStackManager": {
                "numSlices": 1,
                "zs_v3_v4": [[-190, 0], [-110, 0]]
            }
        }
    }

    return im_m


@pytest.fixture
def z_stack_frame_metadata():
    zs_m = {
        'SI': {
            "hChannels": {
                "channelsActive": [[1], [2]]
            },
            "hFastZ": {
                "numFramesPerVolume": 81,
                "numVolumes": 20,
                "userZs": [[85, -11], [155, -111], [165, 69], [245, -31]]
            },
            "hScan2D": {
                "uniformSampling": False,
                "fillFractionTemporal": 0.712867,
                "sampleRate": 80000000.0,
                "linePhase": 6.25e-08,
                "bidirectional": True,
                "mask": [
                    [8], [8], [9], [8], [8], [8], [7], [8], [7], [8], [7], [7],
                    [8], [7], [7], [7], [7], [6], [7], [7], [6], [7], [7], [6],
                    [6], [7], [6], [6], [6], [7], [6], [6], [6], [6], [6], [5],
                    [6], [6], [6], [6], [5], [6], [5], [6], [6], [5], [6], [5],
                    [6], [5], [5], [6], [5], [5], [5], [6], [5], [5], [5], [5],
                    [5], [6], [5], [5], [5], [5], [5], [5], [5], [4], [5], [5],
                    [5], [5], [5], [5], [4], [5], [5], [5], [4], [5], [5], [4],
                    [5], [5], [4], [5], [4], [5], [5], [4], [5], [4], [5], [4],
                    [5], [4], [5], [4], [5], [4], [4], [5], [4], [5], [4], [4],
                    [5], [4], [4], [5], [4], [4], [4], [5], [4], [4], [4], [5],
                    [4], [4], [4], [4], [5], [4], [4], [4], [4], [4], [5], [4],
                    [4], [4], [4], [4], [4], [4], [4], [5], [4], [4], [4], [4],
                    [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4],
                    [4], [4], [4], [4], [3], [4], [4], [4], [4], [4], [4], [4],
                    [4], [4], [4], [3], [4], [4], [4], [4], [4], [4], [4], [3],
                    [4], [4], [4], [4], [4], [3], [4], [4], [4], [4], [4], [3],
                    [4], [4], [4], [4], [3], [4], [4], [4], [4], [3], [4], [4],
                    [4], [3], [4], [4], [4], [4], [3], [4], [4], [4], [3], [4],
                    [4], [4], [3], [4], [4], [3], [4], [4], [4], [3], [4], [4],
                    [4], [3], [4], [4], [4], [3], [4], [4], [3], [4], [4], [4],
                    [3], [4], [4], [3], [4], [4], [4], [3], [4], [4], [3], [4],
                    [4], [4], [3], [4], [4], [3], [4], [4], [4], [3], [4], [4],
                    [3], [4], [4], [4], [3], [4], [4], [3], [4], [4], [4], [3],
                    [4], [4], [3], [4], [4], [4], [3], [4], [4], [4], [3], [4],
                    [4], [4], [3], [4], [4], [3], [4], [4], [4], [3], [4], [4],
                    [4], [3], [4], [4], [4], [4], [3], [4], [4], [4], [3], [4],
                    [4], [4], [4], [3], [4], [4], [4], [4], [3], [4], [4], [4],
                    [4], [4], [3], [4], [4], [4], [4], [4], [3], [4], [4], [4],
                    [4], [4], [4], [4], [3], [4], [4], [4], [4], [4], [4], [4],
                    [4], [4], [4], [3], [4], [4], [4], [4], [4], [4], [4], [4],
                    [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4],
                    [5], [4], [4], [4], [4], [4], [4], [4], [4], [5], [4], [4],
                    [4], [4], [4], [5], [4], [4], [4], [4], [5], [4], [4], [4],
                    [5], [4], [4], [4], [5], [4], [4], [5], [4], [4], [5], [4],
                    [5], [4], [4], [5], [4], [5], [4], [5], [4], [5], [4], [5],
                    [4], [5], [5], [4], [5], [4], [5], [5], [4], [5], [5], [4],
                    [5], [5], [5], [4], [5], [5], [5], [5], [5], [5], [4], [5],
                    [5], [5], [5], [5], [5], [5], [6], [5], [5], [5], [5], [5],
                    [6], [5], [5], [5], [6], [5], [5], [6], [5], [6], [5], [6],
                    [6], [5], [6], [5], [6], [6], [6], [6], [5], [6], [6], [6],
                    [6], [6], [7], [6], [6], [6], [7], [6], [6], [7], [7], [6],
                    [7], [7], [6], [7], [7], [7], [7], [8], [7], [7], [8], [7],
                    [8], [7], [8], [8], [8], [9], [8], [8]
                ]
            },
            "hStackManager": {
                "numSlices": 81,
                "zs_v3_v4": [
                    [55, -41], [55.75, -40.25], [56.5, -39.5], [57.25, -38.75],
                    [58, -38], [58.75, -37.25], [59.5, -36.5], [60.25, -35.75],
                    [61, -35], [61.75, -34.25], [62.5, -33.5], [63.25, -32.75],
                    [64, -32], [64.75, -31.25], [65.5, -30.5], [66.25, -29.75],
                    [67, -29], [67.75, -28.25], [68.5, -27.5], [69.25, -26.75],
                    [70, -26], [70.75, -25.25], [71.5, -24.5], [72.25, -23.75],
                    [73, -23], [73.75, -22.25], [74.5, -21.5], [75.25, -20.75],
                    [76, -20], [76.75, -19.25], [77.5, -18.5], [78.25, -17.75],
                    [79, -17], [79.75, -16.25], [80.5, -15.5], [81.25, -14.75],
                    [82, -14], [82.75, -13.25], [83.5, -12.5], [84.25, -11.75],
                    [85, -11], [85.75, -10.25], [86.5, -9.5], [87.25, -8.75],
                    [8, -8], [88.75, -7.25], [89.5, -6.5], [90.25, -5.75],
                    [91, -5], [91.75, -4.25], [92.5, -3.5], [93.25, -2.75],
                    [94, -2], [94.75, -1.25], [95.5, -0.5], [96.25, 0.25],
                    [97, 1], [97.75, 1.75], [98.5, 2.5], [99.25, 3.25],
                    [100, 4], [100.75, 4.75], [101.5, 5.5], [102.25, 6.25],
                    [103, 7], [103.75, 7.75], [104.5, 8.5], [105.25, 9.25],
                    [106, 10], [106.75, 10.75], [107.5, 11.5], [108.25, 12.25],
                    [109, 13], [109.75, 13.75], [110.5, 14.5], [111.25, 15.25],
                    [112, 16], [112.75, 16.75], [113.5, 17.5], [114.25, 18.25],
                    [115, 19]
                ]
            }
        }
    }

    return zs_m


@pytest.fixture
def timeseries(experiments):
    # The fake data, each page of the Tiff is just a single number
    # to make verifying the results of splitting easier
    ts = np.concatenate([
        np.full((1, 512, 512), ex['experiment_id'])
        for ex in experiments
    ] * 8, axis=0)

    # Turn the data into a TestArray, because it needs to have
    # an asarray method to insert into the downstream code
    return ts.view(TestArray)


@pytest.fixture
def surface_image(experiments, surface_image_roi_metadata):
    # The fake data, each page of the Tiff is just a single number
    # to make verifying the results of splitting easier
    rois = surface_image_roi_metadata['RoiGroups']['imagingRoiGroup']['rois']

    im = np.concatenate([
        np.full((1, 512, 512), roi['zs']) for roi in rois
    ] * 16, axis=0)

    # Turn the data into a TestArray, because it needs to have
    # an asarray method to insert into the downstream code
    return im.view(TestArray)


@pytest.fixture
def z_stack(experiments):
    exps = [e for e in experiments if e['experiment_id'] in [0, 1]]

    zs = np.concatenate([
        np.full((1, 512, 512), ex['experiment_id'])
        for ex in exps
    ] * 16, axis=0)

    # Turn the data into a TestArray, because it needs to have
    # an asarray method to insert into the downstream code
    return zs.view(TestArray)


class TestArray(np.ndarray):
    def asarray(self, key=None):
        if key is None:
            return self[:]
        else:
            if type(key) in [int, slice]:
                return self[key]
            else:
                raise TypeError(
                    f"{key} is of unsupported type {type(key)}"
                )


class MockMesoscopeTiff(MesoscopeTiff):
    def __init__(self,
                 source_tiff,
                 data,
                 experiments,
                 roi_metadata,
                 frame_metadata):
        self.mock_experiments = experiments
        self._frame_data = frame_metadata
        self._roi_data = roi_metadata

        self._tiff = Mock()
        self._tiff.pages = data
        self._tiff.asarray.side_effect = self.asarray

        self._n_pages = None
        self._planes = None
        self._volumes = None
        self._source = source_tiff

    def asarray(self, key=None):
        return self._tiff.pages.asarray(key)


def test_split_timeseries(tmpdir,
                          experiments,
                          timeseries,
                          timeseries_frame_metadata,
                          timeseries_roi_metadata):
    tmpdir.mkdir('resources')

    mesoscope_tiff = MockMesoscopeTiff(
        source_tiff="Mock",
        experiments=experiments,
        roi_metadata=timeseries_roi_metadata,
        data=timeseries,
        frame_metadata=timeseries_frame_metadata)

    run_mesoscope_splitting.split_timeseries(
        mesoscope_tiff, experiments
    )

    for exp in experiments:
        output_h5 = h5py.File(
            f"{exp['storage_directory']}/{exp['experiment_id']}.h5", 'r'
        )
        output_data = np.array(output_h5['data'])

        assert output_data.shape == (8, 512, 512)
        assert np.all(output_data == int(exp['experiment_id']))


def test_split_image_surface(tmpdir,
                             experiments,
                             surface_image,
                             image_frame_metadata,
                             surface_image_roi_metadata):
    mock_mesoscope_tiff = MockMesoscopeTiff(
        source_tiff="Mock",
        experiments=experiments,
        roi_metadata=surface_image_roi_metadata,
        data=surface_image,
        frame_metadata=image_frame_metadata)

    rois = surface_image_roi_metadata['RoiGroups']['imagingRoiGroup']['rois']

    mock_volume_to_tif = create_autospec(volume_to_tif)
    with patch(
        'ophys_etl.modules.mesoscope_splitting.__main__'
        '.volume_to_tif', mock_volume_to_tif
    ):
        run_mesoscope_splitting.split_image(
            mock_mesoscope_tiff,
            experiments,
            "surface"
        )

        for call in mock_volume_to_tif.mock_calls:
            # Check that the data were subsetted to the proper size
            assert call[1][1].asarray().shape == (
                len(surface_image) / len(rois), 512, 512)

            exp_id = int(call[1][0].replace(
                str(tmpdir.realpath()) + '/', '').replace('_surface.tif', '')
            )

            z = rois[experiments[exp_id]['roi_index']]['zs']

            # Check that the correct subset was chosen
            assert np.all(call[1][1].asarray() == z)


def test_split_z_stack(tmpdir,
                       experiments,
                       z_stack,
                       z_stack_frame_metadata,
                       z_stack_roi_metadata):
    tmpdir.mkdir('resources')

    exps = [e for e in experiments if e['experiment_id'] in [0, 1]]

    mesoscope_tiff = MockMesoscopeTiff(
        source_tiff="Mock",
        experiments=experiments,
        roi_metadata=z_stack_roi_metadata,
        data=z_stack,
        frame_metadata=z_stack_frame_metadata)

    for exp in exps:
        run_mesoscope_splitting.split_z(
            mesoscope_tiff, exp
        )

    for exp in exps:
        output_h5 = h5py.File(
            f"{exp['storage_directory']}/{exp['experiment_id']}"
            f"_z_stack_local.h5", 'r'
        )
        output_data = np.array(output_h5['data'])

        assert output_data.shape == (16, 512, 512)
        assert np.all(output_data == int(exp['experiment_id']))
