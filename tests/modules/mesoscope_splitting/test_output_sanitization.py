import pytest
import numpy as np
from ophys_etl.modules.mesoscope_splitting.output_json_sanitizer import (
    _sanitize_element,
    _sanitize_data,
    get_sanitized_json_data)


def test_sanitize_element():
    """
    test that _sanitize_element fixes what it is supposed
    to fix and passes through what it is supposed to pass through
    """
    assert _sanitize_element(2) == 2
    np.testing.assert_allclose(_sanitize_element(2.1), 2.1)
    assert _sanitize_element(float('Infinity')) == "_Inf_"
    assert _sanitize_element(-float('Infinity')) == "_-Inf_"
    assert _sanitize_element(np.inf) == "_Inf_"
    assert _sanitize_element(-np.inf) == "_-Inf_"
    assert _sanitize_element(np.nan) == "_NaN_"


@pytest.mark.parametrize(
        "data, expected",
        [({'a': 'b', 'c': 1, 'd': np.inf},
          {'a': 'b', 'c': 1, 'd': '_Inf_'}),
         ([1, 2, -np.inf, 3, np.nan],
          [1, 2, '_-Inf_', 3, '_NaN_']),
         ({'a': 'b', 'c': [1, np.nan, 3], 'd': np.inf},
          {'a': 'b', 'c': [1, '_NaN_', 3], 'd': '_Inf_'}),
         ({'a': {'b': np.nan, 'c': [1, -np.inf, 2]},
           'b': 2, 'c': 3, 'd': [1, np.inf], 'e': np.nan},
          {'a': {'b': '_NaN_', 'c': [1, '_-Inf_', 2]},
           'b': 2, 'c': 3, 'd': [1, '_Inf_'], 'e': '_NaN_'}),
         ([1, 'b', {'a': np.nan, 'b': [1, 2, np.inf]}, 'c', -np.inf],
          [1, 'b', {'a': '_NaN_', 'b': [1, 2, '_Inf_']}, 'c', '_-Inf_'])
         ])
def test_sanitize_data(data, expected):
    """
    test that _sanitize_data properly sanitizes iterable data elements
    """
    assert _sanitize_data(data=data) == expected


@pytest.fixture
def json_data_fixture0():
    """
    Returns an input json_data and an expected result
    of sanitization
    """
    json_data = {'a': np.inf,
                 'b': [1, 2, -np.inf],
                 'c': {'d': [np.inf, -np.inf, 3],
                       'e': '/path/to/a/file.txt'}}

    expected = {'a': '_Inf_',
                'b': [1, 2, '_-Inf_'],
                'c': {'d': ['_Inf_', '_-Inf_', 3],
                      'e': '/path/to/a/file.txt'}}

    return (json_data, expected)


@pytest.fixture
def json_data_fixture1():
    """
    Returns an input json_data and an expected result
    of sanitization
    """

    json_data = [{'a': '/path/to/file.txt',
                  'b': [np.inf, 2, np.nan]},
                 [-np.inf, 'c', 'd'],
                 4]

    expected = [{'a': '/path/to/file.txt',
                 'b': ['_Inf_', 2, '_NaN_']},
                ['_-Inf_', 'c', 'd'],
                4]

    return json_data, expected


@pytest.fixture
def json_data_fixture_list(json_data_fixture0,
                           json_data_fixture1):
    """
    Returns a list of (input_data, expected_result) pairs
    """
    output = []
    output.append(json_data_fixture0)
    output.append(json_data_fixture1)
    return output


def test_sanitized_json_data(json_data_fixture_list):
    """
    Test that get_sanitized_json_data returns the expected result
    """

    for (input_data, expected_result) in json_data_fixture_list:
        actual_result = get_sanitized_json_data(input_data)
        assert actual_result == expected_result
