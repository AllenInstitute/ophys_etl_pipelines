import pytest
import numpy as np
from scipy.sparse import coo_matrix

from ophys_etl.filters import filter_longest_edge_length


@pytest.mark.parametrize("coo_rois, longest_edge_thrsh, expected_rois",
                         [([coo_matrix(np.array([[0, 0, 0, 0, 0],
                                                 [0, 0.7, 0.85, 0.98, 0.67],
                                                 [0, 0.85, 0.79, 0.82, 0.78],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]]))], 3, []),
                          ([coo_matrix(np.array([[0, 0.76, 0.85, 0, 0],
                                                 [0, 0.88, 0.65, 0, 0],
                                                 [0, 0.95, 0.78, 0, 0],
                                                 [0, 0.98, 0.88, 0, 0],
                                                 [0, 0, 0, 0, 0]]))], 3, []),
                          ([coo_matrix(np.array([[0, 1, 1, 1, 1],
                                                 [0, 1, 1, 1, 1],
                                                 [0, 1, 1, 1, 1],
                                                 [0, 1, 1, 1, 1],
                                                 [0, 0, 0, 0, 0]]))], 3, []),
                          ([coo_matrix(np.array([[0, 0, 0, 0, 0],
                                                 [0.67, 0.89, 0, 0, 0],
                                                 [0, 0, 0, 0, 0.87],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]]))], 3, []),
                          ([coo_matrix(np.array([[0.89, 0, 0, 0, 0],
                                                 [0.79, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0.85, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]]))], 3, []),
                          ([coo_matrix(np.array([[0.89, 0.54, 0, 0, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0.74]]))], 3,
                           []),
                          ([coo_matrix(np.array([[0, 0, 0, 0, 0],
                                                 [0, 1, 0.96, 0, 0],
                                                 [0, 0.67, 0.87, 0, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]]))], 3,
                           [coo_matrix(np.array([[0, 0, 0, 0, 0],
                                                 [0, 1, 0.96, 0, 0],
                                                 [0, 0.67, 0.87, 0, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]]))]),
                          ([coo_matrix(
                              np.array([[0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]]))],
                           3, [coo_matrix(np.array(
                              [[0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]]))]),
                          ([], 3, []),
                          ([coo_matrix(
                              np.array([[0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]])),
                            coo_matrix(
                              np.array([[0, 0, 0, 0, 0],
                                        [0, 1, 0.96, 0, 0],
                                        [0, 0.67, 0.87, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]])),
                            coo_matrix(
                              np.array([[0, 1, 1, 1, 1],
                                        [0, 1, 1, 1, 1],
                                        [0, 1, 1, 1, 1],
                                        [0, 1, 1, 1, 1],
                                        [0, 0, 0, 0, 0]]))],
                           3, [coo_matrix(
                              np.array([[0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]])),
                               coo_matrix(
                                   np.array([[0, 0, 0, 0, 0],
                                             [0, 1, 0.96, 0, 0],
                                             [0, 0.67, 0.87, 0, 0],
                                             [0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0]]))])])
def test_filter_rois_by_longest_edge_length(coo_rois, longest_edge_thrsh,
                                            expected_rois):
    """
    Test Cases (Included in the list of coo_matrices):
    1. Larger than longest edge thrsh in x direction, contiguous
    2. Larger than longest edge thrsh in y direction, contiguous
    3. Larger than longest edge thrsh in x and y, contiguous
    4. Larger than longest edge thrsh in x, non contiguous
    5. Larger than longest edge thrsh in y, non contiguous
    6. Larger than longest edge thrsh in x and y, non contiguous
    7. Smaller than longest edge thrsh in x and y, contiguous
    8. Smaller than longest edge thrsh in x and y, non contiguous
    9. Empty list
    10. List with more than one element two trues and one false
    """
    filtered_rois = filter_longest_edge_length(coo_rois,
                                               longest_edge_thrsh)
    # assume that test is true and the indices line up between calculated and
    # expected
    for i in range(len(filtered_rois)):
        assert np.allclose(filtered_rois[i].toarray(),
                           expected_rois[i].toarray())
