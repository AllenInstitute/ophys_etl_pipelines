import pytest

import ophys_etl.modules.segmentation.modules.create_graph.__main__ as cg


@pytest.mark.parametrize(
        "row_min, row_max, col_min, col_max, kernel, expected",
        [
            # 4 pixels, 6 edges
            (0, 1, 0, 1, None, [[(0, 0), (0, 1)],
                                [(0, 0), (1, 0)],
                                [(1, 1), (1, 0)],
                                [(1, 1), (0, 1)],
                                [(1, 1), (0, 0)],
                                [(0, 1), (1, 0)]]),
            # 4 pixels, 6 edges, offset
            (1, 2, 1, 2, None, [[(1, 1), (1, 2)],
                                [(1, 1), (2, 1)],
                                [(2, 2), (2, 1)],
                                [(2, 2), (1, 2)],
                                [(2, 2), (1, 1)],
                                [(1, 2), (2, 1)]]),
            # 4 pixels, 4 edges
            (0, 1, 0, 1, [(-1, 0), (1, 0), (0, -1), (0, 1)],
                [[(0, 0), (0, 1)],
                 [(0, 0), (1, 0)],
                 [(1, 1), (1, 0)],
                 [(1, 1), (0, 1)]]),
            # 4 pixels, 2 edges
            (0, 1, 0, 1, [(-1, -1), (-1, 1), (1, -1), (1, 1)],
                [[(1, 1), (0, 0)],
                 [(0, 1), (1, 0)]]),
            # 6 pixels, 11 edges
            (0, 1, 0, 2, None, [[(0, 0), (0, 1)],
                                [(0, 0), (1, 0)],
                                [(1, 1), (1, 0)],
                                [(1, 1), (0, 1)],
                                [(1, 1), (0, 0)],
                                [(0, 1), (1, 0)],
                                [(0, 1), (0, 2)],
                                [(0, 1), (1, 2)],
                                [(1, 1), (1, 2)],
                                [(1, 1), (0, 2)],
                                [(1, 2), (0, 2)]])
            ])
def test_create_graph(row_min, row_max, col_min, col_max, kernel, expected):
    graph = cg.create_graph(row_min, row_max, col_min, col_max, kernel)
    assert len(graph.edges) == len(expected)
    for e in expected:
        assert graph.has_edge(*e)
