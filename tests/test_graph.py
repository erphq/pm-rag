import numpy as np
import pytest

from pm_rag import CodeGraph


def test_transition_matrix_row_normalized() -> None:
    g = CodeGraph(
        nodes=["a", "b", "c"],
        edges=[(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)],
    )
    p_t = g.transition_matrix_T()
    p = p_t.T
    # row 0 sums to 1 (two outgoing edges, equal weight)
    assert pytest.approx(p[0].sum()) == 1.0
    # row 2 has no outgoing edges → all zeros
    assert pytest.approx(p[2].sum()) == 0.0


def test_index_of() -> None:
    g = CodeGraph(nodes=["x", "y"], edges=[])
    assert g.index_of("y") == 1


def test_rejects_out_of_range_edges() -> None:
    with pytest.raises(ValueError):
        CodeGraph(nodes=["a"], edges=[(0, 5, 1.0)])


def test_rejects_negative_weights() -> None:
    with pytest.raises(ValueError):
        CodeGraph(nodes=["a", "b"], edges=[(0, 1, -1.0)])


def test_p_t_shape() -> None:
    g = CodeGraph(nodes=list("abcd"), edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
    p_t = g.transition_matrix_T()
    assert p_t.shape == (4, 4)
    assert isinstance(p_t, np.ndarray)
