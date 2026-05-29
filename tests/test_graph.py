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


def test_n_property() -> None:
    g = CodeGraph(nodes=["a", "b", "c"], edges=[])
    assert g.n == 3


def test_self_loop_normalizes_to_one() -> None:
    # Node 0 has only a self-loop; after normalization P[0,0] = 1.0.
    g = CodeGraph(nodes=["a", "b"], edges=[(0, 0, 2.5)])
    p_t = g.transition_matrix_T()
    p = p_t.T
    assert pytest.approx(p[0, 0]) == 1.0
    assert pytest.approx(p[1].sum()) == 0.0  # node 1 is dangling


def test_parallel_edges_weights_accumulate() -> None:
    # Two edges 0→1 with weights 1.0 and 3.0 accumulate to 4.0;
    # the only outgoing direction is 1, so P[0,1] = 1.0.
    g = CodeGraph(nodes=["a", "b"], edges=[(0, 1, 1.0), (0, 1, 3.0)])
    p_t = g.transition_matrix_T()
    p = p_t.T
    assert pytest.approx(p[0, 1]) == 1.0


def test_unequal_weights_produce_correct_probabilities() -> None:
    # Edge 0→1 weight 1.0, edge 0→2 weight 3.0: P[0,1]=0.25, P[0,2]=0.75.
    g = CodeGraph(nodes=["a", "b", "c"], edges=[(0, 1, 1.0), (0, 2, 3.0)])
    p_t = g.transition_matrix_T()
    p = p_t.T
    assert pytest.approx(p[0, 1]) == 0.25
    assert pytest.approx(p[0, 2]) == 0.75


def test_all_dangling_graph_p_t_is_zero() -> None:
    g = CodeGraph(nodes=["a", "b", "c"], edges=[])
    p_t = g.transition_matrix_T()
    assert (p_t == 0.0).all()


def test_single_node_no_edges() -> None:
    g = CodeGraph(nodes=["only"], edges=[])
    assert g.n == 1
    p_t = g.transition_matrix_T()
    assert p_t.shape == (1, 1)
    assert p_t[0, 0] == 0.0


def test_zero_weight_edge_node_remains_dangling() -> None:
    # A zero-weight edge is structurally valid (weight >= 0), but contributes
    # nothing to the row sum, so the node remains dangling in the transition matrix.
    g = CodeGraph(nodes=["a", "b"], edges=[(0, 1, 0.0)])
    p_t = g.transition_matrix_T()
    p = p_t.T
    assert pytest.approx(p[0].sum()) == 0.0
