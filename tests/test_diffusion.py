import numpy as np
import pytest

from pm_rag import personalized_pagerank


def test_ppr_converges_and_normalized() -> None:
    # 3-node ring: a → b → c → a
    p = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    seed = np.array([1.0, 0.0, 0.0])
    r = personalized_pagerank(p.T, seed, alpha=0.15, max_iters=200)
    assert pytest.approx(float(r.sum()), rel=1e-6) == 1.0
    # node a has the highest score because it's the seed
    assert r[0] >= r[1]
    assert r[0] >= r[2]


def test_ppr_seed_must_have_mass() -> None:
    p_t = np.zeros((2, 2))
    with pytest.raises(ValueError):
        personalized_pagerank(p_t, np.zeros(2))


def test_ppr_alpha_validation() -> None:
    p_t = np.zeros((2, 2))
    with pytest.raises(ValueError):
        personalized_pagerank(p_t, np.array([1.0, 0.0]), alpha=0.0)
    with pytest.raises(ValueError):
        personalized_pagerank(p_t, np.array([1.0, 0.0]), alpha=1.0)


def test_ppr_shape_validation() -> None:
    p_t = np.zeros((3, 3))
    with pytest.raises(ValueError):
        personalized_pagerank(p_t, np.zeros(2))


def test_ppr_higher_alpha_concentrates_at_seed() -> None:
    p = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    seed = np.array([1.0, 0.0, 0.0])
    r_low = personalized_pagerank(p.T, seed, alpha=0.05)
    r_high = personalized_pagerank(p.T, seed, alpha=0.9)
    # higher alpha -> more mass at seed
    assert r_high[0] > r_low[0]


def test_ppr_non_square_matrix_raises() -> None:
    p_t = np.zeros((2, 3))
    with pytest.raises(ValueError):
        personalized_pagerank(p_t, np.zeros(2))


def test_ppr_1d_input_raises() -> None:
    with pytest.raises(ValueError):
        personalized_pagerank(np.zeros(3), np.zeros(3))


def test_ppr_max_iters_cap_returns_normalized() -> None:
    # 4-node ring that needs many iterations to converge.
    # max_iters=1 exits before convergence but result must still be
    # a valid L1-normalized probability vector.
    p = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    seed = np.array([1.0, 0.0, 0.0, 0.0])
    r = personalized_pagerank(p.T, seed, alpha=0.15, max_iters=1)
    assert pytest.approx(float(r.sum()), rel=1e-6) == 1.0
    assert float(r.min()) >= 0.0
