"""Personalized PageRank.

Solves `r = alpha * s + (1 - alpha) * P^T r` by power iteration. `alpha`
is the restart probability - higher values bias more toward the seed.
"""
from __future__ import annotations

import numpy as np


def personalized_pagerank(
    p_t: np.ndarray,
    seed: np.ndarray,
    *,
    alpha: float = 0.15,
    max_iters: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """Compute personalized PageRank.

    Args:
        p_t: the transposed transition matrix `P^T` (n x n).
        seed: the personalization vector (n,). Will be L1-normalized.
        alpha: restart probability in (0, 1).
        max_iters: power-iteration cap.
        tol: L1 convergence tolerance.

    Returns:
        The stationary distribution `r` (n,), L1-normalized.
    """
    if p_t.ndim != 2 or p_t.shape[0] != p_t.shape[1]:
        raise ValueError("p_t must be a square 2D matrix")
    if seed.shape != (p_t.shape[0],):
        raise ValueError("seed length must equal p_t side")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    s_total = float(seed.sum())
    if s_total <= 0:
        raise ValueError("seed must have positive mass")
    s = seed / s_total
    r = s.copy()

    for _ in range(max_iters):
        r_new = alpha * s + (1 - alpha) * (p_t @ r)
        total = float(r_new.sum())
        if total > 0:
            r_new = r_new / total
        if float(np.linalg.norm(r_new - r, ord=1)) < tol:
            r = r_new
            break
        r = r_new
    return r
