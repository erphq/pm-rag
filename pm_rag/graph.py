"""Code graph — nodes (symbols), directed weighted edges (calls/imports/types)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CodeGraph:
    """A directed weighted graph over named symbols.

    `edges` is a list of `(src_idx, dst_idx, weight)` tuples. Self-loops
    are allowed. The class makes no assumption about what the symbols
    represent — functions, files, classes, anything indexable.
    """

    nodes: list[str]
    edges: list[tuple[int, int, float]]

    def __post_init__(self) -> None:
        n = len(self.nodes)
        for src, dst, w in self.edges:
            if not 0 <= src < n:
                raise ValueError(f"edge src out of range: {src}")
            if not 0 <= dst < n:
                raise ValueError(f"edge dst out of range: {dst}")
            if w < 0:
                raise ValueError(f"edge weight must be non-negative: {w}")

    @property
    def n(self) -> int:
        return len(self.nodes)

    def index_of(self, name: str) -> int:
        return self.nodes.index(name)

    def transition_matrix_T(self) -> np.ndarray:
        """Return P^T where P is the row-normalized transition matrix.

        Dangling nodes (no outgoing edges) keep zero rows; the diffusion
        loop adds the restart contribution from the seed each iteration,
        so mass is not lost.
        """
        n = self.n
        m = np.zeros((n, n), dtype=np.float64)
        for src, dst, w in self.edges:
            m[src, dst] += w
        row_sums = m.sum(axis=1, keepdims=True)
        nonzero = row_sums.flatten() > 0
        m[nonzero] = m[nonzero] / row_sums[nonzero]
        return m.T
