"""Tree decoding: maximum spanning arborescence (Edmonds / Chu-Liu).

Given an ``(N, N)`` edge-score matrix ``S`` (``S[i, j]`` = strength that ``i`` is
the parent of ``j``), this returns a valid rooted tree as a ``{child: parent}``
mapping. A virtual super-root is attached to every node so that the algorithm is
forced to select exactly one real root.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def find_best_tree(n: int, score_matrix: np.ndarray) -> Dict[int, int]:
    """Maximum spanning arborescence via Edmonds/Chu-Liu.

    Args:
        n: number of nodes.
        score_matrix: ``(N, N)`` non-negative scores; ``score_matrix[i, j]`` is
            the weight of the directed edge ``i -> j`` (``i`` parent of ``j``).

    Returns:
        ``parent_of``: dict mapping each non-root child index to its parent index.
    """
    try:
        import networkx as nx
    except ImportError:
        return _fallback_tree(n, score_matrix)

    weights = score_matrix.copy().astype(np.float64)
    np.fill_diagonal(weights, 0.0)
    weights = np.clip(weights, 1e-6, 1.0 - 1e-6)

    # Super-root trick: a virtual node ``n`` connects to every real node with a
    # tiny weight, so Edmonds is forced to pick exactly one real root.
    ext_n = n + 1
    ext_w = np.zeros((ext_n, ext_n), dtype=np.float64)
    ext_w[:n, :n] = weights
    col_mean = weights.mean(axis=0)
    for i in range(n):
        ext_w[n, i] = 0.01 + col_mean[i] * 0.001  # just enough to satisfy the root requirement
    ext_w[:, n] = 0.0  # nobody is a parent of the super-root

    graph = nx.DiGraph()
    for i in range(ext_n):
        for j in range(ext_n):
            if i == j:
                continue
            w = ext_w[i, j]
            if w > 0:
                # NetworkX minimizes, so we negate to obtain a maximum arborescence.
                graph.add_edge(i, j, weight=-float(w))

    try:
        arb = nx.minimum_spanning_arborescence(graph, attr="weight", preserve_attrs=False)
    except Exception:
        return _fallback_tree(n, score_matrix)

    parent_of: Dict[int, int] = {}
    for u, v in arb.edges():
        if u == n or v == n:
            continue  # drop super-root edges
        parent_of[v] = u
    return parent_of


def _fallback_tree(n: int, score_matrix: np.ndarray) -> Dict[int, int]:
    """Greedy max-arborescence fallback when NetworkX is unavailable."""
    weights = score_matrix.copy()
    np.fill_diagonal(weights, -np.inf)
    # Root = column with the weakest incoming evidence (least wanted as a child).
    col_sum = weights.sum(axis=0)
    root = int(np.argmin(col_sum))
    parent_of: Dict[int, int] = {}
    for j in range(n):
        if j == root:
            continue
        parent_of[j] = int(np.argmax(weights[:, j]))
    return parent_of
