"""PC-algorithm Baseline (PC).

Simplified constraint-based structure learner based on partial
correlation. For every pair (i, j) we compute:

  1. the marginal Pearson correlation ``rho(i, j)``
  2. the *minimum* absolute partial correlation ``rho(i, j | k)`` over all
     possible single conditioning variables ``k != i, j``.

The score ``S[i, j] = S[j, i] = max(0, min_k |rho(i, j | k)|)``
reflects how strong the dependence remains after conditioning on one
other meter -- i.e. the kind of edge the PC algorithm would keep.

This is a deliberately pragmatic PC-style scorer: we do not perform the
full iterative sepset search because the topology is ultimately
extracted by the shared Edmonds decoder, which keeps the comparison with
the GNN fair.

Reference:
    Spirtes, Glymour, Scheines (2000). *Causation, Prediction, and
    Search.* MIT Press.
"""
from __future__ import annotations

import numpy as np

from .base import standardize


def _corr_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix on standardized columns."""
    Z = standardize(X)
    T = Z.shape[0]
    C = (Z.T @ Z) / max(T - 1, 1)
    # Zero-variance columns -> nan -> 0
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def score(data: np.ndarray, names=None) -> np.ndarray:
    """Return symmetric partial-correlation score matrix (N, N).

    ``S[i, j]`` is the minimum absolute order-1 partial correlation between
    meters ``i`` and ``j`` (symmetric); the directed tree decoder orients it.
    """
    T, N = data.shape
    try:
        X = data
        if T > 2000:
            idx = np.linspace(0, T - 1, 2000).astype(np.int64)
            X = data[idx]

        C = _corr_matrix(X)
        abs_C = np.abs(C)

        S = np.zeros((N, N), dtype=np.float64)

        # For each pair, test the minimum partial correlation over single
        # conditioning variables. This is the order-1 PC test.
        for i in range(N):
            for j in range(i + 1, N):
                r_ij = C[i, j]
                if abs(r_ij) < 1e-6:
                    continue

                best_min = abs(r_ij)  # start from order-0
                # Condition on each k != i, j
                for k in range(N):
                    if k == i or k == j:
                        continue
                    r_ik = C[i, k]
                    r_jk = C[j, k]
                    denom = np.sqrt((1.0 - r_ik * r_ik) * (1.0 - r_jk * r_jk))
                    if denom < 1e-9:
                        continue
                    r_ij_k = (r_ij - r_ik * r_jk) / denom
                    r_ij_k = float(np.clip(r_ij_k, -1.0, 1.0))
                    if abs(r_ij_k) < best_min:
                        best_min = abs(r_ij_k)

                S[i, j] = best_min
                S[j, i] = best_min

        return S
    except Exception:
        return np.zeros((N, N), dtype=np.float64)
