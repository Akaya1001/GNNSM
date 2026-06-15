"""Hidden-Load Baseline (HL).

Idea
----
If meter ``i`` is the parent of meter ``j``, we expect that the time
series of ``i`` can be explained by a non-negative linear combination of
its children (plus a hidden-load residual). We therefore solve, for each
candidate parent ``i``:

    argmin_{w >= 0}  || x_i - sum_j w_j * x_j ||^2   s.t.  j != i

using Non-Negative Least Squares (NNLS). The resulting weights
``w_j >= 0`` serve as edge scores ``S[i, j]``: a large ``w_j`` means
"meter j explains a lot of meter i's consumption", i.e. j is a plausible
**child** of i.

This is a single-layer hidden-load decomposition and matches the
"disaggregation" interpretation used in NILM-flavoured hierarchy papers.

We use scipy's ``nnls`` solver if available, and otherwise fall back to a
projected-gradient / soft-thresholded-LASSO proxy.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import nnls as _nnls
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


def _solve_nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return argmin_{w >= 0} || A w - b ||^2."""
    if _HAVE_SCIPY:
        try:
            w, _ = _nnls(A, b, maxiter=200)
            return w
        except Exception:
            pass
    # Fallback: projected least squares (no sign guarantee) -> clip
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(w, 0.0, None)


def score(data: np.ndarray, names=None) -> np.ndarray:
    """Return directed Hidden-Load NNLS score matrix (N, N).

    S[i, j] = weight with which meter j contributes to reconstructing
    meter i's signal. Interpret ``i`` as parent, ``j`` as child.
    """
    T, N = data.shape
    try:
        X = data
        if T > 2000:
            idx = np.linspace(0, T - 1, 2000).astype(np.int64)
            X = data[idx]

        # Scale columns so NNLS weights are comparable
        col_max = np.maximum(np.abs(X).max(axis=0), 1e-9)
        Xs = X / col_max  # (T, N)

        S = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            # Build candidate-child matrix: all columns except i
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            A = Xs[:, mask]  # (T, N-1)
            b = Xs[:, i]
            try:
                w = _solve_nnls(A, b)
            except Exception:
                continue
            # Insert into full-size row
            row = np.zeros(N, dtype=np.float64)
            row[mask] = w
            S[i] = row

        # If a meter never appears as a candidate parent (e.g. very small
        # signal), its row is all zeros -> the decoder can't pick it as
        # parent. That is fine; the super-root trick still yields a valid
        # tree.
        return S
    except Exception:
        return np.zeros((N, N), dtype=np.float64)
