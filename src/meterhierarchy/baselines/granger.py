"""Pairwise Granger-Causality Baseline (Granger).

For every ordered pair (i, j) we fit two OLS models:

  * restricted:   j_t = a0 + sum_{l=1..L} a_l * j_{t-l} + eps
  * unrestricted: j_t = a0 + sum_{l=1..L} a_l * j_{t-l}
                        + sum_{l=1..L} b_l * i_{t-l} + eps

and compute the F-statistic that tests whether adding the lagged values
of meter ``i`` significantly reduces the residual sum of squares when
predicting meter ``j``. A large F-value means "i Granger-causes j",
which in our hierarchy interpretation makes ``i`` a plausible parent of
``j``.

The resulting score matrix ``S[i, j]`` is **directed** (unlike CL / PC).

Reference:
    Granger, C.W.J. (1969). *Investigating causal relations by
    econometric models and cross-spectral methods.* Econometrica 37,
    424-438.
"""
from __future__ import annotations

import numpy as np


def _build_lag_matrix(x: np.ndarray, lag: int) -> np.ndarray:
    """Return (T-lag, lag) matrix of lagged values [x_{t-1}, ..., x_{t-lag}]."""
    T = x.shape[0]
    M = np.empty((T - lag, lag), dtype=np.float64)
    for l in range(1, lag + 1):
        M[:, l - 1] = x[lag - l : T - l]
    return M


def _rss(y: np.ndarray, X: np.ndarray) -> float:
    """Residual sum of squares from OLS(y ~ X)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(resid @ resid)


def score(data: np.ndarray, names=None, lag: int = 3) -> np.ndarray:
    """Return directed Granger F-statistic matrix (N, N).

    S[i, j] = F-statistic that i Granger-causes j (0 on the diagonal),
    i.e. the strength with which meter ``i`` is a candidate parent of ``j``.
    """
    T, N = data.shape
    try:
        # Subsample to keep OLS solves bounded
        X_all = data
        if T > 2000:
            idx = np.linspace(0, T - 1, 2000).astype(np.int64)
            X_all = data[idx]
        T2 = X_all.shape[0]
        if T2 <= 2 * lag + 2:
            return np.zeros((N, N), dtype=np.float64)

        # Pre-compute lag matrices once per column
        lag_mats = [_build_lag_matrix(X_all[:, k], lag) for k in range(N)]
        y_targets = [X_all[lag:, k] for k in range(N)]

        S = np.zeros((N, N), dtype=np.float64)
        ones = np.ones((T2 - lag, 1), dtype=np.float64)

        for j in range(N):
            y = y_targets[j]
            Lj = lag_mats[j]
            # Restricted: only own lags + intercept
            Xr = np.hstack([ones, Lj])
            rss_r = _rss(y, Xr)
            df_r = T2 - lag - (lag + 1)
            if df_r <= 0 or rss_r <= 0:
                continue

            for i in range(N):
                if i == j:
                    continue
                Li = lag_mats[i]
                Xu = np.hstack([ones, Lj, Li])
                rss_u = _rss(y, Xu)
                df_u = T2 - lag - (2 * lag + 1)
                if df_u <= 0 or rss_u <= 0:
                    continue
                num = (rss_r - rss_u) / lag
                den = rss_u / df_u
                if den <= 1e-12:
                    continue
                F = num / den
                if not np.isfinite(F):
                    continue
                S[i, j] = max(F, 0.0)

        return S
    except Exception:
        return np.zeros((N, N), dtype=np.float64)
