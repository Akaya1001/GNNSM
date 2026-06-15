"""NOTEARS Baseline (linear SEM with acyclicity constraint).

Zheng, X., Aragam, B., Ravikumar, P., Xing, E.P. (2018).
*DAGs with NO TEARS: Continuous Optimization for Structure Learning.*
NeurIPS 31.

Learns a weight matrix W in R^{N x N} by solving

    min_W  0.5/T * || X - X W ||_F^2  +  lambda1 * ||W||_1
    s.t.   h(W) = tr(exp(W (.)*(.) W)) - N = 0
           diag(W) = 0

via the augmented Lagrangian. ``|W_ij|`` is returned as the directed
parent-score S[i, j]: if W_ij != 0, meter i regresses meter j on itself,
i.e. i is a plausible *parent* of j.

Implementation follows the reference code of Zheng et al., but is
trimmed to the essentials and uses scipy's L-BFGS-B via
``scipy.optimize.minimize`` on the doubled variable (w_plus, w_minus)
so that the L1 penalty becomes linear and the bounds enforce
non-negativity and diag(W) = 0.

Time budget per case (T<=2000, N<=150): seconds to tens of seconds.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


def _h(W: np.ndarray, d: int) -> tuple[float, np.ndarray]:
    """h(W) = tr(exp(W o W)) - d,  grad = (exp(W o W))^T o 2W."""
    M = W * W
    E = expm(M)
    h = float(np.trace(E)) - d
    G = E.T * (2.0 * W)
    return h, G


def _notears_linear(
    X: np.ndarray,
    lambda1: float = 0.1,
    max_iter: int = 10,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
) -> np.ndarray:
    """Return estimated weight matrix W (d, d)."""
    T, d = X.shape
    if T < 2 or d < 2:
        return np.zeros((d, d), dtype=np.float64)

    # Column-center (NOTEARS assumes zero-mean columns)
    Xc = X - X.mean(axis=0, keepdims=True)
    Xc /= np.maximum(Xc.std(axis=0, keepdims=True), 1e-9)

    def _adj(w: np.ndarray) -> np.ndarray:
        return (w[: d * d] - w[d * d :]).reshape(d, d)

    def _func(w: np.ndarray, rho: float, alpha: float):
        W = _adj(w)
        R = Xc - Xc @ W
        loss = 0.5 / T * float((R * R).sum())
        G_loss = -1.0 / T * (Xc.T @ R)
        h_val, G_h = _h(W, d)
        obj = loss + 0.5 * rho * h_val * h_val + alpha * h_val + lambda1 * float(w.sum())
        G_smooth = G_loss + (rho * h_val + alpha) * G_h
        g = np.concatenate(
            [(G_smooth + lambda1).ravel(), (-G_smooth + lambda1).ravel()]
        )
        return obj, g

    # Bounds: w >= 0 everywhere, diag forced to 0 via (0, 0)
    bnds: list[tuple[float, float | None]] = []
    for _ in range(2):
        for i in range(d):
            for j in range(d):
                bnds.append((0.0, 0.0) if i == j else (0.0, None))

    w_est = np.zeros(2 * d * d, dtype=np.float64)
    rho, alpha, h_prev = 1.0, 0.0, float("inf")

    for _ in range(max_iter):
        while rho < rho_max:
            sol = minimize(
                lambda w: _func(w, rho, alpha),
                w_est,
                method="L-BFGS-B",
                jac=True,
                bounds=bnds,
                options={"maxiter": 100, "ftol": 1e-7, "gtol": 1e-6},
            )
            w_new = sol.x
            h_new, _ = _h(_adj(w_new), d)
            if h_new > 0.25 * h_prev:
                rho *= 10
            else:
                break
        w_est = w_new
        h_prev = h_new
        alpha += rho * h_prev
        if h_prev <= h_tol or rho >= rho_max:
            break

    W = _adj(w_est)
    # Hard threshold like the reference implementation
    W[np.abs(W) < 0.3] = 0.0
    return W


def score(data: np.ndarray, names=None) -> np.ndarray:
    """Return directed NOTEARS score matrix (N, N).

    S[i, j] = |W_ij|, i.e. how strongly meter i regresses meter j's signal
    onto itself -> i is a candidate parent of j.
    """
    T, N = data.shape
    try:
        X = data
        if T > 2000:
            idx = np.linspace(0, T - 1, 2000).astype(np.int64)
            X = data[idx]

        # For very large N, reduce max_iter to stay affordable
        if N <= 30:
            max_iter = 10
        elif N <= 80:
            max_iter = 6
        else:
            max_iter = 4

        W = _notears_linear(X, lambda1=0.1, max_iter=max_iter)

        S = np.abs(W)
        np.fill_diagonal(S, 0.0)
        return S
    except Exception:
        return np.zeros((N, N), dtype=np.float64)
