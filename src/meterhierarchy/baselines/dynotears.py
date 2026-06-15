"""DYNOTEARS Baseline (dynamic NOTEARS for time series).

Pamfil, R., Sriwattanaworachai, N., Desai, S., Pilgerstorfer, P., et al.
(2020). *DYNOTEARS: Structure Learning from Time-Series Data.* AISTATS.

Extends NOTEARS to structural VAR (SVAR) of order p:

    X_t = X_t W  +  sum_{k=1..p} X_{t-k} A_k  +  eps

Here W in R^{N x N} is the contemporaneous parent structure (same
semantics as NOTEARS), while each A_k in R^{N x N} is a lagged coupling
matrix. Acyclicity is enforced only on W via

    h(W) = tr(exp(W (.)* (.) W)) - N = 0.

For meter-hierarchy detection we return the contemporaneous magnitude
blended with a small share of the lagged-coupling magnitudes:

    S[i, j] = |W_ij|  +  beta * max_k |A_k[i, j]|

The contemporaneous part dominates (energy conservation is effectively
instantaneous at 15 min resolution), the lag part gives a small bonus
for consistent temporal precedence, in line with the baseline summary
table in the paper.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


def _h(W: np.ndarray, d: int) -> tuple[float, np.ndarray]:
    """h(W) = tr(exp(W o W)) - d; grad = (exp(W o W))^T o 2W."""
    M = W * W
    E = expm(M)
    h = float(np.trace(E)) - d
    G = E.T * (2.0 * W)
    return h, G


def _build_lagged(X: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y, Z) with Y = X[p:] and Z = [X[p-1:-1] | X[p-2:-2] | ...]."""
    T, d = X.shape
    Y = X[p:]  # (T - p, d)
    blocks = [X[p - k - 1 : T - k - 1] for k in range(p)]
    Z = np.hstack(blocks)  # (T - p, p*d)
    return Y, Z


def _dynotears_linear(
    X: np.ndarray,
    p: int = 1,
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 8,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (W, A) where A has shape (p, d, d)."""
    T, d = X.shape
    if T <= p + 2 or d < 2:
        return np.zeros((d, d), dtype=np.float64), np.zeros((p, d, d), dtype=np.float64)

    # Column-standardize
    Xc = X - X.mean(axis=0, keepdims=True)
    Xc /= np.maximum(Xc.std(axis=0, keepdims=True), 1e-9)

    Y, Z = _build_lagged(Xc, p)  # (T-p, d), (T-p, p*d)
    n = Y.shape[0]

    n_W = d * d
    n_A = p * d * d
    # Parameterization: (W+, W-, A+, A-) all non-negative
    total = 2 * (n_W + n_A)

    def _split(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wp, wn = w[:n_W], w[n_W : 2 * n_W]
        ap, an = w[2 * n_W : 2 * n_W + n_A], w[2 * n_W + n_A :]
        W = (wp - wn).reshape(d, d)
        A = (ap - an).reshape(p * d, d)
        return W, A

    def _func(w: np.ndarray, rho: float, alpha: float):
        W, A = _split(w)
        R = Y - Y @ W - Z @ A
        loss = 0.5 / n * float((R * R).sum())
        G_W = -1.0 / n * (Y.T @ R)
        G_A = -1.0 / n * (Z.T @ R)
        h_val, G_h = _h(W, d)
        obj = (
            loss
            + 0.5 * rho * h_val * h_val
            + alpha * h_val
            + lambda_w * float(w[: 2 * n_W].sum())
            + lambda_a * float(w[2 * n_W :].sum())
        )
        G_W_full = (G_W + (rho * h_val + alpha) * G_h).ravel()
        G_A_full = G_A.ravel()
        g = np.concatenate(
            [
                G_W_full + lambda_w,
                -G_W_full + lambda_w,
                G_A_full + lambda_a,
                -G_A_full + lambda_a,
            ]
        )
        return obj, g

    # Bounds: all non-negative, W-diagonal forced to 0
    bnds: list[tuple[float, float | None]] = []
    # W+ and W- blocks: force diagonals to 0 in both
    for _ in range(2):
        for i in range(d):
            for j in range(d):
                bnds.append((0.0, 0.0) if i == j else (0.0, None))
    # A+ and A- blocks: no zero constraint (lagged self-coupling is allowed)
    for _ in range(2):
        for _idx in range(n_A):
            bnds.append((0.0, None))

    w_est = np.zeros(total, dtype=np.float64)
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
            W_new, _ = _split(w_new)
            h_new, _ = _h(W_new, d)
            if h_new > 0.25 * h_prev:
                rho *= 10
            else:
                break
        w_est = w_new
        h_prev = h_new
        alpha += rho * h_prev
        if h_prev <= h_tol or rho >= rho_max:
            break

    W, A = _split(w_est)
    # Reshape A to (p, d, d)
    A = A.reshape(p, d, d)
    # Hard threshold
    W[np.abs(W) < 0.3] = 0.0
    A[np.abs(A) < 0.3] = 0.0
    return W, A


def score(data: np.ndarray, names=None) -> np.ndarray:
    """Return DYNOTEARS directed score matrix (N, N).

    S[i, j] = |W_ij| + 0.3 * max_k |A_k[i, j]| -- interpret i as parent.
    """
    T, N = data.shape
    try:
        X = data
        if T > 2000:
            idx = np.linspace(0, T - 1, 2000).astype(np.int64)
            X = data[idx]

        # p=1 is sufficient; higher p blows up the variable count quadratically.
        p = 1

        # Keep runtime bounded for large N
        if N <= 30:
            max_iter = 8
        elif N <= 80:
            max_iter = 5
        else:
            max_iter = 3

        W, A = _dynotears_linear(X, p=p, lambda_w=0.1, lambda_a=0.1, max_iter=max_iter)

        S_w = np.abs(W)
        S_a = np.max(np.abs(A), axis=0) if A.size else np.zeros_like(S_w)
        S = S_w + 0.3 * S_a
        np.fill_diagonal(S, 0.0)
        return S
    except Exception:
        return np.zeros((N, N), dtype=np.float64)
