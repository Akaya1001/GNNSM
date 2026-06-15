"""Feature computation for GNN meter hierarchy detection.

Produces, from a ``(T, N)`` consumption matrix:

* ``(N, N, 58)`` directed edge features per ordered meter pair,
* ``(N, 15)`` per-meter node features,
* ``(N, k)`` Laplacian positional encodings.

Fully vectorized NumPy (no Python loops over ``(i, j)`` pairs).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# =====================================================================
# Vectorized helpers
# =====================================================================
def _safe_corrcoef(data: np.ndarray) -> np.ndarray:
    """Correlation matrix with NaN handling."""
    corr = np.corrcoef(data.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _safe_corrcoef_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross-correlation matrix between the columns of ``a`` and ``b``."""
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    a_std = a_c.std(axis=0, keepdims=True) + EPS
    b_std = b_c.std(axis=0, keepdims=True) + EPS
    a_n = a_c / a_std
    b_n = b_c / b_std
    corr = (a_n.T @ b_n) / a.shape[0]
    return np.nan_to_num(corr, nan=0.0)


def _vec_lagged_corr(data: np.ndarray, lag: int) -> np.ndarray:
    """Lagged correlation matrix ``corr(x_i[lag:], x_j[:-lag])``."""
    if lag <= 0 or lag >= data.shape[0]:
        return np.zeros((data.shape[1], data.shape[1]))
    a = data[lag:]
    b = data[:-lag]
    return _safe_corrcoef_cross(a, b)


def _vec_aggregate_corr(data: np.ndarray, factor: int) -> np.ndarray:
    """Correlation on downsampled (block-averaged) data."""
    T, N = data.shape
    n_agg = T // factor
    if n_agg < 3:
        return np.zeros((N, N))
    agg = data[: n_agg * factor].reshape(n_agg, factor, N).mean(axis=1)
    return _safe_corrcoef(agg)


def _vec_spearman(data: np.ndarray) -> np.ndarray:
    """Vectorized Spearman correlation matrix."""
    ranked = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)).astype(float), 0, data)
    return _safe_corrcoef(ranked)


def _vec_entropy(data: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Shannon entropy per column."""
    N = data.shape[1]
    ent = np.zeros(N)
    for i in range(N):
        x = data[:, i]
        hist, _ = np.histogram(x, bins=n_bins)
        p = hist / (hist.sum() + EPS)
        p = p[p > EPS]
        ent[i] = -float((p * np.log(p)).sum())
    return ent


def _vec_mutual_info(data: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Pairwise mutual information matrix (binned joint histograms)."""
    T, N = data.shape
    mi_mat = np.zeros((N, N))

    bins_all = np.zeros((T, N), dtype=np.int32)
    for i in range(N):
        x = data[:, i]
        edges = np.linspace(x.min() - EPS, x.max() + EPS, n_bins + 1)
        bins_all[:, i] = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    marginals = np.zeros((N, n_bins))
    for i in range(N):
        np.add.at(marginals[i], bins_all[:, i], 1)
    marginals /= T

    for i in range(N):
        for j in range(i + 1, N):
            flat_idx = bins_all[:, i] * n_bins + bins_all[:, j]
            joint_flat = np.bincount(flat_idx, minlength=n_bins * n_bins).astype(np.float64)
            joint = joint_flat.reshape(n_bins, n_bins) / T
            px = marginals[i]
            py = marginals[j]
            outer = px[:, None] * py[None, :]
            mask = (joint > EPS) & (outer > EPS)
            mi = np.sum(joint[mask] * np.log(joint[mask] / outer[mask])) if mask.any() else 0.0
            mi_mat[i, j] = mi
            mi_mat[j, i] = mi
    return mi_mat


# =====================================================================
# Laplacian positional encodings (Dwivedi et al., 2023)
# =====================================================================
def compute_laplacian_pe(data: np.ndarray, k: int = 8) -> np.ndarray:
    """Laplacian positional encodings from the (positive) correlation adjacency.

    Always returns ``(N, k)``; for small ``N`` the missing columns are zero
    padded. (Z-normalize per split afterwards so the scale does not dominate the
    first linear layer.)
    """
    N = data.shape[1]
    out_k = k
    usable = min(k, max(N - 2, 0))
    if usable < 1:
        return np.zeros((N, out_k), dtype=np.float32)

    corr = np.corrcoef(data.T)
    corr = np.nan_to_num(corr, nan=0.0)
    A = np.maximum(corr, 0)
    np.fill_diagonal(A, 0)

    D = np.diag(A.sum(axis=1))
    L = D - A
    d_inv_sqrt = np.diag(1.0 / (np.sqrt(A.sum(axis=1)) + 1e-8))
    L_norm = d_inv_sqrt @ L @ d_inv_sqrt

    try:
        _, eigenvectors = np.linalg.eigh(L_norm)
        pe = eigenvectors[:, 1 : 1 + usable].astype(np.float32)
        # Sign convention: first entry non-negative (stabilizes across cases).
        for i in range(pe.shape[1]):
            if pe[0, i] < 0:
                pe[:, i] *= -1.0
    except np.linalg.LinAlgError:
        pe = np.zeros((N, usable), dtype=np.float32)

    if pe.shape[1] < out_k:
        pad = np.zeros((N, out_k - pe.shape[1]), dtype=np.float32)
        pe = np.concatenate([pe, pad], axis=1)
    return pe


# =====================================================================
# 58 directed edge features
# =====================================================================
def compute_pair_features_fast(data: np.ndarray) -> tuple:
    """Compute 58 pairwise edge features for all ``(i, j)`` pairs.

    Returns ``(feat_cube (N, N, 58), abs_means (N,), corr_raw (N, N))``.
    """
    T, N = data.shape
    feat_cube = np.zeros((N, N, 58), dtype=np.float64)

    corr_raw = _safe_corrcoef(data)
    diff_data = np.diff(data, axis=0)
    corr_diff = _safe_corrcoef(diff_data) if T > 2 else np.zeros((N, N))

    abs_means = np.abs(data).mean(axis=0)
    stds = data.std(axis=0) + EPS
    means = data.mean(axis=0)

    q25 = np.percentile(data, 25, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    q05 = np.percentile(data, 5, axis=0)
    q95 = np.percentile(data, 95, axis=0)
    iqr = q75 - q25 + EPS
    medians = np.median(data, axis=0)

    lag1_ij = _vec_lagged_corr(data, 1)
    lag2_ij = _vec_lagged_corr(data, 2)
    hourly_corr = _vec_aggregate_corr(data, 4)
    daily_corr = _vec_aggregate_corr(data, 96)
    spearman = _vec_spearman(data)
    cov_mat = np.cov(data.T)

    try:
        prec = np.linalg.pinv(corr_raw)
    except np.linalg.LinAlgError:
        prec = np.eye(N)
    prec_diag_sqrt = np.sqrt(np.abs(np.diag(prec))) + EPS

    ent = _vec_entropy(data)
    mi_mat = _vec_mutual_info(data, n_bins=20)

    peak_mask = data > np.percentile(data, 90, axis=0, keepdims=True)
    zero_mask = data < np.percentile(data, 10, axis=0, keepdims=True)
    sign_mask = data > medians[None, :]
    peak_counts = peak_mask.sum(axis=0).astype(float) + EPS
    zero_counts = zero_mask.sum(axis=0).astype(float) + EPS

    peak_overlap = peak_mask.astype(float).T @ peak_mask.astype(float)
    zero_overlap = zero_mask.astype(float).T @ zero_mask.astype(float)
    sign_agree = (sign_mask.astype(float).T @ sign_mask.astype(float)) + (
        (~sign_mask).astype(float).T @ (~sign_mask).astype(float)
    )
    sign_agree /= T

    # DTW proxy: RMSE between each pair (downsampled for speed).
    ds_factor = max(1, T // 500)
    data_ds = data[::ds_factor]
    dtw_dist = np.sqrt(((data_ds[:, :, None] - data_ds[:, None, :]) ** 2).mean(axis=0))

    if T > 10:
        x_target = data[1:]
        x_lag = data[:-1]
        cross_cov = _safe_corrcoef_cross(x_target, x_lag)
        granger_r2 = cross_cov ** 2
    else:
        granger_r2 = np.zeros((N, N))

    abs_data = np.abs(data)
    spearman_abs = _vec_spearman(abs_data)
    mad_mat = np.abs(data_ds[:, :, None] - data_ds[:, None, :]).mean(axis=0)

    mi_col = abs_means[:, None]
    mi_row = abs_means[None, :]
    si_col = stds[:, None]
    si_row = stds[None, :]
    iqr_col = iqr[:, None]
    iqr_row = iqr[None, :]
    ent_col = ent[:, None]
    ent_row = ent[None, :]

    # 0-1: raw correlation
    feat_cube[:, :, 0] = corr_raw
    feat_cube[:, :, 1] = np.abs(corr_raw)
    # 2: correlation asymmetry (0 for symmetric corr; kept for schema)
    feat_cube[:, :, 2] = 0.0
    # 3-5: mean ratio features
    feat_cube[:, :, 3] = mi_col / (mi_row + EPS)
    feat_cube[:, :, 4] = mi_row - mi_col
    feat_cube[:, :, 5] = mi_col + mi_row
    # 6-7: diff correlation
    feat_cube[:, :, 6] = corr_diff
    feat_cube[:, :, 7] = np.abs(corr_diff)
    # 8: max of raw and diff
    feat_cube[:, :, 8] = np.maximum(np.abs(corr_raw), np.abs(corr_diff))
    # 9-11: lagged correlations
    feat_cube[:, :, 9] = lag1_ij
    feat_cube[:, :, 10] = lag2_ij
    feat_cube[:, :, 11] = lag1_ij.T
    # 12-13: lag asymmetry
    feat_cube[:, :, 12] = lag1_ij - lag1_ij.T
    feat_cube[:, :, 13] = np.abs(lag1_ij) + np.abs(lag2_ij)
    # 14-17: aggregated correlations
    feat_cube[:, :, 14] = hourly_corr
    feat_cube[:, :, 15] = hourly_corr.T
    feat_cube[:, :, 16] = daily_corr
    feat_cube[:, :, 17] = daily_corr.T
    # 18-19: relative magnitude
    feat_cube[:, :, 18] = mi_col / (mi_row + mi_col + EPS)
    feat_cube[:, :, 19] = np.log(mi_col / (mi_row + EPS) + EPS)
    # 20: sum
    feat_cube[:, :, 20] = mi_col + mi_row
    # 21-23: cross-statistics
    feat_cube[:, :, 21] = mi_col * mi_row
    feat_cube[:, :, 22] = np.minimum(mi_col, mi_row) / (np.maximum(mi_col, mi_row) + EPS)
    feat_cube[:, :, 23] = np.maximum(mi_col, mi_row)
    # 24-26: Spearman
    feat_cube[:, :, 24] = spearman
    feat_cube[:, :, 25] = np.abs(spearman)
    feat_cube[:, :, 26] = np.maximum(np.abs(spearman), feat_cube[:, :, 8])
    # 27: residual correlation
    feat_cube[:, :, 27] = corr_diff
    # 28-29: sign consistency
    feat_cube[:, :, 28] = sign_agree
    feat_cube[:, :, 29] = 1.0 - sign_agree
    # 30-33: variance features
    feat_cube[:, :, 30] = si_col / si_row
    feat_cube[:, :, 31] = np.abs(si_col - si_row) / (si_col + si_row)
    feat_cube[:, :, 32] = np.minimum(si_col, si_row) / (np.maximum(si_col, si_row) + EPS)
    feat_cube[:, :, 33] = np.abs(np.log(si_col / si_row + EPS))
    # 34-36: covariance
    feat_cube[:, :, 34] = cov_mat
    feat_cube[:, :, 35] = cov_mat / (si_col * si_row + EPS)
    feat_cube[:, :, 36] = np.abs(cov_mat)
    # 37-39: IQR features
    feat_cube[:, :, 37] = iqr_col / iqr_row
    feat_cube[:, :, 38] = np.minimum(iqr_col, iqr_row) / (np.maximum(iqr_col, iqr_row) + EPS)
    q75_col = q75[:, None]
    q75_row = q75[None, :]
    feat_cube[:, :, 39] = np.abs(q75_col - q75_row) / (iqr_col + iqr_row)
    # 40-41: peak overlap
    feat_cube[:, :, 40] = peak_overlap / peak_counts[:, None]
    feat_cube[:, :, 41] = peak_overlap / peak_counts[None, :]
    # 42-43: zero overlap
    feat_cube[:, :, 42] = zero_overlap / zero_counts[:, None]
    feat_cube[:, :, 43] = zero_overlap / zero_counts[None, :]
    # 44-46: entropy
    feat_cube[:, :, 44] = ent_col * np.ones((1, N))
    feat_cube[:, :, 45] = np.ones((N, 1)) * ent_row
    feat_cube[:, :, 46] = np.abs(ent_col - ent_row)
    # 47-48: mutual information
    feat_cube[:, :, 47] = mi_mat
    feat_cube[:, :, 48] = mi_mat
    # 49-50: DTW distance
    feat_cube[:, :, 49] = 1.0 / (1.0 + dtw_dist)
    feat_cube[:, :, 50] = dtw_dist
    # 51-52: partial correlation
    pcorr = -prec / (prec_diag_sqrt[:, None] * prec_diag_sqrt[None, :] + EPS)
    np.fill_diagonal(pcorr, 0.0)
    feat_cube[:, :, 51] = pcorr
    feat_cube[:, :, 52] = np.abs(pcorr)
    # 53: Granger R^2
    feat_cube[:, :, 53] = granger_r2
    # 54: Spearman on absolute values
    feat_cube[:, :, 54] = spearman_abs
    # 55: mean absolute difference
    feat_cube[:, :, 55] = mad_mat
    # 56-57: asymmetric magnitude
    feat_cube[:, :, 56] = mi_row - mi_col
    feat_cube[:, :, 57] = (mi_col ** 2 + mi_row ** 2) / (mi_col + mi_row + EPS)

    feat_cube = np.nan_to_num(feat_cube, nan=0.0, posinf=1e6, neginf=-1e6)
    return feat_cube, abs_means, corr_raw


# =====================================================================
# 15 per-node features
# =====================================================================
def compute_node_features_fast(data: np.ndarray, corr_raw: np.ndarray) -> np.ndarray:
    """Compute 15 per-node features -> ``(N, 15)``."""
    T, N = data.shape
    node_feats = np.zeros((N, 15), dtype=np.float64)

    abs_means = np.abs(data).mean(axis=0) + EPS
    stds = data.std(axis=0) + EPS
    means = data.mean(axis=0)

    mean_rank = np.argsort(np.argsort(abs_means)).astype(float) / max(N - 1, 1)

    q05 = np.percentile(data, 5, axis=0)
    q95 = np.percentile(data, 95, axis=0)

    corr_abs = np.abs(corr_raw).copy()
    np.fill_diagonal(corr_abs, 0.0)
    corr_mean = corr_abs.sum(axis=1) / max(N - 1, 1)
    corr_max = corr_abs.max(axis=1) if N > 1 else np.zeros(N)

    ent = _vec_entropy(data)

    if T > 1:
        ac1 = np.array([np.corrcoef(data[:-1, i], data[1:, i])[0, 1] for i in range(N)])
        ac1 = np.nan_to_num(ac1, nan=0.0)
    else:
        ac1 = np.zeros(N)

    if T > 96:
        ac96 = np.array([np.corrcoef(data[:-96, i], data[96:, i])[0, 1] for i in range(N)])
        ac96 = np.nan_to_num(ac96, nan=0.0)
    else:
        ac96 = np.zeros(N)

    frac_above = (data > means[None, :]).mean(axis=0)
    centered = data - means[None, :]
    skew = (centered ** 3).mean(axis=0) / (stds ** 3)
    kurt = (centered ** 4).mean(axis=0) / (stds ** 4) - 3.0
    n_high_corr = (corr_abs > 0.5).sum(axis=1).astype(float) / max(N - 1, 1)

    corr_std = np.zeros(N)
    for i in range(N):
        vals = corr_abs[i, corr_abs[i, :] > 0]
        corr_std[i] = vals.std() if len(vals) > 0 else 0.0

    node_feats[:, 0] = abs_means
    node_feats[:, 1] = mean_rank
    node_feats[:, 2] = stds / abs_means  # coefficient of variation
    node_feats[:, 3] = (np.abs(data) < abs_means[None, :] * 0.01).mean(axis=0)  # near-zero fraction
    node_feats[:, 4] = (q95 - q05) / abs_means  # peak-to-baseline
    node_feats[:, 5] = corr_mean
    node_feats[:, 6] = corr_max
    node_feats[:, 7] = ent
    node_feats[:, 8] = ac1
    node_feats[:, 9] = ac96
    node_feats[:, 10] = frac_above
    node_feats[:, 11] = np.nan_to_num(skew, nan=0.0)
    node_feats[:, 12] = n_high_corr
    node_feats[:, 13] = corr_std
    node_feats[:, 14] = np.nan_to_num(kurt, nan=0.0)
    return node_feats
