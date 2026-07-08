"""Bottom-up aggregation with ADDITIVE hidden load.

Old model (trivial for a GNN):
    X_parent = sum(children) / alpha      # constant scalar
New model:
    X_parent = sum(children) + H(t)       # H has its own dynamics
where H(t) is an independent seasonal + AR(1) process.

The fraction hidden/children naturally varies over time. We still record
    hidden_load_fraction = mean(H) / mean(sum_children)
as a scalar summary in the per-meter metadata (hierarchy.json).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .timeseries import seasonal_component, ar1_noise


def _generate_hidden_load(n: int, target_mean: float,
                          rng: np.random.Generator,
                          shared: dict | None = None) -> np.ndarray:
    """Independent residual-load time series with its own daily/weekly cycle.

    target_mean sets the magnitude scale. Shape: seasonal + AR(1) noise +
    optional mild coupling to the shared temperature signal (residual loads
    often include heating/cooling remnants).
    """
    if target_mean <= 0:
        return np.zeros(n)
    s = seasonal_component(
        n, target_mean,
        amp_day=rng.uniform(0.2, 0.5),
        amp_week=rng.uniform(0.05, 0.2),
        amp_year=rng.uniform(0.1, 0.3),
        rng=rng,
    )
    noise = ar1_noise(n, rng.uniform(0.5, 0.85), target_mean * 0.15, rng)
    H = s + noise
    if shared is not None and rng.random() < 0.5:
        sens = rng.uniform(0.1, 0.4)
        H = H * (1.0 + sens * shared["temperature"])
    H = np.clip(H, 0.0, None)
    # Rescale to exact target mean (otherwise seasonal phase can shift it)
    m = H.mean()
    if m > 1e-9:
        H = H * (target_mean / m)
    return H


def build_parent_series(leaves_df: pd.DataFrame, meters: list,
                        rng: np.random.Generator,
                        shared: dict | None = None,
                        hidden_ratio_range=(1.5, 3.5)
                        ) -> tuple[pd.DataFrame, float, float]:
    """Bottom-up aggregation. Returns (all_df, mean_hidden_ratio, consistency).

    hidden_ratio_range: range for target ratio mean(H) / mean(sum_children).
    Default (1.5, 3.5) -> mean H/S = 2.5 -> non-additive fraction ~= 0.71,
    chosen so roughly 70% of parent consumption carries a non-additive
    hidden load (hidden_load_fraction around 2.17 -> ~68% non-additive).
    """
    all_df = leaves_df.copy()
    inner = [m for m in meters if m["children"]]
    inner_sorted = sorted(inner, key=lambda m: -m["depth"])

    ratios = []
    consistency_errors = []
    for m in inner_sorted:
        child_ids = m["children"]
        S = all_df[child_ids].sum(axis=1)
        mean_S = float(S.mean())
        if mean_S < 1e-9:
            # Degenerate: just copy the sum so parent != 0
            all_df[m["id"]] = S.astype("float64")
            m["has_hidden_load"] = False
            m["hidden_load_fraction"] = 0.0
            continue

        target_ratio = float(rng.uniform(*hidden_ratio_range))
        H = _generate_hidden_load(len(S), mean_S * target_ratio, rng, shared)
        parent = S.values + H
        all_df[m["id"]] = parent.astype("float64")

        # hidden_load_fraction convention: (1 - alpha)/alpha with
        # alpha = mean(S)/mean(parent). For additive H this equals H/S.
        m["has_hidden_load"] = True
        m["hidden_load_fraction"] = float(H.mean() / mean_S)
        ratios.append(target_ratio)

        # Consistency metric: mean |parent - S| / mean(S) = target_ratio on avg.
        diff = np.abs(parent - S.values)
        consistency_errors.append(float(diff.mean() / mean_S))

    # Leaves: no hidden load
    for m in meters:
        if not m["children"]:
            m["has_hidden_load"] = False
            m["hidden_load_fraction"] = 0.0

    # Reorder columns to meter_000, meter_001, ...
    ordered_cols = [m["id"] for m in meters]
    all_df = all_df[ordered_cols]

    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    mean_consistency = float(np.mean(consistency_errors)) if consistency_errors else 0.0
    return all_df, mean_ratio, mean_consistency


def classify_hidden_load_mode(mean_hidden_ratio: float) -> str:
    """Classify the dataset's hidden load magnitude.

    mean_hidden_ratio = mean(H)/mean(children_sum) averaged across inner nodes.
    With range (1.5, 3.5) the overall mean is ~2.5. Thresholds calibrated
    so the three modes split the range roughly evenly.
    """
    if mean_hidden_ratio >= 2.8:
        return "large"
    if mean_hidden_ratio >= 2.0:
        return "medium"
    return "small"
