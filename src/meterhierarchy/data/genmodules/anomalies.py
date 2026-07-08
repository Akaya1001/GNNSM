"""Four anomaly types plus linear meter drift.

Drift and all anomaly types are applied per-leaf at low rates BEFORE
aggregation, so parents are built from the already-perturbed leaves and
the hierarchy remains approximately consistent. Small independent
anomalies can additionally be added to parents afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_type1_zero_gaps(x: np.ndarray, rng: np.random.Generator,
                          rate: float = 0.03, min_len: int = 2,
                          max_len: int = 96) -> np.ndarray:
    """Abrupt NaN gaps (communication outages)."""
    n = len(x)
    n_gaps = int(n * rate / ((min_len + max_len) / 2))
    for _ in range(n_gaps):
        start = int(rng.integers(0, n))
        length = int(rng.integers(min_len, max_len + 1))
        x[start:start + length] = np.nan
    return x


def apply_type2_gradient_break(x: np.ndarray, rng: np.random.Generator,
                               n_events: int = 2) -> np.ndarray:
    """Flat plateau followed by a jump."""
    n = len(x)
    for _ in range(n_events):
        start = int(rng.integers(0, n - 50))
        length = int(rng.integers(10, 50))
        flat_val = x[start] if not np.isnan(x[start]) else 0.0
        x[start:start + length] = flat_val
    return x


def apply_type3_dip(x: np.ndarray, rng: np.random.Generator,
                    n_events: int = 1) -> np.ndarray:
    """Single-step downward spike (meter reset)."""
    n = len(x)
    for _ in range(n_events):
        idx = int(rng.integers(0, n))
        x[idx] = 0.0
    return x


def apply_type4_dst_spike(x: np.ndarray, index: pd.DatetimeIndex,
                          rng: np.random.Generator) -> np.ndarray:
    """Sharp positive spike at DST fall-back (last Sunday of October, 03:00)."""
    n = len(x)
    # naive approach: find end-of-October Sundays in the index
    for yr in np.unique(index.year):
        oct_last_sunday = None
        for day in range(31, 24, -1):
            try:
                dt = pd.Timestamp(year=int(yr), month=10, day=day)
                if dt.dayofweek == 6:
                    oct_last_sunday = dt
                    break
            except ValueError:
                continue
        if oct_last_sunday is None:
            continue
        target = oct_last_sunday + pd.Timedelta(hours=3)
        # find nearest index position
        diffs = np.abs((index - target).total_seconds())
        pos = int(np.argmin(diffs))
        if 0 <= pos < n and not np.isnan(x[pos]):
            x[pos] += float(rng.uniform(3, 6)) * max(1.0, np.nanmean(x))
    return x


def apply_drift(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Linear meter drift (up to +/-5% over the whole series)."""
    n = len(x)
    drift_rate = rng.uniform(-0.05, 0.05)
    drift = 1.0 + drift_rate * np.linspace(0, 1, n)
    return x * drift


def apply_all_to_leaf(x: np.ndarray, index: pd.DatetimeIndex,
                     rng: np.random.Generator,
                     config: dict | None = None) -> np.ndarray:
    cfg = config or {}
    x = x.copy()
    if rng.random() < cfg.get("p_drift", 0.5):
        x = apply_drift(x, rng)
    if rng.random() < cfg.get("p_type1", 0.6):
        x = apply_type1_zero_gaps(x, rng, rate=cfg.get("type1_rate", 0.02))
    if rng.random() < cfg.get("p_type2", 0.3):
        x = apply_type2_gradient_break(x, rng,
                                       n_events=int(rng.integers(1, 3)))
    if rng.random() < cfg.get("p_type3", 0.3):
        x = apply_type3_dip(x, rng, n_events=int(rng.integers(1, 3)))
    if rng.random() < cfg.get("p_type4", 0.4):
        x = apply_type4_dst_spike(x, index, rng)
    return x
