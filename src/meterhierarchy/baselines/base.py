"""Shared numerical helpers for the classical baseline scorers.

Only the small, purely numerical utilities that a baseline's ``score``
actually relies on live here. The evaluation harness, CSV IO, case loading
and tree extraction are intentionally *not* part of this module -- those
concerns live elsewhere in the package.
"""
from __future__ import annotations

import numpy as np


def standardize(x: np.ndarray) -> np.ndarray:
    """Column-wise z-score; zero-variance columns map to zero."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd
