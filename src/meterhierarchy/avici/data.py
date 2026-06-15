"""Adapt a meter consumption case to AVICI's ``(n, d, 2)`` input.

AVICI consumes a 2D set of observations (n samples x d variables). We treat each
timestep as a sample and each meter as a variable; channel 0 is the
per-variable-standardized consumption value, channel 1 is the intervention
indicator (always 0 here -- purely observational data).
"""
from __future__ import annotations

from typing import Optional, Set, Tuple

import numpy as np
import torch


def standardize_columns(data: np.ndarray) -> np.ndarray:
    """Per-variable z-score; zero-variance columns map to 0 (avoids varsortability)."""
    data = np.nan_to_num(data, nan=0.0).astype(np.float32)
    mu = data.mean(axis=0, keepdims=True)
    sd = data.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (data - mu) / sd


def case_to_input(
    data: np.ndarray,
    n_samples: Optional[int] = 400,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Build the ``(n, d, 2)`` AVICI input tensor from a ``(T, N)`` matrix.

    If ``T > n_samples`` a random subset of timesteps is drawn (sorted).
    """
    x = standardize_columns(data)
    T = x.shape[0]
    if n_samples is not None and T > n_samples:
        rng = rng or np.random.default_rng(0)
        idx = np.sort(rng.choice(T, size=n_samples, replace=False))
        x = x[idx]
    interv = np.zeros_like(x)
    inp = np.stack([x, interv], axis=-1)  # (n, d, 2)
    return torch.from_numpy(inp.astype(np.float32))


def adjacency_from_edges(true_edges: Set[Tuple[int, int]], n: int) -> torch.Tensor:
    """``(N, N)`` target adjacency; ``A[p, c] = 1`` iff ``p`` is the parent of ``c``."""
    A = np.zeros((n, n), dtype=np.float32)
    for p, c in true_edges:
        A[p, c] = 1.0
    return torch.from_numpy(A)
