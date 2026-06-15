"""AVICI as a drop-in baseline scorer.

Loads a trained AVICI checkpoint once (cached) and returns the predicted
``(N, N)`` edge-probability matrix for a case. ``theta[i, j]`` = probability that
``i`` is the parent of ``j``, which matches the ``score`` convention directly.

The checkpoint path defaults to ``checkpoints/avici.pt`` and can be overridden
via the ``MH_AVICI_CKPT`` environment variable or the ``checkpoint`` argument.
Train one with ``scripts/train_avici.py``. If no checkpoint exists, a zero
matrix is returned (with a warning) so the evaluation harness stays robust.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..avici.model import AVICIConfig, AVICIModel
from ..avici.data import case_to_input
from ..utils.device import get_device

_DEFAULT_CKPT = "checkpoints/avici.pt"
_CACHE: dict = {}  # checkpoint path -> (model, n_samples, device)


def _load(checkpoint: str, device: torch.device):
    key = str(Path(checkpoint).resolve())
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = AVICIConfig.from_dict(payload["config"])
    model = AVICIModel(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    n_samples = int(payload.get("n_samples", 400))
    _CACHE[key] = (model, n_samples, device)
    return _CACHE[key]


def score(
    data: np.ndarray,
    names: Optional[list] = None,
    checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Return the AVICI ``(N, N)`` parent-score matrix for ``data`` ``(T, N)``."""
    N = data.shape[1]
    ckpt = checkpoint or os.environ.get("MH_AVICI_CKPT", _DEFAULT_CKPT)
    if not Path(ckpt).exists():
        warnings.warn(
            f"AVICI checkpoint not found at {ckpt!r}; returning zeros. "
            f"Train one with scripts/train_avici.py or set MH_AVICI_CKPT.",
            RuntimeWarning,
        )
        return np.zeros((N, N), dtype=np.float64)

    device = device or get_device()
    model, n_samples, dev = _load(ckpt, device)
    inp = case_to_input(data, n_samples=n_samples).to(dev)
    with torch.no_grad():
        probs = model.predict_matrix(inp).cpu().numpy().astype(np.float64)
    return probs
