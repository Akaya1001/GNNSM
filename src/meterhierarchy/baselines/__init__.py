"""Baseline edge scorers.

Every baseline exposes ``score(data, names) -> (N, N)`` where ``S[i, j]`` is the
strength that meter ``i`` is the parent of meter ``j``. The shared
:func:`meterhierarchy.model.find_best_tree` decoder turns that matrix into a
rooted tree, so all methods (and the GNN) are compared identically.

Scorers are resolved lazily so importing this package never requires every
optional dependency at once.
"""
from __future__ import annotations

import importlib
from typing import Callable, Dict, List

# name -> "module:function"
REGISTRY: Dict[str, str] = {
    "CL": "meterhierarchy.baselines.chow_liu:score",
    "PC": "meterhierarchy.baselines.pc:score",
    "Granger": "meterhierarchy.baselines.granger:score",
    "HL": "meterhierarchy.baselines.hidden_load:score",
    "NOTEARS": "meterhierarchy.baselines.notears:score",
    "DYNOTEARS": "meterhierarchy.baselines.dynotears:score",
    "AVICI": "meterhierarchy.baselines.avici:score",
}


def get_scorer(name: str) -> Callable:
    """Return the ``score`` callable for a registered baseline name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(REGISTRY)}")
    module_path, func = REGISTRY[name].split(":")
    mod = importlib.import_module(module_path)
    return getattr(mod, func)


def available() -> List[str]:
    return list(REGISTRY)
