"""Meter hierarchy detection in building energy systems.

This package reconstructs the parent-child wiring hierarchy of energy (sub)meters
from consumption time series alone. It bundles:

* a two-pass Graph Neural Network with directed dual-embedding attention
  (:mod:`meterhierarchy.model`),
* hand-crafted pairwise/node features (:mod:`meterhierarchy.features`),
* a synthetic case generator (:mod:`meterhierarchy.data`),
* classical, optimization-based and amortized (AVICI) baselines
  (:mod:`meterhierarchy.baselines`, :mod:`meterhierarchy.avici`),
* a shared maximum-spanning-arborescence (Edmonds/Chu-Liu) decoder.

All edge scorers (GNN and baselines) emit an ``(N, N)`` score matrix where
``S[i, j]`` is the strength that meter ``i`` is the parent of meter ``j``; the
shared decoder turns that matrix into a valid rooted tree.
"""

__version__ = "0.1.0"
