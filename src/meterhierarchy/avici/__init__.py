"""AVICI: amortized inference for causal/hierarchy structure (Lorch et al., 2022).

A permutation-equivariant axial-attention transformer that maps a consumption
matrix directly to an ``(N, N)`` edge-probability matrix. Trained once on the
synthetic meter-hierarchy corpus, it transfers zero-shot to real installations,
mirroring the GNN's setup so the two are directly comparable.
"""

from .model import AVICIModel, AVICIConfig
from .data import case_to_input, adjacency_from_edges, standardize_columns

__all__ = [
    "AVICIModel",
    "AVICIConfig",
    "case_to_input",
    "adjacency_from_edges",
    "standardize_columns",
]
