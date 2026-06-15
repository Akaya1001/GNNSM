"""Hand-crafted features for the GNN edge scorer.

* :func:`compute_pair_features_fast` -> ``(N, N, 58)`` directed edge features
* :func:`compute_node_features_fast` -> ``(N, 15)`` per-meter features
* :func:`compute_laplacian_pe` -> ``(N, k)`` Laplacian positional encodings
"""

from .pairwise import (
    compute_pair_features_fast,
    compute_node_features_fast,
    compute_laplacian_pe,
)

__all__ = [
    "compute_pair_features_fast",
    "compute_node_features_fast",
    "compute_laplacian_pe",
]
