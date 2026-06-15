"""Edge-level evaluation metrics.

The predicted hierarchy is a ``parent_of`` mapping ``{child_index: parent_index}``
(as returned by the Edmonds decoder); the ground truth is a set of directed
``(parent_index, child_index)`` edges.
"""
from __future__ import annotations

from typing import Dict, Set, Tuple


def evaluate_f1(
    predicted: Dict[int, int],
    true_edges: Set[Tuple[int, int]],
) -> Tuple[float, float, float]:
    """Return ``(f1, precision, recall)`` of the predicted directed edges.

    Args:
        predicted: ``{child: parent}`` mapping from the tree decoder.
        true_edges: set of ground-truth ``(parent, child)`` tuples.
    """
    pred_edges = {(parent, child) for child, parent in predicted.items()}
    tp = len(pred_edges & true_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return f1, precision, recall
