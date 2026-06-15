"""The Edmonds/Chu-Liu decoder recovers a known chain."""
import numpy as np

from meterhierarchy.model.decode import find_best_tree


def test_recovers_chain():
    # Strong edges 0->1 and 1->2; everything else weak.
    S = np.full((3, 3), 0.01)
    S[0, 1] = 0.9
    S[1, 2] = 0.9
    tree = find_best_tree(3, S)
    assert tree.get(1) == 0, tree
    assert tree.get(2) == 1, tree
    # exactly one node is a root (has no parent)
    assert len(tree) == 2, tree


if __name__ == "__main__":
    test_recovers_chain()
    print("test_decode OK")
