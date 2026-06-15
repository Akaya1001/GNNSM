"""Every registered baseline returns an (N, N) matrix and decodes to a tree."""
import warnings

import numpy as np

from meterhierarchy.baselines import available, get_scorer
from meterhierarchy.model.decode import find_best_tree


def test_all_baselines_shapes():
    rng = np.random.default_rng(0)
    N, T = 6, 300
    data = rng.random((T, N)).astype(np.float64)
    for name in available():
        scorer = get_scorer(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # AVICI warns if no checkpoint -> returns zeros
            S = scorer(data)
        assert S.shape == (N, N), f"{name}: bad shape {S.shape}"
        tree = find_best_tree(N, S)
        assert isinstance(tree, dict)


if __name__ == "__main__":
    test_all_baselines_shapes()
    print("test_baselines OK")
