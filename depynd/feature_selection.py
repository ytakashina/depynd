import numpy as np
from .mi import MIEstimator, CMIEstimator


def _mrmr(X, y, alpha=0.0, method=None, options=None):
    n, d = X.shape
    return list(range(d))


def _mifs(X, y, alpha=0.0, method=None, options=None):
    n, d = X.shape
    selected = []
    while True:
        max_cmi = -np.inf
        not_selected = set(range(d)) - set(selected)
        z = X[:, selected]
        for i in not_selected:
            x = X[:, [i]]
            cmi = CMIEstimator(method=method, options=options).fit(x, y, z).cmi
            if max_cmi < cmi:
                max_cmi = cmi
                max_idx = i

        if max_cmi <= alpha or len(selected) == d:
            return selected

        selected += [max_idx]
