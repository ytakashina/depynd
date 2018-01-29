import numpy as np
from .mi import MIEstimator, CMIEstimator


def mrmr(X, y, lamb=0.0, method=None, options=None):
    n, d = X.shape
    return list(range(d))


def mifs(X, y, lamb=0.0, method=None, options=None):
    n, d = X.shape
    selected = []
    y = y.reshape([-1, 1])
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

        if max_cmi <= lamb or len(selected) == d:
            break

        selected += [max_idx]

    while True:
        min_cmi = np.inf
        for i in selected:
            x = X[:, [i]]
            z = X[:, list(set(selected) - set([i]))]
            cmi = CMIEstimator(method=method, options=options).fit(x, y, z).cmi
            if min_cmi > cmi:
                min_cmi = cmi
                min_idx = i

        if min_cmi >= lamb or len(selected) == 0:
            return selected

        selected = list(set(selected) - set([min_idx]))