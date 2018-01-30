import numpy as np
from .mutual_information import conditional_mutual_information


def mrmr(X, y, lamb=0.0, method=None, options=None):
    n, d = X.shape
    return list(range(d))


def mifs(X, y, lamb=0.0, method=None, options=None):
    n, d = X.shape
    selected = []
    while True:
        max_cmi = -np.inf
        not_selected = set(range(d)) - set(selected)
        z = X[:, selected]
        for i in not_selected:
            x = X[:, [i]]
            cmi = conditional_mutual_information(x, y, z, method, options)
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
            cmi = conditional_mutual_information(x, y, z, method, options)
            if min_cmi > cmi:
                min_cmi = cmi
                min_idx = i

        if min_cmi >= lamb or len(selected) == 0:
            return selected

        selected.remove(min_idx)
