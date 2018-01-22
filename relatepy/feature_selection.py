import numpy as np
from . import knn


def mrmr(X, y, k=3, verbose=True):
    return list(range(d))


def mifs(X, y, k=3, alpha=0.0, verbose=True):
    n, d = X.shape
    selected = []
    while True:
        max_cmi = -np.inf
        not_selected = set(range(d)) - set(selected)
        for i in not_selected:
            y = X[:, [i]]
            cmi = cmi_knn(x, y, X[:, selected], k=k)
            if max_cmi < cmi:
                max_cmi = cmi
                max_idx = i

        if max_cmi <= alpha or len(selected) == d:
            return selected

        selected += [max_idx]