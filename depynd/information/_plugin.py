import numpy as np


def _mi_plugin(X, Y):
    XY = np.hstack([X, Y])
    return _entropy_plugin(X) + _entropy_plugin(Y) - _entropy_plugin(XY)


def _entropy_plugin(X):
    n, d = X.shape
    _, cnt = np.unique(X, axis=0, return_counts=True)
    p = cnt / n
    return -np.sum(p * np.log(p))
