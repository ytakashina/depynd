import numpy as np


def _mi_plugin(X, Y):
    XY = np.hstack([X, Y])
    return _h_plugin(X) + _h_plugin(Y) - _h_plugin(XY)


def _h_plugin(X):
    n, d = X.shape
    row_dtype = np.dtype((np.void, X.dtype.itemsize * d))
    b = np.ascontiguousarray(X).view(row_dtype)
    _, cnt = np.unique(b, return_counts=True)
    p = cnt / n
    return -np.sum(p * np.log(p))
