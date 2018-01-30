import numpy as np
from . import mi


def mimat(X, method='knn', options=None):
    n, d = X.shape
    mis = np.eye(d)
    for i, j in [(i, j) for i in range(d) for j in range(i + 1, d)]:
        x = X[:, [i]]
        y = X[:, [j]]
        mis[i, j] = mi.mi(x, y, method, options)

    mis[mis < 0] = 0
    mis = mis + mis.T
    mis[np.eye(d, dtype=bool)] = np.nan
    return mis


def cmimat(X, method='knn', options=None):
    n, d = X.shape
    cmis = np.eye(d)
    for i, j in [(i, j) for i in range(d) for j in range(i + 1, d)]:
        x = X[:, [i]]
        y = X[:, [j]]
        idx_rest = (np.arange(d) != i) & (np.arange(d) != j)
        z = X[:, idx_rest]
        cmis[i, j] = mi.cmi(x, y, z, method, options)

    cmis[cmis < 0] = 0
    cmis = cmis + cmis.T
    cmis[np.eye(d, dtype=bool)] = np.nan
    return cmis
