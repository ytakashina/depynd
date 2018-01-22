import numpy as np
from . import knn
from . import dr


def mimat(X, method='knn', is_discrete=None, k=3):
    n, d = X.shape
    mis = np.eye(d)
    for i, j in [(i, j) for i in range(d) for j in range(d) if i != j]:
        x = X[:, [i]]
        y = X[:, [j]]
        if method == 'dr':
            mis[i, j] = dr.mi_dr(x, y)[0]
        else:
            mis[i, j] = knn.mi_knn(x, y, k=3)

    mis[mis < 0] = 0
    mis[np.eye(d, dtype=bool)] = np.nan
    return mis


def cmimat(X, method='knn', is_discrete=None, k=3):
    n, d = X.shape
    cmis = np.eye(d)
    for i, j in [(i, j) for i in range(d) for j in range(d) if i != j]:
        x = X[:, [i]]
        y = X[:, [j]]
        idx_rest = (np.arange(d) != i) & (np.arange(d) != j)
        z = X[:, idx_rest]
        cmis[i, j] = knn.cmi_knn(x, y, z, k=3)

    cmis[cmis < 0] = 0
    cmis[np.eye(d, dtype=bool)] = np.nan
    return cmis
