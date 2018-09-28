import numpy as np

from depynd.information import conditional_mutual_information


def _iamb(X, lamb=0.0, **kwargs):
    """Learn the structure of Markov random field by finding Markov blanket for each variable with Incremental
    Association Markov Blanket [tsamardinos2003algorithms]_.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Observations of variables.
    lamb: float
        Threshold for independence test.
    kwargs : dict, default None
        Optional parameters for MI estimation.

    Returns
    -------
    adj : array, shape (n_features, n_features)
        Estimated adjacency matrix of an MRF.

    References
    ----------
    .. [tsamardinos2003algorithms] Tsamardinos, Ioannis, et al. "Algorithms for Large Scale Markov Blanket Discovery."
        FLAIRS conference. Vol. 2. 2003.
    """
    n, d = X.shape
    adj = np.zeros([d, d], dtype=bool)
    for i in range(d):
        adj_tmp = np.zeros([d, d], dtype=bool)
        adj_tmp = _grow(adj_tmp, i, X, lamb, **kwargs)
        adj_tmp = _shrink(adj_tmp, i, X, lamb, **kwargs)
        adj |= adj_tmp
    return adj


def _grow(adj, i, X, lamb, **kwargs):
    n, d = X.shape
    x = X[:, i]
    updated = True
    while updated:
        updated = False
        vmax = -np.inf
        z = X[:, adj[i]]
        non_adj = ~adj[i] & (np.arange(d) != i)
        for j in non_adj.nonzero()[0]:
            y = X[:, j]
            cmi = conditional_mutual_information(x, y, z, **kwargs)
            if vmax < cmi:
                vmax = cmi
                imax, jmax = i, j
        if vmax >= lamb:
            adj[imax, jmax] = adj[jmax, imax] = 1
            updated = True
    return adj


def _shrink(adj, i, X, lamb, **kwargs):
    n, d = X.shape
    x = X[:, i]
    for j in adj[i].nonzero()[0]:
        other_adj = adj[i] & (np.arange(d) != j)
        y = X[:, j]
        z = X[:, other_adj]
        cmi = conditional_mutual_information(x, y, z, **kwargs)
        if cmi <= lamb:
            adj[i, j] = adj[j, i] = 0
        if np.count_nonzero(adj[i]) == 0:
            break
    return adj
