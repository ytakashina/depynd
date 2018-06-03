import numpy as np

from depynd.information import conditional_mutual_information


def iamb(X, lamb=0.0, **kwargs):
    """Learn the structure of Markov random field by finding Markov blanket for each variable with Incremental
    Association Markov Blanket [tsamardinos2003algorithms]_.

    Parameters
    ----------
    X : array-like, shape (n_samples, d)
        Observations of variables.
    lamb: float
        Threshold for independence test.
    kwargs : dict, default None
        Optional parameters for MI estimation.

    Returns
    -------
    adj : array, shape (d, d)
        Estimated adjacency matrix of an MRF.

    References
    ----------
    .. [tsamardinos2003algorithms] Tsamardinos, Ioannis, et al. "Algorithms for Large Scale Markov Blanket Discovery."
        FLAIRS conference. Vol. 2. 2003.
    """
    n, d = X.shape
    adj = np.zeros([d, d], dtype=bool)
    for i in range(d):
        x = X[:, i]
        updated = True
        while updated:
            vmax = -np.inf
            updated = False
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
