import numpy as np

from depynd.information import conditional_mutual_information


def iamb(X, lamb=0.0, method=None, options=None):
    """Search Markov blanket in a Bayesian network using Incremental Association Markov Blanket algorithm [1]_.

    Parameters
    ----------
    X : array, shape (n_samples, d)
        Observations of variables.
    lamb: float
        Threshold for independence test.
    method: str, default 'knn'
        Method for MI estimation.
    options : dict, default None
        Optional parameters for MI estimation.

    Returns
    -------
    adj : array, shape (d, d)
        Estimated Markov blanket.

    References
    ----------
    .. [1] Tsamardinos, Ioannis, et al. "Algorithms for Large Scale Markov Blanket Discovery." FLAIRS conference. Vol.
    2. 2003.
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
                cmi = conditional_mutual_information(x, y, z, method, options)
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
            cmi = conditional_mutual_information(x, y, z, method, options)
            if cmi <= lamb:
                adj[i, j] = adj[j, i] = 0
            if np.count_nonzero(adj[i]) == 0:
                break

    return adj
