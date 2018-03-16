import numpy as np
from ..mutual_information import conditional_mutual_information


def gsmn(X, lamb=0.0, method=None, options=None):
    """Search structure of an MRF using glow-shrink
       Markov networks algorithm [1]_.
    Parameters
    ----------
    X : array, shape (n_samples, d)
        Observations of variables.
    lamb: float
        Threshold for independence tests.
    method: str, default 'knn'
        Method for MI estimation.
    options : dict, default None
        Optional parameters for MI estimation.
    Returns
    ----------
    adj : array, shape (d, d)
        Estimated structure of an MRF.
    References
    ----------
    .. [1] Bromberg, Facundo, Dimitris Margaritis, and
           Vasant Honavar. "Efficient Markov network
           structure discovery using independence tests."
           Journal of Artificial Intelligence Research 35
           (2009): 449-484.
    """
    n, d = X.shape
    adj = np.zeros([d, d], dtype=bool)
    for i in range(d):
        x = X[:, i]
        non_adj = ~adj[i] & (np.arange(d) != i)
        for j in non_adj.nonzero()[0]:
            y = X[:, j]
            z = X[:, adj[i]]
            cmi = conditional_mutual_information(x, y, z, method=method, options=options)
            if cmi > lamb:
                adj[i, j] = 1
                adj[j, i] = 1

        for j in adj[i].nonzero()[0]:
            other_adj = adj[i] & (np.arange(d) != j)
            y = X[:, j]
            z = X[:, other_adj]
            cmi = conditional_mutual_information(x, y, z, method=method, options=options)
            if cmi <= lamb:
                adj[i, j] = 0
                adj[j, i] = 0

    return adj
