import numpy as np
from sklearn.covariance import graph_lasso

from .mutual_information import conditional_mutual_information


def skeptic(X, alpha=0.0):
    """Estimate structure of an MRF with nonparanormal
       SKEPTIC using Spearman’s rho [1]_.
    Parameters
    ----------
    X : array, shape (n_samples, d)
        Observations of variables.
    alpha: float
        Regularization parameter for the graphical lasso.
    Returns
    ----------
    precision : array, shape (d, d)
        Estimated precision (inverse covariance) matrix.
    References
    ----------
    .. [1] Liu, Han, et al. "High-dimensional semiparametric
           Gaussian copula graphical models." The Annals of
           Statistics 40.4 (2012): 2293-2326.
    """
    n, d = X.shape
    indices = np.argsort(X, axis=0)
    rank = np.empty_like(indices)
    for r, idx in zip(rank.T, indices.T):
        r[idx] = np.arange(1, len(X) + 1) - (n + 1) / 2
    rho = rank.T @ rank
    stds = np.sqrt(np.diag(rho))
    rho = rho / stds.reshape(1, -1) / stds.reshape(-1, 1)
    cov = 2 * np.sin(np.pi / 6 * rho)
    cov[np.eye(d, dtype=bool)] = 1
    return graph_lasso(cov, alpha)[1]


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
