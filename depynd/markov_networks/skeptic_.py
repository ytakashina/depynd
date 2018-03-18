import numpy as np
from sklearn.covariance import graph_lasso
# In the future, graph_lasso will be renamed to graphical_lasso.


def skeptic(X, alpha):
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
