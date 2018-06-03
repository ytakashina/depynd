import numpy as np
from sklearn.preprocessing import scale
from sklearn.covariance import graph_lasso


def glasso(X, lamb, return_precision=False):
    """Learn the structure of Markov random field with the graphical lasso.

    This function internally uses the implementation of graphical lasso in scikit-learn.

    Parameters
    ----------
    X : array, shape (n_samples, d)
        Observations of variables.
    lamb : float
        Regularization parameter.
    return_precision : bool, default False
        If True, the estimated precision matrix will be returned instead of adjacency matrix.

    Returns
    ----------
    precision : array, shape (d, d)
        Estimated precision (inverse covariance) matrix.
    """
    cov = np.cov(scale(X), rowvar=False)
    pre = graph_lasso(cov, alpha=lamb)[1]
    if return_precision:
        return pre
    else:
        adj = ~np.isclose(pre, 0)
        adj[np.eye(len(adj), dtype=bool)] = 0
        return adj
