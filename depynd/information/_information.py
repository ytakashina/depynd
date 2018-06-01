import numpy as np
from sklearn.utils.validation import check_array

from depynd.information import mi_dr, mi_knn


def mutual_information(X, Y, **kwargs):
    """Estimate mutual information for discrete-continuous mixutres.

    Parameters
    ----------
    X : array-like, shape (n_samples, d_x) or (n_samples)
        The observations of a variable.
    Y : array-like, shape (n_samples, d_y) or (n_samples)
        The observations of the other variable.
    kwargs : dict
        Optional parameters for MI estimation.

    Returns
    -------
    mi : float
        The estimated mutual information between X and Y.
    """
    if np.size(X) == 0 or np.size(Y) == 0:
        return 0
    X = np.atleast_2d(X.T).T
    Y = np.atleast_2d(Y.T).T
    X = check_array(X, ensure_min_samples=2)
    Y = check_array(Y, ensure_min_samples=2)
    assert len(X) == len(Y), 'X and Y must have the same length.'

    mi_estimator = kwargs.get('mi_estimator', 'knn')
    if mi_estimator == 'dr':
        sigma = kwargs.get('sigma', 1)
        n_bases = kwargs.get('n_bases', 200)
        maxiter = kwargs.get('maxiter', 1000)
        return mi_dr(X, Y, sigma=sigma, n_bases=n_bases, maxiter=maxiter)
    elif mi_estimator == 'knn':
        k = kwargs.get('k', 3)
        assert isinstance(k, (int, np.integer)) and k > 0, 'k must be a positive integer.'
        assert k < len(X), '`k` must be smaller than `n_sample`.'
        return mi_knn(X, Y, k)
    else:
        raise NotImplementedError


def conditional_mutual_information(X, Y, Z, **kwargs):
    """Estimate conditional mutual information for discrete-continuous mixutres.

    Parameters
    ----------
    X : array-like, shape (n_samples, d_x)
        Conditioned variable.
    Y : array-like, shape (n_samples, d_y)
        The other conditioned variable.
    Z : array-like, shape (n_samples, d_z)
        Conditioning variable.
    kwargs : dict, default None
        Optional parameters for MI estimation.

    Returns
    -------
    cmi : float
        Estimated conditional mutual information between each X and Y, given Z.
    """
    if np.size(Z) == 0:
        return mutual_information(X, Y, **kwargs)
    assert len(X) == len(Y) == len(Z), 'X, Y and Z must have the same length.'
    X = np.atleast_2d(X.T).T
    Z = np.atleast_2d(Z.T).T
    XZ = np.hstack([X, Z])
    mi_xz_y = mutual_information(XZ, Y, **kwargs)
    mi_y_z = mutual_information(Y, Z, **kwargs)
    return mi_xz_y - mi_y_z


def mimat(X, **kwargs):
    """Dimension-wise mutual information.

    Parameters
    ----------
    X : array-like, shape (n_samples, d)
        Variable.
    kwargs : dict, default None
        Optional parameters for MI estimation.

    Returns
    -------
    mis : array, shape (d, d)
        Estimated pairwise MIs between dimensions.
    """
    n, d = X.shape
    mis = np.eye(d)
    for i, j in [(i, j) for i in range(d) for j in range(i + 1, d)]:
        x = X[:, [i]]
        y = X[:, [j]]
        mis[i, j] = mis[j, i] = mutual_information(x, y, **kwargs)
    mis[mis < 0] = 0
    mis[np.eye(d, dtype=bool)] = np.nan
    return mis


def cmimat(X, **kwargs):
    """Dimension-wise conditional mutual information.

    Parameters
    ----------
    X : array-like, shape (n_samples, d)
        Variable.
    kwargs : dict, default None
        Optional parameters for MI estimation.

    Returns
    -------
    cmis : array, shape (d, d)
        Estimated pairwise CMIs between dimensions, given the other dimensions.
    """
    n, d = X.shape
    cmis = np.eye(d)
    for i, j in [(i, j) for i in range(d) for j in range(i + 1, d)]:
        x = X[:, [i]]
        y = X[:, [j]]
        idx_rest = (np.arange(d) != i) & (np.arange(d) != j)
        z = X[:, idx_rest]
        cmis[i, j] = cmis[j, i] = conditional_mutual_information(x, y, z, **kwargs)
    cmis[cmis < 0] = 0
    cmis[np.eye(d, dtype=bool)] = np.nan
    return cmis
