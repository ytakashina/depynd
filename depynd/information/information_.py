import numpy as np
from .dr_ import _mi_dr
from .knn_ import _mi_knn


def mutual_information(X, Y, method='knn', options=None):
    """Estimate mutual information for discrete-continuous mixutres.
    Parameters
    ----------
    X : array_like, shape (n_samples, d_x)
        Variable.
    Y : array_like, shape (n_samples, d_y)
        The other variable.
    method: str, default 'knn'
        Method for MI estimation.
    options : dict, default None
        Optional parameters for MI estimation.
    Returns
    -------
    mi : float
        Estimated mutual information between each X and Y.
    """
    options = {} if options is None else options
    if X.size == 0 or Y.size == 0:
        return 0
    if np.ndim(X) == 1:
        X = np.reshape(X, [-1, 1])
    if np.ndim(Y) == 1:
        Y = np.reshape(Y, [-1, 1])
    if method == 'dr':
        sigma = options.get('sigma', 1)
        n_bases = options.get('n_bases', 200)
        maxiter = options.get('maxiter', 1000)
        return _mi_dr(X, Y, sigma=sigma, n_bases=n_bases, maxiter=maxiter)
    else:
        k = options.get('k', 3)
        return _mi_knn(X, Y, k)


def conditional_mutual_information(X, Y, Z, method='knn', options=None):
    """Estimate conditional mutual information for discrete-continuous mixutres.
    Parameters
    ----------
    X : array_like, shape (n_samples, d_x)
        Conditioned variable.
    Y : array_like, shape (n_samples, d_y)
        The other conditioned variable.
    Z : array_like, shape (n_samples, d_z)
        Conditioning variable.
    method: str, default 'knn'
        Method for MI estimation.
    options : dict, default None
        Optional parameters for MI estimation.
    Returns
    -------
    cmi : float
        Estimated conditional mutual information between each X and Y, given Z.
    """
    if Z.size == 0:
        return mutual_information(X, Y, method, options)
    if np.ndim(X) == 1:
        X = np.reshape(X, [-1, 1])
    if np.ndim(Y) == 1:
        Y = np.reshape(Y, [-1, 1])
    if np.ndim(Z) == 1:
        Z = np.reshape(Z, [-1, 1])
    XZ = np.hstack([X, Z])
    mi_xz_y = mutual_information(XZ, Y, method, options)
    mi_y_z = mutual_information(Y, Z, method, options)
    return mi_xz_y - mi_y_z


def mimat(X, method='knn', options=None):
    """Dimension-wise mutual information.
    Parameters
    ----------
    X : array_like, shape (n_samples, d)
        Variable.
    method: str, default 'knn'
        Method for MI estimation.
    options : dict, default None
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
        mis[i, j] = mutual_information(x, y, method, options)

    mis[mis < 0] = 0
    mis = mis + mis.T
    mis[np.eye(d, dtype=bool)] = np.nan
    return mis


def cmimat(X, method='knn', options=None):
    """Dimension-wise conditional mutual information.
    Parameters
    ----------
    X : array_like, shape (n_samples, d)
        Variable.
    method: str, default 'knn'
        Method for MI estimation.
    options : dict, default None
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
        cmis[i, j] = conditional_mutual_information(x, y, z, method, options)

    cmis[cmis < 0] = 0
    cmis = cmis + cmis.T
    cmis[np.eye(d, dtype=bool)] = np.nan
    return cmis
