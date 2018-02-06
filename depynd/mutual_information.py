import sys
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import digamma
from scipy.optimize import minimize
from sklearn.cluster import KMeans


def _mi_knn(X, Y, k):
    n, d_x = X.shape
    _, d_y = Y.shape
    distances_x = np.linalg.norm(X - X.reshape([n, -1, d_x]), axis=2)
    distances_y = np.linalg.norm(Y - Y.reshape([n, -1, d_y]), axis=2)
    distances = np.maximum(distances_x, distances_y)
    epsilons = np.partition(distances, k, axis=1)[:, k]
    idx_discrete = np.isclose(epsilons, 0)
    ks = np.repeat(k, n)
    ks[idx_discrete] = np.sum(np.isclose(distances[idx_discrete], 0), axis=1) - 1
    n_x = np.sum(distances_x <= epsilons, axis=0) - 1
    n_y = np.sum(distances_y <= epsilons, axis=0) - 1
    mi = np.log(n) + np.mean(digamma(ks) - np.log(n_x * n_y))
    return mi


def _normal(X, mean, sigma):
    cov = sigma ** 2 * np.eye(len(mean))
    return multivariate_normal.pdf(X, mean=mean, cov=cov)


def _mi_dr(X, Y, sigma, n_bases, maxiter):
    n, d_x = X.shape
    _, d_y = Y.shape
    b = min(n_bases, n)

    XY = np.hstack([X, Y])
    UV = KMeans(b).fit(XY).cluster_centers_
    U, V = np.split(UV, [d_x], axis=1)
    phi_x = np.array([_normal(X, u, sigma) for u in U])
    phi_y = np.array([_normal(Y, v, sigma) for v in V])
    phi = phi_x * phi_y
    h_x = np.sum(phi_x, axis=1)
    h_y = np.sum(phi_y, axis=1)
    h_xy = np.sum(phi, axis=1)
    h = (h_x * h_y - h_xy) / (n ** 2 - n)

    def fun(alpha):
        return -np.sum(np.log(alpha.dot(phi)))

    def jac(alpha):
        return -phi.dot(1 / alpha.dot(phi))

    bounds = [(0, None)] * b
    constraints = [{'type': 'eq', 'fun': lambda alpha: alpha.dot(h) - 1}]

    alpha0 = np.random.uniform(0, 1, b)
    result = minimize(fun=fun, jac=jac, x0=alpha0, bounds=bounds, constraints=constraints,
                      options={'maxiter': maxiter})

    if not result.success:
        print('Optimization failed: %s' % result.message, file=sys.stderr)

    alpha = result.x
    mi = np.mean(np.log(alpha.dot(phi)))
    return mi


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
