import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.cluster import KMeans


def _mi_dr(X, Y, sigma, n_bases, maxiter):
    """Estimate mutual information between X and Y using density ratio estimation.

    Parameters
    ----------
    X : array-like, shape (n_samples, d_x) or (n_samples)
        Observations of a variable.
    Y : array-like, shape (n_samples, d_y) or (n_samples)
        Observations of the other variable.
    sigma : float
        Kernel width for density ratio estimator.
    n_bases : int
        Number of bases used in density ratio estimation.
    maxiter : int
        Maximum number of iteration in density ratio estimation.

    Returns
    -------
    mi : float
        Estimated mutual information between ``X`` and ``Y``.
    """
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

    x0 = np.random.uniform(0, 1, b)
    bounds = [(0, None)] * b
    constraints = [{'type': 'eq', 'fun': lambda alpha: alpha.dot(h) - 1}]
    result = minimize(fun=fun, jac=jac, x0=x0, bounds=bounds, constraints=constraints, options={'maxiter': maxiter})

    if not result.success:
        return np.nan

    mi = np.mean(np.log(result.x.dot(phi)))
    return mi


def _normal(X, mean, sigma):
    cov = sigma ** 2 * np.eye(len(mean))
    return multivariate_normal.pdf(X, mean=mean, cov=cov)
