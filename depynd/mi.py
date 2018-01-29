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
    distances = np.max([distances_x, distances_y], axis=0)
    epsilons = np.sort(distances, axis=1)[:, k]
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


class MIEstimator(object):
    def __init__(self, method='knn', options=None):
        self._method = method
        self._options = {} if options is None else options

    def fit(self, X, Y):
        if self._method == 'dr':
            sigma = self._options.get('sigma', 1)
            n_bases = self._options.get('n_bases', 200)
            maxiter = self._options.get('maxiter', 1000)
            self.mi = _mi_dr(X, Y, sigma=sigma, n_bases=n_bases, maxiter=maxiter)
        else:
            k = self._options.get('k', 3)
            self.mi = _mi_knn(X, Y, k)

        return self


class CMIEstimator(object):
    def __init__(self, method='knn', options=None):
        self._method = method
        self._options = options

    def fit(self, X, Y, Z):
        XZ = np.hstack([X, Z])
        if Z.size == 0:
            self.cmi = MIEstimator(method=self._method, options=self._options).fit(X, Y).mi
        else:
            mi_xz_y = MIEstimator(method=self._method, options=self._options).fit(XZ, Y).mi
            mi_y_z = MIEstimator(method=self._method, options=self._options).fit(Y, Z).mi
            self.cmi = mi_xz_y - mi_y_z

        return self
