import sys
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.cluster import KMeans


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
