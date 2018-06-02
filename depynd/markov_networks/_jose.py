from itertools import combinations

import numpy as np
from sklearn.linear_model import Lasso


def non_diag(A):
    tmp = np.copy(A)
    tmp[np.eye(len(A), dtype=bool)] = 0
    return tmp


def jose(X, lamb=0.01, tol=1e-6, max_iter=100):
    if set(np.ravel(X)) != {0, 1}:
        raise ValueError('Each element of X must be in {0, 1}.')

    n, d = X.shape
    comb = list(combinations(np.arange(d), 2))
    theta = np.eye(d)
    for _ in range(max_iter):
        theta_prev = np.copy(theta)
        z = np.diag(theta)[None, :] + X @ non_diag(theta)
        p = 1 / (1 + np.exp(-z))
        w = p * (1 - p)
        y = np.diag(theta)[None, :] + X @ non_diag(theta) - (p - X) / w
        y_star = np.sqrt(w) * y
        y_mean = np.mean(y_star, axis=0)
        y_star_star = y_star - y_mean[None, :]
        x_star = np.sqrt(w) * X
        x_mean = np.mean(x_star, axis=0)
        x_star_star = x_star - x_mean[None, :]
        x = np.zeros([n * d, d * (d - 1) // 2])
        for k, (i, j) in enumerate(comb):
            x[np.arange(n * i, n * (i + 1)), k] = x_star_star[:, j]
            x[np.arange(n * j, n * (j + 1)), k] = x_star_star[:, i]
        y = np.concatenate(y_star_star.T)
        coef = Lasso(lamb).fit(x, y).coef_
        for k, (i, j) in enumerate(comb):
            theta[i, j] = theta[j, i] = coef[k]
        w_mean = np.mean(np.sqrt(w), axis=0)
        theta[np.eye(d, dtype=bool)] = (y_mean - x_mean @ non_diag(theta)) / w_mean
        diff = np.linalg.norm(theta - theta_prev)
        if diff < tol:
            break

    adj = np.abs(theta) > 1e-6
    adj[np.eye(d, dtype=bool)] = 0
    return adj
