from itertools import combinations

import numpy as np
from scipy.stats import multinomial
from sklearn.linear_model import Lasso


def non_diag(A):
    d = len(A)
    tmp = np.copy(A)
    tmp[np.eye(d, dtype=bool)] = 0
    return tmp


def calc_p(theta, X):
    Z = np.diag(theta)[None, :] + X @ non_diag(theta)
    P = 1 / (1 + np.exp(-Z))
    return P


def calc_w(P):
    W = P * (1 - P)
    return W


def calc_y(theta, X, P, W):
    Y = np.diag(theta)[None, :] + X @ non_diag(theta) - (P - X) / W
    return Y


def calc_ys(W, Y):
    Ys = np.sqrt(W) * Y
    return Ys


def calc_yss(Ys):
    ymean = np.mean(Ys, axis=0)
    Yss = Ys - ymean[None, :]
    return Yss


def calc_xs(W, X):
    Xs = np.sqrt(W) * X
    return Xs


def calc_xss(Xs):
    xmean = np.mean(Xs, axis=0)
    Xss = Xs - xmean[None, :]
    return Xss


def calc_theta_diag(theta, W, Xs, Ys):
    ymean = np.mean(Ys, axis=0)
    xmean = np.mean(Xs, axis=0)
    wmean = np.mean(np.sqrt(W), axis=0)
    diag = (ymean - xmean @ non_diag(theta)) / wmean
    return diag


def jose(X, lamb=0.01, tol=1e-6, max_iter=100):
    if set(np.ravel(X)) != {0, 1}:
        raise ValueError('Each element of X must be in {0, 1}.')

    n, d = X.shape
    comb = list(combinations(np.arange(d), 2))
    # theta = np.zeros([d, d])
    theta = np.eye(d)
    for _ in range(max_iter):
        theta_prev = np.copy(theta)
        P = calc_p(theta, X)
        W = calc_w(P)
        Y = calc_y(theta, X, P, W)
        Ys = calc_ys(W, Y)
        Yss = calc_yss(Ys)
        Xs = calc_xs(W, X)
        Xss = calc_xss(Xs)
        x = np.zeros([n * d, d * (d - 1) // 2])
        for k, (i, j) in enumerate(comb):
            x[np.arange(n * i, n * (i + 1)), k] = Xss[:, j]
            x[np.arange(n * j, n * (j + 1)), k] = Xss[:, i]
        y = np.concatenate(Yss.T)
        coef = Lasso(lamb).fit(x, y).coef_
        for k, (i, j) in enumerate(comb):
            theta[i, j] = theta[j, i] = coef[k]
        theta[np.eye(d, dtype=bool)] = calc_theta_diag(theta, W, Xs, Ys)
        diff = np.linalg.norm(theta - theta_prev)
        if diff < tol:
            break

    adj = np.abs(theta) > 1e-6
    adj[np.eye(d, dtype=bool)] = 0
    return adj


if __name__ == '__main__':
    adj0 = np.zeros([9, 9])
    for i in range(3, 9):
        adj0[i, i - 3] = adj0[i - 3, i] = 1
    adj0[0, 1] = adj0[1, 0] = 1
    adj0[1, 2] = adj0[2, 1] = 1
    adj0[3, 4] = adj0[4, 3] = 1
    adj0[4, 5] = adj0[5, 4] = 1
    adj0[6, 7] = adj0[7, 6] = 1
    adj0[7, 8] = adj0[8, 7] = 1


    def sample(adj, x, i, size=1):
        p = 1 / (1 + np.exp(-adj[i] @ x))
        sample = multinomial(1, [p, 1 - p]).rvs(size)
        return [0, 1][sample.argmax()]


    def gibbs_sampling(adj, L=100):
        X = []
        x = np.ones(9, dtype=int)
        for t in range(L):
            for i in range(9):
                x[i] = sample(adj0, x, i)
            X.append(np.copy(x))
        return np.vstack(X)


    X = gibbs_sampling(adj0)
    print(jose(X))
