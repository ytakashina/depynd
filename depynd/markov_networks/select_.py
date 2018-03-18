import numpy as np
from sklearn.preprocessing import scale
from sklearn.covariance import graph_lasso

from . import skeptic


def _graphical_lasso(X, alpha):
    cov = np.cov(scale(X), rowvar=False)
    return graph_lasso(cov, alpha=alpha)[1]


def _instability(X, estimator, alpha, ratio, rep_num):
    n, p = X.shape
    b = int(ratio * n)
    indices = [np.random.choice(np.arange(n), size=b) for _ in range(rep_num)]
    samples = [scale(sample) for sample in X[indices, :]]
    pres = [estimator(sample, alpha) for sample in samples]
    adjs = np.array([~np.isclose(pre, 0) for pre in pres])
    theta = np.sum(adjs, axis=0) / rep_num
    xi = 2 * theta * (1 - theta)
    d = np.sum(xi) / p / (p - 1)
    return d


def stars(X, estimator, beta=0.1, ratio=None, rep_num=20, lambdas=None, verbose=False):
    n, p = X.shape
    if ratio is None:
        ratio = 10 * (n ** -0.5) if n > 144 else 0.8
    if lambdas is None:
        lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.03, 0.05, 0.08,
                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1e+6]
    lambdas = sorted(lambdas, reverse=True)  # arrange by descending order

    for i, lamb in enumerate(lambdas):
        d = _instability(X, estimator, lamb, ratio=ratio, rep_num=rep_num)
        if d > beta:
            return lambdas[i - 1]
        if verbose:
            print('[stars] lambda: %f, instability: %f' % (lamb, d))
    return 0


def select(X, method='glasso', beta=0.1, ratio=None, rep_num=20, lambdas=None, verbose=False, return_lambda=False):
    if method == 'glasso':
        estimator = _graphical_lasso
    if method == 'skeptic':
        estimator = skeptic
    lamb = stars(X, estimator, beta=beta, ratio=ratio, rep_num=rep_num, lambdas=lambdas, verbose=verbose)
    if return_lambda:
        return estimator(X, lamb), lamb
    else:
        return estimator(X, lamb)
