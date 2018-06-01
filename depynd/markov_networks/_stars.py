import numpy as np
from sklearn.preprocessing import scale


def _instability(X, estimator, alpha, ratio, rep_num):
    n, p = X.shape
    b = int(ratio * n)
    indices = [np.random.choice(np.arange(n), size=b) for _ in range(rep_num)]
    samples = [scale(sample) for sample in X[indices, :]]
    adjs = [estimator(sample, alpha) for sample in samples]
    theta = np.sum(adjs, axis=0) / rep_num
    xi = 2 * theta * (1 - theta)
    d = np.sum(xi) / p / (p - 1)
    return d


def stars(X, estimator, beta, ratio, rep_num, lambdas=None, verbose=False):
    n, p = X.shape
    if lambdas is None:
        lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                   2, 3, 4, 5, 10, 100]
    lambdas = sorted(lambdas, reverse=True)  # sort by descending order

    for i, lamb in enumerate(lambdas):
        instability = _instability(X, estimator, lamb, ratio=ratio, rep_num=rep_num)
        if instability > beta:
            return lambdas[i - 1]
        if verbose:
            print('[stars] lambda: %f, instability: %f' % (lamb, instability))
    return 0
