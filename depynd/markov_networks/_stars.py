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


def stars(X, estimator, lambdas, beta, ratio, rep_num, verbose=False):
    """Return the best regularization parameter using Stability Approach to Regularization Selection
    [liu2010stability]_.

    Parameters
    ----------
    X
    estimator
    lambdas
    beta
    ratio
    rep_num
    verbose

    Returns
    -------

    References
    -------
    .. [liu2010stability] Liu, Han, Kathryn Roeder, and Larry Wasserman. "Stability approach to regularization selection
        (stars) for high dimensional graphical models." Advances in neural information processing systems. 2010.
    """
    for i, lamb in enumerate(lambdas):
        instability = _instability(X, estimator, lamb, ratio=ratio, rep_num=rep_num)
        if instability > beta:
            return lambdas[i - 1]
        if verbose:
            print('[stars] lambda: %f, instability: %f' % (lamb, instability))
    return 0
