import numpy as np
from sklearn.preprocessing import scale


def _instability(X, estimator, lamb, ratio, rep_num, **kwargs):
    n, p = X.shape
    b = int(ratio * n)
    indices = [np.random.choice(np.arange(n), size=b) for _ in range(rep_num)]
    samples = [scale(sample) for sample in X[indices, :]]
    adjs = [estimator(sample, lamb, **kwargs) for sample in samples]
    theta = np.sum(adjs, axis=0) / rep_num
    xi = 2 * theta * (1 - theta)
    d = np.sum(xi) / p / (p - 1)
    return d


def _stars(X, estimator, lamb, beta, ratio, rep_num, verbose=False, **kwargs):
    """Obtain the best regularization parameter using Stability Approach to Regularization Selection
    [liu2010stability]_.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Observations of variables.
    estimator
    lamb
    beta
    ratio
    rep_num
    verbose

    Returns
    -------
    lambda_opt : float
        Optimal regularization parameter.

    References
    -------
    .. [liu2010stability] Liu, Han, Kathryn Roeder, and Larry Wasserman. "Stability approach to regularization selection
        (stars) for high dimensional graphical models." Advances in neural information processing systems. 2010.
    """
    for i, l in enumerate(lamb):
        instability = _instability(X, estimator, l, ratio=ratio, rep_num=rep_num, **kwargs)
        if instability > beta:
            return lamb[i - 1]
        if verbose:
            print('[stars] lambda: %f, instability: %f' % (l, instability))
    return 0
