import numpy as np
from sklearn.preprocessing import scale
from sklearn.covariance import graph_lasso

from depynd.markov_networks import skeptic, stars, gsmn, iamb, gsmple


def _graphical_lasso(X, lamb):
    cov = np.cov(scale(X), rowvar=False)
    pre = graph_lasso(cov, alpha=lamb)[1]
    adj = ~np.isclose(pre, 0)
    adj[np.eye(len(adj), dtype=bool)] = 0
    return adj


def select(X, method='glasso', criteria='stars', lambdas=None, verbose=False, return_lambda=False, **kwargs):
    if method == 'glasso':
        estimator = _graphical_lasso
    elif method == 'skeptic':
        estimator = skeptic
    elif method == 'gsmn':
        estimator = gsmn
    elif method == 'iamb':
        estimator = iamb
    elif method == 'gsmple':
        estimator = gsmple
    else:
        raise NotImplementedError('Method %s is not implemented.' % method)

    n, d = X.shape
    if criteria == 'stars':
        beta = kwargs.get('beta', 0.1)
        ratio = kwargs.get('ratio', 10 * (n ** -0.5) if n > 144 else 0.8)
        rep_num = kwargs.get('rep_num', 20)
        lamb = stars(X, estimator, beta=beta, ratio=ratio, rep_num=rep_num, lambdas=lambdas, verbose=verbose)
    else:
        raise NotImplementedError('Criteria %s is not implemented.' % criteria)

    if return_lambda:
        return estimator(X, lamb), lamb
    else:
        return estimator(X, lamb)
