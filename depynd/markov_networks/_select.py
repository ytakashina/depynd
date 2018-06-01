import numpy as np
from sklearn.preprocessing import scale
from sklearn.covariance import graph_lasso

from depynd.markov_networks import skeptic, stars


def _graphical_lasso(X, alpha):
    cov = np.cov(scale(X), rowvar=False)
    return graph_lasso(cov, alpha=alpha)[1]


def select(X, method='glasso', beta=0.1, ratio=None, rep_num=20, lambdas=None, verbose=False, return_lambda=False):
    if method == 'glasso':
        estimator = _graphical_lasso
    elif method == 'skeptic':
        estimator = skeptic
    else:
        raise NotImplementedError('Method %s is not implemented. Use glasso or skeptic.' % method)

    lamb = stars(X, estimator, beta=beta, ratio=ratio, rep_num=rep_num, lambdas=lambdas, verbose=verbose)
    if return_lambda:
        return estimator(X, lamb), lamb
    else:
        return estimator(X, lamb)
