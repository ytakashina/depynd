from sklearn.utils import check_array

from depynd.markov_networks import _skeptic, _stars, _glasso, _gsmn, _iamb, _gsmple


def select(X, method='glasso', criterion='stars', lambdas=None, verbose=False, return_lambda=False, **kwargs):
    """Learn the structure of Markov random field.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Observations of a set of random variables.
    method : {'glasso', 'skeptic', 'gsmn', 'iamb'}
        Method for structure learning.
    criterion : {'stars'}
        Criteria for selecting regularization parameter.
    lambdas : array-like
        Candidates of regularization parameter.
    verbose : bool
        If True, the objective function is plotted for each regularization parameter.
    return_lambda : bool, default False
        If True, the selected regularization parameter will be returned.
    kwargs : dict
        Optional parameters for MI estimation.

    Returns
    -------
    adj : array, shape (n_features, n_features)
        Estimated adjacency matrix of an MRF.
    """
    X = check_array(X, ensure_min_samples=2, ensure_min_features=2)
    if method == 'glasso':
        estimator = _glasso
    elif method == 'skeptic':
        estimator = _skeptic
    elif method == 'gsmn':
        estimator = _gsmn
    elif method == 'iamb':
        estimator = _iamb
    elif method == 'gsmple':
        estimator = _gsmple
    else:
        raise ValueError('`%s` is not implemented.' % method)

    if lambdas is None:
        lambdas = [1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    lambdas = sorted(lambdas, reverse=True)  # sort by descending order

    n, d = X.shape
    if criterion == 'stars':
        beta = kwargs.get('beta', 0.1)
        ratio = kwargs.get('ratio', 10 * (n ** -0.5) if n > 144 else 0.8)
        rep_num = kwargs.get('rep_num', 20)
        lamb = _stars(X, estimator, beta=beta, ratio=ratio, rep_num=rep_num, lambdas=lambdas, verbose=verbose)
    else:
        raise ValueError('Criteria %s is not implemented.' % criterion)

    if return_lambda:
        return estimator(X, lamb), lamb
    else:
        return estimator(X, lamb)
