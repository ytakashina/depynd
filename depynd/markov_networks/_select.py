import numpy as np
from sklearn.utils import check_array

from depynd.markov_networks import _skeptic, _stars, _glasso, _jose, _gsmn, _iamb, _gsmple


def select(X, method='skeptic', criterion='stars', lamb=None, verbose=False, return_lambda=False, **kwargs):
    """Learn the structure of Markov random field.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Observations of a set of random variables.
    method : {'glasso', 'skeptic', 'gsmn', 'iamb'}
        Method for structure learning.
    criterion : {'stars', 'none'}
        Criteria for selecting regularization parameter.
    lamb : float or array-like or None
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

    if lamb is None:
        lamb = [1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    if method == 'glasso':
        estimator = _glasso
        _check_lamb(lamb, check_non_negative=True, method=method)
    elif method == 'skeptic':
        estimator = _skeptic
        _check_lamb(lamb, check_non_negative=True, method=method)
    elif method == 'jose':
        estimator = _jose
        _check_lamb(lamb, check_non_negative=True, method=method)
    elif method == 'gsmn':
        estimator = _gsmn
        _check_lamb(lamb, check_non_negative=False, method=method)
    elif method == 'iamb':
        estimator = _iamb
        _check_lamb(lamb, check_non_negative=False, method=method)
    elif method == 'gsmple':
        estimator = _gsmple
        _check_lamb(lamb, check_non_negative=False, method=method)
    else:
        raise ValueError('`%s` is not implemented.' % method)

    if criterion == 'none':
        if np.iterable(lamb):
            lamb_opt = next(iter(lamb))
        elif np.isscalar(lamb):
            lamb_opt = lamb
        else:
            raise ValueError('`lamb` should be None, a scalar, or an iterable of scalars.')
    elif criterion == 'stars':
        if np.iterable(lamb) and len(lamb) > 1:
            lamb = sorted(lamb, reverse=True)  # sort by descending order
            if 'beta' not in kwargs:
                kwargs['beta'] = 0.1
            if 'ratio' not in kwargs:
                n = len(X)
                kwargs['ratio'] = 10 * (n ** -0.5) if n > 144 else 0.8
            if 'rep_num' not in kwargs:
                kwargs['rep_num'] = 20
            print(lamb)
            lamb_opt = _stars(X, estimator, lamb=lamb, verbose=verbose, **kwargs)
        elif np.iterable(lamb) and len(lamb) == 1:
            lamb_opt = next(iter(lamb))
        elif np.isscalar(lamb):
            lamb_opt = lamb
        else:
            raise ValueError('`lamb` should be None, a scalar, or an iterable of scalars.')
    else:
        raise ValueError('Criteria %s is not implemented.' % criterion)

    if return_lambda:
        return estimator(X, lamb_opt, **kwargs), lamb_opt
    else:
        return estimator(X, lamb_opt, **kwargs)


def _check_lamb(lamb, check_non_negative, method):
    if np.iterable(lamb):
        if not all(np.isscalar(l) for l in lamb):
            raise ValueError('`lamb` must be None, a scalar, or an iterable of scalars.')
        if check_non_negative and any(l < 0 for l in lamb):
            raise ValueError('Every element in `lamb` must be positive when using %s.' % method)
    else:
        if not np.isscalar(lamb):
            raise ValueError('`lamb` must be None, a scalar, or an iterable of scalars.')
        if check_non_negative and lamb < 0:
            raise ValueError('`lamb` must be positive when using %s.' % method)
