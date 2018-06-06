import numpy as np
from sklearn.utils.validation import check_X_y

from depynd.feature_selection import _mrmr, _mifs


def select(X, y, lamb=0.0, k=None, method='mifs', **kwargs):
    """Select effective features in ``X`` on predicting ``y``.

    Parameters
    ----------
    X : array-like, shape (n_samples, d)
        Observations of feature variables.
    y : array-like, shape (n_samples)
        Observations of the target variable.
    lamb: float or None
        Threshold for independence tests. Ignored if `k` is specified.
    k : int or None
        Number of selected features.
    method: {'mifs', 'mrmr'}, default 'mifs'
        Feature selection method.

    Returns
    -------
    indices : list
        Indices of the selected features.
    """
    y = np.ravel(y)
    X, y = check_X_y(X, y, ensure_min_samples=2, ensure_min_features=2)
    if lamb is None and k is None:
        raise ValueError('At least either `lamb` or `k` should be specified.')
    if k is not None:
        assert isinstance(k, (int, np.integer)) and k > 0, '`k` must be a positive integer.'
        assert k <= X.shape[1], '`k` cannot be larger than number of features.'
    else:
        assert isinstance(lamb, (float, np.float)), '`lamb` must be a real value.'
    if method == 'mifs':
        return _mifs(X, y, lamb=lamb, k=k, **kwargs)
    elif method == 'mrmr':
        return _mrmr(X, y, lamb=lamb, k=k, **kwargs)
    else:
        raise ValueError('`%s` is not implemented.' % method)
