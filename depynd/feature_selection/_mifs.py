import numpy as np

from depynd.information import conditional_mutual_information


def _mifs(X, y, lamb, k, **kwargs):
    """Select effective features in ``X`` on predicting ``y`` using mutual-information-based feature selection
    [brown2012conditional]_.

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
    kwargs : dict
        Optional parameters for MI estimation.

    Returns
    -------
    indices : list
        Indices of the selected features.

    References
    ----------
    .. [brown2012conditional] Brown, Gavin, et al. "Conditional likelihood maximisation: a unifying framework for
        information theoretic feature selection." Journal of machine learning research 13.Jan (2012): 27-66.
    """
    n, d = X.shape
    selected = []
    if k is not None:
        selected = _grow(selected, X, y, -np.inf, k, **kwargs)
    else:
        selected = _grow(selected, X, y, lamb, d, **kwargs)
        selected = _shrink(selected, X, y, lamb, 0, **kwargs)
    return selected


def _grow(selected, X, y, lamb, k, **kwargs):
    n, d = X.shape
    while True:
        max_cmi = -np.inf
        not_selected = set(range(d)) - set(selected)
        z = X[:, selected]
        for i in not_selected:
            x = X[:, i]
            cmi = conditional_mutual_information(x, y, z, **kwargs)
            if max_cmi < cmi:
                max_cmi = cmi
                max_idx = i
        if max_cmi <= lamb or len(selected) == k:
            return selected
        selected.append(max_idx)


def _shrink(selected, X, y, lamb, k, **kwargs):
    while True:
        min_cmi = np.inf
        for i in selected:
            x = X[:, i]
            z = X[:, list(set(selected) - {i})]
            cmi = conditional_mutual_information(x, y, z, **kwargs)
            if min_cmi > cmi:
                min_cmi = cmi
                min_idx = i
        if min_cmi >= lamb or len(selected) == k:
            return selected
        selected.remove(min_idx)
