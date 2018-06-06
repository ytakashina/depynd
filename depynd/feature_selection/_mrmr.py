import numpy as np

from depynd.information import mutual_information


def _mrmr(X, y, lamb, k, **kwargs):
    """Select effective features in ``X`` on predicting ``y`` using minimum redundancy maximum relevance feature
    selection [peng2005feature]_.

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
    .. [peng2005feature] Peng, Hanchuan, Fuhui Long, and Chris Ding. "Feature selection based on mutual information
        criteria of max-dependency, max-relevance, and min-redundancy." IEEE Transactions on pattern analysis and
        machine intelligence 27.8 (2005): 1226-1238.
    """
    n, d = X.shape
    if k is not None:
        return _grow([], X, y, -np.inf, k, **kwargs)
    else:
        return _grow([], X, y, lamb, d, **kwargs)


def _grow(selected, X, y, lamb, k, **kwargs):
    n, d = X.shape
    while True:
        max_obj = -np.inf
        not_selected = set(range(d)) - set(selected)
        for i in not_selected:
            x = X[:, i]
            rel = mutual_information(x, y, **kwargs)
            red = [mutual_information(x, X[:, j], **kwargs) for j in selected]
            obj = rel - (np.mean(red) if red else 0)
            if max_obj < obj:
                max_obj = obj
                max_idx = i
        if max_obj <= lamb or len(selected) == k:
            return selected
        selected.append(max_idx)
