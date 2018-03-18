import numpy as np
from ..mutual_information import conditional_mutual_information


def mifs(X, y, lamb=0.0, method='knn', options=None):
    """Select effective features in X on predinting y using
       mutual-information-based feature selection [1]_.
    Parameters
    ----------
    X : array_like, shape (n_samples, d)
        Features.
    y : array_like, shape (n_samples)
        Target. Can be either continuous or discrete.
    lamb: float
        Threshold for independence tests.
    method: str, default 'knn'
        Method used for MI estimation.
    options : dict, default None
        Optional parameters for MI estimation.
    Returns
    -------
    indices : list
        Indices for the selected features.
    References
    ----------
    .. [1] Brown, Gavin, et al. "Conditional likelihood maximisation:
           a unifying framework for information theoretic feature
           selection." Journal of machine learning research 13.Jan
           (2012): 27-66.
    """
    n, d = X.shape
    selected = []
    while True:
        max_cmi = -np.inf
        not_selected = set(range(d)) - set(selected)
        z = X[:, selected]
        for i in not_selected:
            x = X[:, i]
            cmi = conditional_mutual_information(x, y, z, method, options)
            if max_cmi < cmi:
                max_cmi = cmi
                max_idx = i

        if max_cmi <= lamb or len(selected) == d:
            break

        selected.append(max_idx)

    while True:
        min_cmi = np.inf
        for i in selected:
            x = X[:, i]
            z = X[:, list(set(selected) - set([i]))]
            cmi = conditional_mutual_information(x, y, z, method, options)
            if min_cmi > cmi:
                min_cmi = cmi
                min_idx = i

        if min_cmi >= lamb or len(selected) == 0:
            return selected

        selected.remove(min_idx)
