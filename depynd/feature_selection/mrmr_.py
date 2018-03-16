import numpy as np
from ..mutual_information import mutual_information


def _mrmr(X, y, lamb=0.0, method='knn', options=None):
    """Select effective features in X on predinting y using
       minimum redundancy maximum relevance feature selection [1]_.
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
    .. [1] Peng, Hanchuan, Fuhui Long, and Chris Ding. "Feature
           selection based on mutual information criteria of
           max-dependency, max-relevance, and min-redundancy."
           IEEE Transactions on pattern analysis and machine
           intelligence 27.8 (2005): 1226-1238.
    """
    n, d = X.shape
    selected = []
    while True:
        max_obj = -np.inf
        not_selected = set(range(d)) - set(selected)
        for i in not_selected:
            rel = mutual_information(X[:, i], y)
            red = [mutual_information(X[:, i], X[:, j]) for j in selected]
            obj = rel - (np.mean(red) if red else 0)
            if max_obj < obj:
                max_obj = obj
                max_idx = i

        if max_obj <= 0:
            return selected

        selected.append(max_idx)
