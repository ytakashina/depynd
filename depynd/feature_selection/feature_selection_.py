from .mrmr_ import _mrmr
from .mifs_ import _mifs


def select(X, y, lamb=0.0, method='mifs'):
    """Select effective features in X on predinting y using
    Parameters
    ----------
    X : array_like, shape (n_samples, d)
        Features.
    y : array_like, shape (n_samples)
        Target. Can be either continuous or discrete.
    lamb: float, default 0.0
        Threshold for independence tests.
    method: str, default 'mifs'
        Method used for feature selection. Either 'mifs' or
        'mrmr' can be chosen.
    Returns
    -------
    indices : list
        Indices for the selected features.
    """
    if method == 'mifs':
        return _mifs(X, y, lamb=0.0, method='knn', options=None)
    elif method == 'mrmr':
        return _mrmr(X, y, lamb=0.0, method='knn', options=None)
    else:
        raise NotImplementedError()
