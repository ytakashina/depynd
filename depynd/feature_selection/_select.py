from sklearn.utils.validation import check_X_y
from depynd.feature_selection import mrmr, mifs


def select(X, y, lamb=0.0, method='mifs', **kwargs):
    """Select effective features in X on predinting y using

    Parameters
    ----------
    X : array-like, shape (n_samples, d)
        Features.
    y : array-like, shape (n_samples)
        Target. Can be either continuous or discrete.
    lamb: float, default 0.0
        Threshold for independence tests.
    method: str, default 'mifs'
        Method used for feature selection. Either 'mifs' or 'mrmr' can be chosen.

    Returns
    -------
    indices : list
        Indices for the selected features.
    """
    X, y = check_X_y(X, y)
    if method == 'mifs':
        return mifs(X, y, lamb=lamb, **kwargs)
    elif method == 'mrmr':
        return mrmr(X, y, lamb=lamb, **kwargs)
    else:
        raise NotImplementedError('Method %s is not implemented. Use mifs or mrmr.' % method)
