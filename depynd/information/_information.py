import numpy as np
from sklearn.utils.validation import check_array

from depynd.information import _mi_dr, _mi_knn, _mi_plugin


def mutual_information(X, Y, mi_estimator='auto', is_discrete='auto', force_non_negative=False, **kwargs):
    """Estimate mutual information between ``X`` and ``Y``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features_x) or (n_samples)
        Observations of a variable.
    Y : array-like, shape (n_samples, n_features_y) or (n_samples)
        Observations of the other variable.
    mi_estimator : {'knn', 'dr', 'plugin', 'auto'}, default 'auto'
        MI estimator. If 'auto', MI estimator will be selected depending on whether all features are purely discrete or
        not. If purely discrete, 'plugin' estimator will be used. Otherwise, 'knn' estimator will be selected.
    is_discrete : {'auto', bool}, default 'auto'
        If ``bool``, then it determines whether to consider all features purely discrete or purely continuous. If
        'auto', a column which contains duplicate elements will be considered discrete.
    force_non_negative : bool, default False
        If ``True``, the result will be taken max with zero.
    kwargs : dict
        Optional parameters for MI estimation.

    Returns
    -------
    mi : float
        Estimated mutual information between ``X`` and ``Y``.
    """
    if np.size(X) == 0 or np.size(Y) == 0:
        return 0
    X = np.atleast_2d(X.T).T
    Y = np.atleast_2d(Y.T).T
    X = check_array(X, ensure_min_samples=2)
    Y = check_array(Y, ensure_min_samples=2)
    assert len(X) == len(Y), 'X and Y must have the same length.'

    n = len(X)
    if is_discrete == 'auto':
        is_discrete = all(n != len(set(col)) for col in X.T) and all(n != len(set(col)) for col in Y.T)
        is_continuous = all(n == len(set(col)) for col in X.T) and all(n == len(set(col)) for col in Y.T)
    elif isinstance(is_discrete, bool):
        is_continuous = not is_discrete and \
                        all(n == len(set(col)) for col in X.T) and all(n == len(set(col)) for col in Y.T)
    else:
        raise TypeError("`is_discrete` must be 'auto' or bool.")

    if mi_estimator == 'auto':
        if is_discrete:
            mi_estimator = 'plugin'
        # elif is_continuous:  # unstable
        #     mi_estimator = 'dr'
        else:
            mi_estimator = 'knn'

    if mi_estimator == 'dr':
        sigma = kwargs.get('sigma', 1)
        n_bases = kwargs.get('n_bases', 200)
        maxiter = kwargs.get('maxiter', 1000)
        assert sigma > 0, '`sigma` must be positive.'
        assert isinstance(n_bases, (int, np.integer)) and n_bases > 0, '`n_bases` must be a positive integer.'
        assert isinstance(maxiter, (int, np.integer)) and maxiter > 0, '`maxiter` must be a positive integer.'
        assert is_continuous, 'When using density ratio estimator, all features must be continuous.'
        mi = _mi_dr(X, Y, sigma=sigma, n_bases=n_bases, maxiter=maxiter)
    elif mi_estimator == 'knn':
        n_neighbors = kwargs.get('n_neighbors', 3)
        assert isinstance(n_neighbors, (int, np.integer)), '`n_neighbors` must be an integer.'
        assert n_neighbors > 0, '`n_neighbors` must be positive.'
        assert n_neighbors < len(X), '`n_neighbors` must be smaller than `n_sample`.'
        mi = _mi_knn(X, Y, n_neighbors)
    elif mi_estimator == 'plugin':
        assert is_discrete, 'When using plug-in estimator, all features must be discrete.'
        mi = _mi_plugin(X, Y)
    else:
        raise ValueError('`%s` is not implemented.' % mi_estimator)

    return max(mi, 0) if force_non_negative else mi


def conditional_mutual_information(X, Y, Z, mi_estimator='auto', is_discrete='auto', force_non_negative=False,
                                   **kwargs):
    """Estimate conditional mutual information between ``X`` and ``Y`` given ``Z``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features_x) or (n_samples)
        Observations of a conditioned variable.
    Y : array-like, shape (n_samples, n_features_y) or (n_samples)
        Observations of the other conditioned variable.
    Z : array-like, shape (n_samples, n_features_z) or (n_samples)
        Observations of the conditioning variable.
    mi_estimator : {'knn', 'dr', 'plugin', 'auto'}, default 'auto'
        MI estimator. If 'auto', MI estimator will be selected depending on whether all features are purely discrete or
        not. If purely discrete, 'plugin' estimator will be used. Otherwise, 'knn' estimator will be selected.
    is_discrete : {'auto', bool}, default 'auto'
        If ``bool``, then it determines whether to consider all features purely discrete or purely continuous. If
        'auto', a column which contains duplicate elements will be considered discrete.
    force_non_negative : bool, default False
        If ``True``, the result will be taken max with zero.
    kwargs : dict, default None
        Optional parameters for MI estimation.

    Returns
    -------
    cmi : float
        Estimated conditional mutual information between ``X`` and ``Y``, given ``Z``.
    """
    if np.size(Z) == 0:
        return mutual_information(X, Y, mi_estimator, is_discrete, force_non_negative, **kwargs)
    assert len(X) == len(Y) == len(Z), 'X, Y and Z must have the same length.'
    X = np.atleast_2d(X.T).T
    Z = np.atleast_2d(Z.T).T
    XZ = np.hstack([X, Z])
    mi_xz_y = mutual_information(XZ, Y, mi_estimator, is_discrete, **kwargs)
    mi_y_z = mutual_information(Y, Z, mi_estimator, is_discrete, **kwargs)
    cmi = mi_xz_y - mi_y_z
    return max(cmi, 0) if force_non_negative else cmi
