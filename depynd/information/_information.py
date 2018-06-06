import numpy as np
from sklearn.utils.validation import check_array

from depynd.information import _mi_dr, _mi_knn


def mutual_information(X, Y, mi_estimator='knn', force_non_negative=False, **kwargs):
    """Estimate mutual information between ``X`` and ``Y``.

    Parameters
    ----------
    X : array-like, shape (n_samples, d_x) or (n_samples)
        Observations of a variable.
    Y : array-like, shape (n_samples, d_y) or (n_samples)
        Observations of the other variable.
    mi_estimator : {'knn', 'dr'}, default 'knn'
        MI estimator.
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

    if mi_estimator == 'dr':
        sigma = kwargs.get('sigma', 1)
        n_bases = kwargs.get('n_bases', 200)
        maxiter = kwargs.get('maxiter', 1000)
        assert sigma > 0, '`sigma` must be positive.'
        assert isinstance(n_bases, (int, np.integer)) and n_bases > 0, '`n_bases` must be a positive integer.'
        assert isinstance(maxiter, (int, np.integer)) and maxiter > 0, '`maxiter` must be a positive integer.'
        mi = _mi_dr(X, Y, sigma=sigma, n_bases=n_bases, maxiter=maxiter)
    elif mi_estimator == 'knn':
        k = kwargs.get('k', 3)
        assert isinstance(k, (int, np.integer)) and k > 0, '`k` must be a positive integer.'
        assert k < len(X), '`k` must be smaller than `n_sample`.'
        mi = _mi_knn(X, Y, k)
    else:
        raise ValueError('`%s` is not implemented.' % mi_estimator)

    return max(mi, 0) if force_non_negative else mi


def conditional_mutual_information(X, Y, Z, mi_estimator='knn', force_non_negative=False, **kwargs):
    """Estimate conditional mutual information between ``X`` and ``Y`` given ``Z``.

    Parameters
    ----------
    X : array-like, shape (n_samples, d_x) or (n_samples)
        Observations of a conditioned variable.
    Y : array-like, shape (n_samples, d_y) or (n_samples)
        Observations of the other conditioned variable.
    Z : array-like, shape (n_samples, d_z) or (n_samples)
        Observations of the conditioning variable.
    mi_estimator : {'knn', 'dr'}, default 'knn'
        MI estimator.
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
        return mutual_information(X, Y, mi_estimator, force_non_negative, **kwargs)
    assert len(X) == len(Y) == len(Z), 'X, Y and Z must have the same length.'
    X = np.atleast_2d(X.T).T
    Z = np.atleast_2d(Z.T).T
    XZ = np.hstack([X, Z])
    mi_xz_y = mutual_information(XZ, Y, mi_estimator, force_non_negative, **kwargs)
    mi_y_z = mutual_information(Y, Z, mi_estimator, force_non_negative, **kwargs)
    cmi = mi_xz_y - mi_y_z
    return max(cmi, 0) if force_non_negative else cmi
