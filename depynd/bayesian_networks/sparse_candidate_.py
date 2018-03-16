import numpy as np

from ..mutual_information import conditional_mutual_information


def iamb(X, lamb=0.0, method=None, options=None):
    """Search Markov blanket in a Bayesian network using
       Incremental Association Markov Blanket algorithm [1]_.
    Parameters
    ----------
    X : array, shape (n_samples, d)
        Observations of variables.
    lamb: float
        Threshold for independence test.
    method: str, default 'knn'
        Method for MI estimation.
    options : dict, default None
        Optional parameters for MI estimation.
    Returns
    -------
    mb : array, shape (d, d)
        Estimated Markov blanket.
    References
    ----------
    .. [1] Tsamardinos, Ioannis, et al. "Algorithms for
           Large Scale Markov Blanket Discovery." FLAIRS
           conference. Vol. 2. 2003.
    """
    n, d = X.shape
    mb = np.zeros([d, d], dtype=bool)
    while True:
        cmi_max = -np.inf
        for i in range(d):
            x = X[:, i]
            z = X[:, mb[i]]
            non_mb = ~mb[i] & (np.arange(d) != i)
            for j in non_mb.nonzero()[0]:
                y = X[:, j]
                cmi = conditional_mutual_information(x, y, z, method, options)
                if cmi_max < cmi:
                    cmi_max = cmi
                    idx_max = i, j

        if cmi_max <= lamb:
            break

        mb[idx_max] = 1

    for i in range(d):
        x = X[:, i]
        for j in mb[i].nonzero()[0]:
            other_mb = mb[i] & (np.arange(d) != j)
            y = X[:, j]
            z = X[:, other_mb]
            cmi = conditional_mutual_information(x, y, z, method, options)
            if cmi <= lamb:
                mb[i, j] = 0

    return mb


def mmpc(X, lamb=0.0, method=None, options=None):
    """Search parents and children in a Bayesian network.
    Parameters
    ----------
    X : array, shape (n_samples, d)
        Observations of variables.
    lamb: float
        Threshold for independence test.
    method: str, default 'knn'
        Method for MI estimation.
    options : dict, default None
        Optional parameters for MI estimation.
    Returns
    ----------
    pc : array, shape (d, d)
        Estimated parents and children.
    """
    n, d = X.shape
    pc = np.zeros([d, d], dtype=bool)
    for i in range(d):
        x = X[:, i]
        non_pc = ~pc[i] & (np.arange(d) != i)
        for j in non_pc.nonzero()[0]:
            y = X[:, j]
            z = X[:, pc[i]]
            cmi = conditional_mutual_information(x, y, z, method=method, options=options)
            if cmi > lamb:
                pc[i, j] = 1

        for j in pc[i].nonzero()[0]:
            other_pc = pc[i] & (np.arange(d) != j)
            y = X[:, j]
            z = X[:, other_pc]
            cmi = conditional_mutual_information(x, y, z, method=method, options=options)
            if cmi <= lamb:
                pc[i, j] = 0

    return pc & pc.T
