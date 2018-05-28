import numpy as np

from depynd.information import conditional_mutual_information


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
