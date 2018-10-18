import numpy as np

from depynd.information import mutual_information, conditional_mutual_information


def _gsmple(X, lamb=0.0, **kwargs):
    n, d = X.shape
    adj = np.zeros([d, d], dtype=bool)
    adj = _grow(adj, X, lamb, **kwargs)
    adj = _shrink(adj, X, lamb, **kwargs)
    return adj


def _grow(adj, X, lamb, **kwargs):
    # Initialize CMI cache matrix
    n, d = X.shape
    cmis = np.zeros([d, d])
    cmis[np.eye(d, dtype=bool)] = -np.inf
    for i in range(d):
        x = X[:, i]
        for j in range(i):
            y = X[:, j]
            cmis[i, j] = cmis[j, i] = mutual_information(x, y, **kwargs)

    while np.count_nonzero(adj) < d ** 2 - d:
        scores = cmis + cmis.T
        imax, jmax = np.unravel_index(np.argmax(scores), scores.shape)
        if scores[imax, jmax] <= lamb:
            return adj
        adj[imax, jmax] = adj[jmax, imax] = 1
        cmis[imax, jmax] = cmis[jmax, imax] = -np.inf

        # Re-compute CMIs
        for i in (imax, jmax):
            x = X[:, i]
            z = X[:, adj[i]]
            non_adj = ~adj[i] & (np.arange(d) != i)
            for j in non_adj.nonzero()[0]:
                y = X[:, j]
                cmis[i, j] = conditional_mutual_information(x, y, z, **kwargs)

    return adj


def _shrink(adj, X, lamb, **kwargs):
    # Initialize CMI cache matrix
    n, d = X.shape
    cmis = np.zeros([d, d])
    cmis[np.eye(d, dtype=bool)] = np.inf
    for i in range(d):
        x = X[:, i]
        for j in adj[i].nonzero()[0]:
            other_adj_i = adj[i] & (np.arange(d) != j)
            y = X[:, j]
            z = X[:, other_adj_i]
            cmis[i, j] = conditional_mutual_information(x, y, z, **kwargs)

    while np.count_nonzero(adj) > 0:
        scores = cmis + cmis.T
        imin, jmin = np.unravel_index(np.argmin(scores), scores.shape)
        if scores[imin, jmin] > lamb:
            return adj
        adj[imin, jmin] = adj[jmin, imin] = 0
        cmis[imin, jmin] = cmis[jmin, imin] = np.inf

        # Re-compute CMIs
        for i in range(d):
            x = X[:, i]
            for j in adj[i].nonzero()[0]:
                other_adj_i = adj[i] & (np.arange(d) != j)
                y = X[:, j]
                z = X[:, other_adj_i]
                cmis[i, j] = conditional_mutual_information(x, y, z, **kwargs)

    return adj
