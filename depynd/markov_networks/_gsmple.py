import numpy as np

from depynd.information import conditional_mutual_information


def gsmple(X, lamb=0.0, **kwargs):
    n, d = X.shape
    adj = np.zeros([d, d], dtype=bool)
    while True:
        vmax = -np.inf
        for i in range(d):
            x = X[:, [i]]
            z = X[:, adj[i]]
            non_adj = ~adj[i] & (np.arange(d) != i)
            for j in non_adj.nonzero()[0]:
                if i <= j:
                    continue
                y = X[:, [j]]
                w = X[:, adj[j]]
                cmi = conditional_mutual_information(x, y, z, **kwargs)
                cmi += conditional_mutual_information(x, y, w, **kwargs)
                if vmax < cmi:
                    vmax = cmi
                    imax, jmax = i, j

        if vmax <= lamb:
            break

        adj[imax, jmax] = adj[jmax, imax] = 1

        if np.count_nonzero(adj) == d ** 2 - d:
            break

    while True:
        vmin = np.inf
        for i in range(d):
            x = X[:, [i]]
            for j in adj[i].nonzero()[0]:
                if i <= j:
                    continue
                other_adj_i = adj[i] & (np.arange(d) != j)
                other_adj_j = adj[j] & (np.arange(d) != i)
                y = X[:, [j]]
                z = X[:, other_adj_i]
                w = X[:, other_adj_j]
                cmi = conditional_mutual_information(x, y, z, **kwargs)
                cmi += conditional_mutual_information(x, y, w, **kwargs)
                if vmin > cmi:
                    vmin = cmi
                    imin, jmin = i, j

        if vmin > lamb:
            return adj

        adj[imin, jmin] = adj[jmin, imin] = 0

        if np.count_nonzero(adj) == 0:
            return adj
