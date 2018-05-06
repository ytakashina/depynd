import numpy as np

from depynd.information import conditional_mutual_information


def gplm(X, lamb=0.0, method=None, options=None):
    n, d = X.shape
    adj = np.zeros([d, d], dtype=bool)
    while True:
        vmax = -np.inf
        for i in range(d):
            x = X[:, [i]]
            z = X[:, adj[i]]
            non_adj = ~adj[i] & (np.arange(d) != i)
            for j in non_adj.nonzero()[0]:
                if i == j:
                    continue
                y = X[:, [j]]
                cmi = conditional_mutual_information(x, y, z, method, options)
                if vmax < cmi:
                    vmax = cmi
                    imax, jmax = i, j

        if vmax <= lamb:
            break

        adj[imax, jmax] = adj[jmax, imax] = 1

        if np.count_nonzero(adj) == d * (d - 1) / 2:
            break

    while True:
        vmin = np.inf
        for i in range(d):
            x = X[:, [i]]
            for j in adj[i].nonzero()[0]:
                other_adj = adj[i] & (np.arange(d) != j)
                y = X[:, [j]]
                z = X[:, other_adj]
                cmi = conditional_mutual_information(x, y, z, method, options)
                if vmin > cmi:
                    vmin = cmi
                    imin, jmin = i, j

        if vmin >= lamb:
            return adj

        adj[imin, jmin] = adj[jmin, imin] = 0

        if np.count_nonzero(adj) == 0:
            return adj
