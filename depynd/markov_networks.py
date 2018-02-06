import numpy as np
from .mutual_information import conditional_mutual_information


def gsmn(X, lamb=0.0, method=None, options=None):
    n, d = X.shape
    adj = np.zeros([d, d], dtype=bool)
    for i in range(d):
        x = X[:, i]
        non_adj = ~adj[i] & (np.arange(d) != i)
        for j in non_adj.nonzero()[0]:
            y = X[:, j]
            z = X[:, adj[i]]
            cmi = conditional_mutual_information(x, y, z, method=method, options=options)
            if cmi > lamb:
                adj[i, j] = 1
                adj[j, i] = 1

        for j in adj[i].nonzero()[0]:
            other_adj = adj[i] & (np.arange(d) != j)
            y = X[:, j]
            z = X[:, other_adj]
            cmi = conditional_mutual_information(x, y, z, method=method, options=options)
            if cmi <= lamb:
                adj[i, j] = 0
                adj[j, i] = 0

    return adj