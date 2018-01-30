import numpy as np
from .mutual_information import conditional_mutual_information


def gsmn(X, lamb=0.0, method=None, options=None):
    n, d = X.shape
    adj = [[] for i in range(d)]
    for i in range(d):
        x = X[:, [i]]
        non_adj = set(range(d)) - set(adj[i]) - set([i])
        for j in non_adj:
            y = X[:, [j]]
            z = X[:, adj[i]]
            cmi = conditional_mutual_information(x, y, z, method=None, options=None)
            if cmi > lamb:
                adj[i] += [j]

        for j in adj[i]:
            other_adj = list(set(adj[i]) - set([j]))
            y = X[:, [j]]
            z = X[:, other_adj]
            cmi = conditional_mutual_information(x, y, z, method=None, options=None)
            if cmi < lamb:
                adj[i].remove(j)

    return adj
