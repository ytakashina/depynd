import numpy as np
from .mutual_information import conditional_mutual_information


def markov_blanket(adj, i):
    mb = list(adj[i])
    mb += [j for j, children in enumerate(adj) if i in children]
    mb += [j for j, children in enumerate(adj) for k in adj[i] if k in children]
    return list(set(mb) - set([i]))


def is_complete(adj):
    d = len(adj)
    return all([len(neighbor) == d for neighbor in adj])


def iamb(X, lamb=0.0, method=None, options=None):
    n, d = X.shape
    adj = [[] for i in range(d)]
    while True:
        max_cmi = -np.inf
        for i in range(d):
            non_mb = set(range(d)) - set(adj[i]) - set([i])
            x = X[:, [i]]
            z = X[:, adj[i]]
            for j in non_mb:
                y = X[:, [j]]
                cmi = conditional_mutual_information(x, y, z, method, options)
                if max_cmi < cmi:
                    max_cmi = cmi
                    max_pair = i, j

        if max_cmi <= lamb or is_complete(adj):
            break

        adj[max_pair[0]] += [max_pair[1]]

    for i in range(d):
        x = X[:, [i]]
        for j in adj[i]:
            other_mb = list(set(adj[i]) - set([j]))
            y = X[:, [j]]
            z = X[:, other_mb]
            cmi = conditional_mutual_information(x, y, z, method, options)
            if cmi <= lamb:
                adj[i].remove(j)

    return adj
