import copy
import numpy as np
from .mutual_information import mutual_information, conditional_mutual_information


def markov_blanket(adj, i):
    mb = list(adj[i])
    mb += [j for j, children in enumerate(adj) if i in children]
    mb += [j for j, children in enumerate(adj) for k in adj[i] if k in children]
    return list(set(mb) - set([i]))


def cyclic(adj):
    path = set()

    def visit(i):
        path.add(i)
        for j in adj[i]:
            if j in path or visit(j):
                return True
        path.remove(i)
        return False

    return any(visit(i) for i, _ in enumerate(adj))


def iamb(X, lamb=0.0, method=None, options=None):
    """Search Markov blanket in a Bayesian network.
    Parameters
    ----------
    X : array, shape (n_samples, d)
        Variable.
    method: str, default 'knn'
        Method for MI estimation.
    options : dict, default None
        Optional parameters for MI estimation.
    Returns
    -------
    mb : list of list
        Estimated Markov blanket.
    """
    n, d = X.shape
    mb = [[] for i in range(d)]
    while True:
        max_cmi = -np.inf
        for i in range(d):
            non_mb = set(range(d)) - set(mb[i]) - set([i])
            x = X[:, [i]]
            z = X[:, mb[i]]
            for j in non_mb:
                y = X[:, [j]]
                cmi = conditional_mutual_information(x, y, z, method, options)
                if max_cmi < cmi:
                    max_cmi = cmi
                    max_pair = i, j

        if max_cmi <= lamb or is_complete(mb):
            break

        mb[max_pair[0]] += [max_pair[1]]

    for i in range(d):
        x = X[:, [i]]
        for j in mb[i]:
            other_mb = list(set(mb[i]) - set([j]))
            y = X[:, [j]]
            z = X[:, other_mb]
            cmi = conditional_mutual_information(x, y, z, method, options)
            if cmi <= lamb:
                mb[i].remove(j)

    return mb


def mmpc(X, lamb=0.0, method=None, options=None):
    """Search parents and children in a Bayesian network.
    Parameters
    ----------
    X : array, shape (n_samples, d)
        Variable.
    method: str, default 'knn'
        Method for MI estimation.
    options : dict, default None
        Optional parameters for MI estimation.
    Returns
    -------
    pc : list of list
        Estimated parents and children.
    """
    n, d = X.shape
    pc = [[] for i in range(d)]
    for i in range(d):
        x = X[:, [i]]
        non_pc = set(range(d)) - set(pc[i]) - set([i])
        for j in non_pc:
            y = X[:, [j]]
            z = X[:, pc[i]]
            cmi = conditional_mutual_information(x, y, z, method=method, options=options)
            if cmi > lamb:
                pc[i] += [j]

        for j in pc[i]:
            other_pc = list(set(pc[i]) - set([j]))
            y = X[:, [j]]
            z = X[:, other_pc]
            cmi = conditional_mutual_information(x, y, z, method=method, options=options)
            if cmi < lamb:
                pc[i].remove(j)

    for i in range(d):
        for j in pc[i]:
            if not i in pc[j]:
                pc[i].remove(j)

    return pc


def score_graph(adj, X, criterion, method=None, options=None):
    n, d = X.shape
    if criterion == 'mi':
        mis = [mutual_information(X[:, [i]], X[:, adj[i]], method, options) for i in range(d)]
        return np.sum(mis)
    elif criterion == 'bdeu':
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def mmhc(X, lamb=0.0, criterion='mi', method=None, options=None):
    n, d = X.shape
    adj = [[] for i in range(d)]
    pc = mmpc(X, lamb, method, options)
    score_prev = -np.inf
    while True:
        adj_max = None
        score_max = -np.inf
        for i in range(d):
            x = X[:, [i]]
            for j in range(d):
                if i in adj[j] or j in adj[i]:
                    # Remove
                    adj_new = copy.deepcopy(adj)
                    if i in adj[j]:
                        adj_new[j].remove(i)
                    else:
                        adj_new[i].remove(j)
                    score = score_graph(adj_new, X, criterion, method, options)
                    if score > score_max:
                        score_max = score
                        adj_max = copy.deepcopy(adj_new)
                    # Reverse
                    adj_new = copy.deepcopy(adj)
                    if i in adj[j]:
                        adj_new[j].remove(i)
                        adj_new[i] += [j]
                    else:
                        adj_new[i].remove(j)
                        adj_new[j] += [i]
                    if cyclic(adj_new):
                        continue
                    score = score_graph(adj_new, X, criterion, method, options)
                    if score > score_max:
                        score_max = score
                        adj_max = copy.deepcopy(adj_new)
                elif j in pc[i]:
                    # Add
                    adj_new = copy.deepcopy(adj)
                    adj_new[i] += [j]
                    if cyclic(adj_new):
                        continue
                    score = score_graph(adj_new, X, criterion, method, options)
                    if score > score_max:
                        score_max = score
                        adj_max = copy.deepcopy(adj_new)

        if score_max - score_prev <= lamb:
            return adj

        adj = copy.deepcopy(adj_max)
        score_prev = score_max