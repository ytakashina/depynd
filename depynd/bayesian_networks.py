import numpy as np
from itertools import product

from .mutual_information import mutual_information, conditional_mutual_information


def cyclic(adj):
    path = set()

    def visit(i):
        path.add(i)
        for j in adj[i].nonzero()[0]:
            if j in path or visit(j):
                return True
        path.remove(i)
        return False
    return any(visit(i) for i, _ in enumerate(adj))


def iamb(X, lamb=0.0, method=None, options=None):
    """Search Markov blanket in a Bayesian network using
       Incremental Association Markov Blanket algorithm [1]_.
    Parameters
    ----------
    X : array, shape (n_samples, d)
        Observations of variables.
    lamb: float
        Threshold for independence tests.
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


def _score_graph(adj, X, method=None, options=None):
    # Should be replaced mutual information test (MIT) criterion,
    # where such criterion has not been developed yet in either
    # purely continuous or discrete-continuous mixed settings.
    n, d = X.shape
    mis = [mutual_information(X[:, i], X[:, adj[i]], method, options) for i in range(d)]
    return np.sum(mis)


def _score_graph_diff(adj, X, idx, operation, method=None, options=None):
    i, j = idx
    if operation == 'add':
        diff = -mutual_information(X[:, i], X[:, adj[i]], method, options)
        adj[i, j] = 1
        diff += mutual_information(X[:, i], X[:, adj[i]], method, options)
        adj[i, j] = 0
    elif operation == 'delete':
        diff = -mutual_information(X[:, i], X[:, adj[i]], method, options)
        adj[i, j] = 0
        diff += mutual_information(X[:, i], X[:, adj[i]], method, options)
        adj[i, j] = 1
    elif operation == 'reverse':
        diff = -mutual_information(X[:, i], X[:, adj[i]], method, options)
        diff -= mutual_information(X[:, j], X[:, adj[j]], method, options)
        adj[i, j] = 0
        adj[j, i] = 1
        diff += mutual_information(X[:, i], X[:, adj[i]], method, options)
        diff += mutual_information(X[:, j], X[:, adj[j]], method, options)
        adj[i, j] = 1
        adj[j, i] = 0
    return diff


def mmhc(X, lamb=0.0, method=None, options=None, verbose=False):
    """Greedily search structure of a Bayesian network using
       max-min hill-climbing algorithm [1]_.
    Parameters
    ----------
    X : array, shape (n_samples, d)
        Observations of variables.
    lamb: float
        Threshold for independence tests.
    method: str, default 'knn'
        Method for MI estimation.
    options : dict, default None
        Optional parameters for MI estimation.
    verbose: bool, default False
        Enable verbose output.
    Returns
    ----------
    adj : array, shape (d, d)
        Estimated structure of the Bayesian network.
    References
    ----------
    .. [1] Tsamardinos, Ioannis, Laura E. Brown, and Constantin
           F. Aliferis. "The max-min hill-climbing Bayesian
           network structure learning algorithm." Machine
           learning 65.1 (2006): 31-78.
    """
    n, d = X.shape
    adj = np.zeros([d, d], dtype=bool)
    pc = mmpc(X, lamb, method, options)
    score_prev = -np.inf
    while True:
        score_max = -np.inf
        # Same as "for i in range(d): for j in range(d):"
        for i, j in product(range(d), range(d)):
            if adj[i, j]:
                # Delete
                adj[i, j] = 0
                score = _score_graph(adj, X, method, options)
                if score > score_max:
                    score_max = score
                    idx_max = i, j
                    operation = 'delete'
                # Reverse
                adj[j, i] = 1
                if not cyclic(adj):
                    score = _score_graph(adj, X, method, options)
                    if score > score_max:
                        score_max = score
                        idx_max = i, j
                        operation = 'reverse'
                adj[i, j] = 1
                adj[j, i] = 0
            elif pc[i, j] and not adj[j, i]:
                # Add
                adj[i, j] = 1
                if not cyclic(adj):
                    score = _score_graph(adj, X, method, options)
                    if score > score_max:
                        score_max = score
                        idx_max = i, j
                        operation = 'add'
                adj[i, j] = 0

        if score_max - score_prev <= lamb:
            return adj

        score_prev = score_max
        if verbose:
            print('%s %s: %f' % (operation, idx_max, score_max))
        if operation == 'add':
            adj[idx_max] = 1
        elif operation == 'delete':
            adj[idx_max] = 0
        elif operation == 'reverse':
            adj[idx_max] = 0
            adj[idx_max[::-1]] = 1
