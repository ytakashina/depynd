import numpy as np
from scipy.stats import multinomial

from depynd.markov_networks import jose

adj0 = np.zeros([9, 9])
for i in range(3, 9):
    adj0[i, i - 3] = adj0[i - 3, i] = 1
adj0[0, 1] = adj0[1, 0] = 1
adj0[1, 2] = adj0[2, 1] = 1
adj0[3, 4] = adj0[4, 3] = 1
adj0[4, 5] = adj0[5, 4] = 1
adj0[6, 7] = adj0[7, 6] = 1
adj0[7, 8] = adj0[8, 7] = 1


def sample(adj, x, i, size=1):
    p = 1 / (1 + np.exp(-adj[i] @ x))
    sample = multinomial(1, [p, 1 - p]).rvs(size)
    return sample.argmax()


def gibbs_sampling(adj, L=100):
    X = []
    x = np.ones(9, dtype=int)
    for t in range(L):
        for i in range(9):
            x[i] = sample(adj, x, i)
        X.append(np.copy(x))
    return np.vstack(X)


X = gibbs_sampling(adj0)


def test_jose():
    jose(X, 4.53999298e-05)
