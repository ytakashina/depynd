import numpy as np

from depynd.information import conditional_mutual_information


def fgplm(X, lamb=0.0, method=None, options=None):
    n, d = X.shape
    mb = np.zeros([d, d], dtype=bool)
    while True:
        vmax = -np.inf
        imax = (0, 1)
        for i in range(d):
            x = X[:, [i]]
            z = X[:, mb[i]]
            non_mb = ~mb[i] & (np.arange(d) != i)
            for j in non_mb.nonzero()[0]:
                if i == j:
                    continue
                y = X[:, [j]]
                cmi = conditional_mutual_information(x, y, z, method, options)
                if vmax < cmi:
                    vmax = cmi
                    imax = (i, j)

        if vmax <= lamb or np.count_nonzero(mb) == d * (d - 1) / 2:
            return mb

        mb[imax] = 1
        mb[tuple(reversed(imax))] = 1
